"""A small dependency-aware workflow engine (a DAG runner).

The triage pipeline is not one function: it is a set of named stages, each
declaring which other stages it needs.  This module knows nothing about
tickets - it only knows how to run a graph of stages correctly.

What it gives us that a straight-line function does not
------------------------------------------------------
1. **Declared dependencies.**  A stage says what it needs; the engine works
   out the order.  Adding a stage never means re-reading the call order.
2. **Parallelism for free.**  Stages that do not depend on each other are
   grouped into the same "level" and run on a thread pool.  Classification and
   aspect sentiment both read only the raw ticket, so they run together.
3. **Resumability.**  Every stage result is persisted as it completes.  Re-run
   the same transaction and the already-completed stages are loaded from the
   store instead of being executed again, so a failed stage can be retried on
   its own.
4. **Inspectable state.**  ``GET /ticket/{id}/status`` reads the same rows the
   engine writes, so it shows real progress rather than a guess.

Execution levels
----------------
Level 0 = stages with no dependencies.  Level N = stages all of whose
dependencies are in levels below N.  This is Kahn's topological sort, kept
grouped by level instead of flattened, which is exactly what tells us what may
run at the same time::

    level 0:  classify_ticket   analyse_sentiment      <- run in parallel
    level 1:  score_urgency
    level 2:  select_configuration
    level 3:  retrieve_knowledge
    level 4:  run_agent
    level 5:  compose_response
"""

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from app.workflow.state_store import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_SKIPPED,
    WorkflowStateStore,
)


class StageContext:
    """What a stage is given when it runs."""

    def __init__(self, transaction_id: str, inputs: dict) -> None:
        self.transaction_id = transaction_id
        # The original request (subject, body, customer tier).
        self.inputs = inputs
        # stage name -> that stage's output dictionary.
        self.outputs: dict[str, dict] = {}

    def output_of(self, stage_name: str) -> dict:
        """Read the output of an upstream stage.

        Raises if the stage has not run, which turns a wrong dependency
        declaration into an immediate, obvious error instead of a silent None.
        """
        if stage_name not in self.outputs:
            raise KeyError(
                "stage '" + stage_name + "' has no output yet; is it missing from depends_on?"
            )
        return self.outputs[stage_name]


class Stage:
    """One named unit of work with declared dependencies."""

    def __init__(
        self,
        name: str,
        depends_on: list[str],
        run: Callable[[StageContext], dict],
        description: str = "",
    ) -> None:
        self.name = name
        self.depends_on = list(depends_on)
        # The function that does the work: takes the context, returns a
        # JSON-serialisable dictionary.
        self.run = run
        self.description = description


class WorkflowResult:
    """The outcome of running the whole graph once."""

    def __init__(self, transaction_id: str) -> None:
        self.transaction_id = transaction_id
        self.outputs: dict[str, dict] = {}
        self.succeeded: bool = True
        self.failed_stage: str = ""
        self.error: str = ""
        # Stage names that were loaded from the store instead of executed.
        self.reused_stages: list[str] = []
        self.executed_stages: list[str] = []
        self.duration_seconds: float = 0.0


class WorkflowEngine:
    """Validates a set of stages and runs them in dependency order."""

    def __init__(
        self, stages: list[Stage], store: WorkflowStateStore, max_workers: int = 4
    ) -> None:
        self.stages_by_name: dict[str, Stage] = {}
        for stage in stages:
            if stage.name in self.stages_by_name:
                raise ValueError("duplicate stage name: " + stage.name)
            self.stages_by_name[stage.name] = stage

        self.store = store
        self.max_workers = max_workers

        self._validate_dependencies()
        self.levels = self._compute_levels()

    # ------------------------------------------------------------------
    # Graph validation and ordering
    # ------------------------------------------------------------------
    def _validate_dependencies(self) -> None:
        """Fail loudly if a stage depends on something that does not exist."""
        for stage in self.stages_by_name.values():
            for dependency in stage.depends_on:
                if dependency not in self.stages_by_name:
                    raise ValueError(
                        "stage '" + stage.name + "' depends on unknown stage '" + dependency + "'"
                    )
                if dependency == stage.name:
                    raise ValueError("stage '" + stage.name + "' depends on itself")

    def _compute_levels(self) -> list[list[str]]:
        """Group stages into levels that may run in parallel (Kahn's algorithm).

        Repeatedly take every stage whose dependencies are all already placed
        in an earlier level.  If a pass places nothing but stages remain, the
        leftovers form a cycle.
        """
        remaining = set(self.stages_by_name.keys())
        placed: set[str] = set()
        levels: list[list[str]] = []

        while len(remaining) > 0:
            current_level: list[str] = []

            for name in sorted(remaining):
                stage = self.stages_by_name[name]
                dependencies_ready = True
                for dependency in stage.depends_on:
                    if dependency not in placed:
                        dependencies_ready = False
                        break
                if dependencies_ready:
                    current_level.append(name)

            if len(current_level) == 0:
                raise ValueError(
                    "the workflow graph contains a cycle involving: " + ", ".join(sorted(remaining))
                )

            levels.append(current_level)
            for name in current_level:
                remaining.remove(name)
                placed.add(name)

        return levels

    def describe(self) -> list[dict[str, object]]:
        """A readable description of the graph, used by the API and the README."""
        description = []
        for level_number in range(len(self.levels)):
            stage_details = []
            for name in self.levels[level_number]:
                stage = self.stages_by_name[name]
                stage_details.append(
                    {
                        "name": stage.name,
                        "depends_on": stage.depends_on,
                        "description": stage.description,
                    }
                )
            description.append(
                {
                    "level": level_number,
                    "runs_in_parallel": len(stage_details) > 1,
                    "stages": stage_details,
                }
            )
        return description

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------
    def run(self, transaction_id: str, inputs: dict, resume: bool = True) -> WorkflowResult:
        """Run every stage in dependency order and return the collected outputs.

        With ``resume=True`` a stage that is already marked completed for this
        transaction is loaded from the store rather than executed, which is
        what makes "re-run only the failed stage" work.
        """
        started_at = time.perf_counter()
        result = WorkflowResult(transaction_id)
        context = StageContext(transaction_id, inputs)

        for level in self.levels:
            # If an earlier level failed, mark the rest as skipped and stop.
            if not result.succeeded:
                for name in level:
                    self.store.skip_stage(
                        transaction_id, name, "upstream stage '" + result.failed_stage + "' failed"
                    )
                continue

            # Work out which stages in this level actually need to run.
            stages_to_execute: list[str] = []
            for name in level:
                cached_output = self._cached_output(transaction_id, name, resume)
                if cached_output is not None:
                    context.outputs[name] = cached_output
                    result.reused_stages.append(name)
                else:
                    stages_to_execute.append(name)

            if len(stages_to_execute) == 0:
                continue

            level_results = self._execute_level(stages_to_execute, context)

            # Fold the results into the context, stopping at the first failure.
            for name in stages_to_execute:
                output, error = level_results[name]
                if error == "":
                    context.outputs[name] = output
                    result.executed_stages.append(name)
                elif result.succeeded:
                    result.succeeded = False
                    result.failed_stage = name
                    result.error = error

        result.outputs = context.outputs
        result.duration_seconds = time.perf_counter() - started_at
        return result

    def _cached_output(self, transaction_id: str, stage_name: str, resume: bool) -> dict | None:
        """Return a previously stored successful output, if we are allowed to reuse it."""
        if not resume:
            return None
        record = self.store.get_stage(transaction_id, stage_name)
        if record is None:
            return None
        if record.status != STATUS_COMPLETED:
            return None
        return record.output

    def _execute_level(
        self, stage_names: list[str], context: StageContext
    ) -> dict[str, tuple[dict, str]]:
        """Run every stage of one level, in parallel when there is more than one.

        Returns ``stage name -> (output, error message)``; an empty error
        message means success.
        """
        results: dict[str, tuple[dict, str]] = {}

        # A single stage does not need a thread pool.
        if len(stage_names) == 1:
            name = stage_names[0]
            results[name] = self._execute_stage(name, context)
            return results

        worker_count = min(self.max_workers, len(stage_names))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {}
            for name in stage_names:
                futures[name] = pool.submit(self._execute_stage, name, context)
            for name in stage_names:
                results[name] = futures[name].result()

        return results

    def _execute_stage(self, stage_name: str, context: StageContext) -> tuple[dict, str]:
        """Run one stage, recording its state before and after."""
        stage = self.stages_by_name[stage_name]

        self.store.start_stage(context.transaction_id, stage_name)
        try:
            output = stage.run(context)
        except Exception as error:  # noqa: BLE001 - a stage failure must not kill the run
            message = type(error).__name__ + ": " + str(error)
            self.store.fail_stage(context.transaction_id, stage_name, message)
            return {}, message

        if not isinstance(output, dict):
            message = "stage '" + stage_name + "' must return a dictionary"
            self.store.fail_stage(context.transaction_id, stage_name, message)
            return {}, message

        self.store.complete_stage(context.transaction_id, stage_name, output)
        return output, ""


# Re-exported so callers can compare statuses without importing the store too.
__all__ = [
    "Stage",
    "StageContext",
    "WorkflowEngine",
    "WorkflowResult",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_SKIPPED",
]

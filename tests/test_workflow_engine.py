"""Unit tests for the workflow engine and its state store.

The engine is the part of the system most likely to break silently, so these
tests pin down all four of its promises: correct dependency order, real
parallelism, resumability, and clean failure handling.
"""

import threading
import time

import pytest

from app.workflow.dag import Stage, StageContext, WorkflowEngine
from app.workflow.state_store import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_SKIPPED,
    WorkflowStateStore,
)


def make_stage(name: str, depends_on: list[str], value: int = 1) -> Stage:
    """A stage that simply records its own name and a value."""

    def run(context: StageContext) -> dict:
        return {"name": name, "value": value}

    return Stage(name, depends_on, run)


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------


def test_independent_stages_share_the_first_level(temporary_store):
    engine = WorkflowEngine(
        [make_stage("a", []), make_stage("b", []), make_stage("c", ["a", "b"])], temporary_store
    )

    assert engine.levels == [["a", "b"], ["c"]]


def test_a_chain_produces_one_stage_per_level(temporary_store):
    engine = WorkflowEngine(
        [make_stage("third", ["second"]), make_stage("second", ["first"]), make_stage("first", [])],
        temporary_store,
    )

    assert engine.levels == [["first"], ["second"], ["third"]]


def test_a_diamond_graph_is_ordered_correctly(temporary_store):
    engine = WorkflowEngine(
        [
            make_stage("top", []),
            make_stage("left", ["top"]),
            make_stage("right", ["top"]),
            make_stage("bottom", ["left", "right"]),
        ],
        temporary_store,
    )

    assert engine.levels == [["top"], ["left", "right"], ["bottom"]]


def test_a_missing_dependency_is_rejected(temporary_store):
    with pytest.raises(ValueError) as error:
        WorkflowEngine([make_stage("a", ["does_not_exist"])], temporary_store)
    assert "unknown stage" in str(error.value)


def test_a_self_dependency_is_rejected(temporary_store):
    with pytest.raises(ValueError) as error:
        WorkflowEngine([make_stage("a", ["a"])], temporary_store)
    assert "depends on itself" in str(error.value)


def test_a_cycle_is_rejected(temporary_store):
    with pytest.raises(ValueError) as error:
        WorkflowEngine(
            [make_stage("a", ["b"]), make_stage("b", ["c"]), make_stage("c", ["a"])],
            temporary_store,
        )
    assert "cycle" in str(error.value)


def test_duplicate_stage_names_are_rejected(temporary_store):
    with pytest.raises(ValueError) as error:
        WorkflowEngine([make_stage("a", []), make_stage("a", [])], temporary_store)
    assert "duplicate" in str(error.value)


def test_describe_marks_the_parallel_level(temporary_store):
    engine = WorkflowEngine(
        [make_stage("a", []), make_stage("b", []), make_stage("c", ["a", "b"])], temporary_store
    )

    description = engine.describe()
    assert description[0]["runs_in_parallel"] is True
    assert description[1]["runs_in_parallel"] is False


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def test_stages_run_and_their_outputs_are_collected(temporary_store):
    def sum_stage(context: StageContext) -> dict:
        return {"total": context.output_of("a")["value"] + context.output_of("b")["value"]}

    engine = WorkflowEngine(
        [make_stage("a", [], 3), make_stage("b", [], 4), Stage("c", ["a", "b"], sum_stage)],
        temporary_store,
    )
    result = engine.run("tx-1", {})

    assert result.succeeded is True
    assert result.outputs["c"]["total"] == 7
    assert sorted(result.executed_stages) == ["a", "b", "c"]


def test_reading_an_undeclared_dependency_fails_loudly(temporary_store):
    def bad_stage(context: StageContext) -> dict:
        # "a" exists but was not declared as a dependency of this stage.
        return {"value": context.output_of("a")["value"]}

    engine = WorkflowEngine([make_stage("a", []), Stage("b", [], bad_stage)], temporary_store)
    result = engine.run("tx-undeclared", {})

    assert result.succeeded is False
    assert result.failed_stage == "b"
    assert "depends_on" in result.error


def test_independent_stages_really_run_at_the_same_time(temporary_store):
    thread_ids: list[int] = []
    barrier = threading.Barrier(2, timeout=5)

    def slow_stage(context: StageContext) -> dict:
        thread_ids.append(threading.get_ident())
        # Both stages must reach the barrier, which can only happen if they
        # are running concurrently. A sequential engine would time out here.
        barrier.wait()
        time.sleep(0.05)
        return {"done": True}

    engine = WorkflowEngine(
        [Stage("a", [], slow_stage), Stage("b", [], slow_stage)], temporary_store
    )
    result = engine.run("tx-parallel", {})

    assert result.succeeded is True
    assert len(set(thread_ids)) == 2


def test_inputs_are_visible_to_every_stage(temporary_store):
    def read_input(context: StageContext) -> dict:
        return {"subject": context.inputs["subject"]}

    engine = WorkflowEngine([Stage("a", [], read_input)], temporary_store)
    result = engine.run("tx-inputs", {"subject": "hello"})

    assert result.outputs["a"]["subject"] == "hello"


def test_a_stage_returning_a_non_dictionary_is_a_failure(temporary_store):
    engine = WorkflowEngine([Stage("a", [], lambda context: "not a dict")], temporary_store)
    result = engine.run("tx-bad-return", {})

    assert result.succeeded is False
    assert "must return a dictionary" in result.error


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_a_failing_stage_stops_its_dependents(temporary_store):
    def explode(context: StageContext) -> dict:
        raise ValueError("boom")

    engine = WorkflowEngine(
        [
            make_stage("first", []),
            Stage("second", ["first"], explode),
            make_stage("third", ["second"]),
        ],
        temporary_store,
    )
    result = engine.run("tx-fail", {})

    assert result.succeeded is False
    assert result.failed_stage == "second"
    assert "boom" in result.error

    statuses = {}
    for record in temporary_store.get_stages("tx-fail"):
        statuses[record.stage_name] = record.status

    assert statuses["first"] == STATUS_COMPLETED
    assert statuses["second"] == STATUS_FAILED
    assert statuses["third"] == STATUS_SKIPPED


def test_the_error_message_names_the_exception_type(temporary_store):
    def explode(context: StageContext) -> dict:
        raise KeyError("missing thing")

    engine = WorkflowEngine([Stage("a", [], explode)], temporary_store)
    result = engine.run("tx-error-type", {})

    assert "KeyError" in result.error


# ---------------------------------------------------------------------------
# Resuming
# ---------------------------------------------------------------------------


def test_completed_stages_are_reused_on_a_second_run(temporary_store):
    run_counter = {"count": 0}

    def counted(context: StageContext) -> dict:
        run_counter["count"] = run_counter["count"] + 1
        return {"count": run_counter["count"]}

    engine = WorkflowEngine([Stage("a", [], counted)], temporary_store)

    first = engine.run("tx-resume", {})
    second = engine.run("tx-resume", {})

    assert run_counter["count"] == 1
    assert first.executed_stages == ["a"]
    assert second.executed_stages == []
    assert second.reused_stages == ["a"]
    assert second.outputs["a"]["count"] == 1


def test_only_the_failed_stage_is_re_run(temporary_store):
    upstream_runs = {"count": 0}
    should_fail = {"value": True}

    def upstream(context: StageContext) -> dict:
        upstream_runs["count"] = upstream_runs["count"] + 1
        return {"ok": True}

    def flaky(context: StageContext) -> dict:
        if should_fail["value"]:
            raise RuntimeError("temporary outage")
        return {"ok": True}

    engine = WorkflowEngine(
        [Stage("upstream", [], upstream), Stage("flaky", ["upstream"], flaky)], temporary_store
    )

    first = engine.run("tx-retry", {})
    assert first.succeeded is False
    assert upstream_runs["count"] == 1

    # The outage clears and the pipeline is retried.
    should_fail["value"] = False
    second = engine.run("tx-retry", {})

    assert second.succeeded is True
    # The upstream stage was NOT executed a second time.
    assert upstream_runs["count"] == 1
    assert second.reused_stages == ["upstream"]
    assert second.executed_stages == ["flaky"]


def test_resume_false_re_runs_everything(temporary_store):
    run_counter = {"count": 0}

    def counted(context: StageContext) -> dict:
        run_counter["count"] = run_counter["count"] + 1
        return {"count": run_counter["count"]}

    engine = WorkflowEngine([Stage("a", [], counted)], temporary_store)
    engine.run("tx-force", {})
    engine.run("tx-force", {}, resume=False)

    assert run_counter["count"] == 2


def test_clearing_a_stage_makes_it_run_again(temporary_store):
    run_counter = {"count": 0}

    def counted(context: StageContext) -> dict:
        run_counter["count"] = run_counter["count"] + 1
        return {"count": run_counter["count"]}

    engine = WorkflowEngine([Stage("a", [], counted)], temporary_store)
    engine.run("tx-clear", {})
    temporary_store.clear_stage("tx-clear", "a")
    engine.run("tx-clear", {})

    assert run_counter["count"] == 2


# ---------------------------------------------------------------------------
# The state store on its own
# ---------------------------------------------------------------------------


def test_store_round_trips_a_ticket(tmp_path):
    store = WorkflowStateStore(tmp_path / "state.sqlite3")
    store.create_ticket("tx-9", {"subject": "hello"})

    ticket = store.get_ticket("tx-9")
    assert ticket["request"]["subject"] == "hello"
    assert ticket["feedback_score"] is None


def test_store_returns_none_for_an_unknown_ticket(tmp_path):
    store = WorkflowStateStore(tmp_path / "state.sqlite3")
    assert store.get_ticket("tx-missing") is None
    assert store.get_stage("tx-missing", "a") is None
    assert store.get_stages("tx-missing") == []


def test_store_records_feedback_and_reward(tmp_path):
    store = WorkflowStateStore(tmp_path / "state.sqlite3")
    store.create_ticket("tx-10", {"subject": "hello"})
    store.finish_ticket("tx-10", STATUS_COMPLETED, {"ok": True}, "config_a", "state_key", 1.25)
    store.record_feedback("tx-10", 1, 8.75)

    ticket = store.get_ticket("tx-10")
    assert ticket["status"] == STATUS_COMPLETED
    assert ticket["config_name"] == "config_a"
    assert ticket["latency_seconds"] == pytest.approx(1.25)
    assert ticket["feedback_score"] == 1
    assert ticket["reward"] == pytest.approx(8.75)


def test_stage_duration_is_measured(tmp_path):
    store = WorkflowStateStore(tmp_path / "state.sqlite3")
    store.start_stage("tx-11", "a")
    time.sleep(0.02)
    store.complete_stage("tx-11", "a", {"done": True})

    record = store.get_stage("tx-11", "a")
    assert record.status == STATUS_COMPLETED
    assert record.output == {"done": True}
    assert record.duration_seconds > 0.0


def test_stage_records_survive_a_new_store_object(tmp_path):
    path = tmp_path / "state.sqlite3"
    first_store = WorkflowStateStore(path)
    first_store.create_ticket("tx-12", {"subject": "hello"})
    first_store.complete_stage("tx-12", "a", {"value": 1})

    # A brand new object, as if the process had restarted.
    second_store = WorkflowStateStore(path)
    assert second_store.get_stage("tx-12", "a").output == {"value": 1}
    assert second_store.get_ticket("tx-12") is not None

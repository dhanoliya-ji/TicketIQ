# `app/workflow/` — the DAG engine and the triage pipeline

This is the connective tissue of the project. The triage steps are not one
function: they are named stages with declared dependencies, run by a small
engine that persists their state.

## Files

| File | What it does | Knows about tickets? |
|------|--------------|----------------------|
| `dag.py` | Graph validation, execution levels, parallel execution, resume | **No** — a general graph runner |
| `state_store.py` | SQLite persistence of ticket rows and stage rows | No |
| `pipeline.py` | The seven triage stages and `TriageService`, which owns every component | Yes |

Keeping `dag.py` ignorant of tickets is deliberate: it could run any set of
named stages, which is what makes it independently testable.

## The graph

```
level 0   classify_ticket        analyse_sentiment      <- run in parallel
                 \                    /
level 1              score_urgency
                          |
level 2          select_configuration        <- the bandit chooses here
                          |
level 3          retrieve_knowledge          <- uses the chosen top-K
                          |
level 4               run_agent              <- uses the chosen prompt variant
                          |
level 5           compose_response
```

`classify_ticket` and `analyse_sentiment` both read only the raw ticket, so they
are independent and run on two threads. Everything after that is a real data
dependency: urgency needs both of them, the bandit needs urgency to know which
state it is in, retrieval needs the top-K the bandit picked, and the agent needs
the retrieved knowledge.

`GET /workflow/graph` returns this table live, computed from the declared
dependencies rather than written down anywhere.

## Measured, not just designed

The two level-0 stages really do overlap. The store records `started_at` and
`finished_at` for every stage, so this is read back from a real run rather than
inferred:

```
stage                  thread      started     finished
classify_ticket         17832      6.23ms      15.20ms
analyse_sentiment       34876      0.00ms      11.20ms

different threads : True        wall-clock overlap : 4.97 ms
```

## Execution levels (Kahn's algorithm)

Level 0 is every stage with no dependencies. Level N is every stage whose
dependencies all sit in earlier levels. Repeat until nothing is left; if a pass
places nothing while stages remain, the leftovers form a cycle. Keeping the
result **grouped by level** rather than flattened is what tells us what may run
at the same time.

## The four guarantees

| Guarantee | How |
|-----------|-----|
| **Validated graph** | Missing dependency, self-dependency, cycle and duplicate name are all rejected in the constructor, not at run time. |
| **Parallelism where it is free** | Same-level stages run on a `ThreadPoolExecutor`. |
| **Resumability** | Every stage output is persisted as it completes; a re-run loads completed stages instead of recomputing them. |
| **Honest status** | `GET /ticket/{id}/status` reads the very rows the engine writes, so it cannot drift from reality - including *while the pipeline is still running*. |

## Failure semantics

A stage that raises is recorded `failed`, every downstream stage is marked
`skipped`, and the run returns `succeeded = False` naming the broken stage.
Upstream stages stay `completed` and their outputs stay readable — which is what
makes the retry cheap:

```
first run :  classify [completed]  sentiment [completed]  retrieve [FAILED]  agent [skipped]
retry     :  classify [reused]     sentiment [reused]     retrieve [executed] agent [executed]
```

## Why stages return plain dictionaries

Every stage output is persisted as JSON, so it must be serialisable. This is why
the retrieval stage emits snippet dictionaries and the agent consumes
dictionaries rather than objects — the persisted output *is* what the next stage
receives, which is exactly what makes a mid-pipeline resume correct rather than
approximately correct.

## Why SQLite

Feedback can arrive minutes after the answer, possibly after a restart; the
status endpoint must show real mid-flight state; and a failed stage must be
re-runnable. A file that outlives the process satisfies all three, and SQLite is
in the standard library — no broker, no extra service. Each method opens a
short-lived connection under a lock, because stages run on multiple threads.

### Schema

| Table | Columns |
|-------|---------|
| `tickets` | `transaction_id` (PK), `created_at`, `status`, `request_json`, `result_json`, `config_name`, `state_key`, `latency_seconds`, `feedback_score`, `reward` |
| `stage_runs` | `(transaction_id, stage_name)` (PK), `status`, `output_json`, `error`, `started_at`, `finished_at` |

The `config_name`, `state_key` and `latency_seconds` columns exist purely so
that `POST /feedback` can reconstruct the reward for a ticket it did not handle
in this process.

## `TriageService`

Owns every component (classifier, sentiment analyser, retriever, LLM client,
agent, bandit, state store, engine) and exposes four operations, which map
one-to-one onto the endpoints that change or read ticket state:

| Method | Endpoint |
|--------|----------|
| `handle_ticket` | `POST /ticket` |
| `retry_ticket` | `POST /ticket/{id}/retry` |
| `handle_feedback` | `POST /feedback` |
| `get_status` | `GET /ticket/{id}/status` |

Plus `rl_statistics`, `workflow_description` and `health` for the inspection
endpoints. `startup()` trains the classifier, indexes the knowledge base and
reloads the bandit state; it is called from the FastAPI lifespan hook.

## Retrying

`POST /ticket/{id}/retry` is the resumable engine made usable. It reloads the
stored request, runs the graph again with `resume=True`, and returns the normal
ticket payload plus `stages_reused` and `stages_executed` so the saving is
visible rather than claimed:

```
first attempt : retrieve_knowledge FAILED, agent + compose skipped
retry         : reused classify, sentiment, urgency, select_configuration
                executed retrieve_knowledge, run_agent, compose_response
```

It returns 404 for an unknown transaction and 409 for one that already
completed.

## Tests

`tests/test_workflow_engine.py` (28 tests) covers level computation, all four
rejection cases, failure and skip propagation, resume, retry-of-one-stage, and
the state store. Real parallelism is proved with a `threading.Barrier` that only
passes if two stages genuinely run at the same time.

Two cases worth calling out, because they are where a concurrent engine tends
to go wrong:

* **One concurrent stage fails, its sibling succeeds.** The sibling's output is
  persisted and reused on retry, so the work is not paid for twice. Without
  that, the whole point of the persisted state would be lost precisely when it
  matters.
* **Eight transactions running at once.** They share one store and one engine;
  each still sees only its own inputs and writes exactly its own stage rows.

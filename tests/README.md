# `tests/` — the test suite

**240 tests, 99% coverage of `app/`.** Every test is offline and deterministic:
no model server, no network, no clock dependence.

```bash
pytest                                        # run everything (~3 seconds)
pytest --cov=app --cov-report=term-missing    # with coverage
pytest --cov=app --cov-report=html            # browsable report in htmlcov/
pytest tests/test_workflow_engine.py -v       # one file
pytest -k "bandit and converge"               # one test by name
```

## Files

| File | Covers | Tests |
|------|--------|-------|
| `conftest.py` | Shared fixtures and the fake language model | — |
| `test_ml_classifier.py` | tokenizer, TF-IDF weights, Naive Bayes smoothing/priors/softmax, metric definitions checked against scikit-learn | 33 |
| `test_nlp_and_rag.py` | aspect extraction and precision, sentiment independence, urgency weighting, dataset split, chunking, the FAISS index, TF-IDF vs scikit-learn, category-aware re-ranking | 46 |
| `test_rl_bandit.py` | reward function, incremental average, cold start, explore/exploit, untried arms under negative rewards, per-state isolation, convergence, persistence | 22 |
| `test_workflow_engine.py` | level computation, graph rejection cases, real parallelism, failure + skip, resume, retry-one-stage, concurrent transactions, state store, schema self-healing | 28 |
| `test_agent.py` | mock tools, JSON extraction from prose, the ReAct loop, malformed replies, repeated tool calls and giving up on them, the escalation policy on all three paths, the step limit | 35 |
| `test_llm_client.py` | both prompt variants, template decisions, Ollama request shape, startup and mid-request fallback | 24 |
| `test_api.py` | every endpoint, all error codes, status reflecting real stage state, retry, both console pages | 35 |
| `test_end_to_end.py` | the full pipeline with the LLM mocked out, persistence, feedback, stage failure, retry, mid-flight inspection, a deleted database | 18 |

## How the suite stays offline

`conftest.py` sets two environment variables **at import time**, before
`app.settings` is first imported (which is when `SETTINGS` is built, so setting
them any later would have no effect):

| Variable | Value | Why |
|----------|-------|-----|
| `TICKETIQ_VAR_DIR` | a fresh temp directory | tests never touch a real run's SQLite file or bandit state |
| `TICKETIQ_LLM_BACKEND` | `template` | no test can depend on Ollama being installed |

## The fake language model

`FakeLlmClient` in `conftest.py` is a test double with **no template logic at
all**: it replays a scripted list of decision replies and then a fixed written
reply. That is what lets a test force an exact decision sequence:

```python
fake = fake_llm_factory([
    '{"thought": "check first", "action": "check_refund_eligibility", "action_input": {}}',
    '{"thought": "now I know", "action": "answer", "action_input": {}}',
])
```

`test_end_to_end.py` uses it as the "LLM calls mocked out" end-to-end test: the
assertions are about the plumbing — stages, tool dispatch, persistence, reward —
not about generated prose.

## Things worth pointing at

**Parallelism is proved, not assumed.** `test_independent_stages_really_run_at_the_same_time`
uses a `threading.Barrier(2)` that both stages must reach. A sequential engine
would deadlock and time out; only genuine concurrency passes.

**The maths is checked against hand-computed values,** not just "it returned a
number": the IDF formula, unit-length vectors, the Laplace smoothing
denominator, the harmonic mean in F1, and the bandit's incremental average
against a plain average.

**And against a reference implementation.** The hand-written TF-IDF and metrics
are asserted to agree with scikit-learn — the weights to floating-point epsilon
across the real knowledge base, the metrics exactly, including a category that
is never predicted. That is what makes keeping both worthwhile.

**Failure paths are covered as carefully as happy paths:**

- model output that is not parseable JSON
- a model that only ever calls tools, forever
- Ollama reachable at startup but dying mid-request
- Ollama returning an empty completion
- a stage raising, and the skip propagation that follows
- retrying only the stage that broke
- duplicate feedback on one transaction
- a corrupt bandit state file
- negative rewards, where an untried arm's initial 0.0 used to look like the best score
- an unknown tool name, an unknown action name, a non-object `action_input`
- a model asking for the same tool call twice, and one looping on tool calls forever
- a feature request escalated by the model, by an unparseable reply, and by the step limit
- one concurrent stage failing while its sibling succeeds, and eight transactions running at once
- retrying a transaction that never existed, and one that already succeeded

**Deliberate coverage gaps.** The ~1% not covered is the `except` arm of the
environment-variable parsers in `settings.py` (malformed numeric env vars) and a
couple of defensive branches. Everything with real logic — the bandit, the DAG
engine, the classifier, TF-IDF, chunking and the retriever — is at 100%.

## CI

`.github/workflows/ci.yml` runs black, ruff, mypy and this suite on Python 3.11
and 3.12, checks the dataset regenerates byte-identically, runs both experiment
scripts, and builds and health-checks the Docker image.

# TicketIQ — Self-Optimizing Support Triage Agent

A back-end service that triages free-text support tickets end to end: it
classifies the ticket with a hand-written Naive Bayes model, scores sentiment
**per product aspect**, retrieves the relevant policy from a knowledge base,
reasons with a ReAct agent that can call tools or escalate, and **learns which
pipeline configuration works best** from latency and user feedback using an
online contextual bandit — all orchestrated by a small resumable DAG engine
rather than one monolithic function.

> **Runs with no model server installed.** The language model layer falls back
> to a deterministic offline writer when Ollama is not available, and every
> response says which backend produced it. See
> [Language model setup](#language-model-setup).

| | |
|---|---|
| **Tests** | 191 passing, **98%** coverage of `app/` |
| **Classifier** | 92.5% accuracy / 0.925 macro-F1 on a held-out split |
| **Bandit** | 54% → 76% optimal choices over 5k tickets; **88.8%** at 20k (ε-ceiling is 88.8%) |
| **Demo UI** | A live console at `/`, served by the same app — no build step, no new dependency |
| **Stack** | FastAPI · Pydantic · VADER · SQLite · pytest · black · ruff · Docker · GitHub Actions |

---

## Table of contents

- [Quick start](#quick-start)
- [The live console](#the-live-console)
- [Architecture](#architecture)
- [API reference](#api-reference)
- [Language model setup](#language-model-setup)
- [Results](#results)
- [Design write-ups](#design-write-ups)
- [Testing](#testing)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [Shortcuts and scope notes](#shortcuts-and-scope-notes)

---

## Quick start

### Without Docker

Requires Python 3.10 or newer.

```bash
# 1. clone and enter the repository
cd TicketIQ

# 2. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# 4. run the service
uvicorn app.main:app --reload
```

Then open **<http://localhost:8000/>** — the live console, where you can run
tickets, watch the pipeline stages light up, rate the answers and see the
bandit learn. <http://localhost:8000/docs> is the interactive API documentation.

From the command line:

```bash
curl http://localhost:8000/health
```

```json
{"ready":true,"llm_backend":"template","knowledge_chunks":28,
 "classifier_accuracy":0.925,"bandit_updates":0}
```

Nothing needs to be trained or downloaded first: the classifier trains at
startup from `data/tickets.json` (milliseconds) and the knowledge base is
indexed in memory.

### With Docker

```bash
docker build -t ticketiq .
docker run --rm -p 8000:8000 ticketiq
```

To keep the learned bandit statistics and workflow history between runs, mount
the runtime folder:

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/var:/service/var" ticketiq
```

### Run the experiments

```bash
python scripts/train_and_report.py     # classifier metrics on a held-out split
python scripts/simulate_bandit.py      # RL simulation: does the bandit learn?
python data/generate_tickets.py        # regenerate the labelled dataset
```

---

## The live console

Open **<http://localhost:8000/>** once the service is running. It is three
static files (`app/static/`) served by the same FastAPI process — same origin,
no CORS setup, no `npm install`, no build step, and no extra Python dependency.
Full notes in [app/static/README.md](app/static/README.md).

| Section | What you can demo with it |
|---------|---------------------------|
| **1 · Submit a ticket** | Four one-click samples, each chosen to trigger a different agent behaviour: a refund that calls a tool, an outage that escalates, a lockout that checks account status, a feature request that is never escalated. |
| **2 · Workflow engine** | The DAG drawn from `/workflow/graph`, with each stage lighting up in dependency order and level 0 boxed as "these two run in parallel". |
| **3 · Triage result** | Category and confidence, the urgency meter, per-aspect sentiment with the evidence sentence, the retrieved chunks with similarity bars, the complete reasoning trace including tool observations, the reply, and thumbs up / down. |
| **4 · Reinforcement learning** | Stat tiles, average reward per configuration for any state, and a **30-ticket demo** that makes the bandit visibly abandon the configurations the customer dislikes. |
| **5 · Session activity** | Every HTTP call the page made, so an audience can see there is nothing pre-baked. |

**The stage animation is a replay, labelled as one.** The pipeline finishes in
about 0.1 s, too fast to watch, so the page reads the *real* per-stage durations
from `GET /ticket/{id}/status` and replays them at 26× slower. The milliseconds
shown on each stage are the true measured values.

### Suggested 90-second demo

1. Click **Enterprise outage** → **Run pipeline**. Watch level 0 run two stages
   at once, then the agent decide **escalate to human**.
2. Click **Feature request** → **Run pipeline**. Same pipeline, and the agent
   answers instead of escalating — the escalation policy says a missing feature
   is not an outage.
3. Rate an answer 👍 and point at the reward line: `1 × 10 − latency`.
4. Hit **Run 30-ticket demo** and watch the losing configurations flatten out
   while the favoured one takes over the chart.

---

## Architecture

Full detail, including the reasoning behind each choice, is in
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

```
        POST /ticket          POST /feedback       GET /ticket/{id}/status
             |                      |                       |
             v                      v                       v
   +--------------------------------------------------------------+
   |                        TriageService                         |
   +--------------------------------------------------------------+
        |                       |                        |
        v                       v                        v
  WorkflowEngine        Contextual bandit        WorkflowStateStore
   (DAG runner)          (epsilon-greedy)             (SQLite)
        |                       ^                        ^
        |  runs stages          | reward                 | real state
        v                       |                        |
  +--------------------------------------------------------------+
  | Naive Bayes | VADER aspects | TF-IDF retriever | ReAct agent  |
  +--------------------------------------------------------------+
```

### The pipeline DAG

Each stage declares its dependencies; the engine derives the order and runs
independent stages concurrently.

```mermaid
graph TD
    A[classify_ticket<br/>Naive Bayes] --> C[score_urgency]
    B[analyse_sentiment<br/>VADER per aspect] --> C
    C --> D[select_configuration<br/>contextual bandit]
    A --> D
    D --> E[retrieve_knowledge<br/>cosine top-K]
    A --> E
    E --> F[run_agent<br/>ReAct loop + tools]
    C --> F
    D --> F
    A --> F
    F --> G[compose_response]

    style A fill:#dbeafe,stroke:#1e40af
    style B fill:#dbeafe,stroke:#1e40af
    style D fill:#fef3c7,stroke:#b45309
    style F fill:#dcfce7,stroke:#15803d
```

| Level | Stages | |
|-------|--------|---|
| 0 | `classify_ticket`, `analyse_sentiment` | **run in parallel** — both read only the raw ticket |
| 1 | `score_urgency` | needs category *and* aspect sentiment |
| 2 | `select_configuration` | the bandit's state is the category + urgency + tier |
| 3 | `retrieve_knowledge` | uses the top-K the bandit chose |
| 4 | `run_agent` | uses the prompt variant the bandit chose |
| 5 | `compose_response` | |

`GET /workflow/graph` returns this live, computed from declared dependencies.

---

## API reference

### `POST /ticket`

Runs the full pipeline.

```bash
curl -X POST http://localhost:8000/ticket \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Charged twice on order 4471",
    "body": "My credit card was charged twice for the same monthly invoice. I want a refund.",
    "customer_tier": "enterprise"
  }'
```

<details>
<summary>Response (abridged — real output)</summary>

```json
{
  "transaction_id": "tx-f74f7d434acb",
  "category": "billing",
  "category_confidence": 1.0,
  "aspect_sentiments": [
    {"aspect": "billing", "score": 0.0257, "label": "neutral", "mentions": 3,
     "evidence": "Charged twice on order 4471"}
  ],
  "urgency_score": 0.55,
  "urgency_bucket": "medium",
  "retrieved_knowledge": [
    {"chunk_id": "01_billing_refund_policy#2", "source": "01_billing_refund_policy.md",
     "document_title": "Billing and Refund Policy", "heading": "Duplicate charges",
     "text": "If the same amount was charged twice ...", "score": 0.2433},
    {"chunk_id": "01_billing_refund_policy#1", "heading": "Refund window",
     "text": "A customer may request a full refund within 30 days ...", "score": 0.1332}
  ],
  "action": "answer",
  "response_text": "Category: billing. Applicable policy - Duplicate charges: ...",
  "reasoning_trace": [
    {"step": 1,
     "thought": "The customer is asking for money back, so eligibility must be confirmed before anything is promised.",
     "action": "check_refund_eligibility",
     "action_input": {"order_id": "4471"},
     "observation": "order 4471 is eligible for a refund of 79.0 (duplicate charge, refundable regardless of age)"},
    {"step": 2,
     "thought": "The retrieved knowledge base sections cover this question, so it can be answered directly.",
     "action": "answer", "action_input": {}, "observation": ""}
  ],
  "tool_calls": [
    {"tool": "check_refund_eligibility", "arguments": {"order_id": "4471"},
     "output": {"eligible": true, "amount": 79.0, "days_since_charge": 8, "duplicate_charge": true},
     "summary": "order 4471 is eligible for a refund of 79.0 (duplicate charge, refundable regardless of age)"}
  ],
  "pipeline_config": {"name": "concise_policy|k2", "prompt_variant": "concise_policy",
                      "rag_top_k": 2, "model": "llama3"},
  "config_selection_reason": "cold_start",
  "rl_state_key": "billing|medium|enterprise",
  "llm_backend": "template",
  "latency_seconds": 0.0822,
  "stages_executed": ["analyse_sentiment", "classify_ticket", "score_urgency",
                      "select_configuration", "retrieve_knowledge", "run_agent",
                      "compose_response"],
  "stages_reused": []
}
```
</details>

`customer_tier` is one of `free`, `pro`, `enterprise` (defaults to `free`).

### `POST /feedback`

Binary feedback. This is what closes the reinforcement learning loop.

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "tx-f74f7d434acb", "feedback_score": 1}'
```

```json
{
  "transaction_id": "tx-f74f7d434acb",
  "feedback_score": 1,
  "latency_seconds": 0.0822,
  "reward": 9.9178,
  "state_key": "billing|medium|enterprise",
  "config_name": "concise_policy|k2",
  "updated_average_reward": 9.9178,
  "best_config_for_state": "concise_policy|k2"
}
```

`reward = feedback_score × 10 − latency_seconds`. Feedback is accepted **once**
per transaction; a repeat returns `409`, since applying it twice would bias the
bandit's averages.

### `GET /ticket/{transaction_id}/status`

Real per-stage state, read from the workflow engine's own store.

```bash
curl http://localhost:8000/ticket/tx-f74f7d434acb/status
```

```json
{
  "transaction_id": "tx-eafcfe7d5f8a",
  "status": "completed",
  "request": {
    "subject": "Charged twice on order 4471",
    "body": "My credit card was charged twice for the same monthly invoice. I want a refund.",
    "customer_tier": "enterprise"
  },
  "completed_stages": 7,
  "total_stages": 7,
  "stages": [
    {"stage": "analyse_sentiment", "status": "completed", "error": "", "duration_seconds": 0.0103,
     "output": {"aspects": [{"aspect": "billing", "score": 0.0257, "label": "neutral",
                             "mentions": 3, "evidence": "Charged twice on order 4471"}]}},
    {"stage": "classify_ticket", "status": "completed", "error": "", "duration_seconds": 0.011,
     "output": {"category": "billing", "confidence": 1.0,
                "scores": {"account": -104.2556, "billing": -73.6724,
                           "feature_request": -103.9027, "technical": -115.9191}}},
    {"stage": "score_urgency", "status": "completed", "error": "", "duration_seconds": 0.004,
     "output": {"urgency_score": 0.55, "urgency_bucket": "medium"}}
  ],
  "not_started_stages": [],
  "feedback_score": 1,
  "reward": 9.9304
}
```

*(Abridged: the four remaining stages are omitted. The stage list is ordered by
start time, which is why the two parallel level-0 stages appear first.)*

When a stage fails, its row shows `"status": "failed"` with the error message,
everything downstream shows `"skipped"`, and the completed stages upstream keep
their outputs — which is exactly what makes a retry cheap.

### Inspection endpoints

| Endpoint | Returns |
|----------|---------|
| `GET /health` | readiness, the live LLM backend, chunk count, classifier accuracy |
| `GET /rl/stats` | everything the bandit has learned, per state and per arm |
| `GET /workflow/graph` | the DAG, level by level, showing what runs in parallel |
| `GET /ml/report` | the held-out classifier evaluation and confusion matrix |
| `GET /docs` | interactive OpenAPI documentation |

### Error handling

| Status | When |
|--------|------|
| `422` | invalid tier, empty subject/body, feedback score outside {0, 1} |
| `404` | unknown transaction id |
| `409` | feedback already recorded for that transaction |
| `500` | a pipeline stage failed — the body carries the `transaction_id` and `failed_stage` so the status endpoint can be inspected |

---

## Language model setup

The RL layer needs a genuine choice between configurations, so there are two
distinct prompt variants, each mapped to its own Ollama model:

| Variant | System instruction | Ollama model |
|---------|--------------------|--------------|
| `concise_policy` | at most four sentences, quote the exact policy rule, no pleasantries | `llama3` |
| `empathetic_stepwise` | acknowledge impact, numbered next steps, say who owns it | `mistral` |

Combined with RAG top-K of 2 or 5, that is the bandit's four-arm action space.

**To use real models:**

```bash
ollama serve
ollama pull llama3
ollama pull mistral
TICKETIQ_LLM_BACKEND=auto uvicorn app.main:app
```

**Without Ollama** the service uses `TemplateBackend`, which composes the reply
from the retrieved policy chunks, the chosen prompt variant and the tool
results. It is not a fixed-string stub — the two variants produce visibly
different replies, so the pipeline, the latency measurement and therefore the
reward signal stay meaningful. Every response carries `"llm_backend"`, so
template output can never be mistaken for model output.

This is what lets the test suite and CI run offline and deterministically.

---

## Results

### Classifier (`python scripts/train_and_report.py`)

160 synthetic tickets, stratified 75/25 split, Multinomial Naive Bayes
implemented from scratch:

```
accuracy       : 0.925
macro precision: 0.927
macro recall   : 0.925
macro F1       : 0.925

category          precision  recall     f1         support
account           0.900      0.900      0.900      10
billing           0.909      1.000      0.952      10
feature_request   1.000      0.900      0.947      10
technical         0.900      0.900      0.900      10

Confusion matrix (rows = actual, columns = predicted)
                  account   billing   feature_  technica
account           9         0         0         1
billing           0         10        0         0
feature_request   0         1         9         0
technical         1         0         0         9
```

The first version of the dataset scored a perfect 1.000, which means the task
was too easy to be informative. The generator now mixes a second, unrelated
problem into 45% of tickets ("I cannot log in **and** I was charged twice"),
keeping the subject line's category as the label. The three remaining errors are
exactly those genuinely ambiguous tickets.

### Reinforcement learning (`python scripts/simulate_bandit.py`)

The real bandit and the real reward function against a simulated world where
**no arm is globally best** — fast/concise wins on low urgency, slow/thorough
wins on high urgency:

```
Action distribution              first window   last window
concise_policy|k2                24.1%          36.5%
concise_policy|k5                30.4%          25.5%
empathetic_stepwise|k2           16.3%          6.9%      <- abandoned
empathetic_stepwise|k5           29.2%          31.1%

Share of choices that matched the oracle
first window: 54.4%
last window : 75.9%

What the bandit learned
technical|high|enterprise    learned=empathetic_stepwise|k5  oracle=empathetic_stepwise|k5  match
feature_request|low|free     learned=concise_policy|k2       oracle=concise_policy|k2       match
billing|medium|pro           learned=concise_policy|k5       oracle=concise_policy|k5       match
```

At 20,000 tickets the last window reaches **88.8%** optimal choices — the
ceiling for ε = 0.15 — and average reward per ticket rises from 5.66 to ~6.7.
Note the distribution shift is *per state*: `empathetic_stepwise|k5` is not
globally best, it is retained precisely where high urgency makes it best.

---

## Design write-ups

### Reinforcement learning strategy, and why

Choosing a pipeline configuration is a **one-step** decision — pick an arm, get
a reward, and the next ticket is unrelated to this one. There is no sequence of
states to plan through, so Q-learning's discount factors and next-state
bootstrapping would be machinery with nothing to do. A **contextual bandit** is
the exact model for "one decision, immediate reward, and the right answer
depends on context", and it learns from very few samples, which matters when
every sample costs a real customer interaction. A deep network was explicitly
out of scope, and would be the wrong tool for 36 states and 4 arms anyway.

```
state  = category | urgency bucket | customer tier     (4 × 3 × 3 = 36)
action = prompt variant × RAG top-K                    (2 × 2 = 4)
reward = feedback × 10 − latency seconds
```

Everything continuous is bucketed before it reaches the bandit: a raw urgency
of 0.5512 would create a state that never repeats. Selection is **epsilon-greedy
with a cold start** — every arm is tried once per state before any average is
trusted, then 15% of traffic explores. The average updates incrementally
(`avg += (reward − avg) / n`), which equals re-averaging the whole history
without storing it, and the statistics are persisted to `var/bandit_state.json`
so learning survives a restart.

**Trade-off taken knowingly:** 36 states × 4 arms = 144 cells, and binary
feedback is a noisy signal. Full per-state convergence needs thousands of
tickets. Keeping `category` in the state is still the right modelling call
because a feature request and an outage genuinely want different handling — but
a production deployment with low volume should either drop `category` or share
statistics across states with a linear model (LinUCB).

### Workflow engine design

`app/workflow/dag.py` is a ~150-line DAG runner that knows nothing about
tickets. Stages declare dependencies; the engine computes **execution levels**
with Kahn's algorithm, which naturally answers "what may run at the same time".

Four guarantees, each directly tested:

1. **Validated graph** — missing dependency, self-dependency, cycle and
   duplicate name are all rejected at construction, not at run time.
2. **Parallelism where it is free** — same-level stages run on a thread pool.
   The test proves it with a `threading.Barrier` that only passes if two stages
   are genuinely concurrent.
3. **Resumability** — every stage output is persisted as it completes, so a
   re-run loads finished stages instead of recomputing them. A failed stage can
   be retried alone.
4. **Honest status** — `GET /ticket/{id}/status` reads the very rows the engine
   writes, so it cannot drift from reality.

Failure semantics: a raising stage is recorded `failed`, everything downstream
is `skipped`, upstream outputs stay readable, and the run reports which stage
broke. Hand-rolling this rather than pulling in Airflow/Prefect/Celery was the
brief's explicit preference, and it means no broker, no scheduler and no extra
service to run.

### Why no scikit-learn

The brief allows scikit-learn for vectorisation and metrics but forbids
`model.fit()` for the classifier. Once TF-IDF and the metrics are written out by
hand — which makes the maths visible, the point of the exercise — scikit-learn
has nothing left to do, so it is not a dependency at all. `app/ml/tfidf.py`,
`app/ml/metrics.py` and `app/ml/naive_bayes.py` are the substitutes.

---

## Testing

```bash
pytest                                              # 191 tests
pytest --cov=app --cov-report=term-missing          # coverage report
pytest --cov=app --cov-report=html                  # browsable report in htmlcov/
pytest tests/test_workflow_engine.py -v             # one file
```

Every test is offline and deterministic: `tests/conftest.py` forces the template
LLM backend and a throw-away state directory before `app.settings` is imported.

| File | Covers | Tests |
|------|--------|-------|
| `test_ml_classifier.py` | tokenizer, TF-IDF weights, Naive Bayes smoothing/priors/softmax, metric definitions | 30 |
| `test_nlp_and_rag.py` | aspect extraction, sentiment independence, urgency weighting, dataset split, chunking, cosine search | 33 |
| `test_rl_bandit.py` | reward function, incremental average, cold start, explore/exploit, per-state isolation, convergence, persistence | 18 |
| `test_workflow_engine.py` | level computation, cycle/missing-dependency rejection, real parallelism, failure + skip, resume, retry-one-stage, state store | 24 |
| `test_agent.py` | mock tools, JSON extraction from prose, ReAct loop, malformed replies, step limit | 24 |
| `test_llm_client.py` | both prompt variants, template decisions, Ollama request shape, startup and mid-request fallback | 24 |
| `test_api.py` | every endpoint, all error codes, status reflecting real stage state, the console routes | 26 |
| `test_end_to_end.py` | full pipeline with the LLM mocked out, persistence, feedback, stage failure, selective retry | 12 |

**Coverage: 98% of `app/`** — 100% on the bandit, the DAG engine, the
classifier, TF-IDF, chunking and the vector store.

The suite deliberately covers failure paths, not just happy paths: unparseable
model output, a model that loops forever, Ollama dying mid-request, a stage
raising, duplicate feedback, and a corrupt bandit state file.

### Code quality

```bash
black app tests scripts data          # format
ruff check app tests scripts data     # lint
pre-commit install                    # wire both into git commit
pre-commit run --all-files
```

CI (`.github/workflows/ci.yml`) runs black, ruff and the test suite on Python
3.11 and 3.12, verifies the dataset regenerates byte-identically, runs both
experiment scripts, and builds the Docker image and health-checks the container.

---

## Repository layout

Every directory has its own README explaining what lives in it and why.

```
TicketIQ/
├── app/                        # the service
│   ├── main.py                 # FastAPI endpoints
│   ├── schemas.py              # the whole HTTP contract, in one file
│   ├── settings.py             # every tunable value, read from the environment
│   ├── ml/                     # classifier, TF-IDF, metrics, aspect sentiment, urgency
│   ├── rag/                    # chunking, in-memory vector store, retriever
│   ├── llm/                    # Ollama client, offline fallback, the four configurations
│   ├── agent/                  # ReAct loop and the mock tools
│   ├── rl/                     # contextual bandit, state and reward
│   ├── workflow/               # DAG engine, SQLite state store, the triage pipeline
│   └── static/                 # the live console (plain HTML, CSS, JavaScript)
├── data/
│   ├── generate_tickets.py     # deterministic dataset generator
│   ├── tickets.json            # 160 labelled synthetic tickets
│   └── knowledge_base/         # 5 markdown policy documents (28 chunks)
├── scripts/
│   ├── train_and_report.py     # classifier metrics
│   └── simulate_bandit.py      # RL learning experiment
├── tests/                      # 191 tests, 98% coverage
├── docs/ARCHITECTURE.md        # detailed design and diagrams
├── Dockerfile
├── .github/workflows/ci.yml
├── .pre-commit-config.yaml
├── pyproject.toml              # black, ruff, pytest, coverage configuration
├── requirements.txt
└── requirements-dev.txt
```

---

## Configuration

All settings are environment variables with working defaults
(`app/settings.py`); nothing must be set to run the service.

| Variable | Default | Purpose |
|----------|---------|---------|
| `TICKETIQ_LLM_BACKEND` | `auto` | `auto`, `ollama` or `template` |
| `TICKETIQ_OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `TICKETIQ_OLLAMA_MODEL_A` | `llama3` | model for the concise variant |
| `TICKETIQ_OLLAMA_MODEL_B` | `mistral` | model for the step-by-step variant |
| `TICKETIQ_LLM_TIMEOUT` | `30.0` | seconds before falling back |
| `TICKETIQ_BANDIT_EPSILON` | `0.15` | exploration rate |
| `TICKETIQ_AGENT_MAX_STEPS` | `4` | hard stop for the ReAct loop |
| `TICKETIQ_TEST_SPLIT` | `0.25` | held-out fraction for the classifier report |
| `TICKETIQ_RANDOM_SEED` | `42` | reproducibility |
| `TICKETIQ_VAR_DIR` | `./var` | SQLite state and bandit statistics |
| `TICKETIQ_DATA_DIR` | `./data` | dataset and knowledge base |

---

## Shortcuts and scope notes

Written down deliberately rather than left for a reviewer to discover.

**Dataset is synthetic and template-generated.** Real support data cannot be
shared. 160 tickets built from per-category vocabularies, with a 45%
cross-category noise rate so the task is not trivial. The 92.5% accuracy is
honest for *this* dataset; it says nothing about real traffic, where typos,
multi-language tickets and much longer bodies would all hurt.

**The template LLM backend is a fallback, not a model.** When Ollama is absent
the replies are composed from retrieved policy rather than generated. It exists
so the system is reviewable and testable anywhere; it is labelled in every
response and never silently substituted.

**Aspect sentiment is lexicon-based, and inherits VADER's blind spots.** The
brief allows this ("rule/keyword-based aspect spotting combined with … VADER").
The known gap is negation across a phrase: "nobody has replied for days" is
neutralised by the domain lexicon rather than scored properly negative. A
fine-tuned ABSA model is the real fix and was out of scope.

**Tools are deterministic fakes.** `check_account_status` and
`check_refund_eligibility` derive their answers from the identifier, so they are
reproducible without a billing system. The eligibility logic does mirror the
refund policy document, so the agent's reasoning is at least self-consistent.

**The bandit learns per state, and 36 states is data-hungry.** Discussed under
[Design write-ups](#design-write-ups); LinUCB with shared features is the next
step if traffic is low.

**Feedback is applied once and never revised.** No decay, no handling of a
customer changing their mind. A production system would want a sliding window so
the bandit can track a model that gets better or worse over time.

**Single process, in-memory components.** The classifier, the index and the
bandit live in the process; only the workflow state and bandit statistics are on
disk. Running several replicas would need the bandit statistics moved into shared
storage — the `save`/`load` seam in `app/rl/bandit.py` is where that would go.

**The console is an addition, not a deliverable.** The brief asked for a
back-end service. The UI exists so the system can be demonstrated without a
terminal; it adds no dependency, touches no pipeline code, and every number it
shows comes from a real API call.

**Not implemented by choice:** authentication, rate limiting, multi-language
support, and streaming responses. None were in the brief.

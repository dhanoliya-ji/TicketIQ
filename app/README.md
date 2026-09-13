# `app/` — the service

Everything that runs in production. Each subpackage owns one concern and can be
read on its own; the only module that knows about all of them is
`workflow/pipeline.py`, which wires them into the DAG.

## Files in this directory

| File | What it does |
|------|--------------|
| `main.py` | The FastAPI application: the three required endpoints, four inspection endpoints, and the lifespan hook that trains the classifier and indexes the knowledge base before traffic is served. |
| `schemas.py` | Every request and response model, in one file, so the whole HTTP contract fits on one screen. FastAPI turns these into the OpenAPI docs at `/docs`. |
| `settings.py` | One `Settings` object built from environment variables, with working defaults. Import `SETTINGS`; never read `os.environ` elsewhere. |
| `__init__.py` | Package marker and version string. |
| `static/` | The live console served at `/` — plain HTML, CSS and JavaScript, no build step. See [its README](static/README.md). |

## Subpackages

| Package | Responsibility | Key entry point |
|---------|----------------|-----------------|
| [`ml/`](ml/README.md) | Classical NLP and machine learning: the classifier, TF-IDF, metrics, aspect sentiment, urgency | `TicketClassifierService`, `AspectSentimentAnalyzer` |
| [`rag/`](rag/README.md) | Knowledge base chunking, the in-memory vector store, retrieval | `KnowledgeRetriever` |
| [`llm/`](llm/README.md) | Talking to a language model, plus the four pipeline configurations the RL layer chooses from | `LlmClient`, `ALL_CONFIGS` |
| [`agent/`](agent/README.md) | The ReAct reasoning loop and the mock back-office tools | `TriageAgent` |
| [`rl/`](rl/README.md) | The epsilon-greedy contextual bandit, the state key and the reward function | `EpsilonGreedyContextualBandit` |
| [`workflow/`](workflow/README.md) | The DAG engine, the SQLite state store, and the seven triage stages | `WorkflowEngine`, `TriageService` |

## How a request flows through these packages

```
main.py  ->  workflow/pipeline.py (TriageService)
                 |
                 +-> workflow/dag.py runs the stages in dependency order:
                       ml/classifier_service.py     (classify_ticket)
                       ml/aspect_sentiment.py       (analyse_sentiment)   } in parallel
                       ml/urgency.py                (score_urgency)
                       rl/bandit.py                 (select_configuration)
                       rag/retriever.py             (retrieve_knowledge)
                       agent/react_agent.py + llm/  (run_agent)
                                                    (compose_response)
                 |
                 +-> workflow/state_store.py persists every stage result
```

## Dependency direction

Imports only ever point **downwards** in this list, so there are no cycles:

```
main.py
  -> workflow/pipeline.py
       -> agent/  -> llm/  -> settings.py
       -> rag/    -> ml/tfidf.py
       -> ml/
       -> rl/
       -> workflow/dag.py -> workflow/state_store.py
```

`workflow/dag.py` deliberately imports nothing from `ml/`, `rag/`, `llm/`,
`agent/` or `rl/` — it is a general graph runner that knows nothing about
tickets, which is what makes it independently testable.

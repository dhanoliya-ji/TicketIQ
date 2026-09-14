# `docs/` — design documentation

| File | What it covers |
|------|----------------|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | The full design: component map, the pipeline DAG, the workflow engine, the classifier maths, aspect sentiment, retrieval, the agent loop, the reinforcement learning strategy, the LLM layer, the request lifecycle, and a table of decisions with their alternatives. |

## Where to find what

| Question | Read |
|----------|------|
| How do I run it? | [root README](../README.md#run-it) |
| What does each endpoint return? | [root README](../README.md#api-reference) |
| Why a contextual bandit and not Q-learning? | [ARCHITECTURE.md section 9](ARCHITECTURE.md#9-reinforcement-learning) |
| How does the DAG engine resume a failed run? | [ARCHITECTURE.md section 3](ARCHITECTURE.md#3-the-workflow-engine) |
| What exactly does Naive Bayes compute here? | [ARCHITECTURE.md section 4](ARCHITECTURE.md#4-classical-ml-the-classifier) |
| What happens when the model returns nonsense? | [ARCHITECTURE.md section 8](ARCHITECTURE.md#8-the-agentic-layer) |
| How do I demo this to someone? | [root README](../README.md#the-live-console) |
| What was cut for time? | [root README](../README.md#shortcuts-and-scope-notes) |

Each source directory also has its own README describing the files in it and the
reasoning behind them: [`app/`](../app/README.md), [`app/ml/`](../app/ml/README.md),
[`app/rag/`](../app/rag/README.md), [`app/llm/`](../app/llm/README.md),
[`app/agent/`](../app/agent/README.md), [`app/rl/`](../app/rl/README.md),
[`app/workflow/`](../app/workflow/README.md), [`app/static/`](../app/static/README.md),
[`data/`](../data/README.md),
[`scripts/`](../scripts/README.md), [`tests/`](../tests/README.md).

## Diagrams

The diagrams are Mermaid blocks inside the markdown, so they render on GitHub
and stay diffable in review. There are no binary image files to keep in sync
with the code.

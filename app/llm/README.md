# `app/llm/` — language model access and pipeline configurations

## Files

| File | What it does |
|------|--------------|
| `configs.py` | The two prompt variants and the four `PipelineConfig` arms the RL layer chooses between. |
| `client.py` | `LlmClient` with two interchangeable backends: a real Ollama server, and a deterministic offline writer. |

## The four configurations (the bandit's action space)

A configuration is a **prompt variant** crossed with a **RAG top-K**:

| Variant | System instruction | Ollama model |
|---------|--------------------|--------------|
| `concise_policy` | at most four sentences, quote the exact policy rule, no pleasantries | `TICKETIQ_OLLAMA_MODEL_A` (default `llama3`) |
| `empathetic_stepwise` | acknowledge the impact, numbered next steps, say who owns it | `TICKETIQ_OLLAMA_MODEL_B` (default `mistral`) |

```
concise_policy|k2      concise_policy|k5
empathetic_stepwise|k2 empathetic_stepwise|k5
```

These are genuine alternatives, not labels. The variants differ in system
instruction *and* model; K=5 retrieves more context and is measurably slower
than K=2. That is what gives the bandit something real to learn: the best arm
for a low-urgency free-tier feature request is not the best arm for an angry
enterprise outage.

The configuration name (`prompt_variant|kN`) is the bandit's action key, is
returned in every API response, and is stored on the ticket row so that feedback
arriving later can be attributed to the arm that earned it.

## Two backends, one interface

| Backend | Used when | What it does |
|---------|-----------|--------------|
| `OllamaBackend` | a server answers `GET /api/tags` | posts to `/api/generate` with `stream: false` |
| `TemplateBackend` | no server, or a call fails mid-request | composes the answer from the retrieved chunks, the chosen variant and the tool results |

`TICKETIQ_LLM_BACKEND` selects the policy: `auto` (default, probe once at
startup), `ollama` (force), or `template` (force, never touch the network).

## Why the offline backend exists

Every other part of this project — the classifier, the retriever, the bandit,
the workflow engine — must be testable and reviewable on a machine with no model
server, and CI must be deterministic. The template backend makes that possible
without stubbing out the pipeline.

It is **not** a fixed-string stub. It reads the retrieved policy, the prompt
variant and the tool results, and writes a different answer for each — including
two visibly different reply shapes for the two variants. So the pipeline, the
latency measurement and therefore the reward signal stay meaningful offline.

**Every response reports its backend**, so template output can never be mistaken
for model output. If Ollama was reachable at startup but the call fails (server
stopped, model not pulled, timeout), that single request falls back to the
template rather than failing the ticket, and says so.

## The two tasks

`LlmRequest.task` is either:

- `decide` — the ReAct loop asking for the next action as JSON.
- `write` — turning the decision into the customer-facing reply.

`LlmRequest.facts` carries a structured copy of what is also in the prompt. A
real model ignores it and reads the prompt; the template backend uses it so it
does not have to parse its own prompt back.

## Tests

`tests/test_llm_client.py` (24 tests) monkeypatches `requests` to prove the
Ollama request shape, the startup probe, the mid-request fallback, the
empty-reply fallback, and that forcing `template` never contacts the network.

# `app/static/` — the console

One page, served by the FastAPI application itself at
**<http://localhost:8000/>**.

| File | What it does |
|------|--------------|
| `index.html` | The page. |
| `styles.css` | All styling. |
| `app.js` | API calls and rendering. |

## Why plain HTML, CSS and JavaScript

No React, no bundler, no `npm install`, no new Python dependency. Three static
files served by the same FastAPI process, which means **same origin** (no CORS
to get wrong) and **nothing to build** — clone, start uvicorn, open the page.
The Dockerfile already copies `app/`, so the container serves it too.

The brief asks for a back-end service. The page is an addition, and it is kept
to exactly what the brief specifies the API accepts and returns.

## What is on it, and why each thing is there

**Input** — subject, body, customer tier. Those are the three fields
`POST /ticket` accepts, and the brief calls the ticket *free text*, so the
fields are plain and empty. There are no example or sample tickets: a menu of
canned tickets would imply the endpoint only handles certain kinds, which is
not true.

**Result** — every field requirement 1 lists that `POST /ticket` returns:

| The brief's wording | On the page |
|---|---|
| predicted category | summary |
| per-aspect sentiment scores | collapsed section |
| the retrieved knowledge snippets | collapsed section |
| the agent's chosen action | summary |
| the final response text | the response block |
| the pipeline configuration used (model / prompt variant / RAG top-K) | summary |
| end-to-end latency | summary |
| a unique transaction ID | beside the heading |

Urgency is also shown: requirement 2 asks for it explicitly as a computed
score, and the endpoint returns it.

**The numbers behind the labels.** A conclusion on its own reads as certainty,
so each one is shown with the figure it came from: the category with the
classifier's confidence, the urgency bucket with the score it was banded from,
and each retrieved snippet with the cosine similarity FAISS ranked it by —
which is what makes "top-K" mean anything.

**Reasoning trace and tool calls** — requirement 4 asks for the decision, the
trace *and* the tool calls. The trace gives the thought and the one-line
observation per step; the tool calls section gives what each tool was actually
called with and every field it returned.

**Pipeline stages** — from `GET /ticket/{id}/status`, which requirement 6 asks
to reflect real workflow state. Each row carries the stage's own output,
folded away, because a stage's output is what makes the pipeline inspectable
rather than a row of ticks. Long outputs are cut short: the full reasoning
trace, the retrieved chunks and the finished reply already have their own
sections above, and repeating them whole turned the page into a log dump.

After a retry, the stages that were **reused rather than recomputed** are named
under the table. That is the visible half of requirement 6's "re-run a failed
stage without repeating the upstream stages that already succeeded".

**Feedback** — `POST /feedback`, one binary rating per ticket.

**Retry** — appears only when a stage actually fails. It is how requirement 6's
"a failed stage can be re-run without repeating already-completed upstream
stages" is usable rather than merely implemented.

## What is deliberately not here

A request log, live charts, system statistics, a synthetic-ticket simulator,
ticket history, a service status indicator, and a second page for classifier
metrics and bandit rewards.

The last one was built and then removed. Everything on it is a deliverable of
the *project* rather than of the *interface*, and each is already satisfied:
the classifier report by `scripts/train_and_report.py` and `/ml/report`, the
bandit evidence by `scripts/simulate_bandit.py` — requirement 5 asks for "a
short experiment or simulation", not a screen — and pipeline inspection by
`GET /ticket/{id}/status`. Those endpoints all still exist; they just do not
get a page.

No number on the page is computed in the browser. Everything is served by the
API and rendered as it arrives, so the page cannot drift from the service.

## Design

Neutral greys, one accent, and colour only where it carries meaning.
**Every meaning-bearing colour clears WCAG AA (4.5:1) against the surface it
sits on** — measured, not eyeballed. The first pass used `#2a78d6` for the
accent and it came out at 4.42:1, so it was darkened to `#256abf` (5.2:1).

Colour appears in three places, and in each the word is present too, so colour
is never the only signal: urgency, the action, and sentiment polarity.

Layout leans on whitespace and hairline rules rather than nested boxes. The
only motion is the submit spinner and a short fade when a result arrives, both
disabled under `prefers-reduced-motion: reduce`.

## Behaviour worth knowing

- The previous result is **hidden the moment a new ticket is submitted**, so
  stale figures can never be mistaken for fresh ones.
- Errors are rendered as a sentence, not a status code. A failed pipeline names
  the stage that broke and offers the retry.
- The feedback buttons lock after one use, matching the API.
- When a policy rule overrides the model's chosen action, the reasoning trace
  says so on the step it changed rather than hiding it.

## Editing it

No build step — edit a file and reload the browser. `uvicorn --reload` watches
Python files; static files are read per request, so a plain refresh is enough.

Run `node --check app/static/app.js` after editing the JavaScript. A syntax
error there is otherwise silent: the page renders and simply does nothing, which
has already happened once.

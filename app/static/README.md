# `app/static/` — the console

Two pages, served by the FastAPI application itself.

| Page | URL | Answers |
|------|-----|---------|
| **Triage** | `/` | *What should happen to this ticket?* |
| **Performance** | `/performance` | *Is this system any good?* |

| File | What it does |
|------|--------------|
| `index.html` | The triage page. |
| `performance.html` | The performance page. |
| `styles.css` | Shared styling for both. |
| `app.js` | Triage behaviour. |
| `performance.js` | Performance behaviour. |

## Why plain HTML, CSS and JavaScript

No React, no bundler, no `npm install`, no new Python dependency. Static files
served by the same FastAPI process, which means **same origin** (no CORS to get
wrong) and **nothing to build** — clone, start uvicorn, open the page. The
Dockerfile already copies `app/`, so the container serves it too.

The brief asks for a back-end service, so this is an addition for operating and
demonstrating the system, not a required deliverable.

## Why two pages and not one

They answer different questions for different readers, and only one of them is
about an individual ticket.

The triage page is a **work surface**: someone has a ticket and needs a
decision. It shows the suggested reply first and keeps the justification —
sentiment, sources, reasoning, stage timings — in collapsed sections, opened
when somebody asks *why did it say that?*

The performance page is an **assessment surface**: nothing on it belongs to one
ticket. Classifier metrics are computed over a held-out split; routing rewards
are aggregated across every rating for a whole *kind* of ticket; the pipeline
graph is the same for all of them. Wedging that into the ticket view buried the
most interesting part of the system inside a `<details>` element.

## What is on each page, and what is deliberately absent

**Triage** — subject, description, customer tier, the four-value summary
(category, urgency, action, handling time), the suggested reply, and the
feedback buttons. Collapsed: analysis, agent reasoning, processing steps.

**Performance** — three cards, each backed by one endpoint:

| Card | Endpoint | Shows |
|------|----------|-------|
| Classification quality | `/ml/report` | accuracy, precision, recall, F1, per-category table, confusion matrix |
| Routing | `/rl/stats` | ratings, ticket types seen, explore/exploit, and average reward per configuration for a chosen ticket type |
| Pipeline | `/workflow/graph` | the stages, in the order the engine derived, with the parallel level boxed |

Deliberately **not** here: a request log, live-updating charts, system
statistics, a synthetic-ticket simulator, and a ticket history. Each was either
interesting to build and useless to the reader, or not something the brief asks
the interface to show. `scripts/simulate_bandit.py` is where the learning
experiment belongs.

No number on either page is computed in the browser. Everything is served by
the API and rendered as it arrives, so the pages cannot drift from the service.

## Design

Neutral greys, one accent, and colour used only where it carries meaning.
**Every meaning-bearing colour clears WCAG AA (4.5:1) against the surface it
sits on** — measured, not eyeballed. The first pass used `#2a78d6` for the
accent and it came out at 4.42:1, so it was darkened to `#256abf` (5.2:1).

Colour appears in four places, and in each the word or number is present too,
so colour is never the only signal:

- urgency — low / medium / high
- the action — answer or escalate
- sentiment polarity — positive / neutral / negative
- reward bars — sign is shown by colour *and* by the printed number

The reward bar is a magnitude from a zero baseline. Negative rewards are normal
here, because reward subtracts latency and a local model takes longer than ten
seconds, so the axis recentres whenever any value is below zero.

Layout leans on whitespace and hairline rules rather than nested boxes, so the
page reads as a document rather than a dashboard. The only motion is the submit
spinner and a short fade when a result arrives; both are disabled under
`prefers-reduced-motion: reduce`.

## Behaviour worth knowing

- The previous result is **hidden the moment a new ticket is submitted**, so
  stale figures can never be mistaken for fresh ones.
- Errors are rendered as a sentence, not a status code. A failed pipeline names
  the stage that broke and offers a **retry** that re-runs only that stage.
- The feedback buttons lock after one use, matching the API, which accepts one
  rating per ticket.
- When a policy rule overrides the model's chosen action, the reasoning trace
  says so on the step it changed rather than hiding it.
- The performance page loads its three cards independently, so one unavailable
  endpoint degrades a single card instead of blanking the page. With no
  feedback yet, the routing card explains what to do rather than showing an
  empty table.

## Editing it

No build step — edit a file and reload the browser. `uvicorn --reload` watches
Python files; static files are read per request, so a plain refresh is enough.
Run `node --check app/static/app.js` after editing the JavaScript; a syntax
error there is otherwise silent until the page fails to render.

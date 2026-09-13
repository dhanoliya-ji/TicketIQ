# `app/static/` — the console

The operator interface, served by the FastAPI application itself at
**<http://localhost:8000/>**.

| File | What it does |
|------|--------------|
| `index.html` | Page structure: the ticket form and the result. |
| `styles.css` | All styling. Neutral greys, one accent colour. |
| `app.js` | API calls and rendering. |

## Why plain HTML, CSS and JavaScript

No React, no bundler, no `npm install`, no new Python dependency. Three static
files served by the same FastAPI process, which means:

- **Same origin**, so there is no CORS configuration to get wrong.
- **Nothing to build.** Clone, start uvicorn, open the page. The Dockerfile
  already copies `app/`, so the container serves it too.

The brief asked for a back-end service, so this is an addition for operating and
demonstrating the system, not a required deliverable.

## What is on the page, and what is deliberately not

The page answers one question — *what should happen to this ticket?* — so the
suggested reply is the largest thing on screen and everything that justifies it
is one click away in a collapsed section.

| Always visible | Why |
|----------------|-----|
| Subject, description, customer tier, submit | The only inputs the service takes |
| Category, urgency, action, handling time | The triage decision in four words |
| Suggested reply | The actual work product |
| Was this helpful? | The feedback that trains the routing |

| Collapsed | Opened when someone asks |
|-----------|--------------------------|
| **Analysis** | "Why that category?" — sentiment per aspect, and the exact knowledge base sections quoted |
| **Agent reasoning** | "Why did it escalate?" — each step, and any tool result |
| **Processing steps** | "Is it actually doing all that?" — the seven stages and their real durations, read from the workflow engine's state store |
| **Routing performance** | "What is it learning?" — reward per configuration for this kind of ticket |

Deliberately **not** here: request logs, live charts, system statistics, and a
simulator that fires synthetic tickets. They were interesting to build and
useless to someone doing the job. The same numbers remain available at
`/rl/stats`, `/ml/report` and `/workflow/graph` for anyone who wants them, and
`scripts/simulate_bandit.py` is where the learning experiment belongs.

## Colour

Neutral greys and a single accent (`--accent`, a blue that clears contrast on
white). Colour appears in exactly three places, and in all of them the word is
shown as well, so colour is never the only signal:

- urgency — low / medium / high
- the action — answer or escalate
- sentiment polarity — positive / neutral / negative

No gradients and no decorative animation. The only motion is the submit spinner
and a short fade when the result appears, and both are disabled under
`prefers-reduced-motion: reduce`.

## Behaviour worth knowing

- The previous result is **hidden the moment a new ticket is submitted**, so
  stale figures can never be mistaken for fresh ones.
- Errors are rendered as a sentence, not a status code: a failed pipeline names
  the stage that broke, and a duplicate rating says so.
- The feedback buttons lock after one use, matching the API, which accepts one
  rating per ticket.
- Every figure comes from a real API call. Nothing is pre-computed.

## Editing it

There is no build step — edit a file and reload the browser. `uvicorn --reload`
watches Python files; static files are read per request, so a plain refresh is
enough.

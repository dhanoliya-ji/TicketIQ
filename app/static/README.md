# `app/static/` — the live console

The demo UI, served by the FastAPI application itself at
**<http://localhost:8000/>**.

| File | What it does |
|------|--------------|
| `index.html` | Page structure: the five sections of the console. |
| `styles.css` | All styling, the colour tokens and the animations. |
| `app.js` | Everything behavioural: API calls, the DAG replay, rendering, the charts. |

## Why plain HTML, CSS and JavaScript

No React, no bundler, no `npm install`, no new Python dependency. Three static
files served by the same FastAPI process that serves the API, which means:

- **Same origin**, so there is no CORS configuration to get wrong.
- **Nothing to build.** Clone the repo, start uvicorn, open the page. The
  Dockerfile already copies `app/`, so the container serves it too.
- **Nothing to keep in sync.** The page reads `/workflow/graph`, `/health` and
  `/rl/stats`, so the DAG picture and the statistics *are* the service's real
  state rather than a drawing that drifts.

The brief asked for a back-end service, so this is an addition for
demonstrating the system, not a required deliverable. It adds no dependency and
does not touch any pipeline code.

## What the page shows

| Section | What it demonstrates |
|---------|----------------------|
| 1 Submit a ticket | Four one-click samples chosen to trigger four different agent behaviours |
| 2 Workflow engine | The DAG from `/workflow/graph`, animated per stage, with level 0 boxed as parallel |
| 3 Triage result | Category + confidence, urgency, aspect sentiment, retrieved chunks with scores, the full reasoning trace, the reply, and the feedback buttons |
| 4 Reinforcement learning | Stat tiles, average reward per configuration per state, and a 30-ticket demo that makes the bandit visibly converge |
| 5 Session activity | Every HTTP call the page has made |

## The DAG animation is a replay, not a live feed

The pipeline finishes in about 0.1 seconds, which is far too fast to watch. So
the page:

1. `POST /ticket` and waits for the real result;
2. `GET /ticket/{id}/status` to read the **real recorded per-stage durations**;
3. replays those durations, slowed by `REPLAY_SLOWDOWN` (26x), with each level's
   stages starting together because that is how they actually ran.

The millisecond figure on each finished node is the true measured duration, and
the note under the heading says the replay is slowed. Nothing is invented.

## Two colour systems, deliberately separate

**Data colours carry meaning** and live as tokens in `styles.css`:

- `--series-1..4` are the four bandit configurations. These four hues are a
  validated categorical set for this dark surface — they clear the colour-blind
  separation, chroma, lightness-band and contrast checks *as a group*. **Do not
  substitute them by eye**; re-validate if you change them.
- `--status-*` are used for sentiment polarity and urgency. A status colour is
  never alone: the word ("negative", "high") is always next to it.
- `--seq-*` is a single-hue blue ramp for magnitudes (confidence, similarity).

**Decorative colours carry no meaning** — the drifting hero gradient, the button
sheen — and are free to be vivid.

The reward chart also ships a plain `<table>` of the same numbers underneath, so
the values are readable without relying on colour or bar length at all.

## Accessibility and layout notes

- Every animation is disabled under `prefers-reduced-motion: reduce`, while the
  state changes themselves still happen.
- Bar rows are direct-labelled with their value and pull count; the colour
  swatch never carries identity alone.
- The layout collapses to one column below 720px.
- The page commits to a dark theme and paints its own background explicitly.

## Editing it

There is no build step — edit a file and reload the browser. If you run uvicorn
with `--reload`, note that it watches Python files; static files are read per
request, so a plain browser refresh is enough.

# `app/rl/` — online reinforcement learning

## Files

| File | What it does |
|------|--------------|
| `state.py` | Builds the state key from the context, and computes the reward. |
| `bandit.py` | `EpsilonGreedyContextualBandit`: selection, the update rule, inspection and persistence. |

## Why a contextual bandit

Choosing a pipeline configuration is a **one-step** decision: pick an arm, get a
reward, and the next ticket is unrelated to this one. There is no sequence of
states to plan through, so Q-learning's discount factors and next-state
bootstrapping would be machinery with nothing to do. A contextual bandit is the
exact model for *"one decision, immediate reward, and the right answer depends
on the context"*, and it learns from very few samples — which matters when every
sample costs a real customer interaction.

## State, action, reward

```
state  = category | urgency bucket | customer tier      (4 x 3 x 3 = 36 states)
action = prompt variant x RAG top-K                     (2 x 2 = 4 arms)
reward = (feedback score x 10) - latency in seconds
```

All three context features are already discrete, and they are exactly the three
things that should change the routing: an enterprise customer deserves the
careful answer, a high-urgency ticket justifies more retrieved context, a
feature request needs neither. Everything continuous is **bucketed before it
reaches the bandit** — a raw urgency of 0.5512 would create a state that never
repeats.

The reward is the formula the assignment specifies. A helpful answer is worth 10
points and every second of waiting costs one, so a configuration that is both
slow *and* unhelpful is abandoned first.

## The selection rule

1. **Cold start** — any arm never tried in this state is tried first, so no arm
   is written off on zero evidence.
2. **Explore** — with probability `epsilon` (default 0.15), pick at random.
3. **Exploit** — otherwise pick the arm with the best average reward.

`select_action` returns the reason (`cold_start`, `explore`, `exploit`) alongside
the choice, and it is surfaced as `config_selection_reason` in the API response.

## The update rule

```
new_average = old_average + (reward - old_average) / new_count
```

This produces exactly the same number as re-averaging the whole history, without
storing the history. Updates are guarded by a lock, because FastAPI serves
requests on several threads.

## Where the loop closes

```
POST /ticket    bandit picks an arm; the arm, state key and latency are stored on the ticket row
POST /feedback  that row is read back; reward = feedback*10 - latency; bandit.update(); state saved
```

Feedback can therefore arrive minutes later, or after a restart. It is applied
**once** per transaction (a repeat returns HTTP 409), because applying it twice
would quietly bias the averages. `save` and `load` persist the statistics to
`var/bandit_state.json`; a corrupt file is ignored rather than crashing startup.

## Known trade-off

36 states x 4 arms = 144 cells, and binary feedback is a noisy signal, so full
per-state convergence needs thousands of tickets. Keeping `category` in the
state is still the right modelling call — a feature request and an outage want
genuinely different handling — but a low-volume deployment should either drop
`category` or share statistics across states with a linear model such as LinUCB.
The `save`/`load` seam is also where shared storage would go if the service ever
ran as more than one replica.

## Evidence that it learns

`python scripts/simulate_bandit.py` runs this bandit and this reward function
against a simulated world where no arm is globally best:

| Tickets | Optimal choices, first fifth | Optimal choices, last fifth |
|---------|------------------------------|------------------------------|
| 5,000 | 54.4% | 75.9% |
| 20,000 | 67.7% | **88.8%** |

With `epsilon = 0.15` the ceiling is about 88.8%, so at 20,000 tickets the
bandit is essentially optimal.

## Tests

`tests/test_rl_bandit.py` (18 tests) checks the incremental average against a
plain average, the cold-start rule, the `epsilon = 0` and `epsilon = 1`
extremes, that learning in one state does not leak into another, convergence on
the best arm, and both persistence paths including a corrupt file.

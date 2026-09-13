# `scripts/` — the experiment scripts

Two standalone scripts that produce the numbers quoted in the root README. Both
run from the repository root and need no server.

| Script | Answers |
|--------|---------|
| `train_and_report.py` | How good is the ticket classifier on unseen data? |
| `simulate_bandit.py` | Does the bandit actually shift toward better configurations? |

---

## `train_and_report.py`

```bash
python scripts/train_and_report.py
```

Trains the classifier on a stratified 75/25 split (scikit-learn's
`train_test_split`) and prints accuracy, macro precision/recall/F1, per-category
metrics with support counts, and the confusion matrix - all scored with
`sklearn.metrics`.

It uses **exactly the same code path** the API uses
(`TicketClassifierService`), so the reported numbers cannot drift away from the
running service. The same figures are also served live at `GET /ml/report`.

Current output: **95.0% accuracy, 0.949 macro-F1**. The two errors are both
feature requests read as technical - tickets that genuinely mention both.

---

## `simulate_bandit.py`

```bash
python scripts/simulate_bandit.py          # 5,000 tickets (default)
python scripts/simulate_bandit.py 20000    # longer run
```

### What is real and what is simulated

**Real:** the bandit (`EpsilonGreedyContextualBandit`) and the reward function
(`compute_reward`) are the ones the service uses, imported directly.

**Simulated:** the outside world — a stream of tickets, how long each
configuration takes, and whether the customer found the answer helpful. Running
the real pipeline thousands of times would measure the offline template writer
rather than the learning rule, and would need thousands of humans to click a
thumb.

### The hidden world the bandit is not told about

| | fast + concise | slow + thorough |
|---|---|---|
| **low urgency** | best — the extra time is pure cost | worst |
| **high urgency** | worst — too terse to help | best — the detail pays for the latency |

So **no arm is globally best**; the right answer depends on the state, which is
the whole point of a *contextual* bandit. The differences are deliberately
exaggerated relative to real life so the effect is measurable within a few
thousand tickets — binary feedback times ten is a very noisy reward signal.

### What it prints

1. Action distribution in the first fifth of the run versus the last fifth.
2. Share of choices that matched the oracle, first window versus last.
3. Average reward per ticket across ten blocks, as a bar chart.
4. The arm learned versus the oracle arm for three example states.

### Measured results

| Tickets | Optimal choices, first fifth | Optimal choices, last fifth |
|---------|------------------------------|------------------------------|
| 5,000 | 54.4% | 75.9% |
| 20,000 | 67.7% | **88.8%** |

With `epsilon = 0.15` the theoretical ceiling is about 88.8% — 85% exploiting
plus a quarter of the 15% exploration landing on the best arm by chance — so at
20,000 tickets the bandit is essentially optimal. Average reward per ticket
rises from 5.66 to roughly 6.7 over the same run.

---

## A note on the `sys.path` line

Both scripts start with:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

so that `python scripts/whatever.py` works from a fresh clone without installing
the package first. The `# noqa: E402` comments on the imports below it are there
because those imports must come after that line.

Both scripts run in CI on every push, so neither can silently rot.

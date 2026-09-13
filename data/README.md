# `data/` — the labelled dataset and the knowledge base

## Files

| Path | What it is |
|------|------------|
| `generate_tickets.py` | Deterministic generator for the labelled dataset. |
| `tickets.json` | 160 labelled synthetic tickets, committed so training needs no setup step. |
| `knowledge_base/` | Five markdown policy documents — see [its README](knowledge_base/README.md). |

## The dataset

160 tickets, 40 per category, each with `id`, `subject`, `body`, `category` and
`customer_tier`.

```json
{
  "id": "T-0002",
  "subject": "Change the account owner",
  "body": "My role no longer has permission to open the settings of our workspace. ...",
  "category": "account",
  "customer_tier": "pro"
}
```

The four categories are `billing`, `technical`, `account` and `feature_request`.

### Why it is synthetic

Real support data cannot be shared, and the assignment asks for 100–200 labelled
tickets. Each category has its own vocabulary of subject lines and body
sentences, which are combined with neutral filler sentences.

### Why it is deliberately noisy

The first version scored a **perfect 1.000** accuracy, which means the task was
too easy to be informative — the four vocabularies did not overlap at all. The
generator now mixes one or two sentences from a *different* category into 45% of
tickets:

> "Please add my colleague as a user with administrator permissions. **The
> payment failed in the checkout but the amount left my bank account.**"

The label stays the category of the **subject line**, which is what a human
triager would go by. That brings accuracy to a believable **92.5%**, and the
remaining errors are exactly the genuinely ambiguous tickets. Real tickets are
messy in precisely this way.

### Regenerating it

```bash
python data/generate_tickets.py
```

The random seed is fixed (`RANDOM_SEED = 7`), so the output is byte-identical
every time — CI regenerates the file and fails if `git diff` shows any change.

To change the dataset, edit the `TEMPLATES` dictionary, the noise probability or
`TICKETS_PER_CATEGORY`, then regenerate and re-run
`python scripts/train_and_report.py`.

## How this data is used

| Consumer | Uses |
|----------|------|
| `app/ml/dataset.py` | loads `tickets.json` and does the stratified 75/25 split |
| `app/ml/classifier_service.py` | trains Naive Bayes at service startup |
| `app/rag/chunking.py` | splits `knowledge_base/*.md` into 28 retrievable chunks |
| `scripts/train_and_report.py` | prints the held-out metrics |

Nothing here is written to at run time. Everything the service writes goes to
`var/` (git-ignored).

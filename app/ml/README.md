# `app/ml/` — classical NLP and machine learning

No deep learning, and **no `model.fit()`** — the classifier's counting,
smoothing and log-probability scoring are written out by hand so the maths is
visible.

scikit-learn is used only where the brief permits it: the stratified split, the
TF-IDF vectorisation of the knowledge base, and scoring the classifier's
predictions. The from-scratch TF-IDF and metrics are kept alongside and tested
for agreement with it, so they are verified rather than merely asserted.

## Files

| File | What it does |
|------|--------------|
| `text_utils.py` | Tokenisation: lower-case, strip punctuation, drop stop words and very short tokens. Also `join_ticket_text`, which counts the subject line twice because a support subject is a strong summary of the problem. |
| `tfidf.py` | A TF-IDF vectoriser producing sparse vectors (plain `dict` of word to weight), plus `normalize` and `cosine_similarity`. Used by the retriever. |
| `naive_bayes.py` | Multinomial Naive Bayes from scratch: priors, per-category word counts, Laplace smoothing, log-space scoring and a softmax confidence. |
| `metrics.py` | `accuracy`, `precision`, `recall`, `f1_score`, `classification_report` and `confusion_matrix`, each written from its definition. |
| `dataset.py` | Loads `data/tickets.json` into `LabelledTicket` objects and does the stratified train/test split. |
| `classifier_service.py` | Trains the model at startup, keeps the held-out evaluation report, and answers predictions for new tickets. |
| `aspect_sentiment.py` | Aspect spotting by keyword plus per-sentence VADER scoring, with a domain lexicon that fixes VADER's blind spots on support text. |
| `urgency.py` | The weighted urgency score and its `low`/`medium`/`high` bucket. |

## The classifier, in one box

```
score(category) = log P(category) + sum over words of  count(word) * log P(word | category)

                                  count(word, category) + alpha
P(word | category)  =  --------------------------------------------------
                       total words in category + alpha * vocabulary size
```

Log space stops hundreds of small probabilities underflowing to zero. Laplace
smoothing (`alpha = 1.0`) stops a single unseen word zeroing a whole category.
Words outside the training vocabulary are **skipped** rather than smoothed, so
an unusual word adds no evidence instead of penalising every category equally.

The model trains on the training split to produce an honest quality report, then
retrains on all 160 rows for serving — the held-out split exists to measure, not
to throw data away.

## Aspect sentiment, in three steps

1. Split the ticket into sentences.
2. Spot aspects: `ASPECT_KEYWORDS` matches single words, `ASPECT_PHRASES`
   matches multi-word signals against the raw sentence.
3. Score each sentence with VADER; each aspect gets the mean of its sentences
   and keeps the harshest one as evidence. At most three aspects are returned,
   and at least one (a `general` fallback when nothing matches).

**Why phrases as well as words.** The feature aspect originally listed "add",
"support", "would", "consider" and "ability" as keywords. Those are politeness
words, not feature words: the aspect fired on 73 of the 160 dataset tickets, 33
of them billing, technical or account tickets. Because only three aspects are
returned, a spurious one *pushes a real one off the list* — a ticket praising
billing and complaining about speed lost the billing aspect entirely. The
keyword list is now narrow, the real signal ("would be great", "the ability
to") lives in phrases, and the false-positive count is down from 33 to 15 —
the remainder being tickets that genuinely do mention a feature.

**The domain lexicon matters.** VADER was built for social media, where
"support" is a positive word ("I support you"). On support tickets it is a
neutral product noun, so *"Support has not replied for days"* originally scored
**positive**. `DOMAIN_LEXICON` neutralises those product nouns and adds failure
words VADER does not know (`timeout`, `outage`, `unusable`, `nobody`).

## Urgency

```
urgency = category weight (<= 0.35) + negativity of worst aspect (<= 0.35) + tier weight (<= 0.30)
```

A transparent formula beats a learned model here: a support lead can read it,
disagree with one weight and change one number. The bucket — not the raw score —
feeds the RL state, because a continuous value would create states that never
repeat often enough to learn from.

## Tests

`tests/test_ml_classifier.py` (30 tests) checks the maths against hand-computed
values: the IDF formula, unit-length vectors, the smoothing denominator, and
each metric definition. `tests/test_nlp_and_rag.py` covers aspect independence,
the urgency weighting and the stratified split.

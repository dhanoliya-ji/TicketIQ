# `app/rag/` — retrieval augmented generation

Turns the markdown knowledge base into searchable chunks and finds the ones
relevant to a ticket.

## Files

| File | What it does |
|------|--------------|
| `chunking.py` | Splits each markdown document into one `KnowledgeChunk` per `##` section. |
| `vector_store.py` | `FaissVectorStore`: TF-IDF vectors for every chunk, held in a FAISS index. |
| `retriever.py` | `KnowledgeRetriever`: loads the knowledge base once at startup, searches it, and re-ranks by category. |

## How a chunk becomes something searchable

```
markdown  ──chunking.py──▶  KnowledgeChunk
                                  │
                     tfidf.py ────┤  sparse vector: {word: weight}, length 1
                                  │
              projection ─────────┤  dense float32 row over a fixed vocabulary
                                  │
                  FAISS ──────────▶  IndexFlatIP  ──▶ inner product = cosine
```

**The embeddings are TF-IDF**, produced by the hand-written vectoriser in
[`app/ml/tfidf.py`](../ml/tfidf.py) — no embedding model, no download, no API
call. The vectors are length-normalised, which is what lets FAISS's inner
product stand in for cosine similarity.

## Design choices

**Chunk on headings, not on a character count.** The knowledge base is written
as short, self-contained policy sections, so a section *is* the natural unit of
retrieval. It never cuts a rule in half, and the heading itself ("Refund
window", "Duplicate charges") carries strong keywords. Five documents produce
**28 chunks**.

**`IndexFlatIP`, not an approximate index.** `Flat` means exhaustive — FAISS
compares the query against every stored vector, so the results are exact rather
than approximate. An approximate index such as `IVFFlat` or `HNSW` only starts
paying for itself at hundreds of thousands of vectors, and has to be *trained*
on a sample first. Changing the index type is the one line that would change if
this knowledge base ever grew that far.

**The query is normalised before projection, not after.** A ticket usually
contains words the knowledge base has never seen. They add to the query's own
length but can never match anything, so they are counted when the vector is
normalised and then dropped when it is projected onto the vocabulary. The
result is a query vector of length ≤ 1 whose inner product with a chunk is
exactly the cosine of the two full vectors. Normalising after dropping them
would quietly inflate every score.

**Zero-similarity hits are dropped.** Returning two relevant chunks beats
padding to five with unrelated policy text the model may then quote. FAISS also
pads short results with `-1`, which is filtered out for the same reason.

## Category-aware re-ranking

Plain similarity was not enough. A total outage retrieved the *feature-request*
document — "request", "team" and "explain" are common to both — and the agent
then quoted *"thank the customer for the idea"* at someone whose platform was
down.

So each document declares the category it serves (`DOCUMENT_TOPICS`), and a
chunk from an unrelated document has its score multiplied by
`OFF_TOPIC_MULTIPLIER` (0.45) before the top-K is taken. The escalation rules
apply to every ticket and are never damped.

An off-topic chunk therefore has to be more than twice the textual match to
survive — which keeps it reachable when it genuinely is the better answer (a
refund question misclassified as technical still finds the refund policy) and
out of the way the rest of the time.

## How top-K gets chosen

The retriever does not pick K — the contextual bandit does, in the
`select_configuration` stage that runs before this one. K is 2 or 5, and it is
one of the two dimensions the RL layer learns over, because more context costs
latency and only pays off on harder tickets.

## Worked example

```python
retriever = KnowledgeRetriever(SETTINGS.knowledge_base_dir).load()
hits = retriever.retrieve(
    subject="Charged twice",
    body="My card was charged twice for the same invoice, I want a refund",
    category="billing",
    top_k=3,
)
# 0.246  01_billing_refund_policy.md  | Duplicate charges
# 0.106  01_billing_refund_policy.md  | Invoices and tax
# 0.093  01_billing_refund_policy.md  | Refund window
```

## Tests

`tests/test_nlp_and_rag.py` covers heading splitting, empty sections, unique
chunk ids, that the store really is a `faiss.IndexFlatIP` holding one vector per
chunk, that FAISS's scores equal the cosine computed independently, asking for
more chunks than exist, the zero-overlap filter, top-K limits, and every branch
of the category re-ranking.

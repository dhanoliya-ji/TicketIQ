# `app/rag/` — retrieval augmented generation

Turns the markdown knowledge base into searchable chunks and finds the ones
relevant to a ticket.

## Files

| File | What it does |
|------|--------------|
| `chunking.py` | Splits each markdown document into one `KnowledgeChunk` per `##` section. |
| `vector_store.py` | `InMemoryVectorStore`: one TF-IDF vector per chunk, searched by cosine similarity. |
| `retriever.py` | `KnowledgeRetriever`: loads the knowledge base once at startup and answers top-K queries. |

## Design choices

**Chunk on headings, not on a character count.** The knowledge base is written
as short, self-contained policy sections, so a section *is* the natural unit of
retrieval. It never cuts a rule in half, and the heading itself ("Refund
window", "Duplicate charges") carries strong keywords that improve the match.
Five documents produce **28 chunks**.

**An in-memory cosine index, not FAISS or ChromaDB.** The assignment allows all
three. With 28 chunks, brute force — score every chunk, sort, take the best K —
is faster than either library, adds no native dependency to the Docker image,
and keeps the retrieval maths readable. `InMemoryVectorStore.search()` has the
same shape a FAISS-backed version would, so replacing it would touch one file.

**The predicted category is prepended to the query.** The search text is
`category + subject + body`. The category acts as a cheap hint that pulls a
billing ticket toward the refund policy document.

**Zero-similarity hits are dropped.** Returning two relevant chunks is better
than padding to five with unrelated policy text that the model may then quote.

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
# 0.283  01_billing_refund_policy.md  | Duplicate charges
# 0.094  01_billing_refund_policy.md  | Refund window
# 0.088  05_escalation_rules.md       | When to escalate to a human specialist
```

## Tests

`tests/test_nlp_and_rag.py` covers heading splitting, empty sections, unique
chunk ids, descending score order, the zero-overlap filter, top-K limits, and
the guard that the retriever must be loaded before use.

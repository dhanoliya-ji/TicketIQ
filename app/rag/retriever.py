"""The retrieval stage of the pipeline: ticket text in, knowledge snippets out.

Plain cosine similarity over the whole knowledge base turned out not to be
enough. A technical outage would retrieve the feature-request document, because
words like "request", "team" and "explain" are common to both, and the agent
would then quote *"thank the customer for the idea"* back at someone whose
platform was down.

So retrieval is category aware: each knowledge base document declares which
ticket category it serves, and a chunk from an unrelated document has its score
damped before the top-K is taken. The escalation rules apply to every ticket
and are never damped. This is ordinary metadata-aware retrieval - a document
from the wrong topic has to be a *much* better textual match to survive, rather
than being excluded outright.
"""

from pathlib import Path

from app.rag.chunking import chunk_knowledge_base
from app.rag.vector_store import InMemoryVectorStore, SearchHit

# Which ticket category each knowledge base document serves. The key is a
# prefix of the file name, so renaming a file only means editing this table.
DOCUMENT_TOPICS: dict[str, str] = {
    "01_billing_refund_policy": "billing",
    "02_account_access_troubleshooting": "account",
    "03_technical_troubleshooting": "technical",
    "04_feature_request_handling": "feature_request",
}

# Documents that apply to every ticket, whatever its category.
UNIVERSAL_DOCUMENTS = {"05_escalation_rules"}

# How much a chunk from an unrelated document is damped. At 0.45 an off-topic
# chunk needs to be more than twice as good a textual match to outrank an
# on-topic one, which is the behaviour we want: usually ignored, but still
# reachable when it is clearly the better answer.
OFF_TOPIC_MULTIPLIER = 0.45

# How many candidates to score before re-ranking. Wide enough that damped
# on-topic chunks can still climb into the final top-K.
CANDIDATE_POOL_SIZE = 12


def topic_of_document(source_file_name: str) -> str | None:
    """Return the category a document serves, or None if it serves all of them."""
    stem = source_file_name
    if stem.endswith(".md"):
        stem = stem[: -len(".md")]

    if stem in UNIVERSAL_DOCUMENTS:
        return None
    return DOCUMENT_TOPICS.get(stem)


class KnowledgeRetriever:
    """Loads the knowledge base once and answers top-K queries from memory."""

    def __init__(self, knowledge_base_dir: Path) -> None:
        self.knowledge_base_dir = knowledge_base_dir
        self.store = InMemoryVectorStore()
        self.is_loaded = False

    def load(self) -> "KnowledgeRetriever":
        """Chunk every markdown document and build the vector index."""
        chunks = chunk_knowledge_base(self.knowledge_base_dir)
        self.store.build(chunks)
        self.is_loaded = True
        return self

    def retrieve(self, subject: str, body: str, category: str, top_k: int) -> list[SearchHit]:
        """Find the knowledge chunks most relevant to one ticket.

        The predicted category does two jobs here: it is added to the query
        text as a keyword hint, and it decides which documents are on topic.
        """
        if not self.is_loaded:
            raise RuntimeError("the retriever must be loaded before it can retrieve")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query = category + ". " + subject + ". " + body

        # Score a wide pool first, so the re-ranking has something to work with.
        pool_size = max(CANDIDATE_POOL_SIZE, top_k)
        candidates = self.store.search(query, pool_size)

        reranked = []
        for hit in candidates:
            topic = topic_of_document(hit.chunk.source)

            if topic is None or topic == category:
                adjusted_score = hit.score
            else:
                adjusted_score = hit.score * OFF_TOPIC_MULTIPLIER

            reranked.append(SearchHit(hit.chunk, adjusted_score))

        # Best first; the chunk id breaks ties so results stay stable.
        reranked.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return reranked[:top_k]

    def chunk_count(self) -> int:
        return self.store.size()

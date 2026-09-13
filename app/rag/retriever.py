"""The retrieval stage of the pipeline: ticket text in, knowledge snippets out."""

from pathlib import Path

from app.rag.chunking import chunk_knowledge_base
from app.rag.vector_store import InMemoryVectorStore, SearchHit


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

        The predicted category is added to the query text on purpose: it acts
        as a cheap hint that pulls the search toward the right document, for
        example toward the refund policy for a "billing" ticket.
        """
        if not self.is_loaded:
            raise RuntimeError("the retriever must be loaded before it can retrieve")

        query = category + ". " + subject + ". " + body
        return self.store.search(query, top_k)

    def chunk_count(self) -> int:
        return self.store.size()

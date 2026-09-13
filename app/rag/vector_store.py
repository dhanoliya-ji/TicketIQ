"""A small in-memory vector store with cosine similarity search.

The assignment allows FAISS, ChromaDB or a basic in-memory cosine index.  With
28 knowledge base chunks a hand written index is faster than either library,
adds no native dependency to the Docker image, and - more importantly - the
retrieval maths stays readable: score every chunk, sort, return the best K.

If the knowledge base ever grew to millions of chunks this brute force search
would be the first thing to replace with FAISS; the ``search`` signature is
deliberately the same shape, so only this file would change.
"""

from app.ml.tfidf import SparseVector, TfidfVectorizer, cosine_similarity
from app.rag.chunking import KnowledgeChunk


class SearchHit:
    """One chunk returned by a search, together with its similarity score."""

    def __init__(self, chunk: KnowledgeChunk, score: float) -> None:
        self.chunk = chunk
        self.score = score

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = dict(self.chunk.as_dict())
        result["score"] = round(self.score, 4)
        return result


class InMemoryVectorStore:
    """Holds one TF-IDF vector per chunk and searches them by cosine similarity."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer()
        self.chunks: list[KnowledgeChunk] = []
        self.vectors: list[SparseVector] = []

    def build(self, chunks: list[KnowledgeChunk]) -> "InMemoryVectorStore":
        """Fit the vectoriser on the chunks and embed every one of them."""
        if len(chunks) == 0:
            raise ValueError("cannot build a vector store from zero chunks")

        self.chunks = list(chunks)

        documents = []
        for chunk in self.chunks:
            documents.append(chunk.searchable_text())

        # The IDF values must be learned from the corpus we are going to
        # search, which is why fit() happens here and not at import time.
        self.vectorizer.fit(documents)

        self.vectors = []
        for document in documents:
            self.vectors.append(self.vectorizer.transform(document))

        return self

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        """Return the ``top_k`` chunks most similar to the query text."""
        if len(self.chunks) == 0:
            raise RuntimeError("the vector store must be built before searching")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_vector = self.vectorizer.transform(query)

        hits: list[SearchHit] = []
        for index in range(len(self.chunks)):
            score = cosine_similarity(query_vector, self.vectors[index])
            hits.append(SearchHit(self.chunks[index], score))

        # Highest score first; the chunk id breaks ties so results are stable.
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))

        # Drop chunks with no overlap at all rather than padding the answer
        # with irrelevant policy text.
        useful_hits = []
        for hit in hits[:top_k]:
            if hit.score > 0.0:
                useful_hits.append(hit)
        return useful_hits

    def size(self) -> int:
        return len(self.chunks)

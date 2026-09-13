"""The vector store: a FAISS index over the knowledge base chunks.

How the pieces fit
------------------
1. Each chunk is turned into a **TF-IDF** vector by the hand-written vectoriser
   in ``app/ml/tfidf.py``. Those vectors are sparse (a dictionary of word to
   weight) and length-normalised.
2. The sparse vectors are projected onto a fixed vocabulary to give dense
   ``float32`` rows, which is the shape FAISS works with.
3. The rows go into a ``faiss.IndexFlatIP`` - a flat index scored by **inner
   product**. Because every stored vector has length 1, the inner product of a
   query with a chunk *is* their cosine similarity, so no separate cosine step
   is needed.

Why ``IndexFlatIP`` and not an approximate index
------------------------------------------------
``Flat`` means exhaustive: FAISS compares the query against every stored
vector, so the results are exact rather than approximate. With a knowledge base
of a few dozen chunks that is the right trade - an approximate index such as
``IVFFlat`` or ``HNSW`` only starts paying for itself at hundreds of thousands
of vectors, and it has to be *trained* on a sample first. The index type is the
one line that would change if this knowledge base ever grew that far.

Why the query vector is normalised in sparse space
--------------------------------------------------
A query usually contains words the knowledge base has never seen. Those words
contribute to the query's own length but can never match anything, so they are
included when the vector is normalised and then dropped during projection. The
result is a query vector whose length is at most 1, and whose inner product
with a chunk is exactly the cosine of the two full vectors. Normalising after
dropping them would quietly inflate every score.
"""

import faiss
import numpy as np

from app.ml.tfidf import SparseVector, TfidfVectorizer
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


class FaissVectorStore:
    """A FAISS flat inner-product index over TF-IDF chunk vectors."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer()
        self.chunks: list[KnowledgeChunk] = []

        # The fixed word order that turns a sparse vector into a dense row.
        self.vocabulary: list[str] = []
        self.word_positions: dict[str, int] = {}

        # The FAISS index itself, built in build().
        self.index: faiss.Index | None = None

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def build(self, chunks: list[KnowledgeChunk]) -> "FaissVectorStore":
        """Embed every chunk and load the vectors into a FAISS index."""
        if len(chunks) == 0:
            raise ValueError("cannot build a vector store from zero chunks")

        self.chunks = list(chunks)

        documents = []
        for chunk in self.chunks:
            documents.append(chunk.searchable_text())

        # The IDF values must be learned from the corpus we are going to
        # search, which is why fit() happens here and not at import time.
        self.vectorizer.fit(documents)

        # A stable word order, so a word always lands in the same column.
        self.vocabulary = sorted(self.vectorizer.inverse_document_frequency.keys())
        self.word_positions = {}
        for position in range(len(self.vocabulary)):
            self.word_positions[self.vocabulary[position]] = position

        # One dense row per chunk.
        matrix = np.zeros((len(self.chunks), len(self.vocabulary)), dtype="float32")
        for row in range(len(documents)):
            sparse_vector = self.vectorizer.transform(documents[row])
            self._write_dense_row(sparse_vector, matrix[row])

        # Inner product on unit-length vectors is cosine similarity.
        index = faiss.IndexFlatIP(len(self.vocabulary))
        index.add(matrix)
        self.index = index

        return self

    def _write_dense_row(self, sparse_vector: SparseVector, row: np.ndarray) -> None:
        """Copy a sparse vector into a dense row, dropping unknown words."""
        for word, weight in sparse_vector.items():
            position = self.word_positions.get(word)
            if position is not None:
                row[position] = weight

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int) -> list[SearchHit]:
        """Return the ``top_k`` chunks most similar to the query text."""
        if self.index is None:
            raise RuntimeError("the vector store must be built before searching")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_vector = np.zeros((1, len(self.vocabulary)), dtype="float32")
        self._write_dense_row(self.vectorizer.transform(query), query_vector[0])

        # FAISS cannot return more neighbours than it holds.
        wanted = min(top_k, len(self.chunks))
        scores, positions = self.index.search(query_vector, wanted)

        hits: list[SearchHit] = []
        for column in range(wanted):
            position = int(positions[0][column])
            # FAISS marks "no more results" with -1.
            if position < 0:
                continue

            score = float(scores[0][column])
            # Drop chunks with no word overlap at all rather than padding the
            # answer with irrelevant policy text.
            if score <= 0.0:
                continue

            hits.append(SearchHit(self.chunks[position], score))

        # FAISS returns best-first already; sorting again makes ties stable
        # between runs by falling back to the chunk id.
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return hits

    def size(self) -> int:
        return len(self.chunks)

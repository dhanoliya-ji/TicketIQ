"""The vector store: a FAISS index over the knowledge base chunks.

How the pieces fit
------------------
1. Each chunk is turned into a **TF-IDF** vector by scikit-learn's
   ``TfidfVectorizer``, which the assignment names for exactly this job
   ("scikit-learn ... for vectorization/metrics only"). It is given this
   project's own ``tokenize`` function as its analyser, so the vocabulary is
   the same lower-cased, stop-word-filtered one the classifier sees.
2. The vectors come out L2-normalised and sparse; ``toarray`` turns them into
   the dense ``float32`` rows FAISS indexes.
3. The rows go into a ``faiss.IndexFlatIP`` - a flat index scored by **inner
   product**. Because every stored vector has length 1, the inner product of a
   query with a chunk *is* their cosine similarity, so no separate cosine step
   is needed.

On the hand-written TF-IDF next door
------------------------------------
``app/ml/tfidf.py`` implements the same weighting from its definition, and
``test_hand_written_tfidf_matches_sklearn`` shows the two agree to floating
point epsilon on the real knowledge base. That is not an accident: our term
frequency divides by the document length where scikit-learn does not, and L2
normalisation divides that constant straight back out again. Keeping both means
the from-scratch version is verified against a reference rather than merely
asserted, which is the same arrangement used for the evaluation metrics.

Why ``IndexFlatIP`` and not an approximate index
------------------------------------------------
``Flat`` means exhaustive: FAISS compares the query against every stored
vector, so the results are exact rather than approximate. An approximate index
such as ``IVFFlat`` or ``HNSW`` only starts paying for itself at hundreds of
thousands of vectors, and it has to be *trained* on a sample first. The index
type is the one line that would change if this knowledge base ever grew that
far.
"""

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.ml.text_utils import tokenize
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


def build_vectorizer() -> TfidfVectorizer:
    """The TF-IDF settings used for the knowledge base.

    ``analyzer=tokenize`` hands scikit-learn this project's tokenizer, so it
    does no preprocessing of its own and the vocabulary matches the rest of the
    system. The remaining arguments are spelled out rather than left to default
    so the weighting is readable here instead of in scikit-learn's docs.
    """
    return TfidfVectorizer(
        analyzer=tokenize,
        norm="l2",  # unit length, which is what makes inner product = cosine
        smooth_idf=True,  # idf = log((1 + n) / (1 + df)) + 1
        sublinear_tf=False,  # plain term counts, not 1 + log(count)
    )


class FaissVectorStore:
    """A FAISS flat inner-product index over TF-IDF chunk vectors."""

    def __init__(self) -> None:
        self.vectorizer = build_vectorizer()
        self.chunks: list[KnowledgeChunk] = []
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
        # search, which is why fit happens here and not at import time.
        sparse_matrix = self.vectorizer.fit_transform(documents)

        # FAISS wants a dense, C-contiguous float32 array.
        dense_matrix = np.asarray(sparse_matrix.todense(), dtype="float32")

        index = faiss.IndexFlatIP(dense_matrix.shape[1])
        index.add(dense_matrix)
        self.index = index

        return self

    @property
    def vocabulary(self) -> list[str]:
        """The words the index has a column for, in column order."""
        if self.index is None:
            return []
        return list(self.vectorizer.get_feature_names_out())

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int) -> list[SearchHit]:
        """Return the ``top_k`` chunks most similar to the query text."""
        if self.index is None:
            raise RuntimeError("the vector store must be built before searching")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        # Words the knowledge base has never seen have no column, so they are
        # simply absent from the query vector.
        query_matrix = self.vectorizer.transform([query])
        query_vector = np.asarray(query_matrix.todense(), dtype="float32")

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

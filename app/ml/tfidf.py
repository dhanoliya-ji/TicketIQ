"""A hand written TF-IDF vectoriser.

A document is turned into a *sparse vector*: a plain dictionary that maps a
word to its weight.  Words that are not in the document simply do not appear
in the dictionary, which keeps memory small and the code readable.

The weight of a word is ``term frequency * inverse document frequency``:

* term frequency  - how often the word appears in *this* document.
* inverse document frequency - how rare the word is across *all* documents.
  A word appearing in every document gets a weight close to zero.

The vectors are length-normalised so that cosine similarity between two
vectors is just the sum of their overlapping weights.
"""

import math

from app.ml.text_utils import count_tokens, tokenize

# Type alias: a sparse vector is "word -> weight".
SparseVector = dict[str, float]


class TfidfVectorizer:
    """Learns a vocabulary and IDF weights from a collection of documents."""

    def __init__(self) -> None:
        # How many documents were used during fitting.
        self.document_count: int = 0
        # word -> in how many documents does this word appear at least once.
        self.document_frequency: dict[str, int] = {}
        # word -> inverse document frequency value.
        self.inverse_document_frequency: dict[str, float] = {}

    def fit(self, documents: list[str]) -> "TfidfVectorizer":
        """Learn the vocabulary and the IDF value of every word."""
        self.document_count = len(documents)
        self.document_frequency = {}

        for document in documents:
            tokens = tokenize(document)
            # `set` so that a word repeated inside one document is counted once.
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token in self.document_frequency:
                    self.document_frequency[token] = self.document_frequency[token] + 1
                else:
                    self.document_frequency[token] = 1

        # Smoothed IDF: adding 1 to both parts avoids dividing by zero and
        # keeps the value positive even for a word present in every document.
        self.inverse_document_frequency = {}
        for word, frequency in self.document_frequency.items():
            idf = math.log((1.0 + self.document_count) / (1.0 + frequency)) + 1.0
            self.inverse_document_frequency[word] = idf

        return self

    def transform(self, document: str) -> SparseVector:
        """Turn one document into a normalised sparse TF-IDF vector."""
        tokens = tokenize(document)
        if len(tokens) == 0:
            return {}

        token_counts = count_tokens(tokens)
        total_tokens = len(tokens)

        vector: SparseVector = {}
        for word, count in token_counts.items():
            # A word never seen during fitting gets the highest possible IDF,
            # because "unseen" means "maximally rare".
            if word in self.inverse_document_frequency:
                idf = self.inverse_document_frequency[word]
            else:
                idf = math.log(1.0 + self.document_count) + 1.0

            term_frequency = count / total_tokens
            vector[word] = term_frequency * idf

        return normalize(vector)


def normalize(vector: SparseVector) -> SparseVector:
    """Scale a vector so that its length (magnitude) becomes exactly 1."""
    sum_of_squares = 0.0
    for weight in vector.values():
        sum_of_squares = sum_of_squares + (weight * weight)

    length = math.sqrt(sum_of_squares)
    if length == 0.0:
        return {}

    normalized: SparseVector = {}
    for word, weight in vector.items():
        normalized[word] = weight / length
    return normalized


def cosine_similarity(left: SparseVector, right: SparseVector) -> float:
    """Similarity of two normalised vectors, between 0.0 and 1.0.

    Because both vectors already have length 1, the cosine is simply the sum
    of the products of the weights of the words they have in common.
    """
    # Always loop over the smaller dictionary: the result is the same and the
    # loop is shorter.
    if len(left) > len(right):
        left, right = right, left

    total = 0.0
    for word, weight in left.items():
        if word in right:
            total = total + (weight * right[word])
    return total

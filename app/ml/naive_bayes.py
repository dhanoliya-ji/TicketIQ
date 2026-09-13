"""A Multinomial Naive Bayes text classifier written from scratch.

No ``model.fit()`` from scikit-learn is used anywhere here - the counting,
the smoothing and the log-probability scoring are all implemented directly so
the maths is visible.

The idea in one paragraph
------------------------
For a ticket we want the most likely category.  Bayes' rule says the best
category is the one that maximises ``P(category) * P(words | category)``.
"Naive" means we pretend the words are independent of each other, so
``P(words | category)`` becomes the product of ``P(word | category)`` for every
word.  Multiplying many small probabilities underflows to zero, so we add
logarithms instead of multiplying probabilities.  Unseen words would give a
probability of zero and wipe out the whole product, so we use Laplace
smoothing: pretend every word in the vocabulary was seen ``alpha`` extra times.
"""

import math

from app.ml.text_utils import count_tokens, tokenize


class NaiveBayesTextClassifier:
    """Multinomial Naive Bayes over bag-of-words counts."""

    def __init__(self, alpha: float = 1.0) -> None:
        # Laplace smoothing strength. 1.0 is the classic "add one" choice.
        self.alpha: float = alpha

        # Every category the model has seen, in a stable sorted order.
        self.categories: list[str] = []
        # Every word the model has seen, used as the denominator vocabulary.
        self.vocabulary: set[str] = set()

        # category -> log P(category)
        self.log_prior: dict[str, float] = {}
        # category -> {word -> how many times the word occurred in that category}
        self.word_counts: dict[str, dict[str, int]] = {}
        # category -> total number of word occurrences in that category
        self.total_words: dict[str, int] = {}

        self.is_trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, documents: list[str], labels: list[str]) -> "NaiveBayesTextClassifier":
        """Count words per category and turn the counts into probabilities."""
        if len(documents) != len(labels):
            raise ValueError("documents and labels must have the same length")
        if len(documents) == 0:
            raise ValueError("cannot train on an empty dataset")

        self.categories = sorted(set(labels))
        self.vocabulary = set()
        self.word_counts = {}
        self.total_words = {}

        # Start every category with empty counters.
        documents_per_category: dict[str, int] = {}
        for category in self.categories:
            self.word_counts[category] = {}
            self.total_words[category] = 0
            documents_per_category[category] = 0

        # Walk the training set once and fill the counters.
        for index in range(len(documents)):
            document = documents[index]
            category = labels[index]

            documents_per_category[category] = documents_per_category[category] + 1

            tokens = tokenize(document)
            for word, count in count_tokens(tokens).items():
                self.vocabulary.add(word)

                category_counts = self.word_counts[category]
                if word in category_counts:
                    category_counts[word] = category_counts[word] + count
                else:
                    category_counts[word] = count

                self.total_words[category] = self.total_words[category] + count

        # Prior: how common is each category in the training data.
        total_documents = len(documents)
        self.log_prior = {}
        for category in self.categories:
            prior = documents_per_category[category] / total_documents
            # A category with zero documents cannot happen here because the
            # category list is built from the labels themselves.
            self.log_prior[category] = math.log(prior)

        self.is_trained = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def log_likelihood_of_word(self, word: str, category: str) -> float:
        """log P(word | category) with Laplace smoothing."""
        count = self.word_counts[category].get(word, 0)
        numerator = count + self.alpha
        denominator = self.total_words[category] + (self.alpha * len(self.vocabulary))
        return math.log(numerator / denominator)

    def score_all_categories(self, document: str) -> dict[str, float]:
        """Return the (unnormalised) log score of every category."""
        self._require_trained()

        tokens = tokenize(document)
        token_counts = count_tokens(tokens)

        scores: dict[str, float] = {}
        for category in self.categories:
            # Start from the prior, then add the evidence from every word.
            score = self.log_prior[category]
            for word, count in token_counts.items():
                # Words never seen in training carry no information, so we skip
                # them instead of letting smoothing push every score down.
                if word not in self.vocabulary:
                    continue
                score = score + (count * self.log_likelihood_of_word(word, category))
            scores[category] = score
        return scores

    def predict(self, document: str) -> str:
        """Return the single most likely category for one document."""
        scores = self.score_all_categories(document)

        best_category = self.categories[0]
        best_score = scores[best_category]
        for category, score in scores.items():
            if score > best_score:
                best_category = category
                best_score = score
        return best_category

    def predict_with_confidence(self, document: str) -> tuple[str, float]:
        """Return the best category plus a probability between 0.0 and 1.0.

        The log scores are converted back into probabilities with the softmax
        trick: subtract the largest score first so that ``exp`` never overflows.
        """
        scores = self.score_all_categories(document)

        largest_score = max(scores.values())
        exponentials: dict[str, float] = {}
        total = 0.0
        for category, score in scores.items():
            value = math.exp(score - largest_score)
            exponentials[category] = value
            total = total + value

        probabilities: dict[str, float] = {}
        for category, value in exponentials.items():
            probabilities[category] = value / total

        best_category = self.predict(document)
        return best_category, probabilities[best_category]

    def predict_many(self, documents: list[str]) -> list[str]:
        """Convenience wrapper: predict a whole list of documents."""
        predictions = []
        for document in documents:
            predictions.append(self.predict(document))
        return predictions

    def _require_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError("the classifier must be trained before it can predict")

"""Trains the Naive Bayes classifier once and answers predictions from memory.

The dataset is tiny (160 tickets) so training takes milliseconds.  We simply
train when the service starts instead of saving a model file, which removes a
whole class of "stale artefact" problems.
"""

from pathlib import Path

from app.ml.dataset import CATEGORIES, load_tickets, stratified_split
from app.ml.naive_bayes import NaiveBayesTextClassifier
from app.ml.sklearn_metrics import classification_report as sklearn_classification_report
from app.ml.sklearn_metrics import confusion as sklearn_confusion_matrix
from app.ml.text_utils import join_ticket_text


class ClassificationResult:
    """The outcome of classifying one ticket."""

    def __init__(self, category: str, confidence: float, scores: dict[str, float]) -> None:
        self.category = category
        self.confidence = confidence
        # Raw log scores, useful when debugging a surprising prediction.
        self.scores = scores

    def as_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "scores": {name: round(value, 4) for name, value in self.scores.items()},
        }


class TicketClassifierService:
    """Owns the trained model and the held-out evaluation report."""

    def __init__(self, dataset_path: Path, test_fraction: float, seed: int) -> None:
        self.dataset_path = dataset_path
        self.test_fraction = test_fraction
        self.seed = seed

        self.model = NaiveBayesTextClassifier(alpha=1.0)
        self.evaluation: dict[str, object] = {}
        self.confusion: dict[str, dict[str, int]] = {}

    def train(self) -> "TicketClassifierService":
        """Load the dataset, fit the model on the training split, evaluate it.

        The model is finally re-trained on *all* rows: the held-out split
        exists to produce an honest quality report, but the model that serves
        traffic should learn from every example we have.
        """
        tickets = load_tickets(self.dataset_path)
        training_tickets, test_tickets = stratified_split(tickets, self.test_fraction, self.seed)

        # --- fit on the training split and score the held-out split ---
        training_texts = [ticket.as_text() for ticket in training_tickets]
        training_labels = [ticket.category for ticket in training_tickets]
        self.model.train(training_texts, training_labels)

        test_texts = [ticket.as_text() for ticket in test_tickets]
        test_labels = [ticket.category for ticket in test_tickets]
        predicted_labels = self.model.predict_many(test_texts)

        # The published report comes from scikit-learn, which the assignment
        # allows for evaluation utilities. Only the *scoring* is scikit-learn's;
        # every prediction it is scoring came from the hand-written model above.
        self.evaluation = sklearn_classification_report(test_labels, predicted_labels, CATEGORIES)
        self.evaluation["training_size"] = len(training_tickets)
        self.evaluation["test_size"] = len(test_tickets)
        self.confusion = sklearn_confusion_matrix(test_labels, predicted_labels, CATEGORIES)

        # --- final model: trained on everything ---
        all_texts = [ticket.as_text() for ticket in tickets]
        all_labels = [ticket.category for ticket in tickets]
        self.model.train(all_texts, all_labels)

        return self

    def classify(self, subject: str, body: str) -> ClassificationResult:
        """Predict the category of a new, unseen ticket."""
        text = join_ticket_text(subject, body)
        category, confidence = self.model.predict_with_confidence(text)
        scores = self.model.score_all_categories(text)
        return ClassificationResult(category, confidence, scores)

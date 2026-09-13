"""Loading and splitting the labelled ticket dataset."""

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from app.ml.text_utils import join_ticket_text

# The four categories the classifier can produce.
CATEGORIES = ["account", "billing", "feature_request", "technical"]


class LabelledTicket:
    """One row of the training dataset."""

    def __init__(self, ticket_id: str, subject: str, body: str, category: str, tier: str) -> None:
        self.ticket_id = ticket_id
        self.subject = subject
        self.body = body
        self.category = category
        self.customer_tier = tier

    def as_text(self) -> str:
        """The text the classifier actually sees."""
        return join_ticket_text(self.subject, self.body)


def load_tickets(path: Path) -> list[LabelledTicket]:
    """Read the JSON dataset from disk into LabelledTicket objects."""
    if not path.exists():
        raise FileNotFoundError(
            "Ticket dataset not found at " + str(path) + ". Run: python data/generate_tickets.py"
        )

    raw_rows = json.loads(path.read_text(encoding="utf-8"))

    tickets = []
    for row in raw_rows:
        tickets.append(
            LabelledTicket(
                ticket_id=row["id"],
                subject=row["subject"],
                body=row["body"],
                category=row["category"],
                tier=row.get("customer_tier", "free"),
            )
        )
    return tickets


def stratified_split(
    tickets: list[LabelledTicket], test_fraction: float, seed: int
) -> tuple[list[LabelledTicket], list[LabelledTicket]]:
    """Split into (training set, test set), keeping category balance.

    "Stratified" means the held-out set contains the same proportion of every
    category as the full dataset.  A plain random split could, by chance, leave
    a category out of one side entirely and make the report meaningless.

    This is scikit-learn's ``train_test_split``, which the assignment allows for
    exactly this kind of surrounding utility.  The classifier's own training and
    inference maths is still written out by hand in ``naive_bayes.py`` - no
    ``model.fit()`` anywhere.
    """
    labels = [ticket.category for ticket in tickets]

    training_set, test_set = train_test_split(
        tickets,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels,  # this is what keeps the category balance
        shuffle=True,
    )
    return list(training_set), list(test_set)

"""Loading and splitting the labelled ticket dataset."""

import json
import random
from pathlib import Path

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

    "Stratified" means we split each category separately, so the held-out set
    contains roughly the same proportion of every category as the full
    dataset.  A plain random split could by chance leave a category out.
    """
    random_generator = random.Random(seed)

    # Group the tickets by their category.
    grouped: dict[str, list[LabelledTicket]] = {}
    for ticket in tickets:
        if ticket.category not in grouped:
            grouped[ticket.category] = []
        grouped[ticket.category].append(ticket)

    training_set: list[LabelledTicket] = []
    test_set: list[LabelledTicket] = []

    for category in sorted(grouped.keys()):
        group = list(grouped[category])
        random_generator.shuffle(group)

        test_size = int(round(len(group) * test_fraction))
        # Never let a category disappear completely from either side.
        if test_size == 0 and len(group) > 1:
            test_size = 1
        if test_size >= len(group):
            test_size = len(group) - 1

        test_set.extend(group[:test_size])
        training_set.extend(group[test_size:])

    random_generator.shuffle(training_set)
    random_generator.shuffle(test_set)
    return training_set, test_set

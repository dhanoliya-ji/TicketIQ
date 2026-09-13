"""Trains the ticket classifier and prints the held-out evaluation report.

Run from the repository root::

    python scripts/train_and_report.py

This is the script behind the accuracy / precision / recall / F1 numbers quoted
in the README.  It uses exactly the same code path the API uses, so the numbers
cannot drift apart from the running service.
"""

import sys
from pathlib import Path

# Allow "python scripts/train_and_report.py" without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.classifier_service import TicketClassifierService  # noqa: E402
from app.ml.dataset import CATEGORIES  # noqa: E402
from app.settings import SETTINGS  # noqa: E402

COLUMN_WIDTH = 18


def print_heading(title: str) -> None:
    print("")
    print(title)
    print("-" * len(title))


def main() -> None:
    service = TicketClassifierService(
        dataset_path=SETTINGS.tickets_file,
        test_fraction=SETTINGS.test_split,
        seed=SETTINGS.random_seed,
    )
    service.train()
    evaluation = service.evaluation

    print_heading("Dataset")
    print("file            : " + str(SETTINGS.tickets_file))
    print("training tickets: " + str(evaluation["training_size"]))
    print("held-out tickets: " + str(evaluation["test_size"]))
    print("categories      : " + ", ".join(CATEGORIES))

    print_heading("Held-out performance")
    print("accuracy       : " + _format_number(evaluation["accuracy"]))
    print("macro precision: " + _format_number(evaluation["macro_precision"]))
    print("macro recall   : " + _format_number(evaluation["macro_recall"]))
    print("macro F1       : " + _format_number(evaluation["macro_f1"]))

    print_heading("Per category")
    header = "category".ljust(COLUMN_WIDTH) + "precision  recall     f1         support"
    print(header)
    per_category = evaluation["per_category"]
    for category in CATEGORIES:
        metrics = per_category[category]
        line = (
            category.ljust(COLUMN_WIDTH)
            + _format_number(metrics["precision"]).ljust(11)
            + _format_number(metrics["recall"]).ljust(11)
            + _format_number(metrics["f1"]).ljust(11)
            + str(metrics["support"])
        )
        print(line)

    print_heading("Confusion matrix (rows = actual, columns = predicted)")
    header = "".ljust(COLUMN_WIDTH)
    for category in CATEGORIES:
        header = header + category[:8].ljust(10)
    print(header)
    for actual in CATEGORIES:
        line = actual.ljust(COLUMN_WIDTH)
        for predicted in CATEGORIES:
            line = line + str(service.confusion[actual][predicted]).ljust(10)
        print(line)

    print("")


def _format_number(value: object) -> str:
    return format(float(str(value)), ".3f")


if __name__ == "__main__":
    main()

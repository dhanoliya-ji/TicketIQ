"""Evaluation of the classifier, scored with scikit-learn.

The assignment draws a line in exactly the right place:

    "Implement the core training/inference math yourself (no calling
    model.fit() from scikit-learn for the classifier itself) ... you may use
    scikit-learn or numpy for the surrounding vectorization and evaluation
    utilities."

So the *model* is hand-written (``naive_bayes.py``: the counting, the Laplace
smoothing, the log-probability scoring), and the *scoring of its predictions* -
which is a surrounding utility, not the model - is scikit-learn's. Nothing here
ever sees a ticket; it only sees two lists of labels.

``metrics.py`` next door still implements the same measures from their
definitions. That is not redundant: ``test_hand_written_metrics_match_sklearn``
checks the two agree exactly, which turns the from-scratch implementation into
something verified against a reference rather than merely asserted.

Both functions return the same shapes the hand-written module returns, so the
API response, the training script and the tests are unaffected by which one is
used.
"""

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_report(
    true_labels: list[str], predicted_labels: list[str], categories: list[str]
) -> dict[str, object]:
    """Per-category precision / recall / F1, plus the macro average.

    ``zero_division=0`` makes a category that was never predicted score 0.0
    rather than raising - the same choice the hand-written version makes.
    """
    precisions, recalls, f1_scores, supports = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=categories,
        zero_division=0,
    )

    per_category: dict[str, dict[str, float]] = {}
    for index in range(len(categories)):
        per_category[categories[index]] = {
            "precision": float(precisions[index]),
            "recall": float(recalls[index]),
            "f1": float(f1_scores[index]),
            "support": int(supports[index]),
        }

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=categories,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_category": per_category,
        "sample_count": len(true_labels),
    }


def confusion(
    true_labels: list[str], predicted_labels: list[str], categories: list[str]
) -> dict[str, dict[str, int]]:
    """matrix[actual][predicted] = how many times that mistake was made.

    scikit-learn returns a numpy array indexed by position; this turns it into
    the nested dictionary the API serialises, so the labels travel with the
    numbers instead of the caller having to remember the row order.
    """
    matrix = confusion_matrix(true_labels, predicted_labels, labels=categories)

    result: dict[str, dict[str, int]] = {}
    for row in range(len(categories)):
        actual = categories[row]
        result[actual] = {}
        for column in range(len(categories)):
            result[actual][categories[column]] = int(matrix[row][column])
    return result

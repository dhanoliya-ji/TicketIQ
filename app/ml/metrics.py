"""Evaluation metrics written out by hand so the definitions stay visible.

For one category ("billing", say) every prediction falls into one of these
buckets:

* true  positive - predicted billing, really billing.
* false positive - predicted billing, really something else.
* false negative - predicted something else, really billing.

From those three counts:

* precision = tp / (tp + fp)  -> "when I say billing, how often am I right?"
* recall    = tp / (tp + fn)  -> "of all real billing tickets, how many did I find?"
* f1        = the harmonic mean of precision and recall.
"""


def accuracy(true_labels: list[str], predicted_labels: list[str]) -> float:
    """Fraction of predictions that are exactly right."""
    _require_same_length(true_labels, predicted_labels)
    if len(true_labels) == 0:
        return 0.0

    correct = 0
    for index in range(len(true_labels)):
        if true_labels[index] == predicted_labels[index]:
            correct = correct + 1
    return correct / len(true_labels)


def counts_for_category(
    true_labels: list[str], predicted_labels: list[str], category: str
) -> tuple[int, int, int]:
    """Return (true positives, false positives, false negatives) for a category."""
    _require_same_length(true_labels, predicted_labels)

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for index in range(len(true_labels)):
        actual = true_labels[index]
        predicted = predicted_labels[index]

        if predicted == category and actual == category:
            true_positive = true_positive + 1
        elif predicted == category and actual != category:
            false_positive = false_positive + 1
        elif predicted != category and actual == category:
            false_negative = false_negative + 1

    return true_positive, false_positive, false_negative


def precision(true_labels: list[str], predicted_labels: list[str], category: str) -> float:
    true_positive, false_positive, _ = counts_for_category(true_labels, predicted_labels, category)
    denominator = true_positive + false_positive
    if denominator == 0:
        return 0.0
    return true_positive / denominator


def recall(true_labels: list[str], predicted_labels: list[str], category: str) -> float:
    true_positive, _, false_negative = counts_for_category(true_labels, predicted_labels, category)
    denominator = true_positive + false_negative
    if denominator == 0:
        return 0.0
    return true_positive / denominator


def f1_score(true_labels: list[str], predicted_labels: list[str], category: str) -> float:
    """Harmonic mean of precision and recall for one category."""
    category_precision = precision(true_labels, predicted_labels, category)
    category_recall = recall(true_labels, predicted_labels, category)

    if category_precision + category_recall == 0.0:
        return 0.0
    return (2 * category_precision * category_recall) / (category_precision + category_recall)


def classification_report(
    true_labels: list[str], predicted_labels: list[str], categories: list[str]
) -> dict[str, object]:
    """Per-category metrics plus the macro average and overall accuracy.

    "Macro average" = the plain average over categories, which treats a rare
    category as just as important as a common one.
    """
    per_category: dict[str, dict[str, float]] = {}

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    for category in categories:
        category_precision = precision(true_labels, predicted_labels, category)
        category_recall = recall(true_labels, predicted_labels, category)
        category_f1 = f1_score(true_labels, predicted_labels, category)

        support = 0
        for label in true_labels:
            if label == category:
                support = support + 1

        per_category[category] = {
            "precision": category_precision,
            "recall": category_recall,
            "f1": category_f1,
            "support": support,
        }

        precision_sum = precision_sum + category_precision
        recall_sum = recall_sum + category_recall
        f1_sum = f1_sum + category_f1

    number_of_categories = len(categories)
    if number_of_categories == 0:
        number_of_categories = 1

    return {
        "accuracy": accuracy(true_labels, predicted_labels),
        "macro_precision": precision_sum / number_of_categories,
        "macro_recall": recall_sum / number_of_categories,
        "macro_f1": f1_sum / number_of_categories,
        "per_category": per_category,
        "sample_count": len(true_labels),
    }


def confusion_matrix(
    true_labels: list[str], predicted_labels: list[str], categories: list[str]
) -> dict[str, dict[str, int]]:
    """matrix[actual][predicted] = how many times that mistake was made."""
    _require_same_length(true_labels, predicted_labels)

    matrix: dict[str, dict[str, int]] = {}
    for actual in categories:
        matrix[actual] = {}
        for predicted in categories:
            matrix[actual][predicted] = 0

    for index in range(len(true_labels)):
        actual = true_labels[index]
        predicted = predicted_labels[index]
        if actual in matrix and predicted in matrix[actual]:
            matrix[actual][predicted] = matrix[actual][predicted] + 1

    return matrix


def _require_same_length(true_labels: list[str], predicted_labels: list[str]) -> None:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have the same length")

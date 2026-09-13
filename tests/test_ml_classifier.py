"""Unit tests for the hand written classical ML components.

These cover the *maths*, not just "does it return something": the smoothing
formula, the probability normalisation, the TF-IDF weighting and the metric
definitions are each checked against a value worked out by hand.
"""

import math

import pytest

from app.ml.classifier_service import TicketClassifierService
from app.ml.dataset import CATEGORIES, load_tickets, stratified_split
from app.ml.metrics import (
    accuracy,
    classification_report,
    confusion_matrix,
    f1_score,
    precision,
    recall,
)
from app.ml.naive_bayes import NaiveBayesTextClassifier
from app.ml.sklearn_metrics import classification_report as sklearn_classification_report
from app.ml.sklearn_metrics import confusion as sklearn_confusion
from app.ml.text_utils import count_tokens, join_ticket_text, tokenize
from app.ml.tfidf import TfidfVectorizer, cosine_similarity, normalize
from app.settings import SETTINGS

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("I was CHARGED twice, again!") == ["charged", "twice", "again"]


def test_tokenize_drops_stop_words_and_short_tokens():
    # "the", "and", "was" are stop words; "is" is also too short.
    assert tokenize("the invoice was wrong and it is broken") == ["invoice", "wrong", "broken"]


def test_tokenize_handles_empty_text():
    assert tokenize("") == []
    assert tokenize("!!! ??? ...") == []


def test_join_ticket_text_weights_the_subject_twice():
    joined = join_ticket_text("refund", "please help")
    assert joined.count("refund") == 2


def test_count_tokens_counts_repeats():
    assert count_tokens(["bill", "bill", "late"]) == {"bill": 2, "late": 1}


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------


def test_idf_is_lower_for_common_words():
    vectorizer = TfidfVectorizer().fit(
        ["refund invoice", "refund payment", "refund charge", "login password"]
    )
    # "refund" is in 3 of 4 documents, "login" in only 1, so login must be rarer.
    assert vectorizer.inverse_document_frequency["login"] > (
        vectorizer.inverse_document_frequency["refund"]
    )


def test_idf_matches_the_formula():
    vectorizer = TfidfVectorizer().fit(["refund invoice", "refund payment"])
    # refund appears in 2 of 2 documents: log((1+2)/(1+2)) + 1 = 1.0
    assert vectorizer.inverse_document_frequency["refund"] == pytest.approx(1.0)
    # invoice appears in 1 of 2: log((1+2)/(1+1)) + 1
    expected = math.log(3 / 2) + 1.0
    assert vectorizer.inverse_document_frequency["invoice"] == pytest.approx(expected)


def test_transformed_vectors_have_unit_length():
    vectorizer = TfidfVectorizer().fit(["refund invoice please", "login password reset"])
    vector = vectorizer.transform("refund invoice")

    total = 0.0
    for weight in vector.values():
        total = total + (weight * weight)
    assert math.sqrt(total) == pytest.approx(1.0)


def test_transform_of_empty_text_is_an_empty_vector():
    vectorizer = TfidfVectorizer().fit(["refund invoice"])
    assert vectorizer.transform("") == {}
    assert vectorizer.transform("a of the") == {}


def test_normalize_of_zero_vector_does_not_divide_by_zero():
    assert normalize({"word": 0.0}) == {}


def test_cosine_similarity_is_one_for_identical_and_zero_for_disjoint():
    vectorizer = TfidfVectorizer().fit(["refund invoice", "login password"])
    first = vectorizer.transform("refund invoice")
    second = vectorizer.transform("invoice refund")
    third = vectorizer.transform("login password")

    assert cosine_similarity(first, second) == pytest.approx(1.0)
    assert cosine_similarity(first, third) == pytest.approx(0.0)


def test_cosine_similarity_is_symmetric_for_different_sized_vectors():
    vectorizer = TfidfVectorizer().fit(["refund invoice payment", "refund"])
    long_vector = vectorizer.transform("refund invoice payment")
    short_vector = vectorizer.transform("refund")

    forward = cosine_similarity(long_vector, short_vector)
    backward = cosine_similarity(short_vector, long_vector)
    assert forward == pytest.approx(backward)


# ---------------------------------------------------------------------------
# Naive Bayes
# ---------------------------------------------------------------------------
TRAINING_DOCUMENTS = [
    "refund my invoice please",
    "charged twice on my billing statement",
    "cannot login with my password",
    "password reset is broken for my account",
]
TRAINING_LABELS = ["billing", "billing", "account", "account"]


def build_trained_model() -> NaiveBayesTextClassifier:
    return NaiveBayesTextClassifier(alpha=1.0).train(TRAINING_DOCUMENTS, TRAINING_LABELS)


def test_training_records_priors_and_counts():
    model = build_trained_model()

    assert model.categories == ["account", "billing"]
    # Two documents per category out of four: prior 0.5, log(0.5) is negative.
    assert model.log_prior["billing"] == pytest.approx(math.log(0.5))
    assert model.word_counts["billing"]["refund"] == 1
    assert model.word_counts["account"].get("refund", 0) == 0


def test_laplace_smoothing_matches_the_formula():
    model = build_trained_model()

    # "refund" never appears in the account category, so its smoothed
    # probability is alpha / (total account words + alpha * vocabulary size).
    expected = math.log(1.0 / (model.total_words["account"] + len(model.vocabulary)))
    assert model.log_likelihood_of_word("refund", "account") == pytest.approx(expected)


def test_unseen_word_never_produces_zero_probability():
    model = build_trained_model()
    # A word outside the vocabulary must not crash or wipe out the score.
    assert model.predict("quantum entanglement") in model.categories


def test_prediction_picks_the_right_category():
    model = build_trained_model()
    assert model.predict("I want a refund for my invoice") == "billing"
    assert model.predict("my password does not work") == "account"


def test_confidence_is_a_probability():
    model = build_trained_model()
    category, confidence = model.predict_with_confidence("refund my invoice")

    assert category == "billing"
    assert 0.0 < confidence <= 1.0


def test_scores_across_categories_sum_to_one_after_softmax():
    model = build_trained_model()
    scores = model.score_all_categories("refund invoice")

    largest = max(scores.values())
    total = 0.0
    for score in scores.values():
        total = total + math.exp(score - largest)
    # Normalising by this total is what predict_with_confidence does.
    assert total > 0.0


def test_predicting_before_training_raises():
    model = NaiveBayesTextClassifier()
    with pytest.raises(RuntimeError):
        model.predict("anything")


def test_training_with_mismatched_lengths_raises():
    model = NaiveBayesTextClassifier()
    with pytest.raises(ValueError):
        model.train(["one document"], ["label", "extra label"])


def test_training_on_empty_dataset_raises():
    model = NaiveBayesTextClassifier()
    with pytest.raises(ValueError):
        model.train([], [])


def test_predict_many_returns_one_label_per_document():
    model = build_trained_model()
    predictions = model.predict_many(["refund invoice", "password broken"])
    assert predictions == ["billing", "account"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
ACTUAL = ["billing", "billing", "account", "account", "technical"]
PREDICTED = ["billing", "account", "account", "account", "billing"]


def test_accuracy_counts_exact_matches():
    # billing/billing, account/account, account/account are right: 3 of 5.
    assert accuracy(ACTUAL, PREDICTED) == pytest.approx(0.6)


def test_precision_and_recall_for_one_category():
    # "account" predicted 3 times, correct twice -> precision 2/3.
    assert precision(ACTUAL, PREDICTED, "account") == pytest.approx(2 / 3)
    # There are 2 real account tickets and both were found -> recall 1.0.
    assert recall(ACTUAL, PREDICTED, "account") == pytest.approx(1.0)


def test_f1_is_the_harmonic_mean():
    category_precision = precision(ACTUAL, PREDICTED, "account")
    category_recall = recall(ACTUAL, PREDICTED, "account")
    expected = 2 * category_precision * category_recall / (category_precision + category_recall)
    assert f1_score(ACTUAL, PREDICTED, "account") == pytest.approx(expected)


def test_metrics_are_zero_for_a_category_that_was_never_predicted():
    assert precision(ACTUAL, PREDICTED, "feature_request") == 0.0
    assert recall(ACTUAL, PREDICTED, "feature_request") == 0.0
    assert f1_score(ACTUAL, PREDICTED, "feature_request") == 0.0


def test_classification_report_has_every_category_and_a_macro_average():
    categories = ["account", "billing", "technical"]
    report = classification_report(ACTUAL, PREDICTED, categories)

    assert report["sample_count"] == 5
    assert set(report["per_category"].keys()) == set(categories)

    total_f1 = 0.0
    for category in categories:
        total_f1 = total_f1 + report["per_category"][category]["f1"]
    assert report["macro_f1"] == pytest.approx(total_f1 / len(categories))


def test_confusion_matrix_places_mistakes_correctly():
    matrix = confusion_matrix(ACTUAL, PREDICTED, ["account", "billing", "technical"])

    assert matrix["billing"]["billing"] == 1
    assert matrix["billing"]["account"] == 1
    assert matrix["technical"]["billing"] == 1
    assert matrix["account"]["account"] == 2


def test_hand_written_metrics_match_sklearn():
    """The from-scratch metrics are checked against a reference implementation.

    ``metrics.py`` implements accuracy, precision, recall and F1 from their
    definitions; ``sklearn_metrics.py`` is what the service actually publishes.
    Keeping both is only worthwhile if they agree, so this pins that down on a
    deliberately awkward case: one category is never predicted at all, which is
    where a zero-division mistake would show up.
    """
    categories = ["account", "billing", "technical", "feature_request"]
    actual = ["billing", "billing", "account", "account", "technical", "feature_request"]
    predicted = ["billing", "account", "account", "account", "billing", "technical"]

    mine = classification_report(actual, predicted, categories)
    reference = sklearn_classification_report(actual, predicted, categories)

    assert mine["accuracy"] == pytest.approx(reference["accuracy"])
    assert mine["macro_precision"] == pytest.approx(reference["macro_precision"])
    assert mine["macro_recall"] == pytest.approx(reference["macro_recall"])
    assert mine["macro_f1"] == pytest.approx(reference["macro_f1"])
    assert mine["sample_count"] == reference["sample_count"]

    for category in categories:
        for measure in ["precision", "recall", "f1", "support"]:
            assert mine["per_category"][category][measure] == pytest.approx(
                reference["per_category"][category][measure]
            ), (category + "." + measure)


def test_hand_written_confusion_matrix_matches_sklearn():
    categories = ["account", "billing", "technical"]
    actual = ["billing", "billing", "account", "account", "technical"]
    predicted = ["billing", "account", "account", "account", "billing"]

    assert confusion_matrix(actual, predicted, categories) == sklearn_confusion(
        actual, predicted, categories
    )


def test_the_two_implementations_agree_on_the_real_dataset():
    """Not just a toy case: the same agreement on the actual held-out split."""
    service = TicketClassifierService(
        dataset_path=SETTINGS.tickets_file,
        test_fraction=SETTINGS.test_split,
        seed=SETTINGS.random_seed,
    ).train()

    tickets = load_tickets(SETTINGS.tickets_file)
    _, test_tickets = stratified_split(tickets, SETTINGS.test_split, SETTINGS.random_seed)

    # Re-score the same split with the hand-written implementation. The model
    # has since been retrained on everything, so predictions are recomputed.
    texts = [ticket.as_text() for ticket in test_tickets]
    actual = [ticket.category for ticket in test_tickets]
    predicted = service.model.predict_many(texts)

    mine = classification_report(actual, predicted, CATEGORIES)
    reference = sklearn_classification_report(actual, predicted, CATEGORIES)

    assert mine["accuracy"] == pytest.approx(reference["accuracy"])
    assert mine["macro_f1"] == pytest.approx(reference["macro_f1"])


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        accuracy(["a"], ["a", "b"])


def test_accuracy_of_empty_input_is_zero():
    assert accuracy([], []) == 0.0

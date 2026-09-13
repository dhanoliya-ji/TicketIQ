"""Tests for aspect sentiment, the urgency score, the dataset and retrieval."""

import pytest

from app.ml.aspect_sentiment import (
    AspectSentiment,
    AspectSentimentAnalyzer,
    find_aspects_in_sentence,
    split_into_sentences,
)
from app.ml.dataset import CATEGORIES, load_tickets, stratified_split
from app.ml.urgency import score_urgency, urgency_bucket
from app.rag.chunking import chunk_knowledge_base, chunk_markdown_document
from app.rag.retriever import KnowledgeRetriever
from app.rag.vector_store import InMemoryVectorStore
from app.settings import SETTINGS

# ---------------------------------------------------------------------------
# Aspect sentiment
# ---------------------------------------------------------------------------


def test_split_into_sentences_handles_all_terminators():
    text = "First one. Second one! Third one?\nFourth one"
    assert split_into_sentences(text) == ["First one", "Second one", "Third one", "Fourth one"]


def test_split_into_sentences_of_empty_text():
    assert split_into_sentences("   ") == []


def test_find_aspects_matches_keywords_once_per_aspect():
    aspects = find_aspects_in_sentence("The invoice charge on my invoice is wrong")
    assert aspects.count("billing") == 1


def test_find_aspects_returns_nothing_for_unrelated_text():
    assert find_aspects_in_sentence("The weather outside is pleasant today") == []


def test_analyser_scores_two_aspects_independently():
    analyzer = AspectSentimentAnalyzer()
    aspects = analyzer.analyse(
        "Dashboard unusable",
        "The dashboard is unusable and crashes constantly. The billing team was wonderful and "
        "fixed my invoice quickly.",
    )

    by_name = {aspect.aspect: aspect for aspect in aspects}
    assert "performance" in by_name
    # The two aspects must not share one ticket-wide score.
    assert by_name["performance"].score < 0.0
    if "billing" in by_name:
        assert by_name["billing"].score > by_name["performance"].score


def test_analyser_returns_at_most_three_aspects():
    analyzer = AspectSentimentAnalyzer()
    aspects = analyzer.analyse(
        "Everything is broken",
        "The invoice is wrong. I cannot login. The dashboard is slow. There are errors "
        "everywhere. Nobody replied. My user permissions are missing. Please add a feature.",
    )
    assert len(aspects) <= 3


def test_analyser_falls_back_to_a_general_aspect():
    analyzer = AspectSentimentAnalyzer()
    aspects = analyzer.analyse("Hello", "Just saying hello to the team")

    assert len(aspects) == 1
    assert aspects[0].aspect == "general"


def test_domain_lexicon_keeps_product_nouns_neutral():
    analyzer = AspectSentimentAnalyzer()
    # Without the domain tuning, "support" alone scored positive.
    score = analyzer.vader.polarity_scores("Support")["compound"]
    assert score == pytest.approx(0.0)


def test_sentiment_label_thresholds():
    assert AspectSentiment("billing", -0.5, 1, "x").label == "negative"
    assert AspectSentiment("billing", 0.0, 1, "x").label == "neutral"
    assert AspectSentiment("billing", 0.5, 1, "x").label == "positive"


# ---------------------------------------------------------------------------
# Urgency
# ---------------------------------------------------------------------------


def test_urgency_rises_with_tier():
    aspects = [AspectSentiment("performance", -0.5, 1, "x")]
    free_score = score_urgency("technical", aspects, "free")
    enterprise_score = score_urgency("technical", aspects, "enterprise")
    assert enterprise_score > free_score


def test_urgency_rises_with_negative_sentiment():
    calm = [AspectSentiment("performance", 0.0, 1, "x")]
    angry = [AspectSentiment("performance", -1.0, 1, "x")]
    assert score_urgency("technical", angry, "pro") > score_urgency("technical", calm, "pro")


def test_feature_requests_are_less_urgent_than_technical_issues():
    aspects = [AspectSentiment("performance", -0.5, 1, "x")]
    assert score_urgency("feature_request", aspects, "pro") < score_urgency(
        "technical", aspects, "pro"
    )


def test_urgency_uses_the_most_negative_aspect():
    mixed = [
        AspectSentiment("billing", 0.8, 1, "x"),
        AspectSentiment("performance", -0.9, 1, "x"),
    ]
    only_worst = [AspectSentiment("performance", -0.9, 1, "x")]
    assert score_urgency("technical", mixed, "pro") == score_urgency("technical", only_worst, "pro")


def test_urgency_stays_inside_zero_to_one():
    furious = [AspectSentiment("performance", -1.0, 1, "x")]
    score = score_urgency("technical", furious, "enterprise")
    assert 0.0 <= score <= 1.0


def test_urgency_of_unknown_category_and_tier_uses_defaults():
    aspects = [AspectSentiment("general", 0.0, 1, "x")]
    score = score_urgency("mystery", aspects, "platinum")
    assert 0.0 < score < 1.0


def test_urgency_buckets():
    assert urgency_bucket(0.10) == "low"
    assert urgency_bucket(0.50) == "medium"
    assert urgency_bucket(0.90) == "high"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_dataset_loads_and_covers_every_category():
    tickets = load_tickets(SETTINGS.tickets_file)

    assert len(tickets) >= 100
    found_categories = set()
    for ticket in tickets:
        found_categories.add(ticket.category)
    assert found_categories == set(CATEGORIES)


def test_stratified_split_keeps_every_category_on_both_sides():
    tickets = load_tickets(SETTINGS.tickets_file)
    training_set, test_set = stratified_split(tickets, 0.25, seed=1)

    assert len(training_set) + len(test_set) == len(tickets)

    training_categories = {ticket.category for ticket in training_set}
    test_categories = {ticket.category for ticket in test_set}
    assert training_categories == set(CATEGORIES)
    assert test_categories == set(CATEGORIES)


def test_stratified_split_is_reproducible_for_one_seed():
    tickets = load_tickets(SETTINGS.tickets_file)
    first_training, _ = stratified_split(tickets, 0.25, seed=5)
    second_training, _ = stratified_split(tickets, 0.25, seed=5)

    assert [t.ticket_id for t in first_training] == [t.ticket_id for t in second_training]


def test_loading_a_missing_dataset_raises():
    with pytest.raises(FileNotFoundError):
        load_tickets(SETTINGS.data_dir / "does_not_exist.json")


# ---------------------------------------------------------------------------
# Chunking and retrieval
# ---------------------------------------------------------------------------


def test_chunking_splits_on_headings(tmp_path):
    document = tmp_path / "policy.md"
    document.write_text(
        "# Title\n\n## First section\nFirst body.\n\n## Second section\nSecond body.\n",
        encoding="utf-8",
    )

    chunks = chunk_markdown_document(document)

    assert len(chunks) == 2
    assert chunks[0].document_title == "Title"
    assert chunks[0].heading == "First section"
    assert "First body." in chunks[0].text
    assert chunks[1].heading == "Second section"


def test_chunking_ignores_empty_sections(tmp_path):
    document = tmp_path / "policy.md"
    document.write_text("# Title\n\n## Empty\n\n## Real\nSomething.\n", encoding="utf-8")

    chunks = chunk_markdown_document(document)
    assert len(chunks) == 1
    assert chunks[0].heading == "Real"


def test_chunk_ids_are_unique_across_the_knowledge_base():
    chunks = chunk_knowledge_base(SETTINGS.knowledge_base_dir)
    identifiers = [chunk.chunk_id for chunk in chunks]
    assert len(identifiers) == len(set(identifiers))


def test_chunking_a_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        chunk_knowledge_base(tmp_path / "nope")


def test_chunking_a_folder_without_markdown_raises(tmp_path):
    with pytest.raises(ValueError):
        chunk_knowledge_base(tmp_path)


def test_vector_store_returns_the_most_similar_chunk_first():
    chunks = chunk_knowledge_base(SETTINGS.knowledge_base_dir)
    store = InMemoryVectorStore().build(chunks)

    hits = store.search("refund a duplicate charge on my invoice", top_k=3)

    assert len(hits) > 0
    assert "billing" in hits[0].chunk.source
    # Scores must come back in descending order.
    for index in range(1, len(hits)):
        assert hits[index - 1].score >= hits[index].score


def test_vector_store_drops_chunks_with_no_overlap():
    chunks = chunk_knowledge_base(SETTINGS.knowledge_base_dir)
    store = InMemoryVectorStore().build(chunks)

    hits = store.search("xylophone zebra quasar", top_k=5)
    assert hits == []


def test_vector_store_rejects_bad_usage():
    store = InMemoryVectorStore()
    with pytest.raises(RuntimeError):
        store.search("anything", top_k=1)

    with pytest.raises(ValueError):
        InMemoryVectorStore().build([])

    built = InMemoryVectorStore().build(chunk_knowledge_base(SETTINGS.knowledge_base_dir))
    with pytest.raises(ValueError):
        built.search("refund", top_k=0)


def test_retriever_top_k_limits_the_number_of_snippets():
    retriever = KnowledgeRetriever(SETTINGS.knowledge_base_dir).load()

    two = retriever.retrieve("Cannot log in", "password reset never arrives", "account", top_k=2)
    five = retriever.retrieve("Cannot log in", "password reset never arrives", "account", top_k=5)

    assert len(two) <= 2
    assert len(five) >= len(two)


def test_retriever_finds_the_right_document_per_category():
    retriever = KnowledgeRetriever(SETTINGS.knowledge_base_dir).load()

    account_hits = retriever.retrieve(
        "Password reset email never arrives", "It never lands in my inbox", "account", top_k=1
    )
    billing_hits = retriever.retrieve(
        "Refund request", "I was charged twice for the same invoice", "billing", top_k=1
    )

    assert "account" in account_hits[0].chunk.source
    assert "billing" in billing_hits[0].chunk.source


def test_retriever_must_be_loaded_first():
    retriever = KnowledgeRetriever(SETTINGS.knowledge_base_dir)
    with pytest.raises(RuntimeError):
        retriever.retrieve("subject", "body", "billing", top_k=2)

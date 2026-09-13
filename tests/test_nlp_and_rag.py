"""Tests for aspect sentiment, the urgency score, the dataset and retrieval."""

import pytest

from app.ml.aspect_sentiment import (
    AspectSentiment,
    AspectSentimentAnalyzer,
    find_aspects_in_sentence,
    split_into_sentences,
)
from app.ml.dataset import CATEGORIES, load_tickets, stratified_split
from app.ml.tfidf import cosine_similarity
from app.ml.urgency import score_urgency, urgency_bucket
from app.rag.chunking import chunk_knowledge_base, chunk_markdown_document
from app.rag.retriever import KnowledgeRetriever, topic_of_document
from app.rag.vector_store import FaissVectorStore
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


def test_the_folder_readme_is_not_indexed_as_policy():
    """The knowledge base README documents the corpus; it is not part of it.

    Without this exclusion the developer notes would be retrievable and could
    be quoted back to a customer as if they were policy.
    """
    chunks = chunk_knowledge_base(SETTINGS.knowledge_base_dir)

    sources = {chunk.source.lower() for chunk in chunks}
    assert "readme.md" not in sources
    # The five real policy documents are all still indexed.
    assert len(sources) == 5


def test_readme_exclusion_is_case_insensitive(tmp_path):
    (tmp_path / "README.md").write_text("# Notes\n\n## Dev\nInternal.\n", encoding="utf-8")
    (tmp_path / "policy.md").write_text("# Policy\n\n## Rule\nReal rule.\n", encoding="utf-8")

    chunks = chunk_knowledge_base(tmp_path)

    assert len(chunks) == 1
    assert chunks[0].heading == "Rule"


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
    store = FaissVectorStore().build(chunks)

    hits = store.search("refund a duplicate charge on my invoice", top_k=3)

    assert len(hits) > 0
    assert "billing" in hits[0].chunk.source
    # Scores must come back in descending order.
    for index in range(1, len(hits)):
        assert hits[index - 1].score >= hits[index].score


def test_vector_store_drops_chunks_with_no_overlap():
    chunks = chunk_knowledge_base(SETTINGS.knowledge_base_dir)
    store = FaissVectorStore().build(chunks)

    hits = store.search("xylophone zebra quasar", top_k=5)
    assert hits == []


def test_the_store_really_is_a_faiss_index():
    """The chunks live in FAISS, not in a Python list scanned by hand."""
    import faiss

    chunks = chunk_knowledge_base(SETTINGS.knowledge_base_dir)
    store = FaissVectorStore().build(chunks)

    assert isinstance(store.index, faiss.IndexFlatIP)
    # Every chunk was added to the index, and the index width is the vocabulary.
    assert store.index.ntotal == len(chunks)
    assert store.index.d == len(store.vocabulary)


def test_faiss_scores_match_cosine_similarity():
    """Inner product on unit-length vectors is exactly the cosine.

    This is the property that lets IndexFlatIP stand in for a cosine index, so
    it is worth pinning rather than assuming.
    """
    chunks = chunk_knowledge_base(SETTINGS.knowledge_base_dir)
    store = FaissVectorStore().build(chunks)

    query = "refund a duplicate charge on my invoice"
    hits = store.search(query, top_k=3)

    query_vector = store.vectorizer.transform(query)
    for hit in hits:
        chunk_vector = store.vectorizer.transform(hit.chunk.searchable_text())
        assert hit.score == pytest.approx(cosine_similarity(query_vector, chunk_vector), abs=1e-5)


def test_asking_for_more_chunks_than_exist_is_safe():
    """FAISS pads a short result with -1, which must not become a hit."""
    chunks = chunk_knowledge_base(SETTINGS.knowledge_base_dir)
    store = FaissVectorStore().build(chunks)

    hits = store.search("refund", top_k=len(chunks) + 50)

    assert len(hits) <= len(chunks)
    for hit in hits:
        assert hit.score > 0.0


def test_vector_store_rejects_bad_usage():
    store = FaissVectorStore()
    with pytest.raises(RuntimeError):
        store.search("anything", top_k=1)

    with pytest.raises(ValueError):
        FaissVectorStore().build([])

    built = FaissVectorStore().build(chunk_knowledge_base(SETTINGS.knowledge_base_dir))
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


def test_retriever_rejects_a_bad_top_k():
    retriever = KnowledgeRetriever(SETTINGS.knowledge_base_dir).load()
    with pytest.raises(ValueError):
        retriever.retrieve("subject", "body", "billing", top_k=0)


# --- category aware re-ranking -------------------------------------------


def test_document_topics_are_mapped_and_escalation_is_universal():
    assert topic_of_document("01_billing_refund_policy.md") == "billing"
    assert topic_of_document("03_technical_troubleshooting.md") == "technical"
    # The escalation rules apply to every category, so they have no topic.
    assert topic_of_document("05_escalation_rules.md") is None
    # An unknown document is treated as topic-less rather than crashing.
    assert topic_of_document("99_unknown.md") is None


def test_an_outage_does_not_retrieve_feature_request_policy():
    """The bug this re-ranking exists to fix.

    Plain cosine similarity put "Wording to use" from the feature-request
    document into the top 2 for a total outage, and the agent then quoted
    "thank the customer for the idea" at a customer whose platform was down.
    """
    retriever = KnowledgeRetriever(SETTINGS.knowledge_base_dir).load()

    hits = retriever.retrieve(
        "Total outage, dashboard is down",
        "Every API call returns a 500 error and the whole platform is unusable. "
        "Our team is completely blocked.",
        "technical",
        top_k=2,
    )

    sources = [hit.chunk.source for hit in hits]
    assert not any("feature_request" in source for source in sources)
    assert any("technical" in source for source in sources)


def test_every_category_retrieves_its_own_document_first():
    retriever = KnowledgeRetriever(SETTINGS.knowledge_base_dir).load()

    cases = [
        ("billing", "Charged twice", "My card was charged twice and I want a refund.", "billing"),
        ("account", "Cannot log in", "The password reset email never arrives.", "account"),
        (
            "technical",
            "Dashboard is slow",
            "Every chart takes thirty seconds to load.",
            "technical",
        ),
        (
            "feature_request",
            "Please add dark mode",
            "It would be great to have a dark theme.",
            "feature_request",
        ),
    ]

    for category, subject, body, expected_in_source in cases:
        hits = retriever.retrieve(subject, body, category, top_k=1)
        assert expected_in_source in hits[0].chunk.source, category


def test_escalation_rules_stay_reachable_for_any_category():
    """A universal document must not be damped for any category."""
    retriever = KnowledgeRetriever(SETTINGS.knowledge_base_dir).load()

    hits = retriever.retrieve(
        "Enterprise outage, we want to cancel",
        "We are on the enterprise tier, the service is down and I want to speak to a manager.",
        "technical",
        top_k=3,
    )

    sources = [hit.chunk.source for hit in hits]
    assert any("escalation" in source for source in sources)


def test_a_strongly_matching_off_topic_chunk_can_still_win():
    """Damping lowers off-topic chunks, it does not ban them."""
    retriever = KnowledgeRetriever(SETTINGS.knowledge_base_dir).load()

    # A refund question mislabelled as technical: the billing document is still
    # by far the best textual match, so it must survive the damping.
    hits = retriever.retrieve(
        "Refund for a duplicate charge",
        "The same invoice amount was charged twice to my credit card and I want the "
        "duplicate refunded.",
        "technical",
        top_k=1,
    )

    assert "billing" in hits[0].chunk.source

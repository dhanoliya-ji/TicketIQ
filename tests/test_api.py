"""API tests using FastAPI's TestClient.

The client is created with a ``with`` block so that the lifespan handler runs
and the service is actually started, exactly as it is in production.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def submit_ticket(client, subject: str, body: str, tier: str = "pro"):
    return client.post("/ticket", json={"subject": subject, "body": body, "customer_tier": tier})


# ---------------------------------------------------------------------------
# POST /ticket
# ---------------------------------------------------------------------------


def test_health_reports_a_ready_service(client):
    body = client.get("/health").json()

    assert body["ready"] is True
    assert body["knowledge_chunks"] > 0
    assert 0.0 <= body["classifier_accuracy"] <= 1.0


def test_posting_a_ticket_returns_every_required_field(client):
    response = submit_ticket(
        client,
        "Charged twice on order 4471",
        "My credit card was charged twice for the same monthly invoice.",
        "enterprise",
    )
    assert response.status_code == 200
    body = response.json()

    # The fields the assignment asks the endpoint to return.
    assert body["category"] in ("billing", "technical", "account", "feature_request")
    assert len(body["aspect_sentiments"]) >= 1
    assert isinstance(body["retrieved_knowledge"], list)
    assert body["action"] in (
        "answer",
        "escalate_to_human",
        "check_account_status",
        "check_refund_eligibility",
    )
    assert len(body["response_text"]) > 0
    assert body["pipeline_config"]["name"].count("|") == 1
    assert body["latency_seconds"] > 0.0
    assert body["transaction_id"].startswith("tx-")


def test_a_billing_ticket_is_classified_as_billing(client):
    body = submit_ticket(
        client,
        "Refund request for last month",
        "I would like a refund for the payment taken on the third of the month.",
    ).json()

    assert body["category"] == "billing"


def test_an_account_ticket_retrieves_account_knowledge(client):
    body = submit_ticket(
        client,
        "Password reset email never arrives",
        "The reset email never arrives in my inbox or my spam folder.",
    ).json()

    assert body["category"] == "account"
    sources = [snippet["source"] for snippet in body["retrieved_knowledge"]]
    assert any("account" in source for source in sources)


def test_the_number_of_snippets_respects_the_chosen_top_k(client):
    body = submit_ticket(
        client, "Dashboard is extremely slow", "Every chart takes thirty seconds to load."
    ).json()

    assert len(body["retrieved_knowledge"]) <= body["pipeline_config"]["rag_top_k"]


def test_the_reasoning_trace_is_returned(client):
    body = submit_ticket(
        client, "Cannot log in to my account", "My password is rejected every time."
    ).json()

    assert len(body["reasoning_trace"]) >= 1
    assert "action" in body["reasoning_trace"][0]


def test_an_enterprise_ticket_is_more_urgent_than_a_free_one(client):
    subject = "Dashboard is extremely slow"
    body_text = "The dashboard is unusable and our whole team is blocked."

    free = submit_ticket(client, subject, body_text, "free").json()
    enterprise = submit_ticket(client, subject, body_text, "enterprise").json()

    assert enterprise["urgency_score"] > free["urgency_score"]


def test_an_invalid_tier_is_rejected(client):
    response = submit_ticket(client, "Subject", "Body", "platinum")
    assert response.status_code == 422


def test_an_empty_subject_is_rejected(client):
    response = client.post("/ticket", json={"subject": "", "body": "Body", "customer_tier": "pro"})
    assert response.status_code == 422


def test_a_whitespace_only_ticket_is_rejected(client):
    # min_length alone counts spaces, so this needs its own validator.
    response = client.post(
        "/ticket", json={"subject": "   ", "body": "\t\n ", "customer_tier": "pro"}
    )
    assert response.status_code == 422


def test_surrounding_whitespace_is_trimmed(client):
    response = submit_ticket(client, "  Refund request  ", "  I was charged twice.  ")

    assert response.status_code == 200
    status = client.get("/ticket/" + response.json()["transaction_id"] + "/status").json()
    assert status["request"]["subject"] == "Refund request"


def test_a_missing_body_field_is_rejected(client):
    response = client.post("/ticket", json={"subject": "Subject"})
    assert response.status_code == 422


def test_the_customer_tier_defaults_to_free(client):
    response = client.post("/ticket", json={"subject": "Hello", "body": "Just a question"})
    assert response.status_code == 200
    assert response.json()["rl_state_key"].endswith("|free")


# ---------------------------------------------------------------------------
# GET /ticket/{id}/status
# ---------------------------------------------------------------------------


def test_status_reports_real_stage_state(client):
    transaction_id = submit_ticket(
        client, "Refund request", "I was charged twice for the same invoice."
    ).json()["transaction_id"]

    status = client.get("/ticket/" + transaction_id + "/status").json()

    assert status["status"] == "completed"
    assert status["completed_stages"] == status["total_stages"]
    assert status["not_started_stages"] == []

    stage_names = [stage["stage"] for stage in status["stages"]]
    for expected in [
        "classify_ticket",
        "analyse_sentiment",
        "score_urgency",
        "select_configuration",
        "retrieve_knowledge",
        "run_agent",
        "compose_response",
    ]:
        assert expected in stage_names

    # Each stage carries its own recorded output, not a shared blob.
    by_name = {stage["stage"]: stage for stage in status["stages"]}
    assert by_name["classify_ticket"]["output"]["category"] == "billing"
    assert "aspects" in by_name["analyse_sentiment"]["output"]


def test_status_of_an_unknown_transaction_is_404(client):
    assert client.get("/ticket/tx-does-not-exist/status").status_code == 404


# ---------------------------------------------------------------------------
# POST /ticket/{id}/retry
# ---------------------------------------------------------------------------


def test_retry_reruns_only_what_failed(client):
    """The resumable engine, exercised over HTTP."""
    from app.main import service
    from app.workflow.pipeline import STAGE_RETRIEVE

    original_run = service.engine.stages_by_name[STAGE_RETRIEVE].run

    def explode(context):
        raise RuntimeError("knowledge base briefly unreadable")

    service.engine.stages_by_name[STAGE_RETRIEVE].run = explode
    try:
        failed = submit_ticket(client, "Refund request", "I was charged twice.")
        assert failed.status_code == 500
        transaction_id = failed.json()["detail"]["transaction_id"]
        assert failed.json()["detail"]["failed_stage"] == STAGE_RETRIEVE
    finally:
        service.engine.stages_by_name[STAGE_RETRIEVE].run = original_run

    # Mid-failure, the status endpoint already shows exactly where it stopped.
    status = client.get("/ticket/" + transaction_id + "/status").json()
    by_name = {stage["stage"]: stage["status"] for stage in status["stages"]}
    assert status["status"] == "failed"
    assert by_name["classify_ticket"] == "completed"
    assert by_name[STAGE_RETRIEVE] == "failed"

    response = client.post("/ticket/" + transaction_id + "/retry")
    assert response.status_code == 200
    body = response.json()

    assert body["transaction_id"] == transaction_id
    assert "classify_ticket" in body["stages_reused"]
    assert "analyse_sentiment" in body["stages_reused"]
    assert STAGE_RETRIEVE in body["stages_executed"]
    assert client.get("/ticket/" + transaction_id + "/status").json()["status"] == "completed"


def test_retry_of_an_unknown_transaction_is_404(client):
    assert client.post("/ticket/tx-nope/retry").status_code == 404


def test_retry_of_a_completed_ticket_is_409(client):
    transaction_id = submit_ticket(client, "Refund request", "I was charged twice.").json()[
        "transaction_id"
    ]

    assert client.post("/ticket/" + transaction_id + "/retry").status_code == 409


# ---------------------------------------------------------------------------
# POST /feedback
# ---------------------------------------------------------------------------


def test_feedback_updates_the_bandit(client):
    before = client.get("/rl/stats").json()["total_updates"]

    transaction_id = submit_ticket(
        client, "Please add dark mode", "A dark theme would help our night shift."
    ).json()["transaction_id"]

    response = client.post(
        "/feedback", json={"transaction_id": transaction_id, "feedback_score": 1}
    )
    assert response.status_code == 200
    body = response.json()

    # reward = 1 * 10 - latency
    assert body["reward"] == pytest.approx(10.0 - body["latency_seconds"], abs=0.001)
    assert client.get("/rl/stats").json()["total_updates"] == before + 1


def test_negative_feedback_produces_a_negative_reward(client):
    transaction_id = submit_ticket(
        client, "Sync job crashed overnight", "The nightly job failed again."
    ).json()["transaction_id"]

    body = client.post(
        "/feedback", json={"transaction_id": transaction_id, "feedback_score": 0}
    ).json()

    assert body["reward"] < 0.0


def test_feedback_for_an_unknown_transaction_is_404(client):
    response = client.post("/feedback", json={"transaction_id": "tx-nope", "feedback_score": 1})
    assert response.status_code == 404


def test_feedback_can_only_be_given_once(client):
    transaction_id = submit_ticket(
        client, "Export keeps timing out", "Large exports never finish."
    ).json()["transaction_id"]

    first = client.post("/feedback", json={"transaction_id": transaction_id, "feedback_score": 1})
    second = client.post("/feedback", json={"transaction_id": transaction_id, "feedback_score": 1})

    assert first.status_code == 200
    assert second.status_code == 409


def test_an_out_of_range_feedback_score_is_rejected(client):
    response = client.post("/feedback", json={"transaction_id": "tx-1", "feedback_score": 5})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Inspection endpoints
# ---------------------------------------------------------------------------


def test_rl_stats_lists_every_configuration(client):
    body = client.get("/rl/stats").json()

    assert len(body["actions"]) == 4
    assert len(body["configurations"]) == 4
    assert 0.0 <= body["epsilon"] <= 1.0


def test_the_workflow_graph_shows_the_parallel_level(client):
    levels = client.get("/workflow/graph").json()["levels"]

    assert levels[0]["runs_in_parallel"] is True
    first_level_names = {stage["name"] for stage in levels[0]["stages"]}
    assert first_level_names == {"classify_ticket", "analyse_sentiment"}


def test_the_ml_report_endpoint_returns_metrics(client):
    body = client.get("/ml/report").json()

    assert 0.0 <= body["evaluation"]["accuracy"] <= 1.0
    assert "macro_f1" in body["evaluation"]
    assert "billing" in body["confusion_matrix"]


# ---------------------------------------------------------------------------
# The live console
# ---------------------------------------------------------------------------


def test_the_console_page_is_served_at_the_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "TicketIQ" in response.text


def test_the_console_assets_are_served(client):
    for path in ["/static/styles.css", "/static/app.js"]:
        response = client.get(path)
        assert response.status_code == 200, path
        # A page is useless if its script is empty, so check for real content.
        assert len(response.text) > 1000, path


def test_the_console_shows_every_field_the_brief_requires(client):
    """Requirement 1 lists what POST /ticket returns; the page must show it.

    The configuration was once dropped from the page by accident when a section
    moved, so this checks the markup has a home for each of the eight.
    """
    page = client.get("/").text + client.get("/static/app.js").text

    required = [
        ("predicted category", "out-category"),
        ("per-aspect sentiment", "out-aspects"),
        ("retrieved knowledge snippets", "out-sources"),
        ("chosen action", "out-action"),
        ("final response text", "out-reply"),
        ("pipeline configuration", "pipeline_config"),
        ("end-to-end latency", "out-latency"),
        ("transaction id", "result-ref"),
    ]
    for label, marker in required:
        assert marker in page, label


def test_the_console_shows_the_numbers_behind_the_labels(client):
    """Regression: the page showed conclusions without the figures behind them.

    A category with no confidence, an urgency bucket with no score and a list
    of snippets with no similarity all read as if the service were certain.
    Each number is returned by the API and each is named by the brief, so each
    is on the page.
    """
    page = client.get("/").text + client.get("/static/app.js").text

    for field in ["category_confidence", "urgency_score", "snippet.score"]:
        assert field in page, field


def test_the_console_shows_tool_calls_and_stage_outputs(client):
    """Requirement 4 asks for the tool calls, requirement 6 for stage state.

    The tool calls section was named in the markup and counted in its heading
    but never rendered, so the arguments a tool was called with and what it
    returned were both invisible. The stage table likewise showed only ticks
    and timings, never what a stage produced.
    """
    page = client.get("/").text + client.get("/static/app.js").text

    assert "renderToolCalls" in page
    assert "out-tools" in page
    assert "call.arguments" in page
    assert "call.output" in page
    assert "stage.output" in page


def test_the_console_offers_no_canned_tickets(client):
    """POST /ticket takes any free text, so the page must not imply a menu."""
    page = client.get("/").text + client.get("/static/app.js").text

    assert "example-link" not in page
    assert "EXAMPLES" not in page


def test_there_is_only_one_page(client):
    assert client.get("/performance").status_code == 404
    assert client.get("/static/performance.js").status_code == 404


def test_the_console_does_not_shadow_the_api(client):
    """The static mount must not swallow API routes."""
    assert client.get("/health").status_code == 200
    assert client.get("/workflow/graph").status_code == 200


def test_the_console_is_not_in_the_openapi_schema(client):
    # The UI is not part of the documented API surface.
    schema = client.get("/openapi.json").json()
    assert "/" not in schema["paths"]


def test_the_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()

    assert "/ticket" in schema["paths"]
    assert "/feedback" in schema["paths"]
    assert "/ticket/{transaction_id}/status" in schema["paths"]

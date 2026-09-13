"""End-to-end pipeline tests with the language model mocked out.

These drive ``TriageService`` directly rather than through HTTP, so they can
inject a fake model, force a stage failure, and then inspect the persisted
workflow state afterwards.
"""

import pytest

from app.workflow.pipeline import (
    STAGE_AGENT,
    STAGE_CLASSIFY,
    STAGE_COMPOSE,
    STAGE_RETRIEVE,
    STAGE_SELECT_CONFIG,
    STAGE_SENTIMENT,
    STAGE_URGENCY,
    FeedbackNotAcceptedError,
    PipelineError,
    TriageService,
    UnknownTransactionError,
)

ALL_STAGES = [
    STAGE_CLASSIFY,
    STAGE_SENTIMENT,
    STAGE_URGENCY,
    STAGE_SELECT_CONFIG,
    STAGE_RETRIEVE,
    STAGE_AGENT,
    STAGE_COMPOSE,
]


@pytest.fixture
def service(fake_llm_factory):
    """A fully started service whose language model is a test double."""
    built = TriageService()
    built.startup()

    fake = fake_llm_factory(
        [
            '{"thought": "I should confirm the refund first.",'
            ' "action": "check_refund_eligibility", "action_input": {"order_id": "4471"}}',
            '{"thought": "Now I can answer.", "action": "answer", "action_input": {}}',
        ],
        written_reply="Your duplicate charge will be refunded in full.",
    )
    # Both references have to be replaced: the agent uses one for generation,
    # the service reports the other in /health.
    built.llm_client = fake
    built.agent.llm = fake
    return built


def test_the_whole_pipeline_runs_with_the_model_mocked(service):
    result = service.handle_ticket(
        subject="Duplicate payment on order 4471",
        body="My credit card was charged twice for the same monthly invoice. I want a refund.",
        customer_tier="enterprise",
    )

    # Classification, sentiment and urgency.
    assert result["category"] == "billing"
    assert 0.0 < result["category_confidence"] <= 1.0
    assert len(result["aspect_sentiments"]) >= 1
    assert result["urgency_bucket"] in ("low", "medium", "high")

    # Retrieval.
    assert len(result["retrieved_knowledge"]) > 0
    assert len(result["retrieved_knowledge"]) <= result["pipeline_config"]["rag_top_k"]

    # The agent: one tool call, then an answer written by the fake model.
    assert result["action"] == "answer"
    assert [call["tool"] for call in result["tool_calls"]] == ["check_refund_eligibility"]
    assert result["response_text"] == "Your duplicate charge will be refunded in full."
    assert len(result["reasoning_trace"]) == 2

    # Workflow bookkeeping.
    assert sorted(result["stages_executed"]) == sorted(ALL_STAGES)
    assert result["stages_reused"] == []
    assert result["latency_seconds"] > 0.0


def test_every_stage_output_is_persisted(service):
    result = service.handle_ticket("Cannot log in", "My password is rejected.", "pro")
    status = service.get_status(result["transaction_id"])

    assert status["status"] == "completed"
    assert status["completed_stages"] == len(ALL_STAGES)
    assert status["not_started_stages"] == []

    by_name = {stage["stage"]: stage for stage in status["stages"]}
    assert by_name[STAGE_CLASSIFY]["output"]["category"] == "account"
    assert by_name[STAGE_URGENCY]["output"]["urgency_bucket"] in ("low", "medium", "high")
    assert by_name[STAGE_RETRIEVE]["output"]["top_k"] in (2, 5)
    assert by_name[STAGE_COMPOSE]["output"]["action"] == "answer"

    for stage in status["stages"]:
        assert stage["status"] == "completed"
        assert stage["error"] == ""


def test_the_two_independent_stages_are_on_the_first_level(service):
    levels = service.workflow_description()

    first_level_names = set()
    for stage in levels[0]["stages"]:
        first_level_names.add(stage["name"])

    assert first_level_names == {STAGE_CLASSIFY, STAGE_SENTIMENT}
    assert levels[0]["runs_in_parallel"] is True


def test_re_running_a_transaction_reuses_completed_stages(service):
    result = service.handle_ticket("Refund request", "I was charged twice.", "pro")
    transaction_id = result["transaction_id"]

    # Run the same transaction through the engine again, as a retry would.
    second_run = service.engine.run(transaction_id, dict(status_request(result)))

    assert second_run.succeeded is True
    assert second_run.executed_stages == []
    assert sorted(second_run.reused_stages) == sorted(ALL_STAGES)


def status_request(result: dict) -> dict:
    """The engine inputs for a transaction, read back from the stored request."""
    return {
        "subject": "Refund request",
        "body": "I was charged twice.",
        "customer_tier": "pro",
    }


def test_feedback_updates_the_bandit_for_the_state_that_was_used(service):
    result = service.handle_ticket("Please add dark mode", "A dark theme would help.", "free")
    transaction_id = result["transaction_id"]
    state_key = result["rl_state_key"]
    config_name = result["pipeline_config"]["name"]

    feedback = service.handle_feedback(transaction_id, 1)

    assert feedback["state_key"] == state_key
    assert feedback["config_name"] == config_name
    assert feedback["reward"] == pytest.approx(10.0 - result["latency_seconds"], abs=0.001)

    # The bandit's memory now holds exactly one observation for that arm.
    statistics = service.bandit.statistics[state_key][config_name]
    assert statistics.pulls == 1
    assert statistics.average_reward == pytest.approx(feedback["reward"], abs=0.001)


def test_feedback_is_reflected_in_the_status_endpoint(service):
    result = service.handle_ticket("Charts fail to load", "Nothing renders.", "pro")
    service.handle_feedback(result["transaction_id"], 0)

    status = service.get_status(result["transaction_id"])
    assert status["feedback_score"] == 0
    assert status["reward"] < 0.0


def test_feedback_naming_a_retired_configuration_is_refused_not_a_crash(service):
    """A stale ticket must not take the endpoint down with a KeyError."""
    result = service.handle_ticket("Refund request", "I was charged twice.", "pro")

    # Simulate the configuration line-up changing since the ticket was answered.
    service.bandit.actions = ["some_new_config"]

    with pytest.raises(FeedbackNotAcceptedError) as error:
        service.handle_feedback(result["transaction_id"], 1)
    assert "no longer exists" in error.value.reason


def test_the_pipeline_survives_its_database_being_deleted(service):
    """Losing var/ mid-run must not break every later request."""
    first = service.handle_ticket("Refund request", "I was charged twice.", "pro")
    assert first["category"] == "billing"

    service.store.database_path.unlink()

    # The very next ticket must still work, on a rebuilt schema.
    second = service.handle_ticket("Cannot log in", "My password is rejected.", "pro")
    assert second["category"] == "account"
    assert service.get_status(second["transaction_id"])["status"] == "completed"


def test_feedback_for_an_unknown_transaction_raises(service):
    with pytest.raises(UnknownTransactionError):
        service.handle_feedback("tx-not-real", 1)


def test_status_for_an_unknown_transaction_raises(service):
    with pytest.raises(UnknownTransactionError):
        service.get_status("tx-not-real")


def test_a_failing_stage_fails_the_ticket_and_is_visible_in_the_status(service):
    def explode(context):
        raise RuntimeError("knowledge base unavailable")

    # Break exactly one stage, leaving the rest of the graph intact.
    service.engine.stages_by_name[STAGE_RETRIEVE].run = explode

    with pytest.raises(PipelineError) as error:
        service.handle_ticket("Dashboard is slow", "Everything times out.", "pro")

    transaction_id = error.value.transaction_id
    assert error.value.failed_stage == STAGE_RETRIEVE

    status = service.get_status(transaction_id)
    assert status["status"] == "failed"

    by_name = {stage["stage"]: stage for stage in status["stages"]}
    # Upstream stages still completed and are still readable.
    assert by_name[STAGE_CLASSIFY]["status"] == "completed"
    assert by_name[STAGE_RETRIEVE]["status"] == "failed"
    assert "knowledge base unavailable" in by_name[STAGE_RETRIEVE]["error"]
    # Downstream stages never ran.
    assert by_name[STAGE_AGENT]["status"] == "skipped"
    assert by_name[STAGE_COMPOSE]["status"] == "skipped"


def test_feedback_is_refused_for_a_failed_ticket(service):
    def explode(context):
        raise RuntimeError("still broken")

    service.engine.stages_by_name[STAGE_RETRIEVE].run = explode

    with pytest.raises(PipelineError) as error:
        service.handle_ticket("Dashboard is slow", "Everything times out.", "pro")

    with pytest.raises(FeedbackNotAcceptedError):
        service.handle_feedback(error.value.transaction_id, 1)


def test_a_repaired_stage_is_the_only_one_re_run(service):
    """The point of the resumable engine: retry the broken stage only."""
    original_run = service.engine.stages_by_name[STAGE_RETRIEVE].run

    def explode(context):
        raise RuntimeError("temporary outage")

    service.engine.stages_by_name[STAGE_RETRIEVE].run = explode

    with pytest.raises(PipelineError) as error:
        service.handle_ticket("Refund request", "I was charged twice.", "pro")
    transaction_id = error.value.transaction_id

    # The outage clears.
    service.engine.stages_by_name[STAGE_RETRIEVE].run = original_run
    retry = service.engine.run(
        transaction_id,
        {"subject": "Refund request", "body": "I was charged twice.", "customer_tier": "pro"},
    )

    assert retry.succeeded is True
    # Classification, sentiment and urgency were not recomputed.
    assert STAGE_CLASSIFY in retry.reused_stages
    assert STAGE_SENTIMENT in retry.reused_stages
    assert STAGE_RETRIEVE in retry.executed_stages


def test_health_reports_the_mocked_backend(service):
    health = service.health()

    assert health["ready"] is True
    assert health["llm_backend"] == "fake"
    assert health["knowledge_chunks"] > 0

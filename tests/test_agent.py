"""Tests for the mock tools, the JSON parsing and the ReAct loop.

The loop is driven by a fake language model (see ``conftest.py``) so that each
test can force a specific sequence of decisions.
"""

import pytest

from app.agent.react_agent import (
    TriageAgent,
    extract_identifiers,
    format_snippets,
    parse_first_json_object,
)
from app.agent.tools import check_account_status, check_refund_eligibility, run_tool
from app.llm.configs import get_config

CONFIG = get_config("concise_policy|k2")

SNIPPETS = [
    {
        "document_title": "Billing and Refund Policy",
        "heading": "Duplicate charges",
        "text": "A duplicate charge is always refunded in full.",
    }
]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_account_status_is_deterministic():
    first = check_account_status({"customer_id": "CUST-77"})
    second = check_account_status({"customer_id": "CUST-77"})
    assert first.output == second.output


def test_account_status_reports_a_known_state():
    result = check_account_status({"customer_id": "CUST-77"})
    assert result.output["status"] in ("active", "locked", "suspended")
    assert "CUST-77" in result.summary


def test_refund_eligibility_follows_the_thirty_day_rule():
    result = check_refund_eligibility({"order_id": "4471"})
    output = result.output

    if output["duplicate_charge"]:
        assert output["eligible"] is True
    elif output["days_since_charge"] <= 30:
        assert output["eligible"] is True
    else:
        assert output["eligible"] is False


def test_tools_cope_with_a_missing_identifier():
    result = check_refund_eligibility({})
    assert result.output["order_id"] == "unknown"


def test_running_an_unknown_tool_returns_an_error_result():
    result = run_tool("no_such_tool", {})
    assert result.output["error"] == "unknown tool"
    assert "does not exist" in result.summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_identifiers_are_read_from_the_ticket_text():
    identifiers = extract_identifiers(
        "Duplicate payment on order 4471", "My account 9912 was charged twice"
    )
    assert identifiers["order_id"] == "4471"
    assert identifiers["customer_id"] == "9912"


def test_identifiers_default_to_unknown():
    identifiers = extract_identifiers("Cannot log in", "Nothing works")
    assert identifiers["order_id"] == "unknown"
    assert identifiers["customer_id"] == "unknown"


def test_json_is_parsed_out_of_surrounding_prose():
    text = 'Sure! Here is my decision:\n```json\n{"action": "answer"}\n```\nHope that helps.'
    assert parse_first_json_object(text) == {"action": "answer"}


def test_nested_json_objects_are_parsed_whole():
    text = '{"action": "answer", "action_input": {"order_id": "1"}}'
    parsed = parse_first_json_object(text)
    assert parsed["action_input"]["order_id"] == "1"


def test_unparseable_text_returns_none():
    assert parse_first_json_object("no json here") is None
    assert parse_first_json_object("{not valid json}") is None
    assert parse_first_json_object("[1, 2, 3]") is None


def test_format_snippets_handles_an_empty_list():
    assert "no relevant knowledge base section" in format_snippets([])


def test_format_snippets_numbers_each_chunk():
    rendered = format_snippets(SNIPPETS)
    assert "[1]" in rendered
    assert "Duplicate charges" in rendered


# ---------------------------------------------------------------------------
# The ReAct loop
# ---------------------------------------------------------------------------


def run_agent(fake_llm) -> object:
    agent = TriageAgent(fake_llm)
    return agent.run(
        subject="Duplicate payment on order 4471",
        body="My card was charged twice and I want a refund.",
        customer_tier="pro",
        category="billing",
        urgency_bucket="high",
        snippets=SNIPPETS,
        config=CONFIG,
    )


def test_the_agent_can_answer_immediately(fake_llm_factory):
    fake = fake_llm_factory(['{"thought": "clear", "action": "answer", "action_input": {}}'])
    result = run_agent(fake)

    assert result.decision == "answer"
    assert result.tool_results == []
    assert len(result.trace) == 1
    # One decision call plus one writing call.
    assert result.llm_calls == 2
    assert result.response_text == "FAKE REPLY"


def test_a_tool_call_is_executed_and_fed_back(fake_llm_factory):
    fake = fake_llm_factory(
        [
            '{"thought": "check first", "action": "check_refund_eligibility",'
            ' "action_input": {"order_id": "4471"}}',
            '{"thought": "now I know", "action": "answer", "action_input": {}}',
        ]
    )
    result = run_agent(fake)

    assert result.decision == "answer"
    assert len(result.tool_results) == 1
    assert result.tool_results[0].tool == "check_refund_eligibility"
    # The observation is recorded on the step that made the call.
    assert result.trace[0].observation != ""
    assert "4471" in result.trace[0].observation


def test_a_missing_identifier_is_filled_in_from_the_ticket(fake_llm_factory):
    fake = fake_llm_factory(
        [
            '{"thought": "check", "action": "check_refund_eligibility", "action_input": {}}',
            '{"thought": "done", "action": "answer", "action_input": {}}',
        ]
    )
    result = run_agent(fake)

    # The agent read "order 4471" out of the subject line.
    assert result.tool_results[0].arguments["order_id"] == "4471"


def test_an_identical_repeated_tool_call_is_not_executed_twice(fake_llm_factory):
    """Small models sometimes ask for a call they have already made.

    Re-running it burns a step, and for a tool with real side effects it would
    be worse than wasteful, so the earlier result is replayed instead. Observed
    with llama3.2:1b, which asked for check_refund_eligibility twice in a row.
    """
    same_call = (
        '{"thought": "check", "action": "check_refund_eligibility",'
        ' "action_input": {"order_id": "4471"}}'
    )
    fake = fake_llm_factory(
        [
            same_call,
            same_call,
            '{"thought": "done", "action": "answer", "action_input": {}}',
        ]
    )
    result = run_agent(fake)

    # Three reasoning steps, but the tool only actually ran once.
    assert len(result.trace) == 3
    assert len(result.tool_results) == 1

    # The repeat still gets an observation, marked as a replay.
    assert "already checked" in result.trace[1].observation
    assert result.trace[0].observation in result.trace[1].observation


def test_a_repeated_tool_call_with_different_arguments_does_run(fake_llm_factory):
    """Only identical repeats are suppressed - a different id is a real query."""
    fake = fake_llm_factory(
        [
            '{"thought": "first", "action": "check_refund_eligibility",'
            ' "action_input": {"order_id": "1111"}}',
            '{"thought": "second", "action": "check_refund_eligibility",'
            ' "action_input": {"order_id": "2222"}}',
            '{"thought": "done", "action": "answer", "action_input": {}}',
        ]
    )
    result = run_agent(fake)

    assert len(result.tool_results) == 2
    assert result.tool_results[0].arguments["order_id"] == "1111"
    assert result.tool_results[1].arguments["order_id"] == "2222"


def test_escalation_is_a_terminal_decision(fake_llm_factory):
    fake = fake_llm_factory(
        ['{"thought": "too hard", "action": "escalate_to_human", "action_input": {}}']
    )
    result = run_agent(fake)

    assert result.decision == "escalate_to_human"
    assert len(result.trace) == 1


def test_an_unparseable_reply_escalates_rather_than_guessing(fake_llm_factory):
    fake = fake_llm_factory(["I am not going to answer in JSON today."])
    result = run_agent(fake)

    assert result.decision == "escalate_to_human"
    assert "could not be parsed" in result.trace[0].thought


def test_an_unknown_action_is_downgraded_to_answer(fake_llm_factory):
    fake = fake_llm_factory(['{"thought": "hmm", "action": "launch_rocket", "action_input": {}}'])
    result = run_agent(fake)

    assert result.decision == "answer"
    assert result.trace[0].action == "answer"


def test_a_non_dictionary_action_input_is_ignored(fake_llm_factory):
    fake = fake_llm_factory(['{"thought": "hmm", "action": "answer", "action_input": "oops"}'])
    result = run_agent(fake)

    assert result.trace[0].action_input == {}


def test_the_step_limit_stops_a_looping_model(fake_llm_factory):
    """A model that only ever wants to call tools must not spin forever.

    Each distinct call is a fresh customer id, so the duplicate-call guard does
    not mask the loop - the step limit is what stops it.
    """
    replies = []
    for index in range(20):
        replies.append(
            '{"thought": "again", "action": "check_account_status",'
            ' "action_input": {"customer_id": "' + str(index) + '"}}'
        )

    agent = TriageAgent(fake_llm_factory(replies))
    result = agent.run(
        subject="Cannot log in",
        body="Nothing works",
        customer_tier="free",
        category="account",
        urgency_bucket="low",
        snippets=SNIPPETS,
        config=CONFIG,
    )

    assert result.decision == "escalate_to_human"
    assert len(result.tool_results) == agent.max_steps


def test_a_model_repeating_one_call_forever_also_stops(fake_llm_factory):
    """The same loop, but every call identical: the guard stops re-running the
    tool, and the step limit still ends the loop."""
    tool_call = (
        '{"thought": "again", "action": "check_account_status",'
        ' "action_input": {"customer_id": "1"}}'
    )
    agent = TriageAgent(fake_llm_factory([tool_call] * 20))

    result = agent.run(
        subject="Cannot log in",
        body="Nothing works",
        customer_tier="free",
        category="account",
        urgency_bucket="low",
        snippets=SNIPPETS,
        config=CONFIG,
    )

    assert result.decision == "escalate_to_human"
    # The loop still ran to its limit, but the tool executed exactly once.
    assert len(result.trace) == agent.max_steps + 1
    assert len(result.tool_results) == 1


def test_the_result_dictionary_carries_the_audit_trail(fake_llm_factory):
    fake = fake_llm_factory(
        [
            '{"thought": "check", "action": "check_refund_eligibility", "action_input": {}}',
            '{"thought": "done", "action": "answer", "action_input": {}}',
        ]
    )
    result = run_agent(fake).as_dict()

    assert result["decision"] == "answer"
    assert len(result["reasoning_trace"]) == 2
    assert result["reasoning_trace"][0]["step"] == 1
    assert len(result["tool_calls"]) == 1
    assert result["llm_backend"] == "fake"


# ---------------------------------------------------------------------------
# The offline template backend used when no model server is available
# ---------------------------------------------------------------------------


def test_the_template_backend_drives_the_loop_without_a_server():
    from app.llm.client import LlmClient

    agent = TriageAgent(LlmClient())
    result = agent.run(
        subject="Duplicate payment on order 4471",
        body="My card was charged twice and I want a refund.",
        customer_tier="pro",
        category="billing",
        urgency_bucket="high",
        snippets=SNIPPETS,
        config=CONFIG,
    )

    assert result.llm_backend == "template"
    assert result.decision in ("answer", "escalate_to_human")
    assert len(result.response_text) > 0


def test_the_two_prompt_variants_produce_different_text():
    from app.llm.client import LlmClient

    agent = TriageAgent(LlmClient())
    arguments = {
        "subject": "Dashboard is extremely slow",
        "body": "Everything times out for our whole team.",
        "customer_tier": "enterprise",
        "category": "technical",
        "urgency_bucket": "high",
        "snippets": SNIPPETS,
    }

    concise = agent.run(config=get_config("concise_policy|k2"), **arguments)
    stepwise = agent.run(config=get_config("empathetic_stepwise|k2"), **arguments)

    assert concise.response_text != stepwise.response_text


def test_unknown_configuration_names_are_rejected():
    with pytest.raises(KeyError):
        get_config("no_such_config")

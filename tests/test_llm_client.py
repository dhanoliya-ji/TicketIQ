"""Tests for the language model client.

No test here talks to a real server: ``requests`` is monkeypatched, which lets
us prove both the happy path (Ollama answers) and the important failure path
(Ollama is down, so the template backend takes over without failing the
ticket).
"""

import pytest
import requests

from app.llm import client as client_module
from app.llm.client import (
    BACKEND_OLLAMA,
    BACKEND_TEMPLATE,
    TASK_DECIDE,
    TASK_WRITE,
    LlmClient,
    LlmRequest,
    OllamaBackend,
    TemplateBackend,
)
from app.llm.configs import ALL_CONFIGS, CONCISE_POLICY, EMPATHETIC_STEPWISE, all_config_names

DECIDE_FACTS = {
    "category": "billing",
    "urgency_bucket": "high",
    "customer_tier": "pro",
    "subject": "Refund request",
    "body": "I was charged twice and want a refund.",
    "order_id": "4471",
    "customer_id": "9912",
    "tools_already_used": [],
}

WRITE_FACTS = {
    "prompt_variant": CONCISE_POLICY,
    "category": "billing",
    "decision": "answer",
    "snippets": [{"heading": "Duplicate charges", "text": "Duplicates are always refunded. More."}],
    "tool_results": [{"tool": "check_refund_eligibility", "summary": "eligible for 49.00"}],
}


class FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError("status " + str(self.status_code))


# ---------------------------------------------------------------------------
# Configurations (the bandit's action space)
# ---------------------------------------------------------------------------


def test_there_are_four_distinct_configurations():
    names = all_config_names()
    assert len(names) == 4
    assert len(set(names)) == 4


def test_each_configuration_pairs_a_prompt_variant_with_a_top_k():
    for config in ALL_CONFIGS:
        assert config.prompt_variant in (CONCISE_POLICY, EMPATHETIC_STEPWISE)
        assert config.rag_top_k in (2, 5)
        assert config.name == config.prompt_variant + "|k" + str(config.rag_top_k)


def test_the_two_variants_use_different_system_prompts_and_models():
    concise = [c for c in ALL_CONFIGS if c.prompt_variant == CONCISE_POLICY][0]
    stepwise = [c for c in ALL_CONFIGS if c.prompt_variant == EMPATHETIC_STEPWISE][0]

    assert concise.system_prompt != stepwise.system_prompt
    assert concise.model_name != stepwise.model_name


# ---------------------------------------------------------------------------
# The template backend
# ---------------------------------------------------------------------------


def test_template_decide_checks_a_refund_before_promising_one():
    decision = TemplateBackend().generate(
        LlmRequest("system", "user", TASK_DECIDE, DECIDE_FACTS), "any-model"
    )
    assert '"check_refund_eligibility"' in decision
    assert "4471" in decision


def test_template_decide_does_not_repeat_a_tool_it_already_called():
    facts = dict(DECIDE_FACTS)
    facts["tools_already_used"] = ["check_refund_eligibility"]

    decision = TemplateBackend().generate(
        LlmRequest("system", "user", TASK_DECIDE, facts), "any-model"
    )
    assert '"answer"' in decision


def test_template_decide_escalates_an_enterprise_outage():
    facts = {
        "category": "technical",
        "urgency_bucket": "high",
        "customer_tier": "enterprise",
        "subject": "Total outage",
        "body": "Nothing works at all.",
        "tools_already_used": [],
    }
    decision = TemplateBackend().generate(
        LlmRequest("system", "user", TASK_DECIDE, facts), "any-model"
    )
    assert '"escalate_to_human"' in decision


def test_template_decide_never_escalates_a_feature_request():
    facts = {
        "category": "feature_request",
        "urgency_bucket": "high",
        "customer_tier": "enterprise",
        "subject": "Please add dark mode",
        "body": "We would love a dark theme.",
        "tools_already_used": [],
    }
    decision = TemplateBackend().generate(
        LlmRequest("system", "user", TASK_DECIDE, facts), "any-model"
    )
    assert '"answer"' in decision


def test_template_write_quotes_the_retrieved_policy():
    text = TemplateBackend().generate(
        LlmRequest("system", "user", TASK_WRITE, WRITE_FACTS), "any-model"
    )
    assert "Duplicate charges" in text
    assert "check_refund_eligibility" in text


def test_the_two_variants_write_differently_shaped_replies():
    backend = TemplateBackend()

    concise = backend.generate(LlmRequest("s", "u", TASK_WRITE, WRITE_FACTS), "m")

    stepwise_facts = dict(WRITE_FACTS)
    stepwise_facts["prompt_variant"] = EMPATHETIC_STEPWISE
    stepwise = backend.generate(LlmRequest("s", "u", TASK_WRITE, stepwise_facts), "m")

    assert concise != stepwise
    # The step-by-step variant numbers its steps and is the longer of the two.
    assert "1." in stepwise
    assert len(stepwise) > len(concise)


def test_template_write_copes_with_no_retrieved_knowledge():
    facts = dict(WRITE_FACTS)
    facts["snippets"] = []
    text = TemplateBackend().generate(LlmRequest("s", "u", TASK_WRITE, facts), "m")
    assert "No matching policy" in text


def test_template_write_says_what_happens_on_an_escalation():
    facts = dict(WRITE_FACTS)
    facts["decision"] = "escalate_to_human"
    text = TemplateBackend().generate(LlmRequest("s", "u", TASK_WRITE, facts), "m")
    assert "escalated" in text or "specialist" in text


# ---------------------------------------------------------------------------
# The Ollama backend
# ---------------------------------------------------------------------------


def test_availability_probe_is_true_when_the_server_answers(monkeypatch):
    monkeypatch.setattr(
        client_module.requests, "get", lambda url, timeout: FakeResponse(200, {"models": []})
    )
    assert OllamaBackend("http://localhost:11434", 5.0).is_available() is True


def test_availability_probe_is_false_when_the_server_is_down(monkeypatch):
    def refuse(url, timeout):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(client_module.requests, "get", refuse)
    assert OllamaBackend("http://localhost:11434", 5.0).is_available() is False


def test_generate_sends_the_prompts_and_returns_the_completion(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent["url"] = url
        sent["payload"] = json
        return FakeResponse(200, {"response": "  the model answer  "})

    monkeypatch.setattr(client_module.requests, "post", fake_post)

    backend = OllamaBackend("http://localhost:11434", 5.0)
    text = backend.generate(LlmRequest("system text", "user text", TASK_WRITE, {}), "llama3")

    assert text == "the model answer"
    assert sent["url"].endswith("/api/generate")
    assert sent["payload"]["model"] == "llama3"
    assert sent["payload"]["system"] == "system text"
    assert sent["payload"]["stream"] is False


def test_a_trailing_slash_in_the_base_url_is_handled(monkeypatch):
    monkeypatch.setattr(client_module.requests, "get", lambda url, timeout: FakeResponse(200, {}))
    backend = OllamaBackend("http://localhost:11434/", 5.0)
    assert backend.base_url == "http://localhost:11434"


# ---------------------------------------------------------------------------
# Backend selection and fallback
# ---------------------------------------------------------------------------


def test_the_client_uses_ollama_when_it_is_available(monkeypatch):
    monkeypatch.setattr(client_module.SETTINGS, "llm_backend", "auto")
    monkeypatch.setattr(client_module.requests, "get", lambda url, timeout: FakeResponse(200, {}))
    monkeypatch.setattr(
        client_module.requests,
        "post",
        lambda url, json, timeout: FakeResponse(200, {"response": "model text"}),
    )

    client = LlmClient()
    assert client.active_backend == BACKEND_OLLAMA

    response = client.generate(LlmRequest("s", "u", TASK_WRITE, WRITE_FACTS), "llama3")
    assert response.text == "model text"
    assert response.backend == BACKEND_OLLAMA
    assert response.latency_seconds >= 0.0


def test_auto_falls_back_to_the_template_backend_when_ollama_is_down(monkeypatch):
    monkeypatch.setattr(client_module.SETTINGS, "llm_backend", "auto")

    def refuse(url, timeout):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(client_module.requests, "get", refuse)

    client = LlmClient()
    assert client.active_backend == BACKEND_TEMPLATE


def test_a_mid_request_failure_falls_back_instead_of_failing_the_ticket(monkeypatch):
    """The server was up at startup but dies on the actual call."""
    monkeypatch.setattr(client_module.SETTINGS, "llm_backend", "auto")
    monkeypatch.setattr(client_module.requests, "get", lambda url, timeout: FakeResponse(200, {}))

    def fail_post(url, json, timeout):
        raise requests.Timeout("too slow")

    monkeypatch.setattr(client_module.requests, "post", fail_post)

    client = LlmClient()
    assert client.active_backend == BACKEND_OLLAMA

    response = client.generate(LlmRequest("s", "u", TASK_WRITE, WRITE_FACTS), "llama3")
    # The ticket still gets an answer, clearly labelled as template written.
    assert response.backend == BACKEND_TEMPLATE
    assert "Duplicate charges" in response.text


def test_an_empty_model_reply_falls_back_to_the_template(monkeypatch):
    monkeypatch.setattr(client_module.SETTINGS, "llm_backend", "ollama")
    monkeypatch.setattr(
        client_module.requests,
        "post",
        lambda url, json, timeout: FakeResponse(200, {"response": "   "}),
    )

    client = LlmClient()
    response = client.generate(LlmRequest("s", "u", TASK_WRITE, WRITE_FACTS), "llama3")
    assert response.backend == BACKEND_TEMPLATE


def test_forcing_the_template_backend_never_probes_the_server(monkeypatch):
    monkeypatch.setattr(client_module.SETTINGS, "llm_backend", "template")

    def explode(*args, **kwargs):
        raise AssertionError("the server must not be contacted")

    monkeypatch.setattr(client_module.requests, "get", explode)
    monkeypatch.setattr(client_module.requests, "post", explode)

    client = LlmClient()
    assert client.active_backend == BACKEND_TEMPLATE
    response = client.generate(LlmRequest("s", "u", TASK_DECIDE, DECIDE_FACTS), "llama3")
    assert response.backend == BACKEND_TEMPLATE


def test_the_response_dictionary_reports_the_backend():
    client = LlmClient()
    response = client.generate(LlmRequest("s", "u", TASK_WRITE, WRITE_FACTS), "llama3")
    summary = response.as_dict()

    assert summary["backend"] == BACKEND_TEMPLATE
    assert summary["model"] == "llama3"
    assert summary["latency_seconds"] >= 0.0


def test_forcing_ollama_still_configures_that_backend(monkeypatch):
    monkeypatch.setattr(client_module.SETTINGS, "llm_backend", "ollama")
    client = LlmClient()
    assert client.active_backend == BACKEND_OLLAMA


@pytest.mark.parametrize("task", [TASK_DECIDE, TASK_WRITE])
def test_both_task_types_produce_non_empty_text(task):
    facts = DECIDE_FACTS if task == TASK_DECIDE else WRITE_FACTS
    text = TemplateBackend().generate(LlmRequest("s", "u", task, facts), "m")
    assert len(text) > 0

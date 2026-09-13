"""Talking to a language model, with a deterministic offline fallback.

Two backends implement the same tiny interface:

* ``OllamaBackend``   - posts to a local Ollama server (``llama3`` / ``mistral``).
* ``TemplateBackend`` - writes the answer from the retrieved knowledge itself,
  with no server at all.

Why a fallback exists
---------------------
Everything else in this project (the classifier, the retriever, the bandit,
the workflow engine) must be testable and reviewable on a laptop or in CI
where no model server is running.  ``TICKETIQ_LLM_BACKEND=auto`` therefore
probes Ollama once at startup and quietly uses the template writer when there
is no server.  The response always reports which backend produced it, so a
reviewer can never mistake template output for model output.

The template backend is *not* a stub that returns a fixed string: it reads the
retrieved policy chunks, the chosen prompt variant and the tool results, and
composes a different answer for each, so the rest of the pipeline - including
the reward signal - stays meaningful offline.
"""

import json
import time
from typing import Any

import requests

from app.settings import SETTINGS

# The two things we ever ask a language model to do.
TASK_DECIDE = "decide"  # choose the next agent step, answer as JSON
TASK_WRITE = "write"  # write the customer facing reply

BACKEND_OLLAMA = "ollama"
BACKEND_TEMPLATE = "template"


class LlmRequest:
    """One call to a language model."""

    def __init__(self, system_prompt: str, user_prompt: str, task: str, facts: dict) -> None:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        # TASK_DECIDE or TASK_WRITE.
        self.task = task
        # Structured version of everything that is also written into the
        # prompt.  A real model ignores this and reads the prompt; the template
        # backend uses it so it does not have to parse its own prompt back.
        self.facts = facts


class LlmResponse:
    """What came back, plus how it was produced."""

    def __init__(self, text: str, backend: str, model: str, latency_seconds: float) -> None:
        self.text = text
        self.backend = backend
        self.model = model
        self.latency_seconds = latency_seconds

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model": self.model,
            "latency_seconds": round(self.latency_seconds, 4),
        }


# ---------------------------------------------------------------------------
# Backend 1: a real Ollama server
# ---------------------------------------------------------------------------
class OllamaBackend:
    """Calls the ``/api/generate`` endpoint of a local Ollama server."""

    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        """One cheap probe: does the server answer its model listing endpoint?"""
        try:
            response = requests.get(self.base_url + "/api/tags", timeout=2.0)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, request: LlmRequest, model: str) -> str:
        """Send the prompt and return the raw completion text."""
        payload: dict[str, Any] = {
            "model": model,
            "system": request.system_prompt,
            "prompt": request.user_prompt,
            # We want one complete answer, not a token stream.
            "stream": False,
            "options": {"temperature": 0.2},
        }
        response = requests.post(
            self.base_url + "/api/generate",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        return str(body.get("response", "")).strip()


# ---------------------------------------------------------------------------
# Backend 2: the offline template writer
# ---------------------------------------------------------------------------
class TemplateBackend:
    """Composes an answer from the retrieved knowledge, with no model server."""

    def generate(self, request: LlmRequest, model: str) -> str:
        if request.task == TASK_DECIDE:
            return self._decide(request.facts)
        return self._write(request.facts)

    # -- deciding the next agent step ---------------------------------------
    def _decide(self, facts: dict) -> str:
        """Return the same JSON shape a real model is asked to produce.

        The rules mirror the escalation document: feature requests are never
        escalated, refunds are checked before they are promised, access
        problems need the account status, and an angry high urgency enterprise
        technical ticket goes to a human.
        """
        category = facts.get("category", "technical")
        urgency_bucket = facts.get("urgency_bucket", "low")
        customer_tier = facts.get("customer_tier", "free")
        text = str(facts.get("subject", "") + " " + facts.get("body", "")).lower()
        already_used_tools = facts.get("tools_already_used", [])

        def decision(thought: str, action: str, action_input: dict) -> str:
            return json.dumps({"thought": thought, "action": action, "action_input": action_input})

        # 1. A refund claim must be checked against the policy first.
        wants_refund = "refund" in text or "charged twice" in text or "duplicate" in text
        if (
            category == "billing"
            and wants_refund
            and "check_refund_eligibility" not in already_used_tools
        ):
            return decision(
                "The customer is asking for money back, so eligibility must be "
                "confirmed before anything is promised.",
                "check_refund_eligibility",
                {"order_id": facts.get("order_id", "unknown")},
            )

        # 2. An access problem needs the real account state.
        if category == "account" and "check_account_status" not in already_used_tools:
            return decision(
                "This is an access problem, so the current state of the account "
                "decides which instructions are correct.",
                "check_account_status",
                {"customer_id": facts.get("customer_id", "unknown")},
            )

        # 3. Serious technical pain on a paying tier goes to a specialist.
        if category == "technical" and urgency_bucket == "high" and customer_tier == "enterprise":
            return decision(
                "An enterprise customer reports a high urgency technical failure, "
                "which the escalation rules send to a human specialist.",
                "escalate_to_human",
                {"reason": "enterprise high urgency technical issue"},
            )

        # 4. Everything else can be answered from the retrieved policy.
        return decision(
            "The retrieved knowledge base sections cover this question, so it "
            "can be answered directly.",
            "answer",
            {},
        )

    # -- writing the customer reply -----------------------------------------
    def _write(self, facts: dict) -> str:
        """Compose the reply out of the retrieved chunks and the tool results."""
        prompt_variant = facts.get("prompt_variant", "concise_policy")
        snippets = facts.get("snippets", [])
        tool_results = facts.get("tool_results", [])
        decision = facts.get("decision", "answer")
        category = facts.get("category", "technical")

        # The policy lines we are allowed to rely on.
        policy_lines: list[str] = []
        for snippet in snippets:
            heading = snippet.get("heading", "")
            text = snippet.get("text", "")
            policy_lines.append(heading + ": " + _first_sentence(text))

        observation_lines: list[str] = []
        for result in tool_results:
            observation_lines.append(
                result.get("tool", "tool") + " -> " + result.get("summary", "")
            )

        if prompt_variant == "concise_policy":
            return self._write_concise(category, decision, policy_lines, observation_lines)
        return self._write_stepwise(category, decision, policy_lines, observation_lines)

    def _write_concise(
        self,
        category: str,
        decision: str,
        policy_lines: list[str],
        observation_lines: list[str],
    ) -> str:
        parts = ["Category: " + category + "."]

        if len(policy_lines) > 0:
            parts.append("Applicable policy - " + " ".join(policy_lines[:2]))
        else:
            parts.append("No matching policy section was found for this request.")

        for line in observation_lines:
            parts.append("Checked: " + line + ".")

        if decision == "escalate_to_human":
            parts.append("This ticket is escalated to a human specialist.")
        else:
            parts.append("Applying the rule above resolves this request.")

        return " ".join(parts)

    def _write_stepwise(
        self,
        category: str,
        decision: str,
        policy_lines: list[str],
        observation_lines: list[str],
    ) -> str:
        lines = ["I understand this is disruptive, and I am sorry for the trouble it is causing."]
        lines.append("")
        lines.append("Here is what applies to your " + category + " request:")

        step_number = 1
        for policy_line in policy_lines[:3]:
            lines.append(str(step_number) + ". " + policy_line)
            step_number = step_number + 1

        if len(policy_lines) == 0:
            lines.append("1. I could not find a policy section that covers this exactly.")

        for observation in observation_lines:
            lines.append(
                str(step_number) + ". I already checked this for you: " + observation + "."
            )
            step_number = step_number + 1

        lines.append("")
        if decision == "escalate_to_human":
            lines.append(
                "What happens next: a support specialist now owns this ticket and "
                "will come back to you with a fix or a status update."
            )
        else:
            lines.append(
                "What happens next: follow the steps above and reply to this "
                "ticket if anything is still not working."
            )

        return "\n".join(lines)


def _first_sentence(text: str) -> str:
    """Return the first sentence of a chunk, so replies stay short."""
    stripped = text.strip().replace("\n", " ")
    position = stripped.find(". ")
    if position == -1:
        return stripped
    return stripped[: position + 1]


# ---------------------------------------------------------------------------
# The client the rest of the application uses
# ---------------------------------------------------------------------------
class LlmClient:
    """Chooses a backend once and then serves every generation request."""

    def __init__(self) -> None:
        self.ollama = OllamaBackend(SETTINGS.ollama_base_url, SETTINGS.llm_timeout_seconds)
        self.template = TemplateBackend()
        self.active_backend = self._choose_backend()

    def _choose_backend(self) -> str:
        """Decide once, at startup, which backend to use."""
        configured = SETTINGS.llm_backend

        if configured == BACKEND_TEMPLATE:
            return BACKEND_TEMPLATE
        if configured == BACKEND_OLLAMA:
            return BACKEND_OLLAMA
        # "auto": use Ollama only if it actually answers.
        if self.ollama.is_available():
            return BACKEND_OLLAMA
        return BACKEND_TEMPLATE

    def generate(self, request: LlmRequest, model: str) -> LlmResponse:
        """Run one generation and always return a usable response.

        If the Ollama call fails at request time (server stopped, model not
        pulled, timeout) we fall back to the template backend for that single
        request rather than failing the whole ticket.
        """
        started_at = time.perf_counter()

        if self.active_backend == BACKEND_OLLAMA:
            try:
                text = self.ollama.generate(request, model)
                if text != "":
                    elapsed = time.perf_counter() - started_at
                    return LlmResponse(text, BACKEND_OLLAMA, model, elapsed)
            except (requests.RequestException, ValueError):
                # Fall through to the template backend below.
                pass

        text = self.template.generate(request, model)
        elapsed = time.perf_counter() - started_at
        return LlmResponse(text, BACKEND_TEMPLATE, model, elapsed)

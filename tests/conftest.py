"""Shared pytest fixtures.

The environment variables are set here, at import time, because ``conftest.py``
is imported before any test module and ``app.settings.SETTINGS`` is built once
when ``app.settings`` is first imported.  Setting them later would have no
effect.

Two things are forced for the whole suite:

* ``TICKETIQ_VAR_DIR`` points at a throw-away folder, so tests never touch the
  SQLite file or the bandit state of a real run.
* ``TICKETIQ_LLM_BACKEND=template`` keeps tests offline and deterministic; no
  test ever depends on an Ollama server being installed.
"""

import os
import tempfile
from pathlib import Path

# Must happen before app.settings is imported anywhere.
_TEST_VAR_DIR = Path(tempfile.mkdtemp(prefix="ticketiq-tests-"))
os.environ["TICKETIQ_VAR_DIR"] = str(_TEST_VAR_DIR)
os.environ["TICKETIQ_LLM_BACKEND"] = "template"

import pytest  # noqa: E402

from app.llm.client import LlmResponse  # noqa: E402
from app.workflow.state_store import WorkflowStateStore  # noqa: E402


@pytest.fixture
def temporary_store(tmp_path: Path) -> WorkflowStateStore:
    """A workflow state store backed by a fresh SQLite file per test."""
    return WorkflowStateStore(tmp_path / "workflow.sqlite3")


class FakeLlmClient:
    """A stand-in for the language model with no template logic at all.

    Used by the end-to-end test so that the pipeline is exercised with the
    model *mocked out*: the replies are fixed strings, and the test asserts on
    the plumbing (stages, tools, persistence) rather than on generated text.
    """

    def __init__(self, decision_replies: list[str], written_reply: str = "FAKE REPLY") -> None:
        # One decision reply per loop turn, consumed in order.
        self.decision_replies = list(decision_replies)
        self.written_reply = written_reply
        self.active_backend = "fake"
        self.requests: list[object] = []

    def generate(self, request, model: str) -> LlmResponse:  # noqa: ANN001 - test double
        self.requests.append(request)

        if request.task == "decide":
            if len(self.decision_replies) > 0:
                text = self.decision_replies.pop(0)
            else:
                text = '{"thought": "done", "action": "answer", "action_input": {}}'
            return LlmResponse(text, "fake", model, 0.001)

        return LlmResponse(self.written_reply, "fake", model, 0.001)


@pytest.fixture
def fake_llm_factory():
    """Lets a test build a FakeLlmClient with the decisions it wants."""

    def build(decision_replies: list[str], written_reply: str = "FAKE REPLY") -> FakeLlmClient:
        return FakeLlmClient(decision_replies, written_reply)

    return build

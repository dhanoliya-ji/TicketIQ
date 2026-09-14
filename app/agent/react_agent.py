"""A ReAct style reasoning loop.

ReAct = *Reason* then *Act*.  On every turn the model is shown the ticket, the
retrieved policy and everything observed so far, and must answer with one JSON
object:

    {"thought": "...", "action": "...", "action_input": {...}}

``action`` is one of:

* ``check_account_status``     - call the identity tool, then think again.
* ``check_refund_eligibility`` - call the billing tool, then think again.
* ``answer``                   - enough is known, write the reply.
* ``escalate_to_human``        - hand the ticket to a specialist.

Tool actions feed their result back as an observation and the loop runs again.
Terminal actions end the loop and a second model call writes the customer
reply.  ``agent_max_steps`` is a hard stop so a confused model can never spin.

Everything the loop did - each thought, action and observation - is kept in
``AgentResult.trace`` and returned by the API, because a triage decision that
cannot be audited is not worth much in support operations.
"""

import json
import logging

from app.agent.tools import (
    ALL_ACTION_NAMES,
    ALL_TOOL_NAMES,
    ANSWER,
    ESCALATE_TO_HUMAN,
    ToolResult,
    run_tool,
)
from app.llm.client import TASK_DECIDE, TASK_WRITE, LlmClient, LlmRequest, LlmResponse
from app.llm.configs import PipelineConfig
from app.settings import SETTINGS

# The assignment asks for the agent's decision, its tool calls and its
# reasoning trace to be inspectable in the logs as well as in the API response.
# Logging from the agent itself - rather than from the endpoint - means the
# trace is recorded even when the agent is driven by a script or a test.
logger = logging.getLogger("ticketiq.agent")

# The one category whose escalation the knowledge base forbids outright.
FEATURE_REQUEST_CATEGORY = "feature_request"

# How many times the model may re-ask for a call it has already made before the
# loop stops humouring it and answers with the result already in hand.
REPEATS_BEFORE_GIVING_UP = 2

FEATURE_REQUEST_OVERRIDE_NOTE = (
    "policy: a feature request is never escalated, so this was answered on the "
    "normal queue instead"
)


def enforce_escalation_policy(decision: str, category: str) -> tuple[str, str]:
    """Apply the one escalation rule the knowledge base states outright.

    Returns ``(decision, override note)``; the note is empty when nothing was
    changed. This is a function rather than an inline check because three
    separate paths can end in an escalation - the model choosing it, an
    unparseable reply, and the step limit running out - and the rule has to
    hold on all three, not just the tidy one.
    """
    if decision == ESCALATE_TO_HUMAN and category == FEATURE_REQUEST_CATEGORY:
        return ANSWER, FEATURE_REQUEST_OVERRIDE_NOTE
    return decision, ""


class AgentStep:
    """One turn of the loop, kept for the audit trail."""

    def __init__(self, number: int, thought: str, action: str, action_input: dict) -> None:
        self.number = number
        self.thought = thought
        self.action = action
        self.action_input = action_input
        # Filled in only when the action was a tool call.
        self.observation: str = ""
        # Set when a policy rule changed the model's chosen action. Empty for
        # the ordinary case, so the trace shows plainly when the system
        # overrode the model and why.
        self.override: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.number,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "override": self.override,
        }


class AgentResult:
    """Everything the agent produced for one ticket."""

    def __init__(self) -> None:
        self.decision: str = ANSWER
        self.response_text: str = ""
        self.trace: list[AgentStep] = []
        self.tool_results: list[ToolResult] = []
        self.llm_calls: int = 0
        self.llm_model: str = ""
        # Which backend served each generation, in order. A single ticket can
        # legitimately use both: if a model call times out the client falls
        # back to the template writer for that request only.
        self.backends_used: list[str] = []

    @property
    def llm_backend(self) -> str:
        """The backend that served this ticket.

        Reporting only the last call would have been misleading: a ticket whose
        decision fell back to the template but whose reply was written by the
        model used to report "template", which undersold what actually
        happened. When the calls disagree, both names are reported.
        """
        distinct: list[str] = []
        for backend in self.backends_used:
            if backend not in distinct:
                distinct.append(backend)

        if len(distinct) == 0:
            return ""
        return "+".join(distinct)

    def as_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "reasoning_trace": [step.as_dict() for step in self.trace],
            "tool_calls": [result.as_dict() for result in self.tool_results],
            "llm_calls": self.llm_calls,
            "llm_backend": self.llm_backend,
            "llm_model": self.llm_model,
        }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def extract_identifiers(subject: str, body: str) -> dict[str, str]:
    """Pull an order id and a customer id out of the ticket text.

    Support tickets rarely carry structured fields, so we look for the first
    run of digits after the words "order" or "account"/"customer".  When
    nothing is found the tools still run and report on an unknown identifier,
    which is exactly what a real agent would surface to the human.
    """
    words = (subject + " " + body).replace(",", " ").replace(".", " ").split()

    order_id = "unknown"
    customer_id = "unknown"

    for index in range(len(words)):
        word = words[index].lower()
        digits = "".join(character for character in words[index] if character.isdigit())

        if digits == "":
            continue

        # A number directly after one of these words identifies the thing.
        previous_word = words[index - 1].lower() if index > 0 else ""
        if previous_word in ("order", "invoice", "payment", "charge"):
            order_id = digits
        elif previous_word in ("account", "customer", "workspace", "user"):
            customer_id = digits
        elif order_id == "unknown" and word.startswith("ord"):
            order_id = digits

    return {"order_id": order_id, "customer_id": customer_id}


def parse_first_json_object(text: str) -> dict | None:
    """Find and parse the first ``{...}`` object in a model reply.

    Models often wrap JSON in prose or a code fence, so we scan for balanced
    braces instead of trusting the whole string to be valid JSON.
    """
    start_index = text.find("{")
    if start_index == -1:
        return None

    depth = 0
    for index in range(start_index, len(text)):
        character = text[index]
        if character == "{":
            depth = depth + 1
        elif character == "}":
            depth = depth - 1
            if depth == 0:
                candidate = text[start_index : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                if isinstance(parsed, dict):
                    return parsed
                return None
    return None


def format_snippets(snippets: list[dict]) -> str:
    """Render the retrieved chunks as numbered context for the prompt.

    Snippets are plain dictionaries rather than objects because the workflow
    engine persists every stage output as JSON, and the output of the
    retrieval stage is exactly what is handed to the agent here.
    """
    if len(snippets) == 0:
        return "(no relevant knowledge base section was found)"

    lines = []
    for position in range(len(snippets)):
        snippet = snippets[position]
        lines.append(
            "["
            + str(position + 1)
            + "] "
            + str(snippet.get("document_title", ""))
            + " / "
            + str(snippet.get("heading", ""))
            + "\n"
            + str(snippet.get("text", ""))
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class TriageAgent:
    """Runs the reason/act loop and then writes the customer reply."""

    def __init__(self, llm_client: LlmClient) -> None:
        self.llm = llm_client
        self.max_steps = SETTINGS.agent_max_steps

    def run(
        self,
        subject: str,
        body: str,
        customer_tier: str,
        category: str,
        urgency_bucket: str,
        snippets: list[dict],
        config: PipelineConfig,
    ) -> AgentResult:
        """Decide what to do about one ticket and produce the reply text."""
        result = AgentResult()
        result.llm_model = config.model_name

        identifiers = extract_identifiers(subject, body)
        knowledge_text = format_snippets(snippets)

        tools_already_used: list[str] = []
        observations: list[str] = []
        # tool + arguments -> the summary it returned, so an identical repeat
        # can be answered from memory instead of run again.
        completed_calls: dict[str, str] = {}
        # How many times each call has been asked for again after that.
        repeat_counts: dict[str, int] = {}
        decision = ANSWER

        for step_number in range(1, self.max_steps + 1):
            facts = {
                "category": category,
                "urgency_bucket": urgency_bucket,
                "customer_tier": customer_tier,
                "subject": subject,
                "body": body,
                "order_id": identifiers["order_id"],
                "customer_id": identifiers["customer_id"],
                "tools_already_used": list(tools_already_used),
            }
            request = LlmRequest(
                system_prompt=self._decision_system_prompt(),
                user_prompt=self._decision_user_prompt(
                    subject,
                    body,
                    customer_tier,
                    category,
                    urgency_bucket,
                    knowledge_text,
                    observations,
                    identifiers,
                ),
                task=TASK_DECIDE,
                facts=facts,
            )

            response = self.llm.generate(request, config.model_name)
            result.llm_calls = result.llm_calls + 1
            result.backends_used.append(response.backend)

            parsed = parse_first_json_object(response.text)
            if parsed is None:
                # An unparseable reply is treated as "I cannot decide", which
                # is safest resolved by a human rather than by guessing.
                step = AgentStep(
                    step_number,
                    "The model reply could not be parsed as a decision.",
                    ESCALATE_TO_HUMAN,
                    {"reason": "unparseable model response"},
                )
                result.trace.append(step)
                decision = ESCALATE_TO_HUMAN
                break

            thought = str(parsed.get("thought", "")).strip()
            action = str(parsed.get("action", ANSWER)).strip()
            action_input = parsed.get("action_input", {})
            if not isinstance(action_input, dict):
                action_input = {}

            # An action we do not recognise is downgraded to a plain answer.
            if action not in ALL_ACTION_NAMES:
                action = ANSWER

            # One policy rule is enforced rather than merely asked for. The
            # knowledge base states twice - in the feature request document and
            # in the escalation rules - that a feature request is never
            # escalated, and the system prompt says so too, but a small model
            # ignores it: llama3.2:1b escalated all four sample categories,
            # including a request for dark mode. Quoting a policy to customers
            # while acting against it is not a defensible default, so the
            # decision is corrected here and the correction is recorded in the
            # trace rather than hidden.
            action, override_note = enforce_escalation_policy(action, category)
            if override_note:
                logger.info("step %s | policy override | %s", step_number, override_note)

            step = AgentStep(step_number, thought, action, action_input)
            step.override = override_note
            result.trace.append(step)

            logger.info(
                "step %s | action=%s | input=%s | thought=%s",
                step_number,
                action,
                json.dumps(action_input),
                thought,
            )

            if action in ALL_TOOL_NAMES:
                # Fill in an identifier the model forgot to pass through.
                if action == "check_refund_eligibility" and "order_id" not in action_input:
                    action_input["order_id"] = identifiers["order_id"]
                if action == "check_account_status" and "customer_id" not in action_input:
                    action_input["customer_id"] = identifiers["customer_id"]

                # Smaller models sometimes ask for a call they have already
                # made. Re-running it would burn a step, and for a tool with
                # real side effects it would be worse than wasteful, so the
                # earlier answer is replayed instead.
                call_signature = action + " " + json.dumps(action_input, sort_keys=True)

                if call_signature in completed_calls:
                    previous_summary = completed_calls[call_signature]
                    step.observation = previous_summary + " (already checked at an earlier step)"
                    repeat_counts[call_signature] = repeat_counts.get(call_signature, 0) + 1

                    logger.info(
                        "step %s | tool=%s | repeated call, replaying the earlier result",
                        step_number,
                        action,
                    )

                    # Replaying once gives the model another chance to use the
                    # answer it already has. Asking a third time means it is not
                    # going to, and every further turn is a wasted model call:
                    # llama3.2:1b spent five turns and 19.5 seconds looping on
                    # check_account_status before the step limit escalated it,
                    # which also drags the reward down, since reward subtracts
                    # latency. The tool result and the retrieved policy are
                    # already in hand, so the loop ends with an answer instead.
                    if repeat_counts[call_signature] >= REPEATS_BEFORE_GIVING_UP:
                        step.action = ANSWER
                        step.override = (
                            "the model asked for the same tool call "
                            + str(repeat_counts[call_signature] + 1)
                            + " times; answering with the result it already had"
                        )
                        logger.info("step %s | %s", step_number, step.override)
                        decision = ANSWER
                        break

                    observations.append(action + " -> " + previous_summary)
                    continue

                tool_result = run_tool(action, action_input)
                result.tool_results.append(tool_result)
                step.observation = tool_result.summary
                completed_calls[call_signature] = tool_result.summary

                logger.info(
                    "step %s | tool=%s | arguments=%s | observation=%s",
                    step_number,
                    tool_result.tool,
                    json.dumps(tool_result.arguments),
                    tool_result.summary,
                )

                tools_already_used.append(action)
                observations.append(action + " -> " + tool_result.summary)
                # Loop again so the model can reason about what it just learned.
                continue

            # A terminal action ends the loop.
            decision = action
            break
        else:
            # The loop ran out of steps without a terminal action.
            decision = ESCALATE_TO_HUMAN
            result.trace.append(
                AgentStep(
                    self.max_steps + 1,
                    "The step limit was reached without a conclusion.",
                    ESCALATE_TO_HUMAN,
                    {"reason": "step limit reached"},
                )
            )

        # The two paths above - an unparseable reply and the step limit - set the
        # decision directly, so the rule is applied once more here. Without
        # this, a feature request whose model reply failed to parse would be
        # escalated in defiance of the policy the system quotes to customers.
        decision, final_override = enforce_escalation_policy(decision, category)
        if final_override:
            logger.info("policy override | %s", final_override)
            if len(result.trace) > 0:
                result.trace[-1].action = decision
                result.trace[-1].override = final_override

        logger.info(
            "decision=%s | steps=%s | tools=%s | config=%s",
            decision,
            len(result.trace),
            json.dumps([tool.tool for tool in result.tool_results]),
            config.name,
        )

        result.decision = decision

        written = self._write_reply(
            subject, body, customer_tier, category, decision, snippets, result.tool_results, config
        )
        result.response_text = written.text
        result.backends_used.append(written.backend)
        result.llm_calls = result.llm_calls + 1
        return result

    # -- prompt building ----------------------------------------------------
    def _decision_system_prompt(self) -> str:
        return (
            "You are the reasoning core of a support triage agent. "
            "Reply with exactly one JSON object and no other text. "
            'The object has the keys "thought", "action" and "action_input". '
            '"action" must be one of: ' + ", ".join(ALL_ACTION_NAMES) + ". "
            "Call a tool only when its result would change your reply. "
            "Never escalate a feature request."
        )

    def _decision_user_prompt(
        self,
        subject: str,
        body: str,
        customer_tier: str,
        category: str,
        urgency_bucket: str,
        knowledge_text: str,
        observations: list[str],
        identifiers: dict[str, str],
    ) -> str:
        if len(observations) == 0:
            observation_text = "(no tools have been called yet)"
        else:
            observation_text = "\n".join(observations)

        return (
            "TICKET SUBJECT: " + subject + "\n"
            "TICKET BODY: " + body + "\n"
            "PREDICTED CATEGORY: " + category + "\n"
            "URGENCY: " + urgency_bucket + "\n"
            "CUSTOMER TIER: " + customer_tier + "\n"
            "ORDER ID: " + identifiers["order_id"] + "\n"
            "CUSTOMER ID: " + identifiers["customer_id"] + "\n\n"
            "RETRIEVED KNOWLEDGE BASE SECTIONS:\n" + knowledge_text + "\n\n"
            "OBSERVATIONS SO FAR:\n" + observation_text + "\n\n"
            "What is the next action?"
        )

    def _write_reply(
        self,
        subject: str,
        body: str,
        customer_tier: str,
        category: str,
        decision: str,
        snippets: list[dict],
        tool_results: list[ToolResult],
        config: PipelineConfig,
    ) -> LlmResponse:
        """Second model call: turn the decision into the customer facing text.

        Returns the whole response, not just the text, so the caller can record
        which backend actually wrote the reply.
        """
        tool_summaries = []
        for tool_result in tool_results:
            tool_summaries.append({"tool": tool_result.tool, "summary": tool_result.summary})

        facts = {
            "prompt_variant": config.prompt_variant,
            "category": category,
            "customer_tier": customer_tier,
            "decision": decision,
            "snippets": snippets,
            "tool_results": tool_summaries,
        }

        if decision == ESCALATE_TO_HUMAN:
            instruction = (
                "This ticket is being escalated to a human specialist. Tell the "
                "customer what will happen next without promising a fix."
            )
        else:
            instruction = "Answer the customer using only the knowledge below."

        user_prompt = (
            instruction + "\n\n"
            "TICKET SUBJECT: " + subject + "\n"
            "TICKET BODY: " + body + "\n"
            "CUSTOMER TIER: " + customer_tier + "\n"
            "CATEGORY: " + category + "\n\n"
            "KNOWLEDGE BASE:\n" + format_snippets(snippets) + "\n\n"
            "TOOL RESULTS:\n" + self._format_tool_results(tool_results) + "\n\n"
            "Write the reply to the customer now."
        )

        request = LlmRequest(
            system_prompt=config.system_prompt,
            user_prompt=user_prompt,
            task=TASK_WRITE,
            facts=facts,
        )
        response = self.llm.generate(request, config.model_name)
        return response

    def _format_tool_results(self, tool_results: list[ToolResult]) -> str:
        if len(tool_results) == 0:
            return "(no tools were called)"
        lines = []
        for tool_result in tool_results:
            lines.append(tool_result.tool + ": " + tool_result.summary)
        return "\n".join(lines)

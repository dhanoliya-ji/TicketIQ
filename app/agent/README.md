# `app/agent/` — the agentic layer

## Files

| File | What it does |
|------|--------------|
| `react_agent.py` | The ReAct reasoning loop, the JSON extraction, the identifier extraction, and the second call that writes the reply. |
| `tools.py` | The two mock back-office tools and the registry that dispatches to them. |

## ReAct: reason, then act

Each turn the model sees the ticket, the retrieved policy and every observation
so far, and must reply with exactly one JSON object:

```json
{"thought": "...", "action": "...", "action_input": {...}}
```

| Action | Effect |
|--------|--------|
| `check_account_status` | run the identity tool, feed the result back, think again |
| `check_refund_eligibility` | run the billing tool, feed the result back, think again |
| `answer` | terminal: enough is known, write the reply |
| `escalate_to_human` | terminal: hand the ticket to a specialist |

```
Ticket + policy --> decide --> tool? --> observation --+
                      |                                |
                      |<-------------------------------+
                      v
              answer / escalate --> write the customer reply
```

A terminal action ends the loop, then a **second** model call — using the prompt
variant the bandit chose — turns the decision into the reply text.

## Every failure mode is handled, and tested

| Situation | Behaviour | Why |
|-----------|-----------|-----|
| Reply is not parseable JSON | escalate to a human | a triage agent that cannot state its decision should not guess at one |
| Action name unrecognised | downgrade to `answer` | an invented action must not become a real one |
| `action_input` is not an object | treat as empty | the tool fills in its own defaults |
| Tool called without an identifier | fill it in from the ticket text | models routinely forget to pass arguments through |
| Model only ever calls tools | hard stop at `agent_max_steps` (default 4), then escalate | a confused model can never spin forever |
| Model repeats a call it already made | replay the earlier result instead of re-running the tool | observed with `llama3.2:1b`; re-running burns a step and, for a tool with real side effects, would be worse than wasteful |
| Model escalates a feature request | corrected to `answer`, and the correction is recorded in the trace | the knowledge base forbids it twice over — see below |

## The one rule that is enforced, not asked for

Everything above is the agent recovering from a malformed reply. This one is
different: it overrides a decision the model made cleanly.

`04_feature_request_handling.md` and `05_escalation_rules.md` both state that a
feature request is never escalated, and the decide prompt says so too — yet
`llama3.2:1b` escalated a request for dark mode anyway, and called
`check_account_status` on it for good measure. Quoting a policy to customers
while acting against it is not a defensible default, so
`enforce_escalation_policy` corrects it.

Three separate paths can end in an escalation — the model choosing it, an
unparseable reply, and the step limit running out — and the rule is applied to
**all three**. The first version only covered the tidy path, which meant a
feature request whose reply failed to parse was still escalated.

The correction is never silent: the step keeps the model's own thought, its
`override` field says what changed and why, and it is logged. Measured across
six live runs afterwards: zero escalated feature requests, with the policy
visibly stepping in once.

## The audit trail

`AgentResult.trace` records every step — thought, action, action input and
observation — and it is returned in the API response, written into the
`run_agent` stage output, and logged. A triage decision that cannot be audited
is not worth much in support operations, and it is also the fastest way to debug
a surprising answer.

## The tools are deterministic fakes

`check_account_status` and `check_refund_eligibility` derive their answers from
the identifier by summing its character codes, so the same order id always gives
the same result and no test needs a network.

`hash()` is deliberately **not** used: Python randomises it between processes,
which would make the fake answers change from run to run.

The refund tool mirrors the refund policy document — inside 30 days, or a
duplicate charge at any age — so the agent's reasoning stays consistent with the
knowledge base it quotes.

## Identifier extraction

Support tickets rarely carry structured fields, so `extract_identifiers` looks
for the first run of digits following "order", "invoice", "payment" or "charge"
(an order id), or "account", "customer", "workspace" or "user" (a customer id).
When nothing is found the tools still run and report on an unknown identifier,
which is exactly what a real agent would surface to a human.

```python
extract_identifiers("Duplicate payment on order 4471", "My account 9912 was charged")
# returns order_id 4471 and customer_id 9912
```

## Tests

`tests/test_agent.py` (33 tests) drives the loop with a fake model, so each test
can force a specific decision sequence. It covers every row of the failure table
above, plus the tools and the JSON extraction from prose.

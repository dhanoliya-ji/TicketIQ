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

`tests/test_agent.py` (24 tests) drives the loop with a fake model, so each test
can force a specific decision sequence. It covers every row of the failure table
above, plus the tools and the JSON extraction from prose.

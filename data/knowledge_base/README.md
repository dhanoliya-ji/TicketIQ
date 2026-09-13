# `data/knowledge_base/` — the retrieval corpus

Five short markdown documents holding the policy and troubleshooting knowledge
the agent is allowed to quote. Together they produce **28 retrievable chunks**.

| File | Covers | Sections |
|------|--------|----------|
| `01_billing_refund_policy.md` | refund window, duplicate charges, failed payments, proration, cancellations, invoices and tax | 6 |
| `02_account_access_troubleshooting.md` | sign-in failures, reset emails, locked accounts, two-factor, single sign-on, users and ownership | 6 |
| `03_technical_troubleshooting.md` | slow dashboards, API error codes, export timeouts, webhooks, sync failures, mobile crashes | 6 |
| `04_feature_request_handling.md` | what support should do, wording, duplicates, prioritisation, never escalate a feature request | 5 |
| `05_escalation_rules.md` | when to escalate, when not to, response-time targets, what an escalation must contain, tone | 5 |

## How they are chunked

One chunk per `##` heading. The document title (`#`) and the section heading are
both prepended to the chunk's searchable text, because headings like "Refund
window" and "Duplicate charges" are strong keyword signals.

```
01_billing_refund_policy.md
  -> chunk 01_billing_refund_policy#1  heading "Refund window"
  -> chunk 01_billing_refund_policy#2  heading "Duplicate charges"
  ...
```

Heading-based chunking is used rather than a fixed character window because
these documents are written as short, self-contained rules: a section is the
natural unit of retrieval, and a split never cuts a rule in half.

## Writing conventions

These are not decorative — the pipeline depends on them:

1. **One `#` title per file.** It becomes `document_title`.
2. **Every rule lives under a `##` heading.** Text before the first `##` becomes
   an "Introduction" chunk, which is usually not what you want.
3. **Keep sections short and self-contained.** A chunk is pasted into the model
   prompt whole; a long section crowds out the others.
4. **State the rule, not the rationale.** The agent quotes these back to
   customers.
5. **Name the tools where they apply.** `02` and `01` explicitly mention
   `check_account_status` and `check_refund_eligibility`, which is part of what
   steers the agent toward calling them.

## Consistency with the mock tools

`check_refund_eligibility` in `app/agent/tools.py` implements the same rule
`01_billing_refund_policy.md` states — refundable within 30 days, or a duplicate
charge at any age. If you change the policy document, change the tool too, or
the agent will quote one rule while acting on another.

## Adding a document

Drop a new `.md` file in this folder. It is picked up automatically at startup
(`chunk_knowledge_base` globs `*.md` in sorted order, so the numeric prefixes
keep chunk ids stable). No index needs rebuilding by hand and no configuration
needs updating.

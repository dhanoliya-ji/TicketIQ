"""Mock back-office tools the agent may call.

In a real deployment these would call the billing system and the identity
service.  Here they are deterministic fakes: the answer is derived from the
identifier itself, so the same order id always gives the same result and tests
never need a network.  Every tool returns a dictionary with a ``summary`` key,
which is the single line fed back into the reasoning loop as an observation.
"""

# Tool names, used in prompts, logs and the API response.
CHECK_ACCOUNT_STATUS = "check_account_status"
CHECK_REFUND_ELIGIBILITY = "check_refund_eligibility"

# Non-tool actions the agent may finish with.
ANSWER = "answer"
ESCALATE_TO_HUMAN = "escalate_to_human"

ALL_TOOL_NAMES = [CHECK_ACCOUNT_STATUS, CHECK_REFUND_ELIGIBILITY]
ALL_ACTION_NAMES = ALL_TOOL_NAMES + [ANSWER, ESCALATE_TO_HUMAN]


class ToolResult:
    """The outcome of one tool call."""

    def __init__(self, tool: str, arguments: dict, output: dict, summary: str) -> None:
        self.tool = tool
        self.arguments = arguments
        self.output = output
        self.summary = summary

    def as_dict(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "output": self.output,
            "summary": self.summary,
        }


def _stable_number(identifier: str) -> int:
    """Turn any identifier into a stable number between 0 and 99.

    ``hash()`` is not used because Python randomises it between processes,
    which would make the fake answers change from run to run.  Summing the
    character codes is crude but perfectly stable.
    """
    total = 0
    for character in str(identifier):
        total = total + ord(character)
    return total % 100


def check_account_status(arguments: dict) -> ToolResult:
    """Look up whether an account is active, locked or suspended."""
    customer_id = str(arguments.get("customer_id", "unknown"))
    number = _stable_number(customer_id)

    if number < 60:
        status = "active"
    elif number < 85:
        status = "locked"
    else:
        status = "suspended"

    if status == "active":
        plan = "pro"
    else:
        plan = "free"

    output = {
        "customer_id": customer_id,
        "status": status,
        "plan": plan,
        "failed_login_attempts": number % 12,
        "two_factor_enabled": number % 2 == 0,
    }
    summary = (
        "account "
        + customer_id
        + " is "
        + status
        + " on the "
        + plan
        + " plan with "
        + str(output["failed_login_attempts"])
        + " recent failed sign in attempts"
    )
    return ToolResult(CHECK_ACCOUNT_STATUS, arguments, output, summary)


def check_refund_eligibility(arguments: dict) -> ToolResult:
    """Check whether an order can still be refunded under the 30 day policy."""
    order_id = str(arguments.get("order_id", "unknown"))
    number = _stable_number(order_id)

    days_since_charge = number % 45
    amount = round(19.0 + (number * 7.5), 2)
    is_duplicate = number % 4 == 0

    # Mirrors the refund policy document: inside 30 days, or a duplicate
    # charge at any age.
    eligible = days_since_charge <= 30 or is_duplicate

    if eligible:
        if is_duplicate:
            reason = "duplicate charge, refundable regardless of age"
        else:
            reason = "within the 30 day refund window"
    else:
        reason = "charge is " + str(days_since_charge) + " days old, outside the 30 day window"

    output = {
        "order_id": order_id,
        "eligible": eligible,
        "amount": amount,
        "days_since_charge": days_since_charge,
        "duplicate_charge": is_duplicate,
        "reason": reason,
    }
    summary = (
        "order "
        + order_id
        + " is "
        + ("eligible" if eligible else "not eligible")
        + " for a refund of "
        + str(amount)
        + " ("
        + reason
        + ")"
    )
    return ToolResult(CHECK_REFUND_ELIGIBILITY, arguments, output, summary)


# Name -> function, so the agent can look a tool up by the string the model
# produced.
TOOL_REGISTRY = {
    CHECK_ACCOUNT_STATUS: check_account_status,
    CHECK_REFUND_ELIGIBILITY: check_refund_eligibility,
}


def run_tool(name: str, arguments: dict) -> ToolResult:
    """Execute a tool by name, returning a clear error result if it is unknown."""
    if name not in TOOL_REGISTRY:
        return ToolResult(
            tool=name,
            arguments=arguments,
            output={"error": "unknown tool"},
            summary="tool " + name + " does not exist",
        )
    return TOOL_REGISTRY[name](arguments)

"""Simulation showing the bandit shifting toward the better configurations.

Run from the repository root::

    python scripts/simulate_bandit.py
    python scripts/simulate_bandit.py 2000        # more tickets

What is simulated and what is real
----------------------------------
The **bandit is the real one** (``EpsilonGreedyContextualBandit``) and the
**reward is the real one** (``compute_reward``).  What is simulated is the
outside world: a stream of tickets, how long each configuration takes, and
whether the customer found the answer helpful.  Running the real pipeline
thousands of times would measure the template writer, not the learning rule,
and would need thousands of humans to click a thumb.

The simulated world (deliberately not told to the bandit)
---------------------------------------------------------
* The **empathetic step-by-step** variant writes more text, so it is slower,
  but customers on high urgency tickets like it much more.
* **Top-K = 5** retrieves more context, so it is slower, but it answers
  awkward questions more often - which only matters for high urgency tickets.
* On low urgency tickets the extra time is pure cost, so the fast, concise,
  K=2 configuration wins there.

So there is no globally best arm: the right answer depends on the state, which
is exactly what a *contextual* bandit is for.  The report at the end shows the
action distribution in the first fifth of the run against the last fifth.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm.configs import all_config_names  # noqa: E402
from app.rl.bandit import EpsilonGreedyContextualBandit  # noqa: E402
from app.rl.state import build_state_key, compute_reward  # noqa: E402

DEFAULT_TICKET_COUNT = 5000
RANDOM_SEED = 11
EPSILON = 0.15

CATEGORIES = ["billing", "technical", "account", "feature_request"]
URGENCY_BUCKETS = ["low", "medium", "high"]
TIERS = ["free", "pro", "enterprise"]

# --- the hidden truth of the simulated world -------------------------------
# Base latency in seconds for each configuration.
BASE_LATENCY = {
    "concise_policy|k2": 1.0,
    "concise_policy|k5": 1.6,
    "empathetic_stepwise|k2": 2.2,
    "empathetic_stepwise|k5": 2.9,
}

# Probability that the customer marks the answer helpful, by urgency bucket.
HELPFULNESS = {
    "low": {
        "concise_policy|k2": 0.90,
        "concise_policy|k5": 0.80,
        "empathetic_stepwise|k2": 0.55,
        "empathetic_stepwise|k5": 0.45,
    },
    "medium": {
        "concise_policy|k2": 0.60,
        "concise_policy|k5": 0.85,
        "empathetic_stepwise|k2": 0.70,
        "empathetic_stepwise|k5": 0.80,
    },
    "high": {
        "concise_policy|k2": 0.25,
        "concise_policy|k5": 0.45,
        "empathetic_stepwise|k2": 0.70,
        "empathetic_stepwise|k5": 0.95,
    },
}

# Enterprise customers are a little harder to please on everything.
TIER_HELPFULNESS_ADJUSTMENT = {"free": 0.04, "pro": 0.0, "enterprise": -0.04}


def simulate_latency(config_name: str, generator: random.Random) -> float:
    """Latency with a little noise, never negative."""
    latency = BASE_LATENCY[config_name] + generator.gauss(0.0, 0.15)
    if latency < 0.2:
        return 0.2
    return latency


def simulate_feedback(config_name: str, urgency: str, tier: str, generator: random.Random) -> int:
    """Return 1 (helpful) or 0 (unhelpful) from the hidden probabilities."""
    probability = HELPFULNESS[urgency][config_name] + TIER_HELPFULNESS_ADJUSTMENT[tier]
    if generator.random() < probability:
        return 1
    return 0


def expected_reward(config_name: str, urgency: str, tier: str) -> float:
    """The reward this arm earns on average - used only to grade the bandit."""
    probability = HELPFULNESS[urgency][config_name] + TIER_HELPFULNESS_ADJUSTMENT[tier]
    return (probability * 10.0) - BASE_LATENCY[config_name]


def best_possible_action(urgency: str, tier: str) -> str:
    """The arm an oracle would always pick for this context."""
    actions = all_config_names()
    best = actions[0]
    for action in actions:
        if expected_reward(action, urgency, tier) > expected_reward(best, urgency, tier):
            best = action
    return best


def main() -> None:
    ticket_count = DEFAULT_TICKET_COUNT
    if len(sys.argv) > 1:
        ticket_count = int(sys.argv[1])

    generator = random.Random(RANDOM_SEED)
    actions = all_config_names()
    bandit = EpsilonGreedyContextualBandit(actions, epsilon=EPSILON, seed=RANDOM_SEED)

    window_size = max(1, ticket_count // 5)
    first_window_counts: dict[str, int] = {}
    last_window_counts: dict[str, int] = {}
    for action in actions:
        first_window_counts[action] = 0
        last_window_counts[action] = 0

    total_reward = 0.0
    optimal_choices_first = 0
    optimal_choices_last = 0
    rewards_by_block: list[float] = []
    block_reward_total = 0.0
    block_size = max(1, ticket_count // 10)

    for ticket_number in range(ticket_count):
        # A random ticket arrives.
        category = generator.choice(CATEGORIES)
        urgency = generator.choice(URGENCY_BUCKETS)
        tier = generator.choice(TIERS)
        state_key = build_state_key(category, urgency, tier)

        # The bandit chooses, the world answers.
        action, _reason = bandit.select_action(state_key)
        latency = simulate_latency(action, generator)
        feedback = simulate_feedback(action, urgency, tier, generator)
        reward = compute_reward(feedback, latency)
        bandit.update(state_key, action, reward)

        total_reward = total_reward + reward
        block_reward_total = block_reward_total + reward
        if (ticket_number + 1) % block_size == 0:
            rewards_by_block.append(block_reward_total / block_size)
            block_reward_total = 0.0

        oracle_action = best_possible_action(urgency, tier)
        if ticket_number < window_size:
            first_window_counts[action] = first_window_counts[action] + 1
            if action == oracle_action:
                optimal_choices_first = optimal_choices_first + 1
        if ticket_number >= ticket_count - window_size:
            last_window_counts[action] = last_window_counts[action] + 1
            if action == oracle_action:
                optimal_choices_last = optimal_choices_last + 1

    _print_report(
        ticket_count,
        window_size,
        actions,
        first_window_counts,
        last_window_counts,
        optimal_choices_first,
        optimal_choices_last,
        total_reward,
        rewards_by_block,
        bandit,
    )


def _print_report(
    ticket_count: int,
    window_size: int,
    actions: list[str],
    first_window_counts: dict[str, int],
    last_window_counts: dict[str, int],
    optimal_choices_first: int,
    optimal_choices_last: int,
    total_reward: float,
    rewards_by_block: list[float],
    bandit: EpsilonGreedyContextualBandit,
) -> None:
    print("")
    print("Contextual bandit simulation")
    print("============================")
    print("tickets simulated : " + str(ticket_count))
    print("epsilon           : " + str(EPSILON))
    print("actions           : " + ", ".join(actions))
    print("comparison window : first and last " + str(window_size) + " tickets")

    print("")
    print("Action distribution")
    print("-------------------")
    print("configuration".ljust(26) + "first window   last window")
    for action in actions:
        first_share = 100.0 * first_window_counts[action] / window_size
        last_share = 100.0 * last_window_counts[action] / window_size
        print(
            action.ljust(26)
            + (format(first_share, ".1f") + "%").ljust(15)
            + format(last_share, ".1f")
            + "%"
        )

    print("")
    print("Share of choices that matched the oracle")
    print("----------------------------------------")
    print("first window: " + format(100.0 * optimal_choices_first / window_size, ".1f") + "%")
    print("last window : " + format(100.0 * optimal_choices_last / window_size, ".1f") + "%")

    print("")
    print("Average reward per ticket, by block of " + str(len(rewards_by_block)) + " blocks")
    print("-------------------------------------------------")
    for index in range(len(rewards_by_block)):
        value = rewards_by_block[index]
        bar = "#" * int(max(0.0, value) * 4)
        print("block " + str(index + 1).rjust(2) + ": " + format(value, "6.2f") + "  " + bar)

    print("")
    print("overall average reward: " + format(total_reward / ticket_count, ".3f"))
    print("states learned        : " + str(len(bandit.statistics)))
    print(
        "explore / exploit     : "
        + str(bandit.exploration_count)
        + " / "
        + str(bandit.exploitation_count)
    )

    print("")
    print("What the bandit learned for a few states")
    print("----------------------------------------")
    example_states = [
        ("technical", "high", "enterprise"),
        ("feature_request", "low", "free"),
        ("billing", "medium", "pro"),
    ]
    for category, urgency, tier in example_states:
        state_key = build_state_key(category, urgency, tier)
        learned = bandit.best_action(state_key)
        oracle = best_possible_action(urgency, tier)
        verdict = "match" if learned == oracle else "MISMATCH"
        print(
            state_key.ljust(36)
            + "learned="
            + learned.ljust(26)
            + "oracle="
            + oracle.ljust(26)
            + verdict
        )
    print("")


if __name__ == "__main__":
    main()

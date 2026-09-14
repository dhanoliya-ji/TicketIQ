"""Unit tests for the contextual bandit, focused on the update rule.

The update rule is the heart of the learning loop, so it is checked against
hand-computed averages rather than only "the number moved".
"""

import random

import pytest

from app.rl.bandit import ArmStatistics, EpsilonGreedyContextualBandit
from app.rl.state import build_state_key, compute_reward

ACTIONS = ["config_a", "config_b", "config_c"]
STATE = "billing|high|pro"


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def test_reward_is_feedback_times_ten_minus_latency():
    assert compute_reward(1, 2.5) == pytest.approx(7.5)
    assert compute_reward(0, 2.5) == pytest.approx(-2.5)
    assert compute_reward(1, 0.0) == pytest.approx(10.0)


def test_unhelpful_answers_always_score_below_helpful_ones():
    # Even a very slow helpful answer beats a fast unhelpful one, as long as
    # the answer took less than ten seconds.
    assert compute_reward(1, 9.0) > compute_reward(0, 0.1)


def test_state_key_joins_the_three_context_features():
    assert build_state_key("billing", "high", "pro") == "billing|high|pro"


# ---------------------------------------------------------------------------
# The incremental average
# ---------------------------------------------------------------------------


def test_arm_statistics_average_matches_a_plain_average():
    arm = ArmStatistics()
    rewards = [10.0, 4.0, 7.0, 1.0]
    for reward in rewards:
        arm.update(reward)

    assert arm.pulls == 4
    assert arm.average_reward == pytest.approx(sum(rewards) / len(rewards))


def test_first_update_sets_the_average_to_that_reward():
    arm = ArmStatistics()
    arm.update(6.5)
    assert arm.average_reward == pytest.approx(6.5)


def test_bandit_update_changes_only_the_chosen_arm_in_the_chosen_state():
    bandit = EpsilonGreedyContextualBandit(ACTIONS, epsilon=0.0, seed=1)
    bandit.update(STATE, "config_a", 8.0)

    assert bandit.statistics[STATE]["config_a"].average_reward == pytest.approx(8.0)
    assert bandit.statistics[STATE]["config_b"].pulls == 0
    # A different state must be untouched.
    assert "technical|low|free" not in bandit.statistics


def test_updating_an_unknown_action_raises():
    bandit = EpsilonGreedyContextualBandit(ACTIONS, epsilon=0.0, seed=1)
    with pytest.raises(KeyError):
        bandit.update(STATE, "config_does_not_exist", 1.0)


def test_bandit_rejects_bad_construction():
    with pytest.raises(ValueError):
        EpsilonGreedyContextualBandit([], epsilon=0.1)
    with pytest.raises(ValueError):
        EpsilonGreedyContextualBandit(ACTIONS, epsilon=1.5)


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------


def test_every_arm_is_tried_once_before_any_is_exploited():
    bandit = EpsilonGreedyContextualBandit(ACTIONS, epsilon=0.0, seed=3)

    chosen = set()
    for _ in range(len(ACTIONS)):
        action, reason = bandit.select_action(STATE)
        assert reason == "cold_start"
        chosen.add(action)
        bandit.update(STATE, action, 1.0)

    assert chosen == set(ACTIONS)


def test_with_epsilon_zero_the_best_arm_is_always_chosen():
    bandit = EpsilonGreedyContextualBandit(ACTIONS, epsilon=0.0, seed=3)
    bandit.update(STATE, "config_a", 1.0)
    bandit.update(STATE, "config_b", 9.0)
    bandit.update(STATE, "config_c", 5.0)

    for _ in range(20):
        action, reason = bandit.select_action(STATE)
        assert action == "config_b"
        assert reason == "exploit"


def test_with_epsilon_one_the_choice_is_always_exploration():
    bandit = EpsilonGreedyContextualBandit(ACTIONS, epsilon=1.0, seed=3)
    for action in ACTIONS:
        bandit.update(STATE, action, 1.0)

    for _ in range(10):
        _action, reason = bandit.select_action(STATE)
        assert reason == "explore"


def test_learning_in_one_state_does_not_leak_into_another():
    bandit = EpsilonGreedyContextualBandit(ACTIONS, epsilon=0.0, seed=3)
    first_state = "billing|low|free"
    second_state = "technical|high|enterprise"

    for action in ACTIONS:
        bandit.update(first_state, action, 1.0)
        bandit.update(second_state, action, 1.0)

    bandit.update(first_state, "config_a", 20.0)
    bandit.update(second_state, "config_c", 20.0)

    assert bandit.best_action(first_state) == "config_a"
    assert bandit.best_action(second_state) == "config_c"


def test_bandit_converges_on_the_best_arm():
    """The end-to-end learning property: the action distribution must shift."""
    bandit = EpsilonGreedyContextualBandit(ACTIONS, epsilon=0.1, seed=7)
    noise = random.Random(7)
    true_average = {"config_a": 2.0, "config_b": 8.0, "config_c": 4.0}

    early_best_count = 0
    late_best_count = 0

    for round_number in range(600):
        action, _reason = bandit.select_action(STATE)
        bandit.update(STATE, action, true_average[action] + noise.gauss(0.0, 1.0))

        if round_number < 100 and action == "config_b":
            early_best_count = early_best_count + 1
        if round_number >= 500 and action == "config_b":
            late_best_count = late_best_count + 1

    assert bandit.best_action(STATE) == "config_b"
    # Late on, the best arm should dominate far more than at the start.
    assert late_best_count > early_best_count
    assert late_best_count > 80  # out of 100, allowing for epsilon exploration


def test_an_untried_arm_is_never_reported_as_the_best_one():
    """Regression: negative rewards used to lose to an unmeasured arm.

    The reward is feedback * 10 - latency, so any answer slower than ten
    seconds scores below zero even when the customer said it helped - the
    normal case against a local language model. An untried arm still carries
    its initial 0.0, which beat the only arm that had actually been measured.
    """
    bandit = EpsilonGreedyContextualBandit(ACTIONS, epsilon=0.0, seed=1)
    bandit.update(STATE, "config_b", -3.03)  # helpful answer, 13.0s latency

    assert bandit.best_action(STATE) == "config_b"


def test_the_best_arm_is_the_best_among_those_measured():
    bandit = EpsilonGreedyContextualBandit(ACTIONS, epsilon=0.0, seed=1)
    bandit.update(STATE, "config_a", -8.0)
    bandit.update(STATE, "config_c", -2.0)
    # config_b is still untried and still sits at 0.0.

    assert bandit.best_action(STATE) == "config_c"


def test_best_action_is_stable_when_nothing_has_been_tried():
    bandit = EpsilonGreedyContextualBandit(ACTIONS, epsilon=0.0, seed=1)

    # No evidence at all: an arbitrary but stable answer, not a crash.
    assert bandit.best_action("a|state|nobody|has|seen") == ACTIONS[0]


def test_exploitation_also_ignores_untried_arms():
    """select_action and best_action must not disagree."""
    bandit = EpsilonGreedyContextualBandit(ACTIONS, epsilon=0.0, seed=1)
    for action in ACTIONS:
        bandit.update(STATE, action, -5.0)
    bandit.update(STATE, "config_c", -1.0)

    action, reason = bandit.select_action(STATE)
    assert reason == "exploit"
    assert action == bandit.best_action(STATE)


def test_total_pulls_counts_every_update():
    bandit = EpsilonGreedyContextualBandit(ACTIONS, epsilon=0.0, seed=1)
    bandit.update("state_one", "config_a", 1.0)
    bandit.update("state_one", "config_a", 1.0)
    bandit.update("state_two", "config_b", 1.0)

    assert bandit.total_pulls() == 3


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_saved_statistics_survive_a_reload(tmp_path):
    path = tmp_path / "bandit.json"

    original = EpsilonGreedyContextualBandit(ACTIONS, epsilon=0.2, seed=1)
    original.update(STATE, "config_b", 9.0)
    original.update(STATE, "config_b", 7.0)
    original.save(path)

    restored = EpsilonGreedyContextualBandit(ACTIONS, epsilon=0.2, seed=1)
    assert restored.load(path) is True
    assert restored.statistics[STATE]["config_b"].pulls == 2
    assert restored.statistics[STATE]["config_b"].average_reward == pytest.approx(8.0)


def test_loading_a_missing_file_returns_false(tmp_path):
    bandit = EpsilonGreedyContextualBandit(ACTIONS, epsilon=0.2, seed=1)
    assert bandit.load(tmp_path / "not_here.json") is False


def test_loading_a_corrupt_file_does_not_crash(tmp_path):
    path = tmp_path / "bandit.json"
    path.write_text("this is not json", encoding="utf-8")

    bandit = EpsilonGreedyContextualBandit(ACTIONS, epsilon=0.2, seed=1)
    assert bandit.load(path) is False
    assert bandit.total_pulls() == 0


def test_snapshot_lists_every_action_for_every_seen_state():
    bandit = EpsilonGreedyContextualBandit(ACTIONS, epsilon=0.2, seed=1)
    bandit.update(STATE, "config_a", 3.0)

    snapshot = bandit.snapshot()
    assert snapshot["epsilon"] == 0.2
    assert set(snapshot["states"][STATE].keys()) == set(ACTIONS)
    assert snapshot["states"][STATE]["config_a"]["pulls"] == 1

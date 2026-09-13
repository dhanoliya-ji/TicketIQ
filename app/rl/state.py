"""The bandit state (the "context") and the reward function.

State
-----
``category | urgency bucket | customer tier`` = 4 x 3 x 3 = 36 possible states.
All three are already discrete, and they are exactly the three things that
should change which configuration we route to:

* an enterprise customer deserves the careful, step-by-step answer;
* a high urgency ticket justifies retrieving more context;
* a feature request needs neither.

Everything continuous (the raw urgency number, the sentiment scores) is
bucketed before it reaches the bandit, because a bandit learns per state and
continuous states would never repeat often enough to learn anything.

Reward
------
The assignment defines it exactly:

    reward = (user feedback score x 10) - (latency in seconds)

So a helpful answer is worth 10 points and every second of waiting costs one
point.  An unhelpful answer scores a negative reward equal to its latency,
which is what makes slow *and* unhelpful configurations get abandoned first.
"""

# How much one unit of positive feedback is worth.
FEEDBACK_WEIGHT = 10.0


def build_state_key(category: str, urgency_bucket: str, customer_tier: str) -> str:
    """Combine the three context features into one dictionary key."""
    return category + "|" + urgency_bucket + "|" + customer_tier


def compute_reward(feedback_score: int, latency_seconds: float) -> float:
    """reward = feedback x 10 - latency."""
    return (feedback_score * FEEDBACK_WEIGHT) - latency_seconds

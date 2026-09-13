"""Turns category + aspect sentiment + customer tier into one urgency number.

The score is a deliberately simple weighted sum of three signals, each
contributing a fixed share of the final 0.0 - 1.0 value:

    urgency = category weight        (up to 0.35)
            + negativity of the worst aspect (up to 0.35)
            + customer tier weight   (up to 0.30)

A transparent formula is worth more here than a learned model: a support lead
can read it, disagree with a weight, and change one number.  The bucket
("low" / "medium" / "high") is what the reinforcement learning layer uses as
part of its state, because a continuous score would create far too many
distinct states for a bandit to learn from.
"""

from app.ml.aspect_sentiment import AspectSentiment

# How intrinsically urgent each category is.
CATEGORY_WEIGHT: dict[str, float] = {
    "technical": 0.35,
    "account": 0.30,
    "billing": 0.25,
    "feature_request": 0.05,
}

# How much a paying customer is prioritised.
TIER_WEIGHT: dict[str, float] = {
    "free": 0.05,
    "pro": 0.18,
    "enterprise": 0.30,
}

# The negativity part can contribute at most this much.
SENTIMENT_MAXIMUM = 0.35

# Cut points that turn the number into a bucket.
MEDIUM_THRESHOLD = 0.40
HIGH_THRESHOLD = 0.65


def score_urgency(category: str, aspects: list[AspectSentiment], customer_tier: str) -> float:
    """Return an urgency score between 0.0 and 1.0."""
    category_part = CATEGORY_WEIGHT.get(category, 0.20)
    tier_part = TIER_WEIGHT.get(customer_tier, 0.05)

    # Find the most negative aspect. A score of -1.0 is maximum anger, +1.0 is
    # maximum happiness, so negativity = how far below zero we are.
    worst_score = 0.0
    for aspect in aspects:
        if aspect.score < worst_score:
            worst_score = aspect.score

    negativity = -worst_score  # now between 0.0 and 1.0
    sentiment_part = negativity * SENTIMENT_MAXIMUM

    total = category_part + sentiment_part + tier_part

    # Clamp into range, in case the weights are edited to something larger.
    if total < 0.0:
        return 0.0
    if total > 1.0:
        return 1.0
    return total


def urgency_bucket(score: float) -> str:
    """Turn the continuous score into 'low', 'medium' or 'high'."""
    if score >= HIGH_THRESHOLD:
        return "high"
    if score >= MEDIUM_THRESHOLD:
        return "medium"
    return "low"

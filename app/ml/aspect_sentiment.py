"""Aspect level sentiment: which parts of the product is the customer unhappy about.

A single sentiment score for a whole ticket hides too much.  "Billing is fine
but the dashboard is unusable" is positive about billing and strongly negative
about performance.  So we do two simple steps:

1. Aspect spotting - keyword matching against a small hand written dictionary.
2. Sentiment scoring - VADER scores every *sentence*, and each aspect gets the
   average score of the sentences that mention it.

VADER is a rule based sentiment tool (no model download, no training) which is
a good fit for short, blunt support text.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Aspect name -> the words that signal that aspect.  Keeping this as a plain
# dictionary makes it obvious how to add a new aspect later.
ASPECT_KEYWORDS: dict[str, list[str]] = {
    "billing": [
        "invoice",
        "charge",
        "charged",
        "billing",
        "payment",
        "refund",
        "price",
        "card",
        "subscription",
        "receipt",
        "coupon",
        "proration",
        "renewal",
        "paid",
    ],
    # Also deliberately narrow. An earlier version included the bare words
    # "log" and "sign", which are not login words on their own: "the euro
    # sign shows as a question mark", "please sign the agreement" and "the
    # audit log is empty" all fired the login aspect. The real signal is the
    # word with its particle ("sign in", "log in"), so those live in
    # ASPECT_PHRASES below.
    "login": [
        "login",
        "logins",
        "signin",
        "password",
        "passwords",
        "authentication",
        "authenticate",
        "locked",
        "lockout",
        "credentials",
        "sso",
        "mfa",
        "otp",
        "2fa",
    ],
    "performance": [
        "slow",
        "latency",
        "timeout",
        "timing",
        "freeze",
        "freezes",
        "crash",
        "crashes",
        "hang",
        "performance",
        "degraded",
        "seconds",
        "loading",
    ],
    "reliability": [
        "error",
        "errors",
        "failed",
        "failure",
        "broken",
        "crashed",
        "bug",
        "down",
        "outage",
        "unavailable",
        "500",
    ],
    "support_response_time": [
        "waiting",
        "reply",
        "replied",
        "responded",
        "urgent",
        "asap",
        "escalate",
        "nobody",
        "ignored",
        "unanswered",
    ],
    "account_management": [
        "account",
        "user",
        "users",
        "permission",
        "permissions",
        "role",
        "owner",
        "workspace",
        "teammate",
        "admin",
        "administrator",
        "seat",
        "seats",
    ],
    # Deliberately narrow. An earlier version included "add", "support",
    # "would", "consider" and "ability", which are politeness words rather
    # than feature words: the aspect then fired on 73 of the 160 dataset
    # tickets, 33 of them billing, technical or account tickets, and crowded
    # genuine aspects out of the top three. The real signal for a feature
    # request is usually a phrase, so those live in ASPECT_PHRASES below.
    "feature_availability": [
        "feature",
        "features",
        "roadmap",
        "integration",
        "idea",
        "suggestion",
    ],
}

# Some aspects are signalled by a phrase rather than a single word. Matching
# "would" alone is hopeless; matching "would be great" is not. These are
# checked as substrings of the lower-cased sentence, so word order matters and
# punctuation between the words does not.
ASPECT_PHRASES: dict[str, list[str]] = {
    "feature_availability": [
        "would be great",
        "would be nice",
        "nice to have",
        "feature request",
        "please consider",
        "would love",
        "the ability to",
        "add support for",
    ],
    "login": [
        "sign in",
        "signed in",
        "signing in",
        "sign-in",
        "log in",
        "logged in",
        "logging in",
        "log-in",
        "logged out",
        "log out",
        "sign out",
        "signed out",
        # "logging me out" puts the object between the two words, so the
        # plain "log out" phrase does not match it.
        "logging me out",
        "logged me out",
        "logs me out",
        "signing me out",
        "signed me out",
        "signs me out",
    ],
    "support_response_time": [
        "response time",
        "still waiting",
        "has not replied",
        "no one has replied",
        "nobody has replied",
    ],
}

# ---------------------------------------------------------------------------
# Domain tuning of the VADER lexicon.
#
# VADER was built for social media, so a few words score badly on support
# tickets.  The clearest example: "support" is a positive word in the general
# lexicon ("I support you"), but in our data it is a neutral product noun, so
# "Support has not replied for days" came out *positive*.  We therefore
# neutralise a few product nouns and give our own failure words a negative
# valence.  The scale is the VADER one: roughly -4.0 (awful) .. +4.0 (great).
# ---------------------------------------------------------------------------
DOMAIN_LEXICON: dict[str, float] = {
    # Product nouns that should carry no sentiment of their own.
    "support": 0.0,
    "help": 0.0,
    "please": 0.0,
    "like": 0.0,
    # Failure words that VADER does not know about.
    "timeout": -1.8,
    "timeouts": -1.8,
    "outage": -2.5,
    "downtime": -2.0,
    "unusable": -3.0,
    "unresponsive": -2.2,
    "laggy": -1.8,
    "buggy": -2.0,
    "overcharged": -2.5,
    "duplicate": -1.2,
    "unauthorised": -1.5,
    "unauthorized": -1.5,
    "escalate": -1.0,
    "blocked": -2.0,
    "nobody": -1.8,
    "ignored": -2.0,
    "unanswered": -2.0,
    "repeatedly": -1.0,
}

# Never return more than this many aspects, per the assignment (1-3 aspects).
MAXIMUM_ASPECTS = 3

# A VADER compound score below this counts as negative, above it as positive.
NEGATIVE_THRESHOLD = -0.05
POSITIVE_THRESHOLD = 0.05


class AspectSentiment:
    """Sentiment toward one aspect of the product."""

    def __init__(self, aspect: str, score: float, mentions: int, evidence: str) -> None:
        self.aspect = aspect
        # VADER "compound" score: -1.0 (very negative) .. +1.0 (very positive).
        self.score = score
        # How many sentences mentioned this aspect.
        self.mentions = mentions
        # The sentence that best illustrates the score, for the API response.
        self.evidence = evidence

    @property
    def label(self) -> str:
        """Human readable polarity."""
        if self.score < NEGATIVE_THRESHOLD:
            return "negative"
        if self.score > POSITIVE_THRESHOLD:
            return "positive"
        return "neutral"

    def as_dict(self) -> dict[str, object]:
        return {
            "aspect": self.aspect,
            "score": round(self.score, 4),
            "label": self.label,
            "mentions": self.mentions,
            "evidence": self.evidence,
        }


def split_into_sentences(text: str) -> list[str]:
    """Split text on the usual sentence enders, without a regex."""
    sentences = []
    current_characters: list[str] = []

    for character in text:
        if character in ".!?\n":
            sentence = "".join(current_characters).strip()
            if sentence != "":
                sentences.append(sentence)
            current_characters = []
        else:
            current_characters.append(character)

    # Whatever is left after the last full stop is also a sentence.
    last_sentence = "".join(current_characters).strip()
    if last_sentence != "":
        sentences.append(last_sentence)

    return sentences


def find_aspects_in_sentence(sentence: str) -> list[str]:
    """Return every aspect whose keywords appear in this sentence."""
    lowered_words = set()
    for word in sentence.lower().split():
        # Strip punctuation stuck to the word ("slow," -> "slow").
        cleaned = "".join(character for character in word if character.isalnum())
        if cleaned != "":
            lowered_words.add(cleaned)

    found = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lowered_words:
                found.append(aspect)
                break  # one keyword is enough to say the aspect was mentioned

    # Phrases are matched against the raw sentence, because their meaning comes
    # from the words being adjacent.
    lowered_sentence = sentence.lower()
    for aspect, phrases in ASPECT_PHRASES.items():
        if aspect in found:
            continue
        for phrase in phrases:
            if phrase in lowered_sentence:
                found.append(aspect)
                break

    return found


class AspectSentimentAnalyzer:
    """Extracts aspects from a ticket and scores sentiment for each one."""

    def __init__(self) -> None:
        self.vader = SentimentIntensityAnalyzer()
        # Apply the domain tuning described above on top of the stock lexicon.
        self.vader.lexicon.update(DOMAIN_LEXICON)

    def analyse(self, subject: str, body: str) -> list[AspectSentiment]:
        """Return up to three aspects with their sentiment scores."""
        text = subject + ". " + body
        sentences = split_into_sentences(text)

        # aspect -> list of sentence scores, and the most negative sentence seen.
        scores_by_aspect: dict[str, list[float]] = {}
        evidence_by_aspect: dict[str, str] = {}

        for sentence in sentences:
            aspects = find_aspects_in_sentence(sentence)
            if len(aspects) == 0:
                continue

            sentiment = self.vader.polarity_scores(sentence)
            compound_score = sentiment["compound"]

            for aspect in aspects:
                if aspect not in scores_by_aspect:
                    scores_by_aspect[aspect] = []
                    evidence_by_aspect[aspect] = sentence
                scores_by_aspect[aspect].append(compound_score)

                # Keep the harshest sentence as the evidence, because that is
                # the one a support agent needs to read first.
                previous_worst = self.vader.polarity_scores(evidence_by_aspect[aspect])["compound"]
                if compound_score < previous_worst:
                    evidence_by_aspect[aspect] = sentence

        # No keyword matched at all: fall back to one overall "general" aspect
        # so that downstream stages always have something to work with.
        if len(scores_by_aspect) == 0:
            overall = self.vader.polarity_scores(text)["compound"]
            return [AspectSentiment("general", overall, 1, subject.strip())]

        results = []
        for aspect, aspect_scores in scores_by_aspect.items():
            total = 0.0
            for score in aspect_scores:
                total = total + score
            average = total / len(aspect_scores)
            results.append(
                AspectSentiment(aspect, average, len(aspect_scores), evidence_by_aspect[aspect])
            )

        # Most mentioned first; ties broken by the most negative score, because
        # an angry aspect matters more than a neutral one.
        results.sort(key=lambda item: (-item.mentions, item.score))
        return results[:MAXIMUM_ASPECTS]

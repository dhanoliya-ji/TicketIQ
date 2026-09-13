"""Very small text preparation helpers shared by every NLP component.

We deliberately keep this simple and dependency free: lower-casing, splitting
on non-letters, dropping stop words and dropping very short tokens.  That is
enough signal for short support tickets and it keeps the behaviour easy to
reason about when a prediction looks wrong.
"""

# Words that appear in almost every ticket and therefore carry no information
# about which category the ticket belongs to.
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "which",
    "who",
    "will",
    "with",
    "would",
    "you",
    "your",
}

# Tokens shorter than this are ignored ("a", "ok", "hi", ...).
MINIMUM_TOKEN_LENGTH = 3


def tokenize(text: str) -> list[str]:
    """Turn a free text string into a list of clean lower-case word tokens.

    Example:
        >>> tokenize("I was CHARGED twice for my invoice!")
        ['charged', 'twice', 'invoice']
    """
    lowered = text.lower()

    # Replace every character that is not a letter or a digit with a space,
    # then split on whitespace.  This removes punctuation without needing a
    # regular expression.
    cleaned_characters = []
    for character in lowered:
        if character.isalnum():
            cleaned_characters.append(character)
        else:
            cleaned_characters.append(" ")
    raw_tokens = "".join(cleaned_characters).split()

    tokens = []
    for token in raw_tokens:
        if len(token) < MINIMUM_TOKEN_LENGTH:
            continue
        if token in STOP_WORDS:
            continue
        tokens.append(token)
    return tokens


def join_ticket_text(subject: str, body: str) -> str:
    """Combine the subject and body of a ticket into one string.

    The subject is repeated twice because a support subject line is usually a
    very strong summary of the problem, so we let it count double.
    """
    return subject + " " + subject + " " + body


def count_tokens(tokens: list[str]) -> dict[str, int]:
    """Count how many times each token appears.

    Example:
        >>> count_tokens(["bill", "bill", "late"])
        {'bill': 2, 'late': 1}
    """
    counts: dict[str, int] = {}
    for token in tokens:
        if token in counts:
            counts[token] = counts[token] + 1
        else:
            counts[token] = 1
    return counts

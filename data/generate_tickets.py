"""Builds the synthetic labelled ticket dataset used to train the classifier.

Run it from the repository root:

    python data/generate_tickets.py

It writes ``data/tickets.json``.  The output is committed to the repository so
that training does not depend on running this script first, but regenerating
it is deterministic: the random seed is fixed, so the same file comes out
every time.

Why synthetic data?  The assignment asks for 100-200 labelled tickets and real
support data cannot be shared.  Each category gets its own vocabulary of
subjects and bodies, mixed with shared "filler" sentences so that the
categories overlap a little and the classification task is not trivial.
"""

import json
import random
from pathlib import Path

# Where the finished dataset is written.
OUTPUT_FILE = Path(__file__).resolve().parent / "tickets.json"

# Fixed seed keeps the dataset reproducible.
RANDOM_SEED = 7

# How many tickets to build per category (4 categories x 40 = 160 tickets).
TICKETS_PER_CATEGORY = 40

# Chance that a ticket also mentions a problem from a *different* category.
# Real tickets are messy like this ("I cannot log in AND I was charged twice"),
# and without this noise the four categories separate perfectly, which would
# make the accuracy report meaningless.  The label stays the category of the
# subject line, which is what a human triager would also go by.
CROSS_CATEGORY_NOISE_PROBABILITY = 0.45

CUSTOMER_TIERS = ["free", "pro", "enterprise"]

# Sentences that could appear in any ticket.  They add noise so that the
# classifier has to rely on the category specific words.
SHARED_FILLERS = [
    "Please let me know as soon as possible.",
    "I have already tried the usual steps from your help centre.",
    "Our team is blocked on this right now.",
    "Thanks in advance for the help.",
    "I am happy to share more details if you need them.",
    "This started happening earlier this week.",
]

# ---------------------------------------------------------------------------
# Per category building blocks.
# ---------------------------------------------------------------------------
TEMPLATES: dict[str, dict[str, list[str]]] = {
    "billing": {
        "subjects": [
            "Charged twice on my invoice",
            "Refund request for last month",
            "Wrong amount on my credit card",
            "Invoice does not match my plan",
            "Cancel subscription and refund",
            "Payment failed but money was taken",
            "Unexpected renewal charge",
            "VAT missing from the receipt",
            "Question about proration on my bill",
            "Duplicate payment on order 4471",
        ],
        "bodies": [
            "My credit card was charged twice for the same monthly invoice.",
            "I would like a refund for the payment taken on the third of the month.",
            "The billing page shows a different price than the plan I subscribed to.",
            "We were invoiced for ten seats but our account only uses six seats.",
            "The renewal charge appeared even though I cancelled the subscription.",
            "The payment failed in the checkout but the amount left my bank account.",
            "Please send a corrected invoice with the tax details for our finance team.",
            "I need a receipt for the annual payment for our accounting records.",
            "The discount coupon was not applied to the final price at checkout.",
            "Can you explain the proration amount added to this billing cycle?",
        ],
    },
    "technical": {
        "subjects": [
            "Dashboard is extremely slow",
            "API returns 500 errors",
            "Export keeps timing out",
            "Webhook events are not delivered",
            "Charts fail to load",
            "Sync job crashed overnight",
            "Latency spike on the reports page",
            "Integration stopped working after update",
            "Search returns no results",
            "Mobile app crashes on launch",
        ],
        "bodies": [
            "The reporting dashboard takes over thirty seconds to load any chart.",
            "Every call to the public API responds with a 500 internal server error.",
            "The CSV export times out before it finishes for large date ranges.",
            "Webhook deliveries stopped arriving at our endpoint since yesterday.",
            "Performance has degraded badly and the page freezes while loading data.",
            "The nightly synchronisation job failed with a connection timeout error.",
            "Our integration broke right after the latest release was deployed.",
            "The application throws an error whenever we upload a large file.",
            "Response times are very slow and requests occasionally return a gateway error.",
            "The mobile application crashes immediately after the splash screen.",
        ],
    },
    "account": {
        "subjects": [
            "Cannot log in to my account",
            "Password reset email never arrives",
            "Locked out after too many attempts",
            "Two factor authentication not working",
            "Need to add a teammate",
            "Change the account owner",
            "Single sign on login fails",
            "Remove a deactivated user",
            "Email address on the account is wrong",
            "Access permissions are missing",
        ],
        "bodies": [
            "I cannot log in, the login page says my password is invalid.",
            "The password reset email never arrives in my inbox or my spam folder.",
            "My account is locked after several failed sign in attempts.",
            "The two factor authentication code is rejected every time I enter it.",
            "Please add my colleague as a user with administrator permissions.",
            "We need to transfer ownership of the workspace to a different person.",
            "Single sign on redirects back to the login screen without signing me in.",
            "A former employee still has access and their user should be removed.",
            "The email address registered on the account has a typo and needs updating.",
            "My role no longer has permission to open the settings of our workspace.",
        ],
    },
    "feature_request": {
        "subjects": [
            "Please add dark mode",
            "Request: bulk edit support",
            "Suggestion for a Slack integration",
            "Would love an audit log",
            "Feature idea: scheduled reports",
            "Can you support custom fields",
            "Roadmap question about the mobile app",
            "Request for a public API endpoint",
            "Idea: saved filters on the list view",
            "Please consider multi language support",
        ],
        "bodies": [
            "It would be great if the product supported a dark theme for night work.",
            "Could you add the ability to edit many records at once in bulk?",
            "We would really like an integration that posts notifications into Slack.",
            "An audit log showing who changed what would help our compliance review.",
            "Please consider adding scheduled reports delivered by email every week.",
            "Custom fields on the record page would let us model our own process.",
            "Is a native mobile application on the roadmap for the next quarters?",
            "We would like a public endpoint so we can pull the data ourselves.",
            "Saved filters on the list view would save our team a lot of clicks.",
            "Support for additional languages in the interface would help our region.",
        ],
    },
}


def build_dataset() -> list[dict[str, str]]:
    """Create the full list of labelled ticket dictionaries."""
    random_generator = random.Random(RANDOM_SEED)
    tickets: list[dict[str, str]] = []
    ticket_number = 1

    for category in sorted(TEMPLATES.keys()):
        subjects = TEMPLATES[category]["subjects"]
        bodies = TEMPLATES[category]["bodies"]

        for _ in range(TICKETS_PER_CATEGORY):
            subject = random_generator.choice(subjects)

            # Most tickets get one category sentence, some get two, so the
            # ticket lengths vary the way real tickets do.
            chosen_bodies = [random_generator.choice(bodies)]
            if random_generator.random() < 0.5:
                chosen_bodies.append(random_generator.choice(bodies))

            # Some tickets also mention an unrelated problem from another
            # category, which is what makes the dataset non-trivial.
            if random_generator.random() < CROSS_CATEGORY_NOISE_PROBABILITY:
                other_categories = []
                for name in sorted(TEMPLATES.keys()):
                    if name != category:
                        other_categories.append(name)
                noise_category = random_generator.choice(other_categories)
                noise_bodies = TEMPLATES[noise_category]["bodies"]

                # One or two sentences about the unrelated problem.  Two makes
                # the ticket genuinely ambiguous, which is where a classifier
                # is allowed to be wrong.
                number_of_noise_sentences = random_generator.choice([1, 2])
                for _ in range(number_of_noise_sentences):
                    chosen_bodies.append(random_generator.choice(noise_bodies))

            # Half of the tickets also get a neutral filler sentence.
            if random_generator.random() < 0.5:
                chosen_bodies.append(random_generator.choice(SHARED_FILLERS))

            body = " ".join(chosen_bodies)

            tickets.append(
                {
                    "id": "T-" + str(ticket_number).zfill(4),
                    "subject": subject,
                    "body": body,
                    "category": category,
                    "customer_tier": random_generator.choice(CUSTOMER_TIERS),
                }
            )
            ticket_number = ticket_number + 1

    # Shuffle so that the categories are not grouped together in the file.
    random_generator.shuffle(tickets)
    return tickets


def main() -> None:
    tickets = build_dataset()
    OUTPUT_FILE.write_text(json.dumps(tickets, indent=2) + "\n", encoding="utf-8")
    print("Wrote " + str(len(tickets)) + " tickets to " + str(OUTPUT_FILE))


if __name__ == "__main__":
    main()

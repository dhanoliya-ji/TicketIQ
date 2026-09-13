"""The pipeline configurations that the reinforcement learning layer chooses between.

Each configuration ("arm" in bandit language) is a combination of:

* a **prompt variant** - a distinct system instruction *and*, when a real
  Ollama server is available, a distinct model.  The two variants really do
  behave differently: one quotes policy tersely, the other explains step by
  step and is noticeably slower because it writes more tokens.
* a **RAG top-K** - how many knowledge base chunks are put in the context.
  K=2 is fast and focused, K=5 is slower but covers edge cases.

Two variants x two K values = four arms.  That is a small enough action space
for a bandit to learn quickly from real feedback, and large enough that the
choice genuinely matters: the best arm for a simple free-tier feature request
is not the best arm for an angry enterprise outage.
"""

from app.settings import SETTINGS

# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------
CONCISE_POLICY = "concise_policy"
EMPATHETIC_STEPWISE = "empathetic_stepwise"

PROMPT_VARIANTS: dict[str, str] = {
    CONCISE_POLICY: (
        "You are TicketIQ, a support triage assistant for a B2B SaaS company. "
        "Answer in at most four sentences. Quote the exact policy rule you are "
        "relying on. Do not apologise, do not add pleasantries, do not invent "
        "any rule that is not in the provided context."
    ),
    EMPATHETIC_STEPWISE: (
        "You are TicketIQ, a support triage assistant for a B2B SaaS company. "
        "Open by acknowledging the impact on the customer in one sentence, then "
        "give clearly numbered next steps, then say what happens next and who "
        "owns it. Stay warm and concrete. Never invent a rule that is not in "
        "the provided context."
    ),
}


class PipelineConfig:
    """One complete choice of how to answer a ticket."""

    def __init__(self, prompt_variant: str, rag_top_k: int) -> None:
        self.prompt_variant = prompt_variant
        self.rag_top_k = rag_top_k

    @property
    def name(self) -> str:
        """Stable identifier used as the bandit action key and in the API response."""
        return self.prompt_variant + "|k" + str(self.rag_top_k)

    @property
    def system_prompt(self) -> str:
        return PROMPT_VARIANTS[self.prompt_variant]

    @property
    def model_name(self) -> str:
        """Which Ollama model this variant uses when a server is available."""
        if self.prompt_variant == CONCISE_POLICY:
            return SETTINGS.ollama_model_a
        return SETTINGS.ollama_model_b

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "prompt_variant": self.prompt_variant,
            "rag_top_k": self.rag_top_k,
            "model": self.model_name,
        }


# The full action space, in a fixed order so that reports are comparable.
ALL_CONFIGS: list[PipelineConfig] = [
    PipelineConfig(CONCISE_POLICY, 2),
    PipelineConfig(CONCISE_POLICY, 5),
    PipelineConfig(EMPATHETIC_STEPWISE, 2),
    PipelineConfig(EMPATHETIC_STEPWISE, 5),
]

# Fast lookup from the action name back to the configuration object.
CONFIGS_BY_NAME: dict[str, PipelineConfig] = {}
for _config in ALL_CONFIGS:
    CONFIGS_BY_NAME[_config.name] = _config


def all_config_names() -> list[str]:
    """The bandit action space as a list of names."""
    names = []
    for config in ALL_CONFIGS:
        names.append(config.name)
    return names


def get_config(name: str) -> PipelineConfig:
    """Look up a configuration by name, with a clear error if it is unknown."""
    if name not in CONFIGS_BY_NAME:
        raise KeyError("unknown pipeline configuration: " + name)
    return CONFIGS_BY_NAME[name]

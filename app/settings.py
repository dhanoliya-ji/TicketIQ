"""Central place for every tunable value and file path in the service.

Everything is read from environment variables so that the same code runs
locally, inside Docker and inside CI without edits.  Import ``SETTINGS``
(a single shared instance) instead of reading ``os.environ`` directly.
"""

import os
from pathlib import Path

# The repository root = the folder that contains the "app" package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    """Read a string environment variable, falling back to a default."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back to a default."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to a default."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class Settings:
    """All configuration for one running instance of the service."""

    def __init__(self) -> None:
        # ---------------- Data locations ----------------
        # Folder holding the labelled ticket dataset and the knowledge base.
        self.data_dir: Path = Path(_env_str("TICKETIQ_DATA_DIR", str(PROJECT_ROOT / "data")))
        self.tickets_file: Path = self.data_dir / "tickets.json"
        self.knowledge_base_dir: Path = self.data_dir / "knowledge_base"

        # Folder for files the service writes at runtime (never committed).
        self.var_dir: Path = Path(_env_str("TICKETIQ_VAR_DIR", str(PROJECT_ROOT / "var")))
        self.workflow_db_file: Path = self.var_dir / "workflow_state.sqlite3"
        self.bandit_state_file: Path = self.var_dir / "bandit_state.json"

        # ---------------- Machine learning ----------------
        # Fraction of the dataset held out for the evaluation report.
        self.test_split: float = _env_float("TICKETIQ_TEST_SPLIT", 0.25)
        # Seed used everywhere we need randomness, so runs are reproducible.
        self.random_seed: int = _env_int("TICKETIQ_RANDOM_SEED", 42)

        # ---------------- Reinforcement learning ----------------
        # Probability of trying a random configuration instead of the best one.
        self.bandit_epsilon: float = _env_float("TICKETIQ_BANDIT_EPSILON", 0.15)

        # ---------------- Large language model ----------------
        # "ollama"   -> call a local Ollama server over HTTP.
        # "template" -> deterministic offline writer, no server needed.
        # "auto"     -> use Ollama when it answers, otherwise template.
        self.llm_backend: str = _env_str("TICKETIQ_LLM_BACKEND", "auto").lower()
        self.ollama_base_url: str = _env_str("TICKETIQ_OLLAMA_URL", "http://localhost:11434")
        # The two models the RL layer can choose between when Ollama is used.
        #
        # These defaults are small on purpose: together they are about 2.3 GB,
        # so following the README takes minutes rather than an hour, and both
        # are from families the brief names (Llama 3 and Qwen). Point them at
        # anything Ollama serves - "llama3", "mistral", "qwen2.5:7b" - with the
        # environment variables; nothing else in the code needs to change.
        self.ollama_model_a: str = _env_str("TICKETIQ_OLLAMA_MODEL_A", "llama3.2:1b")
        self.ollama_model_b: str = _env_str("TICKETIQ_OLLAMA_MODEL_B", "qwen2.5:1.5b")
        # Generous, because a local model's *first* request also pays for
        # loading several gigabytes of weights into memory. 30 seconds was not
        # enough for that and produced a spurious fall back to the template
        # writer on the first ticket after every restart; warm calls take 2-3s.
        self.llm_timeout_seconds: float = _env_float("TICKETIQ_LLM_TIMEOUT", 120.0)

        # ---------------- Agent ----------------
        # Safety stop for the ReAct loop, so it can never spin forever.
        self.agent_max_steps: int = _env_int("TICKETIQ_AGENT_MAX_STEPS", 4)

    def ensure_var_dir(self) -> None:
        """Create the runtime folder if it does not exist yet."""
        self.var_dir.mkdir(parents=True, exist_ok=True)


# One shared instance imported by the rest of the application.
SETTINGS = Settings()

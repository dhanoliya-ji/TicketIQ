"""An epsilon-greedy contextual bandit.

Why a contextual bandit and not Q-learning or a policy network
--------------------------------------------------------------
Choosing a pipeline configuration is a *one step* decision: we pick an arm,
we get a reward, and the next ticket is unrelated to this one.  There is no
sequence of states to plan through, so the machinery of Q-learning (discount
factors, next-state bootstrapping) would be dead weight.  A contextual bandit
is the exact model for "one decision, immediate reward, and the right decision
depends on the context".  It also learns from very few samples, which matters
because every sample costs a real customer interaction.

The rule
--------
For each (state, action) pair we keep a running average of the reward.  Then,
for each new ticket:

* with probability ``epsilon`` pick a random arm  -> **explore**;
* otherwise pick the arm with the best average    -> **exploit**;
* any arm never tried in this state is tried first, so no arm is written off
  on zero evidence.

The running average is updated incrementally:

    new_average = old_average + (reward - old_average) / new_count

which is the same number as re-averaging the whole history, without storing it.
"""

import json
import random
import threading
from pathlib import Path

# Value reported for a state/action pair that has never been tried.
NO_DATA = 0.0


class ArmStatistics:
    """How one action has performed in one state."""

    def __init__(self, pulls: int = 0, average_reward: float = 0.0) -> None:
        # How many times this arm was chosen *and* rewarded in this state.
        self.pulls = pulls
        self.average_reward = average_reward

    def update(self, reward: float) -> None:
        """Fold one new reward into the running average."""
        self.pulls = self.pulls + 1
        difference = reward - self.average_reward
        self.average_reward = self.average_reward + (difference / self.pulls)

    def as_dict(self) -> dict[str, float]:
        return {"pulls": self.pulls, "average_reward": round(self.average_reward, 4)}


class EpsilonGreedyContextualBandit:
    """Learns which action is best in each state, online, from rewards."""

    def __init__(self, actions: list[str], epsilon: float, seed: int = 0) -> None:
        if len(actions) == 0:
            raise ValueError("the bandit needs at least one action")
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("epsilon must be between 0.0 and 1.0")

        self.actions = list(actions)
        self.epsilon = epsilon
        self.random_generator = random.Random(seed)

        # state key -> action name -> statistics
        self.statistics: dict[str, dict[str, ArmStatistics]] = {}

        # How many times each *reason* for a choice was used, which is what the
        # simulation report and the /rl/stats endpoint show.
        self.exploration_count = 0
        self.exploitation_count = 0

        # The API serves requests on several threads, so every mutation of the
        # statistics is guarded.
        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    # Choosing an action
    # ------------------------------------------------------------------
    def select_action(self, state_key: str) -> tuple[str, str]:
        """Pick an action for this state.

        Returns ``(action name, why)`` where *why* is "explore", "exploit" or
        "cold_start", so the decision can be explained in the API response.
        """
        with self.lock:
            arms = self._arms_for_state(state_key)

            # 1. Cold start: try every arm once before trusting any average.
            untried_actions = []
            for action in self.actions:
                if arms[action].pulls == 0:
                    untried_actions.append(action)
            if len(untried_actions) > 0:
                chosen = self.random_generator.choice(untried_actions)
                self.exploration_count = self.exploration_count + 1
                return chosen, "cold_start"

            # 2. Explore with probability epsilon.
            if self.random_generator.random() < self.epsilon:
                chosen = self.random_generator.choice(self.actions)
                self.exploration_count = self.exploration_count + 1
                return chosen, "explore"

            # 3. Otherwise exploit the best known arm.
            best_action = self.actions[0]
            best_average = arms[best_action].average_reward
            for action in self.actions:
                if arms[action].average_reward > best_average:
                    best_action = action
                    best_average = arms[action].average_reward

            self.exploitation_count = self.exploitation_count + 1
            return best_action, "exploit"

    # ------------------------------------------------------------------
    # Learning from a reward
    # ------------------------------------------------------------------
    def update(self, state_key: str, action: str, reward: float) -> None:
        """Record the reward an action earned in a state."""
        if action not in self.actions:
            raise KeyError("unknown action: " + action)

        with self.lock:
            arms = self._arms_for_state(state_key)
            arms[action].update(reward)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def best_action(self, state_key: str) -> str:
        """The arm with the highest average reward in this state, ignoring epsilon."""
        with self.lock:
            arms = self._arms_for_state(state_key)
            best = self.actions[0]
            for action in self.actions:
                if arms[action].average_reward > arms[best].average_reward:
                    best = action
            return best

    def total_pulls(self) -> int:
        with self.lock:
            total = 0
            for arms in self.statistics.values():
                for arm in arms.values():
                    total = total + arm.pulls
            return total

    def snapshot(self) -> dict[str, object]:
        """A JSON friendly view of everything the bandit has learned."""
        with self.lock:
            states: dict[str, dict[str, dict[str, float]]] = {}
            for state_key in sorted(self.statistics.keys()):
                arms = self.statistics[state_key]
                states[state_key] = {}
                for action in self.actions:
                    states[state_key][action] = arms[action].as_dict()

            return {
                "epsilon": self.epsilon,
                "actions": list(self.actions),
                "exploration_count": self.exploration_count,
                "exploitation_count": self.exploitation_count,
                "states": states,
            }

    # ------------------------------------------------------------------
    # Persistence, so learning survives a restart
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Write the learned statistics to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2) + "\n", encoding="utf-8")

    def load(self, path: Path) -> bool:
        """Load statistics written by ``save``. Returns False if there is no file."""
        if not path.exists():
            return False

        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A corrupt state file should not stop the service from starting;
            # we simply begin learning again from scratch.
            return False

        with self.lock:
            self.exploration_count = int(stored.get("exploration_count", 0))
            self.exploitation_count = int(stored.get("exploitation_count", 0))

            self.statistics = {}
            stored_states = stored.get("states", {})
            for state_key, arms in stored_states.items():
                self.statistics[state_key] = {}
                for action in self.actions:
                    values = arms.get(action, {})
                    self.statistics[state_key][action] = ArmStatistics(
                        pulls=int(values.get("pulls", 0)),
                        average_reward=float(values.get("average_reward", NO_DATA)),
                    )
        return True

    # ------------------------------------------------------------------
    def _arms_for_state(self, state_key: str) -> dict[str, ArmStatistics]:
        """Return (creating if needed) the statistics for one state.

        The caller already holds the lock.
        """
        if state_key not in self.statistics:
            fresh: dict[str, ArmStatistics] = {}
            for action in self.actions:
                fresh[action] = ArmStatistics()
            self.statistics[state_key] = fresh
        return self.statistics[state_key]

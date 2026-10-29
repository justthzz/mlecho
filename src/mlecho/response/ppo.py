"""PPO response policy: a learned alternative to the rule table (step 7).

The agent sees the same `State` as the rules, turned into 10 numbers by
`observation`, and picks one of the 5 actions. It is trained in the lab's
practice world (`mlecho.bench.env`) with Stable-Baselines3 PPO; `PPOPolicy`
loads the trained network and plugs into `Guard(policy=...)` like `RulePolicy`.

Needs the optional extra: `pip install "mlecho[rl]"` (only for `PPOPolicy`;
`observation` is plain numpy).
"""

from pathlib import Path

import numpy as np

from mlecho.response.base import Action, State

OBSERVATION_SIZE = 10
_ACTIONS = list(Action)


def observation(state: State) -> np.ndarray:
    """The State as numbers in [-1, 1]: 5 probabilities, confidence, streak (capped
    at 5 windows), acted, escalated, loss change (clipped)."""
    return np.array(
        [
            *state.probabilities,
            state.confidence,
            min(state.streak, 5) / 5,
            float(state.acted),
            float(state.escalated),
            float(np.clip(state.loss_change, -1.0, 1.0)),
        ],
        dtype=np.float32,
    )


class PPOPolicy:
    """A trained PPO agent behind the `ResponsePolicy` interface (deterministic).

    `path`: the agent's network, saved with `agent.policy.save(path)` (only the
    network is needed to decide; the trainer's state isn't kept).
    """

    def __init__(self, path: str | Path, device: str = "cpu"):
        try:
            from stable_baselines3.common.policies import ActorCriticPolicy
        except ImportError as error:
            raise ImportError("PPOPolicy needs: pip install 'mlecho[rl]'") from error
        self.network = ActorCriticPolicy.load(str(path), device=device)

    def decide(self, state: State) -> Action:
        action, _ = self.network.predict(observation(state), deterministic=True)
        return _ACTIONS[int(action)]

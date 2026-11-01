"""Warm start for the PPO agent: copy the rules first, then let PPO improve on them.

PPO from scratch (experiment 19, first run) had to discover from random
actions that acting at all pays off, and after 12,000 steps its deterministic
choice still lost to the rules. Here the network first learns, by supervised
learning, to pick the rules' action in every window the rules met in the
practice world (behaviour cloning), and its value head learns the rules'
discounted return from that window on. PPO then starts from a policy that is
already as good as the rules and only has to find where to do better.

The practice windows alone are too few where the rules act (a smoke test copied
them perfectly yet played worse on new episodes), so the copy also learns the
rules' action on made-up States (`made_up_states`): the rules are a fixed
formula of the State, so any State can be labelled. Those carry no return, so
they train the action head only.
"""

import numpy as np
import torch

from mlecho.bench.env.env import ACTIONS, ResponseEnv
from mlecho.classification.classifier import ATTACK_TYPES
from mlecho.response.base import State
from mlecho.response.ppo import observation


def demonstrations(
    env: ResponseEnv, decide, episodes: list[int], gamma: float
) -> dict[str, np.ndarray]:
    """Play `episodes` with `decide(observation, state) -> Action` and record every
    window: the observation, the action taken and the discounted return from there."""
    observations, actions, returns = [], [], []
    for episode in episodes:
        obs, info = env.reset(options={"episode": episode})
        rewards, done = [], False
        while not done:
            action = ACTIONS.index(decide(obs, info["state"]))
            observations.append(obs)
            actions.append(action)
            obs, reward, done, _, info = env.step(action)
            rewards.append(reward)
        to_go, episode_returns = 0.0, []
        for reward in reversed(rewards):
            to_go = reward + gamma * to_go
            episode_returns.append(to_go)
        returns.extend(reversed(episode_returns))
    return {
        "observations": np.array(observations, np.float32),
        "actions": np.array(actions, np.int64),
        "returns": np.array(returns, np.float32),
    }


def made_up_states(decide, n: int, seed: int = 0) -> dict[str, np.ndarray]:
    """`n` random States (sharp and flat probabilities, any streak and flags) with
    the action `decide(state)` picks for each."""
    rng = np.random.default_rng(seed)
    observations, actions = [], []
    for _ in range(n):
        probabilities = rng.dirichlet(np.full(len(ATTACK_TYPES), rng.choice([0.2, 1.0, 5.0])))
        top = int(probabilities.argmax())
        state = State(
            ATTACK_TYPES[top],
            float(probabilities[top]),
            tuple(float(x) for x in probabilities),
            0 if top == 0 else int(rng.integers(0, 8)),
            bool(rng.random() < 0.5),
            bool(rng.random() < 0.5),
            float(rng.normal(0, 0.5)),
        )
        observations.append(observation(state))
        actions.append(ACTIONS.index(decide(state)))
    return {"observations": np.array(observations, np.float32), "actions": np.array(actions)}


def imitate(
    policy,
    demos: dict[str, np.ndarray],
    extra: dict[str, np.ndarray] | None = None,
    steps: int = 5000,
    lr: float = 1e-3,
    seed: int = 0,
) -> dict[str, float]:
    """Train an SB3 `ActorCriticPolicy` on `demos`: cross-entropy towards the shown
    action plus the squared error of the value against the shown return; on
    `extra` (made-up States without returns), the cross-entropy only. Each of the
    `steps` uses 256 random demo windows (and 256 made-up States).
    Returns the final losses and how often the most likely action matches (overall,
    and in the rare windows where the shown action isn't continue)."""
    generator = torch.Generator().manual_seed(seed)
    obs = torch.as_tensor(demos["observations"])
    actions = torch.as_tensor(demos["actions"])
    returns = torch.as_tensor(demos["returns"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    for _ in range(steps):
        rows = torch.randint(len(actions), (256,), generator=generator)
        values, log_prob, _ = policy.evaluate_actions(obs[rows], actions[rows])
        action_loss = -log_prob.mean()
        value_loss = ((values.flatten() - returns[rows]) ** 2).mean()
        if extra:
            more = torch.randint(len(extra["actions"]), (256,), generator=generator)
            _, extra_log_prob, _ = policy.evaluate_actions(
                torch.as_tensor(extra["observations"][more]),
                torch.as_tensor(extra["actions"][more]),
            )
            action_loss = (action_loss - extra_log_prob.mean()) / 2
        optimizer.zero_grad()
        (action_loss + 0.5 * value_loss).backward()
        optimizer.step()
    policy.eval()
    with torch.no_grad():
        values, _, _ = policy.evaluate_actions(obs, actions)
    result = {"value_loss": float(((values.flatten() - returns) ** 2).mean())}
    result |= _agreement(policy, obs, actions, "")
    if extra:
        extra_obs, extra_actions = (torch.as_tensor(extra[k]) for k in ("observations", "actions"))
        result |= _agreement(policy, extra_obs, extra_actions, "made_up_")
    return result


@torch.no_grad()
def _agreement(policy, obs: torch.Tensor, actions: torch.Tensor, prefix: str) -> dict:
    """How often the most likely action is the shown one (overall, and where the
    shown action isn't continue, which is rare)."""
    chosen = policy.get_distribution(obs).distribution.probs.argmax(dim=1)
    acting = actions != 0  # ACTIONS[0] is continue
    return {
        f"{prefix}agreement": float((chosen == actions).float().mean()),
        f"{prefix}agreement_acting": float((chosen == actions)[acting].float().mean()),
    }

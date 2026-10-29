"""Experiment 19: train the PPO response agent in the practice world (step 7).

Uses the practice episodes of experiment 18. The last `--held-out` episodes of
every kind are never trained on: they pick the best checkpoint (model
selection) and compare the agent with the other policies inside the practice
world. The real attacks of experiment 15 are not used here at all; the agent
meets them only in the final test (experiment 15 with `--policy ppo`).

Before training, the baselines play the held-out episodes: no response, the
rules, quarantine-everything and random actions. If the rules don't beat no
response clearly, the reward is wrong.

Settings fixed before training (ROADMAP step 7): reward weights `RewardWeights()`,
PPO with 2 x 64 hidden units, 512 steps per update, gamma 0.95 (a window's
consequences matter for ~20 windows), entropy bonus 0.01, the rest
Stable-Baselines3 defaults; 24,000 steps (~600 practice episodes).

Run:   python experiments/19_train_ppo.py
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import mlflow
import numpy as np
import torch
from common import add_detection_args, add_guard_args, detection_data, guard_parts
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor as EpisodeStats

from mlecho.bench.baselines.response import NoResponse, QuarantineEverything
from mlecho.bench.env.env import ACTIONS, ResponseEnv, play
from mlecho.bench.env.practice import KINDS
from mlecho.bench.models import SmallCNN
from mlecho.core import get_device, load_weights, seed_everything
from mlecho.response.rules import RulePolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_detection_args(parser)
    add_guard_args(parser)
    parser.add_argument("--practice", default="data/env/practice-seed7.pt")
    parser.add_argument(
        "--burn-in", default="data/checkpoints/burn-in-smallcnn-w64-sgd0.01-seed42.pt"
    )
    parser.add_argument("--held-out", type=int, default=4, help="per kind, never trained on")
    parser.add_argument("--timesteps", type=int, default=24_000)
    parser.add_argument("--check-every", type=int, default=6_000, help="timesteps")
    parser.add_argument("--agent-seed", type=int, default=0)
    parser.add_argument("--out", default="data/checkpoints/ppo-response-seed0.pt")
    return parser.parse_args()


def split(episodes, held_out: int) -> tuple[list[int], list[int]]:
    """Episode numbers to train on and to hold out (the last `held_out` of each kind)."""
    by_kind = defaultdict(list)
    for k, episode in enumerate(episodes):
        by_kind[episode.attack.kind].append(k)
    train = [k for kind in KINDS for k in by_kind[kind][:-held_out]]
    test = [k for kind in KINDS for k in by_kind[kind][-held_out:]]
    return train, test


def evaluate(env: ResponseEnv, decide, episodes: list[int]) -> dict:
    """Play every held-out episode once: mean return per kind, end effect, actions."""
    returns, effects, actions = defaultdict(list), defaultdict(list), Counter()
    for k in episodes:
        taken = []

        def record(obs, state, taken=taken):
            action = decide(obs, state)
            taken.append(action.value)
            return action

        total, info = play(env, record, k)
        kind = env.episodes[k].attack.kind
        returns[kind].append(total)
        effects[kind].append(info["effect"])
        actions.update(f"{kind}:{a}" for a in taken if a != "continue")
    result = {f"return_{kind}": float(np.mean(v)) for kind, v in returns.items()}
    result |= {f"end_effect_{kind}": float(np.mean(v)) for kind, v in effects.items()}
    result["return_mean"] = float(np.mean([np.mean(v) for v in returns.values()]))
    return result | {"actions": dict(actions)}


def policy_decider(policy):
    return lambda obs, state: policy.decide(state)


class Progress(BaseCallback):
    """Logs the training return per rollout, and every `every` steps plays the
    held-out episodes and keeps the best network."""

    def __init__(self, env: ResponseEnv, held_out: list[int], every: int, out: str):
        super().__init__()
        self.env, self.held_out, self.every, self.out = env, held_out, every, out
        self.best, self.next_check = -np.inf, every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_check:
            self.next_check += self.every
            result = evaluate(self.env, self._decide, self.held_out)
            mlflow.log_metric("held_out_return", result["return_mean"], step=self.num_timesteps)
            print(f"  {self.num_timesteps} steps: held-out return {result['return_mean']:.2f}")
            if result["return_mean"] > self.best:
                self.best = result["return_mean"]
                self.model.policy.save(self.out)
                mlflow.log_metric("best_at_step", self.num_timesteps)
        return True

    def _on_rollout_end(self) -> None:
        rewards = [info["r"] for info in self.model.ep_info_buffer]
        if rewards:
            mlflow.log_metric("train_return", float(np.mean(rewards)), step=self.num_timesteps)

    def _decide(self, obs, state):
        action, _ = self.model.policy.predict(obs, deterministic=True)
        return ACTIONS[int(action)]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    data = detection_data(args.seed, args.fit_size, args.gm_cache)
    _, calibration, classifier = guard_parts(args, data, device)
    trusted = np.concatenate([data.fit, data.held_out])
    validation_raw = data.full.data[trusted]
    validation_labels = np.asarray(data.full.targets)[trusted]
    train_raw = data.full.data[data.train]

    episodes = torch.load(args.practice, weights_only=False)  # our own file (experiment 18)
    train, held_out = split(episodes, args.held_out)

    def make_env(numbers: list[int]) -> ResponseEnv:
        return ResponseEnv(
            [episodes[k] for k in numbers],
            train_raw,
            calibration,
            classifier,
            load_weights(SmallCNN(width=64), args.burn_in, device),
            validation_raw,
            validation_labels,
            device,
            window=args.window,
            seed=args.seed,
        )

    test_env = make_env(held_out)
    test_numbers = list(range(len(held_out)))

    mlflow.set_experiment("mlecho-ppo")
    with mlflow.start_run(run_name=f"train-ppo-seed{args.agent_seed}"):
        mlflow.log_params(vars(args) | {"device": str(device), "train_episodes": len(train)})

        rng = np.random.default_rng(args.agent_seed)
        baselines = {
            "none": policy_decider(NoResponse()),
            "rules": policy_decider(RulePolicy()),
            "quarantine-everything": policy_decider(QuarantineEverything()),
            "random": lambda obs, state: ACTIONS[int(rng.integers(len(ACTIONS)))],
        }
        table = {}
        for name, decide in baselines.items():
            start = time.perf_counter()
            table[name] = evaluate(test_env, decide, test_numbers)
            seconds = (time.perf_counter() - start) / len(test_numbers)
            print(
                f"{name:22s} held-out return {table[name]['return_mean']:.2f} "
                f"({seconds:.1f} s/episode)"
            )
            mlflow.log_metrics({f"{name}_{k}": v for k, v in table[name].items() if k != "actions"})

        env = EpisodeStats(make_env(train))
        agent = PPO(
            "MlpPolicy",
            env,
            n_steps=512,
            batch_size=64,
            gamma=0.95,
            ent_coef=0.01,
            policy_kwargs={"net_arch": [64, 64]},
            seed=args.agent_seed,
            device="cpu",
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        agent.learn(
            args.timesteps, callback=Progress(test_env, test_numbers, args.check_every, args.out)
        )
        mlflow.log_metric("train_minutes", (time.perf_counter() - start) / 60)

        best = PPO("MlpPolicy", env, device="cpu")
        best.policy = best.policy.load(args.out, device="cpu")
        table["ppo"] = evaluate(
            test_env,
            lambda obs, state: ACTIONS[int(best.policy.predict(obs, deterministic=True)[0])],
            test_numbers,
        )
        mlflow.log_metrics({f"ppo_{k}": v for k, v in table["ppo"].items() if k != "actions"})
        mlflow.log_artifact(args.out)
        mlflow.log_text(json.dumps(table, indent=2), "held_out_practice.json")
        print("\nheld-out practice episodes (mean return per kind; higher is better)")
        for name, result in table.items():
            row = " ".join(f"{kind[:8]} {result.get(f'return_{kind}', 0):6.2f}" for kind in KINDS)
            print(f"{name:22s} {row} | mean {result['return_mean']:.2f}")
        for name, result in table.items():
            print(f"{name:22s} actions {result['actions']}")


if __name__ == "__main__":
    main()

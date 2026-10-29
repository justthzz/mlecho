"""The PPO agent's practice world (a Gymnasium environment, needs `mlecho[rl]`).

One episode = one cached practice stream (`practice.py`) replayed through a
real `Guard`: the victim starts from the burn-in model and learns online, one
SGD step per batch, exactly as in experiment 15. One step = one window: the
agent sees the window's `State` (`response.ppo.observation`) and picks an
action, which the guard carries out (`Guard.respond`); then the next window
is trained.

The reward is measured after every window on trusted validation images (never
the test set), with knowledge only the practice world has (the attack):

    reward = - effect   x (attack effect above the start)
             - accuracy x (clean accuracy drop since the start)
             - data     x (share of clean images left out in the window)
             - the action's own cost (rollback, retrain; escalating when nothing happened)

Attack effect: the share of triggered images the model calls the trigger's
target (backdoor, slow-drip), or the share of the flipped class it calls the
flip's target (label-flip); 0 for harmless shifts and clean streams.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from mlecho.attacks.backdoor import add_trigger
from mlecho.bench.datasets import normalise
from mlecho.bench.env.practice import CachedEpisode, PracticeAttack, batch_pixels
from mlecho.bench.online import online_step
from mlecho.classification.classifier import AttackClassifier
from mlecho.classification.features import Calibration
from mlecho.classification.monitor import Evidence, Monitor
from mlecho.guard import Guard
from mlecho.response.base import Action
from mlecho.response.ppo import OBSERVATION_SIZE, observation

ACTIONS = list(Action)


@dataclass(frozen=True)
class RewardWeights:
    """Fixed before training (ROADMAP step 7)."""

    effect: float = 1.0
    accuracy: float = 1.0
    data: float = 1.0
    rollback: float = 0.02
    retrain: float = 0.02
    false_escalation: float = 0.1  # a human's time, wasted when nothing happened


class _ReplayMonitor(Monitor):
    """A Monitor whose evidence comes from the cache instead of the fold models."""

    def __init__(self, episode: CachedEpisode, calibration, classifier, window: int):
        super().__init__([], [], calibration, classifier, window)
        self.episode, self.batch = episode, 0

    def evidence(self, images: torch.Tensor) -> Evidence:
        evidence = self.episode.evidence(self.batch)
        self.batch += 1
        return evidence


class _Outcome:
    """Clean accuracy and attack effect on trusted validation images."""

    def __init__(self, raw: np.ndarray, labels: np.ndarray, device, images: int, seed: int):
        rows = np.random.default_rng(seed).permutation(len(labels))[:images]
        self.raw, self.labels, self.device = raw[rows], labels[rows], device
        self.clean = normalise(self.raw).to(device)
        self._effect_images: dict[int, tuple[torch.Tensor, int]] = {}

    @torch.no_grad()
    def predict(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        model.eval()
        return model(images).argmax(dim=1).cpu()

    def accuracy(self, model: nn.Module) -> float:
        return (
            (self.predict(model, self.clean) == torch.as_tensor(self.labels)).float().mean().item()
        )

    def effect(self, model: nn.Module, attack: PracticeAttack, key: int) -> float:
        if key not in self._effect_images:
            self._effect_images[key] = self._attack_images(attack)
        images, target = self._effect_images[key]
        if images is None:
            return 0.0
        return (self.predict(model, images) == target).float().mean().item()

    def _attack_images(self, attack: PracticeAttack):
        if attack.trigger is not None:  # backdoor, slow-drip: triggered non-target images
            target = attack.trigger.target_class
            raw = self.raw[self.labels != target]
            return normalise(add_trigger(raw, attack.trigger)).to(self.device), target
        if attack.flip is not None:  # label-flip: the source class
            source, target = attack.flip
            return normalise(self.raw[self.labels == source]).to(self.device), target
        return None, -1


def play(env: "ResponseEnv", decide, episode: int) -> tuple[float, dict]:
    """One whole episode with `decide(observation, state) -> Action`. Returns
    (total reward, the last step's info)."""
    obs, info = env.reset(options={"episode": episode})
    total, done = 0.0, False
    while not done:
        action = ACTIONS.index(decide(obs, info["state"]))
        obs, reward, done, _, info = env.step(action)
        total += reward
    return total, info


class ResponseEnv(gym.Env):
    """Practice world for a response policy: observations are `State`s, actions `Action`s."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        episodes: list[CachedEpisode],
        train_raw: np.ndarray,
        calibration: Calibration,
        classifier: AttackClassifier,
        burn_in: nn.Module,
        validation_raw: np.ndarray,
        validation_labels: np.ndarray,
        device: torch.device,
        window: int = 10,
        lr: float = 0.01,
        momentum: float = 0.9,
        weights: RewardWeights | None = None,
        measure_images: int = 1000,
        seed: int = 0,
    ):
        super().__init__()
        self.episodes, self.train_raw = episodes, train_raw
        self.calibration, self.classifier, self.window = calibration, classifier, window
        self.burn_in = {k: v.detach().clone() for k, v in burn_in.state_dict().items()}
        self.model = burn_in
        self.lr, self.momentum, self.device = lr, momentum, device
        self.weights = weights or RewardWeights()
        self.outcome = _Outcome(validation_raw, validation_labels, device, measure_images, seed)
        self.start_accuracy = None
        self._start_effect: dict[int, float] = {}
        self.loss_fn = nn.CrossEntropyLoss()
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(-1.0, 1.0, (OBSERVATION_SIZE,), np.float32)

    # ---------- one episode ----------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options and "episode" in options:
            self.index = int(options["episode"])
        else:
            self.index = int(self.np_random.integers(len(self.episodes)))
        self.episode = self.episodes[self.index]
        self.model.load_state_dict(self.burn_in)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=self.momentum
        )
        if self.start_accuracy is None:
            self.start_accuracy = self.outcome.accuracy(self.model)
        if self.index not in self._start_effect:
            self._start_effect[self.index] = self.outcome.effect(
                self.model, self.episode.attack, self.index
            )
        monitor = _ReplayMonitor(self.episode, self.calibration, self.classifier, self.window)
        self.guard = Guard(monitor)
        self.guard.watch(self.model, self.optimizer)
        self.batch, self.totals = 0, {"clean_dropped": 0, "poison_trained": 0}
        self._window_clean_dropped = self._window_images = 0
        state = self._train_until_window_end()
        return observation(state), {"state": state}

    def step(self, action: int):
        chosen = ACTIONS[int(action)]
        attacked = self.episode.attack.kind != "clean" and self.batch > self.episode.attack.onset
        self.guard.respond(chosen)
        self._window_clean_dropped = 0
        self._window_images = 0
        state = self._train_until_window_end()
        reward, info = self._reward(chosen, attacked)
        terminated = state is None
        obs = np.zeros(OBSERVATION_SIZE, np.float32) if terminated else observation(state)
        info["state"] = state  # for policies that read the State itself (the rules)
        return obs, reward, terminated, False, info

    # ---------- inside ----------

    def _train_until_window_end(self):
        """Train batch by batch until the guard has a window's State (None at the end)."""
        ep = self.episode
        while self.batch < len(ep.rows):
            b = self.batch
            self.batch += 1
            pixels = batch_pixels(ep.attack, self.train_raw[ep.rows[b]], ep.changed[b])
            images, labels = normalise(pixels).to(self.device), torch.as_tensor(ep.labels[b])
            keep = self.guard.check(images, labels).cpu().numpy()
            clean_dropped = int((~ep.poisoned[b] & ~keep).sum())
            self._window_clean_dropped += clean_dropped
            self._window_images += len(keep)
            self.totals["clean_dropped"] += clean_dropped
            self.totals["poison_trained"] += int((ep.poisoned[b] & keep).sum())
            loss = None
            if keep.sum() > 1:
                keep_t = torch.as_tensor(keep)
                loss = online_step(
                    self.model,
                    self.optimizer,
                    self.loss_fn,
                    images[keep_t.to(self.device)],
                    labels[keep_t].to(self.device),
                )
            state = self.guard.observe(loss)
            if state is not None:
                return state
        return None

    def _reward(self, action: Action, attacked: bool) -> tuple[float, dict]:
        w = self.weights
        accuracy = self.outcome.accuracy(self.model)
        effect = self.outcome.effect(self.model, self.episode.attack, self.index)
        effect_rise = max(0.0, effect - self._start_effect[self.index])
        accuracy_drop = max(0.0, self.start_accuracy - accuracy)
        data_loss = self._window_clean_dropped / max(1, self._window_images)
        cost = {Action.ROLLBACK: w.rollback, Action.RETRAIN: w.retrain}.get(action, 0.0)
        if action == Action.ESCALATE and not attacked:
            cost += w.false_escalation
        reward = -(w.effect * effect_rise + w.accuracy * accuracy_drop + w.data * data_loss) - cost
        info = {
            "effect": effect,
            "accuracy": accuracy,
            "data_loss": data_loss,
            "action": action.value,
            **self.totals,
        }
        return float(reward), info

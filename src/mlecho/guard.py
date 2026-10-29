"""Guard: the one object a user touches.

    guard = Guard.calibrate(trusted_images, trusted_labels, num_classes=10, device=device)
    guard.watch(model, optimizer)               # lets rollback and retrain work
    for images, labels in loader:
        keep = guard.check(images, labels)      # True = train on this image
        loss = train_step(images[keep], labels[keep])
        guard.update(loss)                      # after every window: verdict + action

Give `check` the batches as they arrive, **before** any random augmentation
(shifts, flips): poison is in the incoming data, and augmented images look
"different from normal" to the picture check, so the guard would treat
every window as a harmless shift (measured: experiment 10).

`calibrate` learns what "normal" means from the user's own trusted clean
images (cross-fitted CNN-VAEs and reference models, thresholds, made-up
attacks and the attack classifier), so the guard is not tied to CIFAR-10.
Every window's verdict and every action is sent to the event loggers
(`mlecho.tracking`: a JSON audit log, MLflow).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mlecho.actions.checkpoints import CheckpointManager
from mlecho.actions.quarantine import quarantine_mask
from mlecho.classification.classifier import AttackClassifier, fit_classifier
from mlecho.classification.features import FEATURES, Calibration, calibrate
from mlecho.classification.monitor import Evidence, Monitor, Verdict
from mlecho.classification.training_data import made_up_set, made_up_settings
from mlecho.detection.folds import fold_numbers, fold_scores, train_fold_models
from mlecho.detection.label_check import ReferenceCNN
from mlecho.detection.vae import ConvVAE
from mlecho.response.base import Action, ResponsePolicy, State
from mlecho.response.rules import RulePolicy


@dataclass(frozen=True)
class GuardConfig:
    """Settings of `Guard.calibrate` (defaults: the ones evaluated in steps 3-5)."""

    pixel_mean: tuple[float, ...]  # how the user's model normalises images
    pixel_std: tuple[float, ...]
    folds: int = 5
    vae_epochs: int = 60
    reference_epochs: int = 100
    patch: int = 4  # picture score: worst patch x patch square (experiment 07)
    rate: float = 0.01  # share of clean images each threshold flags
    window: int = 10  # batches per verdict
    batch_size: int = 128
    made_up_configs: int = 40  # random triggers and shifts for the classifier
    made_up_windows: int = 600  # per attack type
    seed: int = 42
    check_every: int = 1  # score every n-th batch while nothing is suspicious (speed)


@dataclass
class _Incident:
    """Bookkeeping between windows (one incident = a run of suspicious windows)."""

    streak: int = 0
    start: int = 0
    clean_streak: int = 0
    active: bool = False
    acted: bool = False
    escalated: bool = False
    last_type: str = "clean"
    losses: list[float] = field(default_factory=list)
    previous_loss: float | None = None


class Guard:
    """Detector + classifier + policy + actions around one training run."""

    def __init__(
        self,
        monitor: Monitor,
        policy: ResponsePolicy | None = None,
        loggers: list | None = None,
        config: GuardConfig | None = None,
        check_every: int | None = None,
    ):
        """`check_every`: in quiet times, score only every n-th batch (a window is then
        `window` scored batches). Once a window looks suspicious, or a quarantine is
        on, every batch is scored. Default: the config's value, else 1 (every batch)."""
        self.monitor = monitor
        self.policy = policy if policy is not None else RulePolicy()
        self.loggers = loggers or []
        self.config = config
        self.quarantine: Verdict | None = None  # the verdict that started the quarantine
        self.batches = 0
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.checkpoints = CheckpointManager()
        self._incident = _Incident()
        self._evidence: tuple[Evidence, torch.Tensor] | None = None
        if check_every is None:
            check_every = config.check_every if config is not None else 1
        self.check_every = check_every
        self._in_window = 0  # scored batches in the current window
        self._window_start = 0  # batches trained before the current window began

    # ---------- during training ----------

    def watch(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Track this model: a checkpoint now and after every window."""
        self.model, self.optimizer = model, optimizer
        self.checkpoints.save(model, optimizer, batch=self.batches, clean=True)

    def check(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Score one incoming batch (normalised images). Returns which images to train on."""
        alert = self.quarantine is not None or self._incident.active
        if not (alert or self.batches % self.check_every == 0):
            self._evidence = None  # a quiet batch that isn't sampled
            return torch.ones(len(labels), dtype=torch.bool, device=labels.device)
        evidence = self.monitor.evidence(images)
        self._evidence = (evidence, labels)
        keep = torch.ones(len(labels), dtype=torch.bool)
        if self.quarantine is not None:
            keep = ~quarantine_mask(evidence, labels, self.monitor.calibration, self.quarantine)
        return keep.to(labels.device)

    def update(self, loss: float | None = None) -> Action | None:
        """Call after the training step. At the end of a window: verdict, action, checkpoint.

        Returns the action taken at a window's end (None in between).
        """
        self.batches += 1
        incident = self._incident
        if loss is not None:
            incident.losses.append(loss)
        if self._evidence is None:
            return None
        if self._in_window == 0:
            self._window_start = self.batches - 1
        self._in_window += 1
        verdict = self.monitor.add(*self._evidence)
        if verdict is None:
            return None
        self._in_window = 0
        state = self._next_state(verdict)
        action = self.policy.decide(state)
        self._carry_out(action, verdict)
        if self.model is not None:
            self.checkpoints.save(
                self.model, self.optimizer, self.batches, clean=verdict.attack_type == "clean"
            )
        event = {"batch": self.batches, "action": action.value} | asdict(verdict) | asdict(state)
        for logger in self.loggers:
            logger.log(event)
        return action

    def _next_state(self, verdict: Verdict) -> State:
        incident = self._incident
        if verdict.attack_type == "clean":
            incident.streak, incident.clean_streak = 0, incident.clean_streak + 1
            if incident.clean_streak >= 2:  # two clean windows in a row: the incident is over
                incident.active = incident.acted = incident.escalated = False
        else:
            incident.clean_streak = 0
            same = verdict.attack_type == incident.last_type
            incident.streak = incident.streak + 1 if same else 1
            if not incident.active:  # first suspicious window: remember where it began
                incident.active, incident.start = True, self._window_start
        incident.last_type = verdict.attack_type
        mean_loss = float(np.mean(incident.losses)) if incident.losses else 0.0
        previous = incident.previous_loss
        incident.previous_loss, incident.losses = mean_loss, []
        return State(
            attack_type=verdict.attack_type,
            confidence=verdict.confidence,
            probabilities=verdict.probabilities,
            streak=incident.streak,
            acted=incident.acted,
            escalated=incident.escalated,
            loss_change=0.0 if previous is None else mean_loss - previous,
        )

    def _carry_out(self, action: Action, verdict: Verdict) -> None:
        if action in (Action.ROLLBACK, Action.RETRAIN) and self.model is not None:
            checkpoint = (
                self.checkpoints.last_clean(before=self._incident.start - self.monitor.window)
                if action == Action.ROLLBACK
                else self.checkpoints.oldest_clean()
            )
            self.checkpoints.restore(checkpoint, self.model, self.optimizer)
        if action in (Action.QUARANTINE, Action.ROLLBACK, Action.RETRAIN):
            self.quarantine, self._incident.acted = verdict, True
        if action == Action.ESCALATE:
            self._incident.escalated = True

    def close(self) -> None:
        """Finish the event logs (e.g. write the audit log to MLflow)."""
        for logger in self.loggers:
            logger.close()

    # ---------- before training ----------

    @classmethod
    def calibrate(
        cls,
        raw: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
        device: torch.device,
        config: GuardConfig,
        policy: ResponsePolicy | None = None,
        loggers: list | None = None,
    ) -> Guard:
        """Learn "normal" from trusted clean images: raw pixels (N, H, W, 3), 0-255.

        Trains the fold models, measures the calibration, makes up attacks on the
        trusted images and trains the attack classifier on them (steps 3-4).
        """
        generator = torch.Generator().manual_seed(config.seed)
        rng = np.random.default_rng(config.seed)
        normalise = _normaliser(config)
        images, targets = normalise(raw), torch.as_tensor(labels)
        fold = fold_numbers(len(targets), config.folds, generator)
        vaes, references = train_fold_models(
            images,
            targets,
            fold,
            num_classes,
            device,
            config.vae_epochs,
            config.reference_epochs,
            config.batch_size,
            seed=config.seed,
        )

        def score(changed_raw: np.ndarray) -> Evidence:
            return Evidence(
                *fold_scores(vaes, references, normalise(changed_raw), fold, config.patch)
            )

        trusted = score(raw)
        window_size = config.window * config.batch_size
        calibration = calibrate(
            trusted.picture, trusted.log_probs, targets, window_size, config.rate, generator
        )
        settings = made_up_settings(score, raw, num_classes, config.made_up_configs, rng)
        x, y = made_up_set(
            trusted,
            targets,
            settings,
            calibration,
            config.made_up_windows,
            config.window,
            config.batch_size,
            rng,
            generator,
        )
        classifier = AttackClassifier(len(FEATURES))
        fit_classifier(classifier, x, y)
        monitor = Monitor(
            vaes, references, calibration, classifier, config.window, config.patch, config.seed
        )
        return cls(monitor, policy, loggers, config)

    # ---------- saving ----------

    def save(self, path: str | Path) -> None:
        """Write everything `calibrate` learned to one file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        monitor = self.monitor
        torch.save(
            {
                "config": asdict(self.config),
                "calibration": asdict(monitor.calibration),
                "num_classes": monitor.calibration.reference_counts.shape[0],
                "vaes": [m.state_dict() for m in monitor.vaes],
                "references": [m.state_dict() for m in monitor.references],
                "classifier": monitor.classifier.state_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: torch.device,
        policy: ResponsePolicy | None = None,
        loggers: list | None = None,
        check_every: int | None = None,
    ) -> Guard:
        """A guard saved with `save`, ready to `watch` a new training run."""
        saved = torch.load(path, map_location="cpu", weights_only=True)
        config = GuardConfig(
            **{k: tuple(v) if isinstance(v, list) else v for k, v in saved["config"].items()}
        )
        num_classes = saved["num_classes"]

        def build(model: nn.Module, state: dict) -> nn.Module:
            model.load_state_dict(state)
            return model.to(device).eval()

        vaes = [build(ConvVAE(), s) for s in saved["vaes"]]
        references = [build(ReferenceCNN(num_classes=num_classes), s) for s in saved["references"]]
        classifier = AttackClassifier(len(FEATURES))
        classifier.load_state_dict(saved["classifier"])
        monitor = Monitor(
            vaes,
            references,
            Calibration(**saved["calibration"]),
            classifier,
            config.window,
            config.patch,
            config.seed,
        )
        return cls(monitor, policy, loggers, config, check_every)


def _normaliser(config: GuardConfig):
    """Raw pixels (N, H, W, C), 0-255 -> normalised tensors (N, C, H, W)."""
    mean = torch.tensor(config.pixel_mean).view(1, -1, 1, 1)
    std = torch.tensor(config.pixel_std).view(1, -1, 1, 1)

    def normalise(raw: np.ndarray) -> torch.Tensor:
        return (torch.from_numpy(raw).permute(0, 3, 1, 2).float() / 255 - mean) / std

    return normalise

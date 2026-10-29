import numpy as np
import pytest
import torch
from torch import nn

from mlecho.classification.classifier import ATTACK_TYPES
from mlecho.detection.folds import fold_numbers, fold_scores, train_fold_models
from mlecho.guard import Guard, GuardConfig
from mlecho.response.base import Action
from mlecho.tracking.audit import AuditLog

DEVICE = torch.device("cpu")
TINY = GuardConfig(
    pixel_mean=(0.5, 0.5, 0.5),
    pixel_std=(0.25, 0.25, 0.25),
    folds=2,
    vae_epochs=1,
    reference_epochs=1,
    window=2,
    batch_size=16,
    made_up_configs=2,
    made_up_windows=3,
)


def _trusted(n: int = 120):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8), np.arange(n) % 10


def test_fold_numbers_split_evenly():
    fold = fold_numbers(10, 5, torch.Generator().manual_seed(0))
    assert sorted(torch.bincount(fold).tolist()) == [2] * 5


def test_fold_models_score_every_image_with_its_own_fold():
    raw, labels = _trusted(40)
    images = torch.randn(40, 3, 32, 32)
    fold = torch.arange(40) % 2
    vaes, refs = train_fold_models(images, torch.as_tensor(labels), fold, 10, DEVICE, 1, 1, 16)
    assert len(vaes) == len(refs) == 2
    picture, log_probs = fold_scores(vaes, refs, images, fold)
    assert picture.shape == (40,) and log_probs.shape == (40, 10)
    assert torch.allclose(log_probs.exp().sum(dim=1), torch.ones(40), atol=1e-5)


class _AlwaysRollback:
    def decide(self, state):
        return Action.ROLLBACK if not state.acted else Action.CONTINUE


def test_guard_calibrates_saves_loads_and_protects_a_training_loop(tmp_path):
    raw, labels = _trusted()
    guard = Guard.calibrate(raw, labels, 10, DEVICE, TINY)
    path = tmp_path / "guard.pt"
    guard.save(path)

    audit = AuditLog(tmp_path / "events.jsonl")
    loaded = Guard.load(path, DEVICE, policy=_AlwaysRollback(), loggers=[audit])
    assert loaded.monitor.calibration.picture_threshold == pytest.approx(
        guard.monitor.calibration.picture_threshold
    )

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    start = model[1].weight.detach().clone()
    loaded.watch(model, optimizer)
    images, targets = torch.randn(16, 3, 32, 32), torch.arange(16) % 10
    actions, after_window = [], None
    for b in range(4):
        keep = loaded.check(images, targets)
        assert keep.dtype == torch.bool and keep.shape == (16,)
        loss = nn.functional.cross_entropy(model(images[keep]), targets[keep])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        actions.append(loaded.update(loss.item()))
        if b == 1:
            after_window = model[1].weight.detach().clone()
    loaded.close()

    assert actions[0] is None and actions[2] is None  # mid-window
    assert actions[1] == Action.ROLLBACK  # first window: rolled back to the start...
    events = AuditLog.read(tmp_path / "events.jsonl")
    assert [e["batch"] for e in events] == [2, 4]
    assert events[0]["action"] == "rollback" and events[0]["attack_type"] in ATTACK_TYPES
    assert loaded.quarantine is not None  # ...and quarantine is on
    assert torch.equal(after_window, start)  # the rollback undid the first window's training


def test_mlflow_logger_records_metrics_tags_and_the_event_list(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from mlecho.tracking.mlflow_logger import MLflowLogger

    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    event = {
        "batch": 10,
        "attack_type": "backdoor",
        "confidence": 0.9,
        "probabilities": (0.02, 0.9, 0.03, 0.03, 0.02),
        "action": "rollback",
    }
    with mlflow.start_run() as run:
        logger = MLflowLogger()
        logger.log(event)
        logger.close()
    data = mlflow.get_run(run.info.run_id).data
    mlflow.set_tracking_uri(previous)
    assert data.metrics["guard_p_backdoor"] == pytest.approx(0.9)
    assert "rollback" in data.tags["guard_action_1"]


def test_sampled_checking_scores_every_nth_quiet_batch(tmp_path):
    raw, labels = _trusted()
    path = tmp_path / "guard.pt"
    Guard.calibrate(raw, labels, 10, DEVICE, TINY).save(path)

    class _Never:
        def decide(self, state):
            return Action.CONTINUE

    guard = Guard.load(path, DEVICE, policy=_Never(), check_every=3)
    guard._incident.active = False
    images, targets = torch.randn(16, 3, 32, 32), torch.arange(16) % 10
    verdict_at, scored = [], 0
    for b in range(12):
        guard.check(images, targets)
        scored += guard._evidence is not None
        if guard.update(0.0) is not None:
            verdict_at.append(b + 1)
        guard._incident.active = False  # keep it quiet: random images may look odd
    assert scored == 4  # batches 1, 4, 7, 10
    assert verdict_at == [4, 10]  # a window = 2 scored batches


def test_filter_returns_the_same_batch_until_a_quarantine_starts(tmp_path):
    raw, labels = _trusted()
    path = tmp_path / "guard.pt"
    Guard.calibrate(raw, labels, 10, DEVICE, TINY).save(path)
    guard = Guard.load(path, DEVICE, policy=_AlwaysRollback())
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    guard.watch(model, optimizer)
    images, targets = torch.randn(16, 3, 32, 32), torch.arange(16) % 10
    kept, _ = guard.filter(images, targets)
    assert kept is images  # no quarantine yet: the very same tensor
    loss = nn.functional.cross_entropy(model(kept), targets)
    guard.update(loss)  # a tensor is fine
    guard.filter(images, targets)
    assert guard.update(loss) == Action.ROLLBACK
    kept, kept_labels = guard.filter(images, targets)
    assert len(kept) == len(kept_labels) == int(guard.check(images, targets).sum())

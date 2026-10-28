import numpy as np
import pytest
import torch
from torch import nn

from mlecho.actions.checkpoints import CheckpointManager
from mlecho.actions.quarantine import quarantine_mask
from mlecho.bench.baselines.response import QuarantineEverything
from mlecho.bench.stream import Scenario, run_episode
from mlecho.classification.classifier import ATTACK_TYPES, AttackClassifier
from mlecho.classification.features import FEATURES, calibrate
from mlecho.classification.monitor import Evidence, Monitor
from mlecho.detection.label_check import ReferenceCNN
from mlecho.detection.vae import ConvVAE
from mlecho.response.base import Action, State
from mlecho.response.rules import RULES, RulePolicy


def _state(kind="backdoor", confidence=0.9, streak=2, acted=False, escalated=False):
    return State(kind, confidence, (0.2,) * 5, streak, acted, escalated, 0.0)


@pytest.mark.parametrize("kind", ["backdoor", "accumulative", "label_flip"])
def test_rules_follow_the_table_once_the_answer_persists(kind):
    policy = RulePolicy()
    assert policy.decide(_state(kind, streak=1)) == Action.CONTINUE  # one window is not enough
    assert policy.decide(_state(kind, streak=2)) == RULES[kind]
    assert policy.decide(_state(kind, streak=3, acted=True)) == Action.CONTINUE  # already done


def test_rules_escalate_on_low_confidence_or_a_harmless_shift_but_only_once():
    policy = RulePolicy()
    assert policy.decide(_state("backdoor", confidence=0.4)) == Action.ESCALATE
    assert policy.decide(_state("benign_shift")) == Action.ESCALATE
    assert policy.decide(_state("benign_shift", escalated=True)) == Action.CONTINUE
    assert policy.decide(_state("clean", streak=0)) == Action.CONTINUE


def test_checkpoints_restore_the_right_copy():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    manager = CheckpointManager(keep=3)
    values = {}
    for batch, clean in [(0, True), (10, True), (20, False), (30, True), (40, False)]:
        with torch.no_grad():
            model.weight.fill_(batch)
        values[batch] = model.weight.clone()
        manager.save(model, optimizer, batch, clean)
    assert [c.batch for c in manager.saved] == [0, 20, 30, 40]  # newest 3 + the first
    manager.restore(manager.last_clean(before=25), model, optimizer)
    assert torch.equal(model.weight, values[0])  # 10 was dropped, 20 not clean
    manager.restore(manager.last_clean(before=40), model, optimizer)
    assert torch.equal(model.weight, values[30])
    assert manager.oldest_clean().batch == 0


def _calibration():
    g = torch.Generator().manual_seed(0)
    labels = torch.arange(1000) % 10
    logits = torch.randn(1000, 10, generator=g)
    logits[torch.arange(1000), labels] += 3
    return calibrate(torch.rand(1000, generator=g), logits.log_softmax(1), labels, 200, 0.01, g)


def test_quarantine_drops_flagged_pictures_or_the_suspicious_pair():
    cal = _calibration()
    log_probs = torch.full((4, 10), -5.0)
    log_probs[:, 3] = -0.01  # the reference model thinks every image is a cat
    labels = torch.tensor([3, 5, 5, 3])  # two cats labelled dog
    picture = torch.tensor([0.0, 0.0, 0.0, 10.0])  # image 3 looks strange
    evidence = Evidence(picture, log_probs)
    flip = quarantine_mask(evidence, labels, cal, "label_flip", (5, 3))
    assert flip.tolist() == [False, True, True, False]
    trigger = quarantine_mask(evidence, labels, cal, "backdoor", (5, 3))
    assert trigger[3] and not trigger[0]


def _tiny_monitor(device):
    torch.manual_seed(0)
    cal = _calibration()
    classifier = AttackClassifier(len(FEATURES))
    return Monitor([ConvVAE()], [ReferenceCNN()], cal, classifier.to(device), window=2)


def test_monitor_gives_a_verdict_every_window():
    monitor = _tiny_monitor(torch.device("cpu"))
    images, labels = torch.randn(8, 3, 32, 32), torch.arange(8) % 10
    evidence = monitor.evidence(images)
    assert evidence.picture.shape == (8,) and evidence.log_probs.shape == (8, 10)
    assert monitor.add(evidence, labels) is None
    verdict = monitor.add(evidence, labels)
    assert verdict.attack_type in ATTACK_TYPES
    assert sum(verdict.probabilities) == pytest.approx(1.0)


class _AlwaysRollback:
    def decide(self, state):
        return Action.ROLLBACK if not state.acted else Action.CONTINUE


def test_episode_runs_counts_poison_and_carries_out_actions():
    device = torch.device("cpu")
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=(64, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(64) % 10
    poisoned = np.zeros(64, dtype=bool)
    poisoned[::4] = True
    scenario = Scenario("test", raw, labels, poisoned)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    log = run_episode(
        model,
        optimizer,
        raw,
        labels,
        scenario,
        _tiny_monitor(device),
        _AlwaysRollback(),
        batches=8,
        onset=2,
        batch_size=8,
        measure=lambda m: {"ok": 1.0},
        measure_every=4,
        device=device,
        seed=0,
    )
    assert len(log.windows) == 4  # 8 batches / window of 2
    assert [c["batch"] for c in log.curve] == [0, 4, 8]
    # The 6 batches after the onset are rows 16-63 of the episode's shuffled order.
    streamed = poisoned[np.random.default_rng(0).permutation(64)[16:]].sum()
    assert log.poison_trained + log.poison_dropped == streamed
    assert log.actions and log.actions[0][1] == "rollback"


def test_quarantine_everything_baseline_ignores_the_attack_type():
    policy = QuarantineEverything()
    assert policy.decide(_state("benign_shift")) == Action.QUARANTINE
    assert policy.decide(_state("backdoor", acted=True)) == Action.CONTINUE
    assert policy.decide(_state("clean", streak=0)) == Action.CONTINUE

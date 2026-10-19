from types import SimpleNamespace

import numpy as np
from torch.utils.data import Subset

from mlecho.bench.attacks.backdoor import (
    BackdoorConfig,
    add_trigger,
    make_triggered_test_set,
    poison_train_set,
)

CONFIG = BackdoorConfig(target_class=0, poison_rate=0.25, patch_size=4)


def _fake_cifar(n: int = 40) -> SimpleNamespace:
    """A tiny stand-in for CIFAR-10: black images, labels cycling 0-9."""
    return SimpleNamespace(
        data=np.zeros((n, 32, 32, 3), dtype=np.uint8),
        targets=[i % 10 for i in range(n)],
    )


def test_trigger_paints_only_the_corner_and_keeps_the_original():
    images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    stamped = add_trigger(images, CONFIG)

    assert (stamped[:, -4:, -4:] == CONFIG.patch_color).all()  # corner is yellow
    assert stamped[:, :-4, :].sum() == 0  # rows above the corner untouched
    assert stamped[:, :, :-4].sum() == 0  # columns left of the corner untouched
    assert images.sum() == 0  # the input was copied, not changed


def test_poisoning_changes_the_right_images_only():
    fake = _fake_cifar()
    train_set = Subset(fake, indices=list(range(20)))  # positions 20-39 act as val

    poisoned = poison_train_set(train_set, CONFIG, seed=0)

    assert len(poisoned) == 5  # 25% of 20 train images
    assert all(i < 20 for i in poisoned)  # never touches val positions
    for i in poisoned:
        assert fake.targets[i] == CONFIG.target_class
        assert (fake.data[i, -4:, -4:] == CONFIG.patch_color).all()
    clean = [i for i in range(40) if i not in poisoned]
    assert all(fake.data[i].sum() == 0 for i in clean)


def test_poisoning_never_picks_images_already_in_the_target_class():
    fake = _fake_cifar()
    original = list(fake.targets)
    poisoned = poison_train_set(Subset(fake, list(range(40))), CONFIG, seed=1)
    assert all(original[i] != CONFIG.target_class for i in poisoned)


def test_poisoning_is_repeatable_with_the_same_seed():
    first = poison_train_set(Subset(_fake_cifar(), list(range(40))), CONFIG, seed=3)
    second = poison_train_set(Subset(_fake_cifar(), list(range(40))), CONFIG, seed=3)
    assert first == second


def test_triggered_test_set_drops_target_class_and_labels_rest_as_target():
    fake = _fake_cifar(n=20)  # labels 0-9 twice: two of them are the target class
    triggered = make_triggered_test_set(fake, CONFIG)

    assert len(triggered.targets) == 18
    assert set(triggered.targets) == {CONFIG.target_class}
    assert (triggered.data[:, -4:, -4:] == CONFIG.patch_color).all()
    assert fake.data.sum() == 0  # original test set unchanged

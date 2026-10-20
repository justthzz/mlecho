from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Subset

from mlecho.attacks.backdoor import (
    BackdoorConfig,
    add_trigger,
    add_trigger_tensor,
    make_triggered_test_set,
    poison_train_set,
)
from mlecho.attacks.labelflip import LabelFlipConfig, flip_labels
from mlecho.attacks.slowdrip import drip_backdoor

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


FLIP = LabelFlipConfig(source_class=3, target_class=5, poison_rate=0.05)


def test_label_flip_relabels_only_source_class_train_images():
    fake = _fake_cifar(n=200)  # 20 images of each class
    original = list(fake.targets)
    train_set = Subset(fake, indices=list(range(100)))  # 10 cats in train

    flipped = flip_labels(train_set, FLIP, seed=0)

    assert len(flipped) == 5  # 5% of 100 train images
    assert all(i < 100 for i in flipped)  # never touches val positions
    assert all(original[i] == FLIP.source_class for i in flipped)
    assert all(fake.targets[i] == FLIP.target_class for i in flipped)
    unchanged = [i for i in range(200) if i not in flipped]
    assert all(fake.targets[i] == original[i] for i in unchanged)


def test_label_flip_never_changes_pixels():
    fake = _fake_cifar(n=200)
    flip_labels(Subset(fake, list(range(200))), FLIP, seed=0)
    assert fake.data.sum() == 0


def test_label_flip_is_repeatable_with_the_same_seed():
    first = flip_labels(Subset(_fake_cifar(n=200), list(range(200))), FLIP, seed=4)
    second = flip_labels(Subset(_fake_cifar(n=200), list(range(200))), FLIP, seed=4)
    assert first == second


def test_label_flip_refuses_a_budget_bigger_than_the_source_class():
    too_much = LabelFlipConfig(poison_rate=0.5)  # 50 of 100 images, but only 10 cats
    with pytest.raises(ValueError):
        flip_labels(Subset(_fake_cifar(n=100), list(range(100))), too_much, seed=0)


MEAN, STD = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)


def test_tensor_trigger_matches_the_raw_pixel_trigger():
    raw = np.random.default_rng(0).integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)
    expected = (add_trigger(raw, CONFIG) / 255 - 0.5) / 0.25  # stamp, then normalise
    tensors = torch.tensor((raw / 255 - 0.5) / 0.25).permute(0, 3, 1, 2)  # normalise first
    stamped = add_trigger_tensor(tensors, CONFIG, MEAN, STD).permute(0, 2, 3, 1).numpy()
    assert np.allclose(stamped, expected)


def test_drip_poisons_the_right_share_and_reports_which():
    images, labels = torch.zeros(100, 3, 32, 32), torch.arange(100) % 10
    generator = torch.Generator().manual_seed(0)

    dripped, new_labels, mask = drip_backdoor(images, labels, 0.05, CONFIG, MEAN, STD, generator)

    assert int(mask.sum()) == 5
    assert (labels[mask] != CONFIG.target_class).all()  # never picks the target class
    assert (new_labels[mask] == CONFIG.target_class).all()
    assert torch.equal(new_labels[~mask], labels[~mask])  # clean ones unchanged
    assert torch.equal(dripped[~mask], images[~mask])
    assert not torch.equal(dripped[mask], images[mask])  # poisoned ones got the patch
    assert images.sum() == 0  # the input batch itself was not changed


def test_drip_with_rate_zero_changes_nothing():
    images, labels = torch.randn(10, 3, 32, 32), torch.arange(10)
    generator = torch.Generator().manual_seed(0)
    dripped, new_labels, mask = drip_backdoor(images, labels, 0.0, CONFIG, MEAN, STD, generator)
    assert not mask.any()
    assert torch.equal(dripped, images) and torch.equal(new_labels, labels)

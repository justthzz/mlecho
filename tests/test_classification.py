import numpy as np
import pytest
import torch

from mlecho.classification.classifier import ATTACK_TYPES, AttackClassifier, fit_classifier
from mlecho.classification.features import (
    FEATURES,
    calibrate,
    stream_rows,
    surprise,
    window_features,
)
from mlecho.classification.synthetic import (
    Shift,
    apply_shift,
    random_backdoor,
    random_flip,
    random_shift,
)


def _trusted(n: int = 2000, classes: int = 10, seed: int = 0):
    """Fake trusted scores: random picture scores, a reference model that is always right."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(n) % classes
    picture = torch.rand(n, generator=g)
    logits = torch.randn(n, classes, generator=g)
    logits[torch.arange(n), labels] += 3.0
    return picture, torch.log_softmax(logits, dim=1), labels


def test_surprise_is_minus_log_probability_of_the_given_label():
    log_probs = torch.log(torch.tensor([[0.5, 0.25, 0.25]]))
    assert surprise(log_probs, torch.tensor([1])).item() == pytest.approx(np.log(4))


def test_calibration_thresholds_flag_the_chosen_share():
    picture, log_probs, labels = _trusted()
    cal = calibrate(picture, log_probs, labels, 500, 0.01, torch.Generator().manual_seed(0))
    assert (picture > cal.picture_threshold).float().mean().item() == pytest.approx(0.01, abs=0.002)
    assert cal.reference_counts.sum().item() == len(labels)
    assert cal.mix_level > 0


def _window(picture, log_probs, labels, batches=4):
    return (
        picture.view(batches, -1),
        log_probs.view(batches, -1, log_probs.shape[1]),
        labels.view(batches, -1),
    )


def test_clean_window_features_look_normal():
    picture, log_probs, labels = _trusted()
    cal = calibrate(picture, log_probs, labels, 400, 0.05, torch.Generator().manual_seed(0))
    features = window_features(*_window(picture[:400], log_probs[:400], labels[:400]), cal)
    named = dict(zip(FEATURES, features.tolist(), strict=True))
    assert len(features) == len(FEATURES)
    assert 0.3 < named["picture_rate"] < 3  # about the clean share
    assert 0.5 < named["picture_median"] < 1.5
    assert named["label_top_label"] < 0.6  # surprising labels spread over the classes


def test_backdoor_window_flags_pictures_with_one_label_evenly_spread():
    picture, log_probs, labels = _trusted()
    cal = calibrate(picture, log_probs, labels, 400, 0.01, torch.Generator().manual_seed(0))
    picture, log_probs, labels = _window(picture[:400], log_probs[:400], labels[:400])
    picture, labels = picture.clone(), labels.clone()
    picture[:, :5] = 10.0  # 5 squared images in every batch...
    labels[:, :5] = 0  # ...all labelled class 0
    named = dict(
        zip(FEATURES, window_features(picture, log_probs, labels, cal).tolist(), strict=True)
    )
    assert named["picture_rate"] > 3
    assert named["picture_top_label"] > 0.8
    assert named["picture_dispersion"] < 1  # same count in every batch: steadier than chance
    assert named["picture_surprise"] > 0  # their labels don't fit the pictures


def test_random_backdoor_fits_inside_the_image():
    rng = np.random.default_rng(0)
    for _ in range(50):
        config = random_backdoor(rng, 32, 10)
        row, col = config.position
        assert 2 <= config.patch_size <= 6
        assert row + config.patch_size <= 32 and col + config.patch_size <= 32
        assert 0 <= config.target_class < 10


def test_shifts_change_pixels_but_keep_the_shape():
    images = np.random.default_rng(0).integers(50, 200, size=(3, 8, 8, 3), dtype=np.uint8)
    brighter = apply_shift(images, Shift("brightness", 1.3))
    assert brighter.shape == images.shape and brighter.dtype == np.uint8
    assert brighter.mean() > images.mean()
    assert apply_shift(images, Shift("blur", 1.0)).std() < images.std()
    shift = random_shift(np.random.default_rng(1))
    assert shift.kind in ("brightness", "contrast", "blur")


def test_random_flip_avoids_the_excluded_pairs():
    rng = np.random.default_rng(0)
    excluded = {(3, 5), (4, 7)}
    pairs = {random_flip(rng, 10, excluded) for _ in range(300)}
    assert not pairs & excluded
    assert all(s != t for s, t in pairs)


def test_classifier_learns_separable_types_and_gives_probabilities():
    g = torch.Generator().manual_seed(0)
    types = torch.arange(500) % len(ATTACK_TYPES)
    features = torch.randn(500, 3, generator=g) * 0.3 + types.view(-1, 1).float() * 10
    model = AttackClassifier(num_features=3)
    fit_classifier(model, features, types)
    probabilities = model.probabilities(features)
    assert probabilities.sum(dim=1) == pytest.approx(torch.ones(500))
    assert (probabilities.argmax(dim=1) == types).float().mean() > 0.95
    copy = AttackClassifier(num_features=3)
    copy.load_state_dict(model.state_dict())  # the feature scaling is saved too
    assert torch.equal(copy.probabilities(features), probabilities)


def test_stream_rows_repeat_images_more_than_plain_sampling():
    g = torch.Generator().manual_seed(0)
    rows = stream_rows(1000, 1000, g)
    assert rows.min() >= 0 and rows.max() < 1000
    # Two rounds of resampling leave fewer distinct images than one (about 50% vs 63%).
    assert len(rows.unique()) < 600

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mlecho.detection.label_check import (
    ReferenceCNN,
    agreement_counts,
    label_scores,
    pair_surprise,
    pair_threshold,
    train_reference_epoch,
)
from mlecho.detection.thresholds import count_threshold, score_threshold
from mlecho.detection.vae import (
    ConvVAE,
    VAEConfig,
    reconstruction_scores,
    train_vae_epoch,
    vae_loss,
)

SMALL = VAEConfig(channels=8, latent=8)


def _smooth_images(n: int, size: int = 32, seed: int = 0) -> torch.Tensor:
    """Plain images: each one a random colour that fades left to right (no sharp details)."""
    generator = torch.Generator().manual_seed(seed)
    colours = torch.rand(n, 3, 1, 1, generator=generator) * 2 - 1
    fade = torch.linspace(0.5, 1.0, size).view(1, 1, 1, size)
    return (colours * fade).expand(n, 3, size, size).clone()


def test_vae_rebuilds_images_of_the_same_shape_any_size():
    model = ConvVAE(SMALL).eval()
    for size in (32, 48):
        images = torch.zeros(2, 3, size, size)
        rebuilt, mean, log_var = model(images)
        assert rebuilt.shape == images.shape
        assert mean.shape == log_var.shape == (2, SMALL.latent)


def test_scores_are_one_per_image_and_repeatable():
    model = ConvVAE(SMALL)
    images = _smooth_images(5)
    first, second = reconstruction_scores(model, images), reconstruction_scores(model, images)
    assert first.shape == (5,)
    assert torch.equal(first, second)  # eval mode uses the mean code: no randomness


def test_loss_is_positive_and_training_lowers_it():
    torch.manual_seed(0)
    model = ConvVAE(SMALL)
    images = _smooth_images(64)
    loader = DataLoader(TensorDataset(images, torch.zeros(64)), batch_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = train_vae_epoch(model, loader, optimizer, torch.device("cpu"))
    for _ in range(20):
        last = train_vae_epoch(model, loader, optimizer, torch.device("cpu"))
    assert first > 0 and last < first
    assert vae_loss(images, *model(images)).item() > 0


def test_unusual_square_scores_higher_after_training_on_plain_images():
    torch.manual_seed(0)
    model = ConvVAE(SMALL)
    images = _smooth_images(128)
    loader = DataLoader(TensorDataset(images, torch.zeros(128)), batch_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(30):
        train_vae_epoch(model, loader, optimizer, torch.device("cpu"))

    plain = _smooth_images(8, seed=1)
    squared = plain.clone()
    squared[:, :, -6:, -6:] = 3.0  # a bright square in the corner, never seen in training
    assert (reconstruction_scores(model, squared) > reconstruction_scores(model, plain)).all()


def test_patch_score_reacts_more_than_mean_score_to_a_small_square():
    torch.manual_seed(0)
    model = ConvVAE(SMALL)
    plain = _smooth_images(4, seed=2)
    squared = plain.clone()
    squared[:, :, -4:, -4:] = 3.0
    mean_gain = reconstruction_scores(model, squared) / reconstruction_scores(model, plain)
    patch_gain = reconstruction_scores(model, squared, patch=4) / reconstruction_scores(
        model, plain, patch=4
    )
    assert (patch_gain > mean_gain).all()  # the square is diluted by a whole-image average


def test_score_threshold_leaves_the_requested_share_above():
    scores = torch.arange(1000, dtype=torch.float)  # 0 .. 999
    threshold = score_threshold(scores, 0.01)
    assert (scores > threshold).float().mean().item() == pytest.approx(0.01, abs=0.002)


def test_count_threshold_matches_chance_for_clean_groups():
    scores = torch.arange(1000, dtype=torch.float)
    threshold = score_threshold(scores, 0.01)  # each clean image flagged with chance ~1%
    needed = count_threshold(scores, threshold, 128, 0.01, torch.Generator().manual_seed(0))
    # Binomial(128, 0.01): P(>= 4) is about 3.9%, P(>= 5) about 0.9%
    assert needed == 5
    assert count_threshold(scores, threshold, 128, 0.0, torch.Generator().manual_seed(0)) > 5


def test_reference_model_any_size_and_label_scores_match_cross_entropy():
    model = ReferenceCNN(width=8)
    for size in (32, 40):
        assert model(torch.zeros(2, 3, size, size)).shape == (2, 10)
    images, labels = torch.randn(4, 3, 32, 32), torch.tensor([0, 3, 5, 9])
    scores = label_scores(model, images, labels)
    expected = torch.nn.functional.cross_entropy(model.eval()(images), labels, reduction="none")
    assert torch.allclose(scores, expected, atol=1e-6)


def test_label_check_scores_a_wrong_label_higher_after_training():
    torch.manual_seed(0)
    model = ReferenceCNN(width=8)
    images = _smooth_images(64)
    labels = (images[:, 0, 0, -1] > 0).long()  # class 1 = reddish images, class 0 = the rest
    loader = DataLoader(TensorDataset(images, labels), batch_size=16, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(15):
        train_reference_epoch(model, loader, optimizer, torch.device("cpu"))
    right = label_scores(model, images, labels)
    wrong = label_scores(model, images, 1 - labels)
    assert wrong.mean() > right.mean() and (wrong > right).float().mean() > 0.9


def test_agreement_counts_follow_label_and_prediction():
    counts = agreement_counts(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 0, 1]), 2)
    assert torch.equal(counts, torch.tensor([[2.0, 1.0], [0.0, 1.0]], dtype=torch.float64))


def test_pair_surprise_finds_the_flipped_pair_and_clean_groups_stay_quiet():
    generator = torch.Generator().manual_seed(0)

    def sample(n):  # a reference model that is wrong 20% of the time, to either other class
        labels = torch.randint(0, 3, (n,), generator=generator)
        shift = torch.randint(1, 3, (n,), generator=generator)
        wrong = torch.rand(n, generator=generator) < 0.2
        return torch.where(wrong, (labels + shift) % 3, labels), labels

    reference = agreement_counts(*sample(600), 3)  # a small trusted set, 200 per class
    level = pair_threshold(3, 0.05)
    clean_alarms = sum(pair_surprise(*sample(300), reference)[0] > level for _ in range(300))
    assert clean_alarms / 300 < 0.1  # about 5% or less, despite the small trusted set

    predictions, labels = sample(300)
    flip = (labels == 0) & (torch.rand(300, generator=generator) < 0.5)
    labels[flip] = 2  # half the class-0 images relabelled as class 2
    z, given, looks_like = pair_surprise(predictions, labels, reference)
    assert (given, looks_like) == (2, 0) and z > level


def test_pair_threshold_grows_with_more_pairs_and_stricter_rates():
    assert pair_threshold(10, 0.01) > pair_threshold(3, 0.01) > pair_threshold(3, 0.1)

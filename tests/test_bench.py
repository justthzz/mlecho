import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlecho.bench.datasets import cifar10_transform, split_indices
from mlecho.bench.evaluation import alarm_rates, image_metrics, roc_auc, simulate_flag_counts
from mlecho.bench.models import MaxPool2x2, SmallCNN
from mlecho.bench.online import online_step
from mlecho.bench.training import evaluate, predict, train_one_epoch


def test_split_has_right_sizes_and_no_overlap():
    train, val = split_indices(n=100, val_size=20, seed=0)
    assert len(train) == 80
    assert len(val) == 20
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(range(100))


def test_split_is_repeatable_with_same_seed():
    assert split_indices(100, 20, seed=7) == split_indices(100, 20, seed=7)
    assert split_indices(100, 20, seed=7) != split_indices(100, 20, seed=8)


def _fake_photo() -> Image.Image:
    pixels = np.random.default_rng(0).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def test_plain_transform_always_gives_the_same_tensor():
    photo, transform = _fake_photo(), cifar10_transform(augment=False)
    first = transform(photo)
    assert first.shape == (3, 32, 32)
    assert torch.equal(first, transform(photo))


def test_augmented_transform_varies_but_keeps_the_shape():
    photo, transform = _fake_photo(), cifar10_transform(augment=True)
    versions = [transform(photo) for _ in range(10)]
    assert all(v.shape == (3, 32, 32) for v in versions)
    assert any(not torch.equal(versions[0], v) for v in versions[1:])


def test_small_cnn_gives_one_score_per_class():
    model = SmallCNN(num_classes=10)
    images = torch.randn(4, 3, 32, 32)  # 4 fake CIFAR-sized images
    assert model(images).shape == (4, 10)


def test_small_cnn_width_controls_size_not_output_shape():
    narrow, wide = SmallCNN(width=16), SmallCNN(width=32)
    images = torch.randn(2, 3, 32, 32)
    assert narrow(images).shape == wide(images).shape == (2, 10)
    assert _weight_count(wide) > _weight_count(narrow)


def _weight_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _fake_loader(n: int = 32) -> DataLoader:
    images = torch.randn(n, 3, 32, 32)
    labels = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=8)


def test_training_changes_weights_and_evaluate_does_not():
    device = torch.device("cpu")
    model = SmallCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = _fake_loader()

    before = [p.clone() for p in model.parameters()]
    result = train_one_epoch(model, loader, loss_fn, optimizer, device)
    assert any(not torch.equal(b, a) for b, a in zip(before, model.parameters(), strict=True))
    assert result.loss > 0
    assert 0.0 <= result.accuracy <= 1.0

    after_training = [p.clone() for p in model.parameters()]
    evaluate(model, loader, loss_fn, device)
    assert all(torch.equal(b, a) for b, a in zip(after_training, model.parameters(), strict=True))


def test_predict_returns_one_guess_and_one_label_per_image():
    loader = _fake_loader(n=20)  # batches of 8, 8, 4
    predictions, truths = predict(SmallCNN(), loader, torch.device("cpu"))
    assert predictions.shape == truths.shape == (20,)
    assert int(predictions.min()) >= 0 and int(predictions.max()) <= 9
    assert torch.equal(truths, torch.cat([labels for _, labels in loader]))


def test_online_step_updates_weights_once_and_returns_the_loss():
    model = SmallCNN(width=8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    images, labels = torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))

    before = [p.clone() for p in model.parameters()]
    loss = online_step(model, optimizer, nn.CrossEntropyLoss(), images, labels)

    assert loss > 0
    assert any(not torch.equal(b, a) for b, a in zip(before, model.parameters(), strict=True))


def test_maxpool2x2_matches_pytorch_maxpool_and_supports_second_derivatives():
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    assert torch.equal(MaxPool2x2()(x), nn.MaxPool2d(2)(x))

    (first,) = torch.autograd.grad(MaxPool2x2()(x).pow(2).sum(), x, create_graph=True)
    (second,) = torch.autograd.grad(first.sum(), x)
    assert second.shape == x.shape


def test_small_cnn_group_norm_has_no_running_statistics_and_same_output_shape():
    model = SmallCNN(width=8, norm="group")
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert list(model.buffers()) == []  # nothing that training could quietly change
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 10)


def test_roc_auc_perfect_coin_flip_and_reversed():
    low, high = torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.7, 0.8, 0.9])
    assert roc_auc(low, high) == 1.0  # every poison above every clean image
    assert roc_auc(high, low) == 0.0
    assert roc_auc(torch.tensor([0.5, 0.5]), torch.tensor([0.5, 0.5])) == 0.5  # all tied
    # 1 of 4 clean-poison pairs ranks the poison higher
    assert roc_auc(torch.tensor([0.2, 0.6]), torch.tensor([0.1, 0.3])) == 0.25


def test_image_metrics_counts_recall_false_positives_and_precision():
    clean = torch.tensor([0.1, 0.2, 0.3, 0.9])  # 1 of 4 clean above 0.5
    poison = torch.tensor([0.8, 0.95])  # both above
    metrics = image_metrics(clean, poison, threshold=0.5)
    assert metrics == {"recall": 1.0, "fpr": 0.25, "precision": 2 / 3}


def test_simulated_stream_and_alarm_rates():
    population = torch.zeros(100)
    population[:10] = 1.0  # 10% of images score above 0.5
    generator = torch.Generator().manual_seed(0)
    counts = simulate_flag_counts(population, 0.5, 400, 20, generator)
    assert counts.float().mean().item() == pytest.approx(2.0, abs=0.3)
    dripped = simulate_flag_counts(torch.zeros(100), 0.5, 10, 20, generator, torch.ones(5), 3)
    assert (dripped == 3).all()  # exactly the 3 dripped images per batch are flagged
    assert alarm_rates(torch.tensor([0, 5, 0, 0]), 5, 2, 5) == (0.25, 0.5)

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlecho.bench.datasets import split_indices
from mlecho.bench.models import SmallCNN
from mlecho.bench.training import evaluate, train_one_epoch


def test_split_has_right_sizes_and_no_overlap():
    train, val = split_indices(n=100, val_size=20, seed=0)
    assert len(train) == 80
    assert len(val) == 20
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(range(100))


def test_split_is_repeatable_with_same_seed():
    assert split_indices(100, 20, seed=7) == split_indices(100, 20, seed=7)
    assert split_indices(100, 20, seed=7) != split_indices(100, 20, seed=8)


def test_small_cnn_gives_one_score_per_class():
    model = SmallCNN(num_classes=10)
    images = torch.randn(4, 3, 32, 32)  # 4 fake CIFAR-sized images
    assert model(images).shape == (4, 10)


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

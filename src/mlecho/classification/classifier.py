"""The attack classifier: which kind of poison is this window showing?

A small softmax (multinomial logistic) regression on the window features. It
is simple on purpose: few features, so a linear model is enough; it gives a
probability per type, which the response step uses as its confidence; and
its weights show which feature points to which attack.
"""

import torch
import torch.nn.functional as F
from torch import nn

ATTACK_TYPES = ("clean", "backdoor", "label_flip", "accumulative", "benign_shift")


class AttackClassifier(nn.Module):
    """Features -> one score per attack type. `fit_classifier` trains it."""

    def __init__(self, num_features: int, num_types: int = len(ATTACK_TYPES)):
        super().__init__()
        self.linear = nn.Linear(num_features, num_types)
        # Saved with the weights: every feature is rescaled to mean 0, spread 1.
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("std", torch.ones(num_features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear((features - self.mean) / self.std)

    @torch.no_grad()
    def probabilities(self, features: torch.Tensor) -> torch.Tensor:
        """(windows, types): the chance of each attack type, rows sum to 1."""
        return F.softmax(self(features), dim=-1)


def fit_classifier(
    model: AttackClassifier,
    features: torch.Tensor,
    types: torch.Tensor,
    l2: float = 1e-3,
    steps: int = 200,
) -> float:
    """Train on (windows, features) and their type numbers. Returns the final loss.

    Full-batch L-BFGS (fine for a few thousand windows); `l2` keeps the weights
    small so no single feature is trusted too much.
    """
    model.mean.copy_(features.mean(dim=0))
    model.std.copy_(features.std(dim=0).clamp_min(1e-6))
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=steps, line_search_fn="strong_wolfe")

    def loss_fn() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(model(features), types) + l2 * model.linear.weight.square().sum()
        loss.backward()
        return loss

    model.train()
    optimizer.step(loss_fn)
    model.eval()
    return loss_fn().item()

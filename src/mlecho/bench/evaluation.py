"""Numbers for judging a detector (lab only; the guard itself doesn't need them)."""

import torch


def roc_auc(clean_scores: torch.Tensor, poison_scores: torch.Tensor) -> float:
    """Chance that a random poisoned image scores higher than a random clean one.

    1.0 = the score always ranks poison above clean (perfect), 0.5 = no better
    than a coin flip, below 0.5 = poison tends to look *more* normal. Ties
    count as half. This equals the area under the ROC curve.
    """
    clean = clean_scores.flatten().double()
    poison = poison_scores.flatten().double()
    scores = torch.cat([clean, poison])
    order = scores.argsort()
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    for value in scores.unique():  # tied scores share their average rank
        tied = scores == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    poison_rank_sum = ranks[len(clean) :].sum()
    wins = poison_rank_sum - len(poison) * (len(poison) + 1) / 2
    return (wins / (len(clean) * len(poison))).item()

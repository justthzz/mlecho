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


def image_metrics(
    clean_scores: torch.Tensor, poison_scores: torch.Tensor, threshold: float
) -> dict[str, float]:
    """Recall, false-positive rate and precision of "score > threshold" for one attack.

    Precision is for the given mix: len(poison_scores) poisoned images among
    len(clean_scores) clean ones, as in the real poisoned training set.
    """
    recall = (poison_scores > threshold).float().mean().item()
    false_positive = (clean_scores > threshold).float().mean().item()
    caught, wrong = recall * len(poison_scores), false_positive * len(clean_scores)
    precision = caught / (caught + wrong) if caught + wrong else 0.0
    return {"recall": recall, "fpr": false_positive, "precision": precision}


def simulate_flag_counts(
    population: torch.Tensor,
    threshold: float,
    batches: int,
    batch_size: int,
    generator: torch.Generator,
    drip_pool: torch.Tensor | None = None,
    per_batch: int = 0,
) -> torch.Tensor:
    """Flagged images per batch, for `batches` random batches of a training stream.

    `population` holds one score per training image, as the stream would see it
    (poisoned images already carry their poisoned score). For slow-drip, the
    first `per_batch` images of every batch are replaced by random scores from
    `drip_pool`.
    """
    picks = torch.randint(len(population), (batches, batch_size), generator=generator)
    scores = population[picks]
    if per_batch:
        drip = torch.randint(len(drip_pool), (batches, per_batch), generator=generator)
        scores[:, :per_batch] = drip_pool[drip]
    return (scores > threshold).sum(dim=1)


def alarm_rates(
    counts: torch.Tensor, batch_needed: int, window: int, window_needed: int
) -> tuple[float, float]:
    """Share of batches with >= batch_needed flags, and of consecutive windows of
    `window` batches with >= window_needed flags in total."""
    batch_rate = (counts >= batch_needed).float().mean().item()
    windows = counts[: len(counts) // window * window].view(-1, window).sum(dim=1)
    return batch_rate, (windows >= window_needed).float().mean().item()

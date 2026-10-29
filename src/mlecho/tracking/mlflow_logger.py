"""Guard events into MLflow (optional: `pip install mlecho[mlflow]`).

Per window: the confidence and every type's probability as metrics (step =
batch number), so they can be plotted in the MLflow UI. Every action other
than "continue" also becomes a tag. On `close`, the full audit log is saved
to the run as `guard_events.json`.
"""

from mlecho.classification.classifier import ATTACK_TYPES


class MLflowLogger:
    """Logs into the active MLflow run (start one with `mlflow.start_run()` first)."""

    def __init__(self, prefix: str = "guard"):
        import mlflow  # only needed when this logger is used

        self.mlflow, self.prefix = mlflow, prefix
        self.events: list[dict] = []
        self.actions = 0

    def log(self, event: dict) -> None:
        self.events.append(event)
        p, step = self.prefix, event["batch"]
        metrics = {f"{p}_confidence": event["confidence"]}
        metrics |= {
            f"{p}_p_{t}": v for t, v in zip(ATTACK_TYPES, event["probabilities"], strict=True)
        }
        metrics[f"{p}_alarm"] = float(event["attack_type"] != "clean")
        self.mlflow.log_metrics(metrics, step=step)
        if event["action"] != "continue":
            self.actions += 1
            self.mlflow.set_tag(
                f"{p}_action_{self.actions}",
                f"batch {step}: {event['action']} ({event['attack_type']}, "
                f"confidence {event['confidence']:.2f})",
            )

    def close(self) -> None:
        self.mlflow.log_dict({"events": self.events}, f"{self.prefix}_events.json")

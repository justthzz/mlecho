"""MLEcho: Real-Time CNN-VAE and Reinforcement Learning Framework for
Detecting and Responding to Training Data Poisoning in MLOps Pipelines.

MLEcho watches training batches as they arrive. A CNN-VAE picture check and
a label check score every image; every few batches an attack classifier
names the likely poisoning type, and a response policy chooses an action
(continue, quarantine, rollback, retrain, escalate). Every event can be
logged to MLflow.

    from mlecho import Guard, GuardConfig

    guard = Guard.calibrate(trusted_images, trusted_labels, num_classes, device,
                            GuardConfig(pixel_mean=..., pixel_std=...))
    guard.watch(model, optimizer)
    for images, labels in loader:
        keep = guard.check(images, labels)
        loss = train_step(images[keep], labels[keep])
        guard.update(loss)

Research code from a Final Year Project; see the README for what is
evaluated and its limits.
"""

from mlecho.guard import Guard, GuardConfig

__version__ = "0.1.0"
__all__ = ["Guard", "GuardConfig", "__version__"]

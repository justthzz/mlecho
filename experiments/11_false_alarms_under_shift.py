"""Experiment 11: false alarms on clean data that changed for normal reasons (step 3.6).

Proposal 2.3.2: a poisoning alarm must not fire just because the data
legitimately changed (a new camera, different lighting). We take clean training
images, apply harmless changes (brighter, darker, more or less contrast or
colour, a slight hue shift, blur, sensor noise) and run both checks with the
thresholds calibrated on the original trusted data (as in experiments 08, 09).

No image here is poisoned, so every alarm is a false alarm. Targets on
unchanged data: 1% of images, batches and windows.

Run:   python experiments/11_false_alarms_under_shift.py
View:  mlflow ui --port 5050   (then open http://127.0.0.1:5050)
"""

import argparse
import re

import mlflow
import numpy as np
import torch
from common import (
    SHIFTS,
    add_detection_args,
    detection_data,
    label_check_scores,
    picture_scores,
    shifted,
)

from mlecho.bench.evaluation import alarm_rates, simulate_flag_counts
from mlecho.core import get_device, load_weights, seed_everything
from mlecho.detection.label_check import (
    ReferenceCNN,
    agreement_counts,
    pair_surprise,
    pair_threshold,
)
from mlecho.detection.thresholds import count_threshold, score_threshold
from mlecho.detection.vae import ConvVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_detection_args(parser)
    parser.add_argument("--vae", default="data/checkpoints/vae-c16-z32-seed42.pt")
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--reference-prefix", default="data/checkpoints/reference-w32-seed42-fold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--images", type=int, default=10000, help="clean training images used")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    generator = torch.Generator().manual_seed(args.seed)
    data = detection_data(args.seed, args.fit_size, args.gm_cache)
    targets = np.asarray(data.full.targets)
    vae = load_weights(ConvVAE(), args.vae, device)
    models = [
        load_weights(ReferenceCNN(), f"{args.reference_prefix}{f}.pt", device).eval()
        for f in range(args.folds)
    ]

    # Thresholds from the trusted data, exactly as in experiments 08 and 09.
    held = picture_scores(vae, data.full.data[data.held_out], args.patch, device)
    pic_t = score_threshold(held, args.image_rate)
    pic_batch = count_threshold(held, pic_t, args.batch_size, args.batch_rate, generator)
    trusted = np.concatenate([data.fit, data.held_out])
    fold_of_trusted = np.arange(len(trusted)) % args.folds
    ref_scores, ref_pred = label_check_scores(
        models, data.full.data[trusted], targets[trusted], fold_of_trusted, device
    )
    lab_t = score_threshold(ref_scores, args.image_rate)
    lab_batch = count_threshold(ref_scores, lab_t, args.batch_size, args.batch_rate, generator)
    window_images = args.batch_size * args.window
    pic_window = count_threshold(held, pic_t, window_images, args.batch_rate, generator)
    lab_window = count_threshold(ref_scores, lab_t, window_images, args.batch_rate, generator)
    reference_counts = agreement_counts(ref_pred, torch.as_tensor(targets[trusted]), 10)
    level = pair_threshold(10, args.batch_rate)

    rows = np.random.default_rng(args.seed).choice(len(data.train), args.images, replace=False)
    raw, labels = data.full.data[data.train[rows]], targets[data.train[rows]]
    which = np.random.default_rng(args.seed).integers(0, args.folds, len(rows))

    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name="false-alarms-under-shift"):
        mlflow.log_params(vars(args) | {"device": str(device)})
        print(
            f"{'change':22s} | picture: image  batch  window | label: image  batch  window"
            "  pattern(10)"
        )
        for name, change in SHIFTS.items():
            changed = shifted(raw, change)
            pic = picture_scores(vae, changed, args.patch, device)
            lab, pred = label_check_scores(models, changed, labels, which, device)
            results = {"picture_image": (pic > pic_t).float().mean().item()}
            pic_counts = simulate_flag_counts(pic, pic_t, args.streams, args.batch_size, generator)
            results["picture_batch"], results["picture_window"] = alarm_rates(
                pic_counts, pic_batch, args.window, pic_window
            )
            results["label_image"] = (lab > lab_t).float().mean().item()
            lab_counts = simulate_flag_counts(lab, lab_t, args.streams, args.batch_size, generator)
            results["label_batch"], results["label_window"] = alarm_rates(
                lab_counts, lab_batch, args.window, lab_window
            )
            picks = torch.randint(len(rows), (100, window_images), generator=generator)
            label_t = torch.as_tensor(labels)
            results["label_pattern"] = float(
                np.mean(
                    [pair_surprise(pred[p], label_t[p], reference_counts)[0] > level for p in picks]
                )
            )
            key = re.sub(r"[^a-z0-9]+", "_", name).strip("_")  # e.g. "blur_sigma_0_8"
            mlflow.log_metrics({f"{k}_{key}": v for k, v in results.items()})
            r = results
            print(
                f"{name:22s} | {r['picture_image']:6.1%} {r['picture_batch']:6.1%} "
                f"{r['picture_window']:6.1%} | {r['label_image']:6.1%} {r['label_batch']:6.1%} "
                f"{r['label_window']:6.1%}  {r['label_pattern']:6.1%}"
            )


if __name__ == "__main__":
    main()

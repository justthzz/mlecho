import numpy as np
import pytest
import torch

from mlecho.bench.baselines.drift import image_statistics, ks_alarm, ks_p_value
from mlecho.detection.label_check import psi


def test_image_statistics_are_channel_means_and_spreads():
    raw = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    raw[1, :, :, 0] = 200  # second image: red channel full
    stats = image_statistics(raw)
    assert stats.shape == (2, 6)
    assert stats[1, 0] == 200 and stats[0, 0] == 0 and (stats[:, 3:] == 0).all()


def test_ks_p_value_same_vs_shifted():
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(2000, generator=generator)
    b = torch.randn(2000, generator=generator)
    assert ks_p_value(a, b) > 0.05  # same distribution
    assert ks_p_value(a, b + 0.5) < 1e-6  # clearly shifted


def test_ks_alarm_and_its_false_alarm_rate():
    generator = torch.Generator().manual_seed(1)
    reference = torch.randn(3000, 3, generator=generator)
    alarms = [ks_alarm(reference, torch.randn(500, 3, generator=generator)) for _ in range(200)]
    assert sum(alarms) / 200 <= 0.03  # about 1% expected
    assert ks_alarm(reference, torch.randn(500, 3, generator=generator) + 0.5)


def test_psi_zero_for_same_mix_and_grows_with_change():
    same = torch.tensor([100.0, 100.0, 100.0])
    assert psi(same, same * 2) == pytest.approx(0.0)
    small = psi(same, torch.tensor([110.0, 100.0, 90.0]))
    large = psi(same, torch.tensor([50.0, 150.0, 100.0]))
    assert 0 < small < 0.1 < large

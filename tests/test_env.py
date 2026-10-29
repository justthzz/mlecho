import numpy as np
import pytest
import torch
from torch import nn

from mlecho.attacks.backdoor import BackdoorConfig
from mlecho.bench.env.practice import (
    KINDS,
    PracticeAttack,
    attack_batch,
    batch_pixels,
    cache_episode,
    random_practice_attack,
)
from mlecho.classification.synthetic import Shift
from mlecho.guard import Guard, GuardConfig
from mlecho.response.base import Action, State
from mlecho.response.ppo import OBSERVATION_SIZE, observation

gym = pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

DEVICE = torch.device("cpu")
TINY = GuardConfig(
    pixel_mean=(0.5, 0.5, 0.5),
    pixel_std=(0.25, 0.25, 0.25),
    folds=2,
    vae_epochs=1,
    reference_epochs=1,
    window=2,
    batch_size=16,
    made_up_configs=2,
    made_up_windows=3,
)


def _images(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8), np.arange(n) % 10


def test_attack_batch_poisons_as_each_attack_should():
    rng = np.random.default_rng(0)
    labels = np.arange(200) % 10
    trigger = BackdoorConfig(target_class=0)
    new, poisoned, changed = attack_batch(PracticeAttack("backdoor", 0, trigger, 0.5), labels, rng)
    assert poisoned.any() and (labels[poisoned] != 0).all() and (new[poisoned] == 0).all()
    assert (changed == poisoned).all()
    new, poisoned, _ = attack_batch(PracticeAttack("accumulative", 0, trigger, drip=3), labels, rng)
    assert poisoned.sum() == 3 and (new[poisoned] == 0).all()
    new, poisoned, changed = attack_batch(
        PracticeAttack("label_flip", 0, rate=0.05, flip=(3, 5)), labels, rng
    )
    assert (labels[poisoned] == 3).all() and (new[poisoned] == 5).all() and not changed.any()
    shift = PracticeAttack("benign_shift", 0, rate=0.5, shift=Shift("brightness", 1.3))
    new, poisoned, changed = attack_batch(shift, labels, rng)
    assert changed.any() and not poisoned.any() and (new == labels).all()
    raw, _ = _images(200)
    assert (batch_pixels(shift, raw, changed)[~changed] == raw[~changed]).all()


def test_random_practice_attacks_keep_out_the_test_pair():
    rng = np.random.default_rng(0)
    for kind in KINDS:
        attack = random_practice_attack(rng, kind, 32, 10, (5, 10), excluded={(3, 5)})
        assert attack.kind == kind and 5 <= attack.onset < 10
    flips = {
        random_practice_attack(rng, "label_flip", 32, 10, (5, 10), {(3, 5)}).flip
        for _ in range(300)
    }
    assert (3, 5) not in flips


def test_observation_is_ten_numbers_in_range():
    state = State("backdoor", 0.8, (0.1, 0.8, 0.05, 0.03, 0.02), 9, True, False, -3.0)
    obs = observation(state)
    assert obs.shape == (OBSERVATION_SIZE,) and obs.dtype == np.float32
    assert obs.min() >= -1 and obs.max() <= 1 and obs[6] == 1.0 and obs[9] == -1.0


@pytest.fixture(scope="module")
def tiny_world(tmp_path_factory):
    raw, labels = _images(120)
    guard = Guard.calibrate(raw, labels, 10, DEVICE, TINY)
    train_raw, train_labels = _images(160, seed=1)
    attack = PracticeAttack("backdoor", 2, BackdoorConfig(target_class=0), rate=0.3)
    episode = cache_episode(
        attack,
        train_raw,
        train_labels,
        guard.monitor,
        batches=8,
        batch_size=16,
        device=DEVICE,
        seed=0,
    )
    return guard, episode, train_raw


def test_cache_episode_stores_every_batch(tiny_world):
    _, episode, _ = tiny_world
    assert episode.rows.shape == episode.labels.shape == (8, 16)
    assert episode.picture.shape == (8, 16) and episode.log_probs.shape == (8, 16, 10)
    assert not episode.poisoned[:2].any() and episode.poisoned[2:].any()


def test_env_runs_an_episode_and_passes_the_gymnasium_check(tiny_world):
    from gymnasium.utils.env_checker import check_env

    from mlecho.bench.env.env import ACTIONS, ResponseEnv

    guard, episode, train_raw = tiny_world
    val_raw, val_labels = _images(50, seed=2)
    env = ResponseEnv(
        [episode],
        train_raw,
        guard.monitor.calibration,
        guard.monitor.classifier,
        nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10)),
        val_raw,
        val_labels,
        DEVICE,
        window=2,
        measure_images=50,
    )
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    steps, done = 0, False
    while not done:
        obs, reward, done, _, info = env.step(ACTIONS.index(Action.ROLLBACK) if steps == 0 else 0)
        assert np.isfinite(reward) and reward <= 0
        steps += 1
    assert steps == 4  # 8 batches / windows of 2
    assert info["poison_trained"] + info["clean_dropped"] >= 0


def test_ppo_policy_loads_and_decides(tiny_world, tmp_path):
    from stable_baselines3 import PPO

    from mlecho.bench.env.env import ResponseEnv
    from mlecho.response.ppo import PPOPolicy

    guard, episode, train_raw = tiny_world
    val_raw, val_labels = _images(50, seed=2)
    env = ResponseEnv(
        [episode],
        train_raw,
        guard.monitor.calibration,
        guard.monitor.classifier,
        nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10)),
        val_raw,
        val_labels,
        DEVICE,
        window=2,
        measure_images=50,
    )
    PPO("MlpPolicy", env, n_steps=8, batch_size=8, device="cpu").policy.save(tmp_path / "agent.pt")
    policy = PPOPolicy(tmp_path / "agent.pt")
    state = State("clean", 0.9, (0.9, 0.025, 0.025, 0.025, 0.025), 0, False, False, 0.0)
    assert policy.decide(state) in list(Action)

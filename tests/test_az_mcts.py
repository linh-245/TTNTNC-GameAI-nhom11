"""Test bổ sung cho AZ-MCTS (cấp 7): nước đi hợp lệ, chạy được với checkpoint thật,
và có mặt trong menu của giao diện Pygame.

Bổ sung cho tests/test_az_mcts.py (kiểm tra logic tìm kiếm bằng mạng giả).
"""
import os

import numpy as np
import pytest

pytest.importorskip("torch")

import agents.az_mcts_agent as az
from game.connect4_env import ConnectFourEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "weights", "az_iter_10.pt")


def _play_random_moves(n, seed=0):
    rng = np.random.default_rng(seed)
    env = ConnectFourEnv()
    env.reset()
    for _ in range(n):
        env.step(int(rng.choice(env.validinputs)))
    return env


def test_returns_legal_move_with_fake_net(monkeypatch):
    """Với mạng giả (prior đều, value 0), agent luôn trả về cột còn đi được."""
    monkeypatch.setattr(az, "predict", lambda model, env: (np.ones(7) / 7, 0.0))
    monkeypatch.setattr(az, "_get_model", lambda ckpt: None)
    for n in (0, 5, 12):
        env = _play_random_moves(n, seed=n)
        move = az.az_mcts_agent(env, n_simulations=50)
        assert move in env.validinputs


@pytest.mark.skipif(not os.path.exists(CKPT), reason="chưa có weights/az_iter_10.pt")
def test_real_checkpoint_returns_legal_move():
    """Với checkpoint thật, agent chạy được và trả về nước hợp lệ."""
    import inspect
    kw = {"n_simulations": 30}
    if "ckpt" in inspect.signature(az.az_mcts_agent).parameters:
        kw["ckpt"] = CKPT
    env = _play_random_moves(6, seed=1)
    move = az.az_mcts_agent(env, **kw)
    assert move in env.validinputs


@pytest.mark.skipif(not os.path.exists(CKPT), reason="chưa có weights/az_iter_10.pt")
def test_ui_menu_lists_az(monkeypatch):
    """Giao diện Pygame phải có mục AZ-MCTS khi có checkpoint."""
    pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from ui import pygame_ui
    assert any(name.startswith("AZ-MCTS") for name in pygame_ui.AGENT_NAMES)
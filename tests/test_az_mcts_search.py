"""Kiểm tra logic tìm kiếm của AZ-MCTS (cấp 7) bằng mạng giả: prior đều, value 0.
Chỉ có kết quả thắng/thua ở nút kết thúc dẫn đường, nên agent phải tự thấy nước thắng ngay
và nước phải chặn. Lỗi dấu (đảo Q, đảo giá trị theo độ sâu) làm các test này sai."""
import numpy as np
import pytest

pytest.importorskip("torch")

import agents.az_mcts_agent as az
from game.connect4_env import ConnectFourEnv


@pytest.fixture(autouse=True)
def fake_net(monkeypatch):
    monkeypatch.setattr(az, "predict", lambda model, env: (np.ones(7) / 7, 0.0))
    monkeypatch.setattr(az, "_get_model", lambda ckpt: None)


def pos(moves):
    env = ConnectFourEnv()
    for m in moves:
        env.step(m)
    return env


def test_takes_immediate_win():
    env = pos([1, 2, 1, 2, 1, 2])          # đỏ có 3 quân cột 1
    assert az.az_mcts_agent(env, n_simulations=200) == 1


def test_blocks_vertical_threat():
    env = pos([2, 1, 4, 1, 6, 1])          # vàng có 3 quân cột 1, đỏ không thắng được ngay
    assert az.az_mcts_agent(env, n_simulations=400) == 1


def test_blocks_horizontal_threat():
    env = pos([1, 4, 1, 5, 2, 6, 7, 7])    # vàng có 4, 5, 6 ở hàng dưới: phải chặn cột 3
    assert az.az_mcts_agent(env, n_simulations=400) == 3

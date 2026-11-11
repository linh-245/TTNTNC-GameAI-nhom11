"""Test phần học máy (mã hóa trạng thái). Chạy: python -m pytest tests/test_learning.py -v"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from game.connect4_env import ConnectFourEnv
from training.encode import encode_state, legal_mask, N_CHANNELS, ROWS, COLS


def make(moves):
    env = ConnectFourEnv(); env.reset()
    for m in moves:
        env.step(m)
    return env


def test_shape_and_dtype():
    x = encode_state(make([]))
    assert x.shape == (N_CHANNELS, ROWS, COLS) == (3, 6, 7)
    assert x.dtype == np.float32


def test_empty_board():
    x = encode_state(make([]))
    assert x[0].sum() == 0 and x[1].sum() == 0
    assert (x[2][0] == 1).all() and x[2][1:].sum() == 0


def test_perspective_flips_with_turn():
    env = make([4])  # đỏ đánh cột 4, tới lượt vàng
    x = encode_state(env)
    assert x[0].sum() == 0          # vàng chưa có quân
    assert x[1][0, 3] == 1          # quân đỏ là của đối thủ
    assert x[1].sum() == 1


def test_stacking_and_legal_plane():
    env = make([4, 4])  # đỏ ở hàng 0, vàng ở hàng 1, tới lượt đỏ
    x = encode_state(env)
    assert x[0][0, 3] == 1 and x[1][1, 3] == 1
    assert x[2][2, 3] == 1 and x[2][0, 3] == 0


def test_full_column_is_not_legal():
    env = make([1] * 6)  # cột 1 đầy, không ai thắng
    x = encode_state(env)
    assert x[2][:, 0].sum() == 0
    mask = legal_mask(env)
    assert mask[0] == 0 and mask.sum() == 6


def test_encode_does_not_change_env():
    env = make([4, 3, 4])
    before = env.state.copy()
    turn = env.turn
    encode_state(env)
    assert (env.state == before).all() and env.turn == turn
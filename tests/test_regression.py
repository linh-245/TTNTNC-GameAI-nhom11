"""Test hồi quy cho các lỗi đã sửa. Chạy: python -m pytest tests/ -v"""
import os, sys, random
from functools import partial
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.connect4_env import ConnectFourEnv
from game.game_utils import one_conn_game, run_games
from agents.random_agent import conn_random
from agents.heuristic_agent import conn_think1, conn_think2, conn_think3
from agents.mcts_agent import mcts_agent
from agents.alphabeta_agent import alphabeta_agent


def make(moves):
    env = ConnectFourEnv(); env.reset()
    for m in moves:
        env.step(m)
    return env


# Vàng có 3 quân dọc ở cột 1, đến lượt đỏ -> phải chặn cột 1
BLOCK_POS = [2, 1, 3, 1, 2, 1]
# Đỏ có 3 quân dọc ở cột 4, đến lượt đỏ -> thắng ngay ở cột 4
WIN_POS = [4, 1, 4, 1, 4, 2]


def test_one_conn_game_reward_is_from_player1_view():
    always_col = lambda env: 1
    # P1 xếp 4 quân dọc cột 4 (P2 đi cột 1 và 2 xen kẽ) -> P1 thắng => +1
    seq1 = iter([4, 4, 4, 4]); seq2 = iter([1, 2, 1, 2])
    r, _ = one_conn_game(lambda e: next(seq1), lambda e: next(seq2))
    assert r == 1
    # P2 thắng => -1 (trước đây hàm trả +1 cho cả hai trường hợp)
    seq1 = iter([1, 2, 5, 7]); seq2 = iter([4, 4, 4, 4])
    r, _ = one_conn_game(lambda e: next(seq1), lambda e: next(seq2))
    assert r == -1


def test_run_games_distinguishes_strong_from_weak():
    random.seed(0)
    res = run_games(conn_think2, conn_random, n_games=40)
    assert res['score'] > 0.7, res


def test_think3_blocks_and_wins():
    assert conn_think3(make(BLOCK_POS)) == 1
    assert conn_think3(make(WIN_POS)) == 4


def test_think3_beats_think2_or_ties_not_loses_all():
    random.seed(1)
    res = run_games(conn_think3, conn_random, n_games=40)
    assert res['score'] > 0.8, res


def test_mcts_blocks_and_wins():
    random.seed(0)
    assert mcts_agent(make(BLOCK_POS), n_simulations=400) == 1
    assert mcts_agent(make(WIN_POS), n_simulations=400) == 4


def test_mcts_beats_random():
    random.seed(0)
    res = run_games(partial(mcts_agent, n_simulations=100), conn_random, n_games=10)
    assert res['score'] >= 0.8, res


def test_alphabeta_blocks_and_wins():
    assert alphabeta_agent(make(BLOCK_POS), depth=3) == 1
    assert alphabeta_agent(make(WIN_POS), depth=3) == 4

"""
Test cho các agent tìm kiếm (M4)
================================
Alpha-Beta: thắng ngay, chặn thế thua, bật/tắt cắt tỉa cho cùng điểm gốc,
            budget_ms trả nước hợp lệ.
MCTS      : thắng ngay, chặn, cùng seed cùng nước, thắng Random trong 10 ván.

Chạy: python -m pytest tests/test_search.py -q
"""

import os
import sys
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from game.connect4_env import ConnectFourEnv
from agents.alphabeta_agent import alphabeta_agent, _negamax
from agents.mcts_agent import mcts_agent

INF = float('inf')


# ─────────────────────────────────────────────────────────────────────────────
# Dựng thế cờ bằng chuỗi nước đi (cột 1-7). Người đi trước đi nước lẻ, người đi sau
# đi nước chẵn, nên không phụ thuộc vào nhãn màu của env.
# ─────────────────────────────────────────────────────────────────────────────

def play(seq):
    env = ConnectFourEnv()
    for m in seq:
        env.step(m)
    assert not env.done
    return env


# Người sắp đi có 3 quân chồng ở cột 1 -> thắng ngay ở cột 1
WIN_NOW = [1, 2, 1, 2, 1, 3]
# Đối thủ có 3 quân chồng ở cột 2, người sắp đi không có nước thắng -> phải chặn cột 2
BLOCK_VERTICAL = [1, 2, 3, 2, 5, 2]
# Đối thủ có 3 quân hàng đáy ở cột 4-5-6 (cột 3 đã bị chặn) -> phải chặn cột 7
BLOCK_HORIZONTAL = [3, 4, 1, 5, 1, 6]

BLOCK_CASES = [(BLOCK_VERTICAL, 2), (BLOCK_HORIZONTAL, 7)]


def sample_positions(plies_list=(0, 3, 7, 11, 15, 19), seed=0):
    """Các thế cờ chưa kết thúc, sinh bằng nước ngẫu nhiên có seed cố định."""
    rng = random.Random(seed)
    out = []
    for plies in plies_list:
        while True:
            env = ConnectFourEnv()
            for _ in range(plies):
                env.step(rng.choice(env.validinputs))
                if env.done:
                    break
            if not env.done:
                out.append(env)
                break
    return out


def play_vs_random(agent_fn, n_games=10, seed=0):
    """
    agent_fn(env, game_idx, ply) -> cột. Chia đều đi trước/đi sau.
    Trả về số ván agent thắng. Dùng random.Random riêng cho đối thủ Random.
    """
    rng = random.Random(seed)
    wins = 0
    for g in range(n_games):
        env = ConnectFourEnv()
        agent_first = (g % 2 == 0)
        ply = 0
        agent_won = False
        while not env.done:
            agent_to_move = ((ply % 2 == 0) == agent_first)
            if agent_to_move:
                move = agent_fn(env.copy(), g, ply)
            else:
                move = rng.choice(env.validinputs)
            assert move in env.validinputs
            env.step(move)
            if env.done and env.reward != 0 and agent_to_move:
                agent_won = True
            ply += 1
        wins += agent_won
    return wins


# ─────────────────────────────────────────────────────────────────────────────
# Alpha-Beta
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_alphabeta_takes_immediate_win(depth):
    assert alphabeta_agent(play(WIN_NOW), depth=depth) == 1


@pytest.mark.parametrize("seq,expected", BLOCK_CASES)
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_alphabeta_blocks_losing_threat(seq, expected, depth):
    assert alphabeta_agent(play(seq), depth=depth) == expected


@pytest.mark.parametrize("use_ordering", [True, False])
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_pruning_on_off_same_root_score_and_move(depth, use_ordering):
    """Cắt tỉa không được đổi điểm ở gốc (cũng như nước được chọn)."""
    for env in sample_positions():
        v_pruned = _negamax(env, depth, -INF, INF, True, True, use_ordering)
        v_plain = _negamax(env, depth, -INF, INF, True, False, use_ordering)
        assert v_pruned == v_plain

        m_pruned = alphabeta_agent(env, depth=depth, use_pruning=True,
                                   use_ordering=use_ordering)
        m_plain = alphabeta_agent(env, depth=depth, use_pruning=False,
                                  use_ordering=use_ordering)
        assert m_pruned == m_plain


def test_ordering_on_off_same_root_score():
    """Sắp xếp nước đi cũng không được đổi điểm gốc."""
    for env in sample_positions():
        assert (_negamax(env, 3, -INF, INF, True, True, True) ==
                _negamax(env, 3, -INF, INF, True, True, False))


def test_pruning_reduces_nodes_and_counter_works():
    env = ConnectFourEnv()
    alphabeta_agent(env, depth=4, use_pruning=False)
    nodes_plain = alphabeta_agent.last_stats["nodes"]
    alphabeta_agent(env, depth=4, use_pruning=True)
    nodes_pruned = alphabeta_agent.last_stats["nodes"]
    assert 0 < nodes_pruned < nodes_plain


@pytest.mark.parametrize("budget_ms", [0, 1, 50])
def test_alphabeta_budget_returns_valid_move_in_time(budget_ms):
    for env in sample_positions(plies_list=(0, 8, 16)):
        t0 = time.perf_counter()
        move = alphabeta_agent(env, depth=12, budget_ms=budget_ms)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert move in env.validinputs
        assert elapsed_ms < budget_ms + 200          # dư địa cho máy CI chậm
        stats = alphabeta_agent.last_stats
        assert 0 <= stats["depth"] <= 12
        assert {"nodes", "depth", "time_ms", "timed_out"} <= set(stats)


def test_alphabeta_budget_still_takes_immediate_win():
    assert alphabeta_agent(play(WIN_NOW), depth=5, budget_ms=0) == 1


def test_alphabeta_default_signature_unchanged():
    """Gọi theo chữ ký cũ (UI/notebook) vẫn chạy."""
    env = play(WIN_NOW)
    assert alphabeta_agent(env) == 1
    assert alphabeta_agent(env, 3) == 1
    assert alphabeta_agent(env, 3, True) == 1


def test_alphabeta_beats_random():
    wins = play_vs_random(lambda env, g, ply: alphabeta_agent(env, depth=3),
                          n_games=10, seed=1)
    assert wins >= 9


# ─────────────────────────────────────────────────────────────────────────────
# MCTS
# ─────────────────────────────────────────────────────────────────────────────

def test_mcts_takes_immediate_win():
    env = play(WIN_NOW)
    for seed in range(5):
        assert mcts_agent(env, n_simulations=800, seed=seed) == 1


@pytest.mark.parametrize("seq,expected", BLOCK_CASES)
def test_mcts_blocks(seq, expected):
    env = play(seq)
    for seed in range(5):
        assert mcts_agent(env, n_simulations=800, seed=seed) == expected


def test_mcts_same_seed_same_move():
    for env in sample_positions():
        moves = {mcts_agent(env, n_simulations=300, seed=7) for _ in range(3)}
        assert len(moves) == 1


def test_mcts_does_not_touch_global_random():
    env = play(BLOCK_VERTICAL)
    random.seed(123)
    expected = random.random()
    random.seed(123)
    mcts_agent(env, n_simulations=100, seed=5)
    assert random.random() == expected


def test_mcts_runs_exact_n_simulations_without_budget():
    mcts_agent(play(BLOCK_VERTICAL), n_simulations=123, seed=0)
    stats = mcts_agent.last_stats
    assert stats["sims"] == 123
    assert stats["latency_ms"] >= 0


@pytest.mark.parametrize("budget_ms", [0, 1, 50])
def test_mcts_budget_valid_move_and_stats(budget_ms):
    env = play(BLOCK_VERTICAL)
    t0 = time.perf_counter()
    move = mcts_agent(env, n_simulations=10**9, budget_ms=budget_ms, seed=3)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert move in env.validinputs
    assert elapsed_ms < budget_ms + 200
    stats = mcts_agent.last_stats
    assert set(stats) == {"sims", "latency_ms"}
    if budget_ms >= 50:
        assert stats["sims"] > 0


def test_mcts_beats_random_in_10_games():
    wins = play_vs_random(
        lambda env, g, ply: mcts_agent(env, n_simulations=300, seed=1000 * g + ply),
        n_games=10, seed=2)
    assert wins >= 9

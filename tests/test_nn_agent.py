"""
Test cho agents/nn_agent.py.
Dùng một mạng nhỏ khởi tạo ngẫu nhiên (chưa huấn luyện) chỉ để kiểm tra agent chạy đúng
và luôn đi nước hợp lệ. Không kiểm tra sức mạnh chơi.
Tự bỏ qua nếu máy không có torch (ví dụ CI chưa cài torch).
"""
import os
import random
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.nn_agent import nn_agent
from game.connect4_env import ConnectFourEnv
from training.model import PolicyValueNet


@pytest.fixture
def ckpt(tmp_path):
    """Lưu một mạng nhỏ chưa huấn luyện theo đúng định dạng checkpoint của train_sl.py."""
    torch.manual_seed(0)
    model = PolicyValueNet(channels=8, n_blocks=1)
    path = tmp_path / "tiny.pt"
    torch.save({"model": model.state_dict(),
                "config": {"channels": 8, "blocks": 1}}, path)
    return str(path)


def new_env():
    env = ConnectFourEnv()
    env.reset()
    return env


def test_returns_column_in_range(ckpt):
    move = nn_agent(new_env(), checkpoint=ckpt)
    assert isinstance(move, int)
    assert 1 <= move <= 7


def test_never_picks_full_column(ckpt):
    env = new_env()
    for _ in range(6):          # xen kẽ hai màu trong cột 1 nên không ai thắng, cột đầy
        env.step(1)
    assert 1 not in env.validinputs
    assert nn_agent(env, checkpoint=ckpt) != 1


def test_plays_only_legal_moves_in_full_games(ckpt):
    rng = random.Random(0)
    for g in range(10):
        env = new_env()
        agent_turn = "red" if g % 2 == 0 else "yellow"
        while not env.done:
            if env.turn == agent_turn:
                move = nn_agent(env, checkpoint=ckpt)
            else:
                move = rng.choice(env.validinputs)
            assert move in env.validinputs
            env.step(move)


def test_last_stats_updated(ckpt):
    nn_agent(new_env(), checkpoint=ckpt)
    stats = nn_agent.last_stats
    assert "latency_ms" in stats and "value" in stats
    assert -1.0 <= stats["value"] <= 1.0


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        nn_agent(new_env(), checkpoint=str(tmp_path / "khong_co.pt"))
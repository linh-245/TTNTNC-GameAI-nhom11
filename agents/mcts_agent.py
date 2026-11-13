import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import random
import time
from typing import Optional, List, Dict
from game.connect4_env import ConnectFourEnv

"""
Cấp 5: Monte Carlo Tree Search (MCTS/UCT)
==========================================
Theo sách Liu (2024), Ch. 8.

Triển khai MCTS với UCB1 (Upper Confidence Bound for Trees):
  UCT = W/N + c * sqrt(ln(N_parent) / N)

Bốn pha:
  1. Selection: chọn node theo UCT
  2. Expansion: mở rộng node chưa được thăm hết
  3. Simulation: rollout ngẫu nhiên đến kết thúc
  4. Backpropagation: cập nhật W, N ngược lên root

Tùy chọn:
  - seed      : dùng random.Random(seed) riêng, không đụng random toàn cục
  - budget_ms : có giá trị thì lặp mô phỏng đến hết giờ (bỏ qua n_simulations);
                None thì chạy đủ n_simulations
  - mcts_agent.last_stats = {"sims", "latency_ms"}
"""


class MCTSNode:
    """Một node trong cây MCTS."""

    __slots__ = ['env', 'parent', 'action', 'children',
                 'untried_actions', 'wins', 'visits', 'turn', 'mover']

    def __init__(self, env: ConnectFourEnv, parent: Optional['MCTSNode'] = None,
                 action: Optional[int] = None, mover: Optional[str] = None,
                 copy_env: bool = True):
        # copy_env=False: node nhận quyền sở hữu env (env vừa được copy ở nơi gọi)
        self.env = env.copy() if copy_env else env
        self.mover = mover  # người đã đi nước dẫn tới node này
        self.parent = parent
        self.action = action  # hành động dẫn đến node này
        self.children: List['MCTSNode'] = []
        self.untried_actions: List[int] = env.get_valid_actions() if not env.done else []
        self.wins: float = 0.0
        self.visits: int = 0
        self.turn: str = env.turn  # lượt của người chơi TẠI node này

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        return self.env.done

    def uct_score(self, c: float = math.sqrt(2)) -> float:
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        exploit = self.wins / self.visits
        explore = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploit + explore

    def best_child(self, c: float = math.sqrt(2)) -> 'MCTSNode':
        return max(self.children, key=lambda node: node.uct_score(c))

    def expand(self, rng: random.Random) -> 'MCTSNode':
        """Mở rộng một node chưa được thăm bằng đối tượng random riêng (rng)."""
        action = self.untried_actions.pop(
            rng.randint(0, len(self.untried_actions) - 1)
        )
        new_env = self.env.copy()
        mover = new_env.turn
        new_env.step(action)
        # new_env đã là bản copy riêng -> giao thẳng cho node con, không copy lần nữa
        child = MCTSNode(new_env, parent=self, action=action, mover=mover,
                         copy_env=False)
        self.children.append(child)
        return child

    def update(self, result: float):
        """Cập nhật thống kê node."""
        self.visits += 1
        self.wins += result


def _simulate(env: ConnectFourEnv, root_turn: str, rng: random.Random) -> float:
    """
    Rollout ngẫu nhiên đến khi game kết thúc bằng rng riêng.
    
    Returns
    -------
    float : +1 nếu root_turn thắng, -1 nếu thua, 0 nếu hòa.
    """
    # Node đã kết thúc thì không cần copy: vòng lặp bên dưới không chạy, env không bị sửa
    sim_env = env if env.done else env.copy()
    while not sim_env.done:
        # Lấy danh sách action hợp lệ từ env và chọn ngẫu nhiên bằng rng riêng
        valid_actions = sim_env.get_valid_actions()
        if not valid_actions:
            break
        action = rng.choice(valid_actions)
        _, reward, done, _ = sim_env.step(action)

    if sim_env.reward == 0:
        return 0.0

    winner_turn = sim_env.turn  # Người đã thắng
    if winner_turn == root_turn:
        return 1.0
    return -1.0


def mcts_agent(env: ConnectFourEnv, n_simulations: int = 800,
               c: float = math.sqrt(2), budget_ms: Optional[int] = None,
               seed: Optional[int] = None) -> int:
    """
    Cấp 6 - MCTS/UCT agent.

    Parameters
    ----------
    n_simulations : số mô phỏng khi budget_ms=None.
    budget_ms     : có giá trị -> lặp mô phỏng đến hết giờ (không giới hạn bởi
                    n_simulations). Kết quả khi đó phụ thuộc tốc độ máy.
    seed          : cùng seed + cùng n_simulations (không budget) -> cùng nước đi.

    Side effect
    -----------
    mcts_agent.last_stats = {"sims": số mô phỏng đã chạy, "latency_ms": thời gian}
    """
    # Khởi tạo đối tượng random riêng theo seed (nếu có)
    rng = random.Random(seed)

    root_turn = env.turn
    root = MCTSNode(env)

    start_time = time.perf_counter()
    deadline = None if budget_ms is None else start_time + budget_ms / 1000.0
    sims = 0

    while True:
        if deadline is None:
            if sims >= n_simulations:
                break
        elif time.perf_counter() >= deadline:
            break

        node = root

        # 1. Selection
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(c)

        # 2. Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand(rng)

        # 3. Simulation
        result = _simulate(node.env, root_turn, rng)

        # 4. Backpropagation
        while node is not None:
            if node.mover == root_turn:
                node.update(result)
            else:
                node.update(-result)
            node = node.parent

        sims += 1

    # Chọn nước đi tốt nhất theo số lần thăm (visits)
    if root.children:
        move = max(root.children, key=lambda n: n.visits).action
    elif root.untried_actions:          # hết giờ trước khi kịp chạy mô phỏng nào
        move = rng.choice(root.untried_actions)
    else:                               # ván đã kết thúc
        move = env.sample()

    mcts_agent.last_stats = {
        "sims": sims,
        "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
    }
    return move


# Khởi tạo giá trị mặc định ban đầu cho last_stats
mcts_agent.last_stats = {"sims": 0, "latency_ms": 0.0}

import sys
import os


import math
import random
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
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MCTSNode:
    """Một node trong cây MCTS."""

    __slots__ = ['env', 'parent', 'action', 'children',
                 'untried_actions', 'wins', 'visits', 'turn', 'mover']

    def __init__(self, env: ConnectFourEnv, parent: Optional['MCTSNode'] = None,
                 action: Optional[int] = None, mover: Optional[str] = None):
        self.env = env.copy()
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

    def expand(self) -> 'MCTSNode':
        """Mở rộng một node chưa được thăm."""
        action = self.untried_actions.pop(
            random.randint(0, len(self.untried_actions) - 1)
        )
        new_env = self.env.copy()
        mover = new_env.turn
        new_env.step(action)
        child = MCTSNode(new_env, parent=self, action=action, mover=mover)
        self.children.append(child)
        return child

    def update(self, result: float):
        """Cập nhật thống kê node."""
        self.visits += 1
        self.wins += result


def _simulate(env: ConnectFourEnv, root_turn: str) -> float:
    """
    Rollout ngẫu nhiên đến khi game kết thúc.
    
    Returns
    -------
    float : +1 nếu root_turn thắng, -1 nếu thua, 0 nếu hòa.
    """
    sim_env = env.copy()
    while not sim_env.done:
        action = sim_env.sample()
        _, reward, done, _ = sim_env.step(action)

    if sim_env.reward == 0:
        return 0.0

    # Xác định người thắng
    # Người vừa đi (trước khi done=True) là người thắng
    # Sau step() cuối, turn đã không đổi (vì done=True)
    winner_turn = sim_env.turn  # Người đã thắng (done=True sau khi họ đi)
    if winner_turn == root_turn:
        return 1.0
    return -1.0


def mcts_agent(env: ConnectFourEnv, n_simulations: int = 800,
               c: float = math.sqrt(2)) -> int:
    """
    Cấp 6 - MCTS/UCT agent.
    
    Theo sách Liu (2024), Ch. 8, mcts_agent().
    
    Parameters
    ----------
    env : ConnectFourEnv
    n_simulations : int
        Số mô phỏng (rollout). Mặc định 800.
    c : float
        Hằng số khám phá trong UCT. Mặc định sqrt(2).
    
    Returns
    -------
    int : cột tốt nhất (1-7)
    """
    root_turn = env.turn
    root = MCTSNode(env)

    for _ in range(n_simulations):
        node = root

        # 1. Selection
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(c)

        # 2. Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        # 3. Simulation
        result = _simulate(node.env, root_turn)

        # 4. Backpropagation
        while node is not None:
            # result từ góc nhìn root_turn
            # Nếu node thuộc lượt root_turn → wins là tốt
            # Nếu node thuộc lượt đối thủ → wins là xấu cho root
            if node.mover == root_turn:
                node.update(result)
            else:
                node.update(-result)
            node = node.parent

    # Chọn nước đi tốt nhất (theo số lần thăm, không theo UCT score)
    if not root.children:
        return env.sample()

    best = max(root.children, key=lambda n: n.visits)
    return best.action

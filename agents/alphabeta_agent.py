import sys
import os


import numpy as np
from typing import Tuple
from game.connect4_env import ConnectFourEnv


"""
Cấp 4 & 5: Minimax với Alpha-Beta Pruning + Position Evaluation
================================================================
Theo sách Liu (2024):
  - Ch. 5: Depth-Pruned MiniMax (MiniMax_depth)
  - Ch. 6: Alpha-Beta Pruning (MiniMax_ab)
  - Ch. 7: Position Evaluation (eval_payoff_conn, MiniMax_conn_eval)

Triển khai Negamax với alpha-beta pruning, hàm đánh giá vị trí,
sắp xếp nước đi từ giữa ra (move ordering).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Tuple
from game.connect4_env import ConnectFourEnv

# ─────────────────────────────────────────────────────────────────────────────
# Hàm đánh giá vị trí (theo Ch. 7)
# ─────────────────────────────────────────────────────────────────────────────

# Ma trận trọng số vị trí: ưu tiên cột giữa và hàng dưới
# shape (7 cols, 6 rows) - giá trị cho quân đỏ (1)
POSITION_WEIGHTS = np.array([
    [3, 4, 5, 5, 4, 3],
    [4, 6, 8, 8, 6, 4],
    [5, 8,11,11, 8, 5],
    [7,10,13,13,10, 7],
    [5, 8,11,11, 8, 5],
    [4, 6, 8, 8, 6, 4],
    [3, 4, 5, 5, 4, 3],
], dtype=np.float32)


def _count_windows(board: np.ndarray, piece: int, length: int) -> int:
    """Đếm số cửa sổ 4 ô có thể giúp piece thắng với `length` quân đã có."""
    rows, cols = 6, 7
    count = 0

    def check_window(window):
        return (np.sum(window == piece) == length and
                np.sum(window == 0) == 4 - length)

    # Ngang
    for r in range(rows):
        for c in range(cols - 3):
            window = np.array([board[c+i][r] for i in range(4)])
            if check_window(window):
                count += 1

    # Dọc
    for c in range(cols):
        for r in range(rows - 3):
            window = np.array([board[c][r+i] for i in range(4)])
            if check_window(window):
                count += 1

    # Chéo /
    for c in range(cols - 3):
        for r in range(rows - 3):
            window = np.array([board[c+i][r+i] for i in range(4)])
            if check_window(window):
                count += 1

    # Chéo \
    for c in range(cols - 3):
        for r in range(3, rows):
            window = np.array([board[c+i][r-i] for i in range(4)])
            if check_window(window):
                count += 1

    return count


def eval_payoff_conn(env: ConnectFourEnv) -> float:
    """
    Hàm đánh giá vị trí (position evaluation function).
    Theo sách Ch. 7, eval_payoff_conn().
    
    Tính điểm từ góc nhìn của người chơi hiện tại (env.turn).
    Điểm dương = có lợi, điểm âm = bất lợi.
    """
    board = env.state
    piece = 1 if env.turn == 'red' else -1
    opp_piece = -piece

    score = 0.0

    # Điểm vị trí
    score += float(np.sum(board * piece * POSITION_WEIGHTS))

    # Điểm cửa sổ 3 quân (sắp thắng)
    score += 50 * _count_windows(board, piece, 3)
    score -= 80 * _count_windows(board, opp_piece, 3)

    # Điểm cửa sổ 2 quân
    score += 10 * _count_windows(board, piece, 2)
    score -= 15 * _count_windows(board, opp_piece, 2)

    return score


def _order_moves(moves):
    """Sắp xếp nước đi từ cột giữa (4) ra hai bên để alpha-beta hiệu quả hơn."""
    center = 4
    return sorted(moves, key=lambda m: abs(m - center))


def _negamax(env: ConnectFourEnv, depth: int, alpha: float, beta: float,
             use_eval: bool = True) -> float:
    """
    Negamax với alpha-beta pruning.
    
    Returns: điểm tốt nhất từ góc nhìn của người chơi HIỆN TẠI.
    """
    if env.done:
        # Người chơi vừa đi đã thắng → người hiện tại thua
        if env.reward != 0:
            return -1000 - depth  # thua sớm hơn = tệ hơn
        return 0  # hòa

    if depth == 0:
        if use_eval:
            return eval_payoff_conn(env) / 1000.0
        return 0

    best = -float('inf')
    for m in _order_moves(env.validinputs):
        env_copy = env.copy()
        env_copy.step(m)
        score = -_negamax(env_copy, depth - 1, -beta, -alpha, use_eval)
        if score > best:
            best = score
        alpha = max(alpha, best)
        if alpha >= beta:
            break  # beta cutoff

    return best


def alphabeta_agent(env: ConnectFourEnv, depth: int = 5,
                    use_eval: bool = True) -> int:
    """
    Cấp 5 - Alpha-Beta Pruning MiniMax agent.
    
    Theo sách Liu (2024), Ch. 6-7.
    
    Parameters
    ----------
    env : ConnectFourEnv
    depth : int
        Độ sâu tìm kiếm (mặc định 5)
    use_eval : bool
        True = dùng hàm đánh giá vị trí (Ch. 7), False = chỉ thắng/thua/hòa
    
    Returns
    -------
    int : cột tốt nhất (1-7)
    """
    best_move = env.validinputs[0]
    best_score = -float('inf')
    alpha = -float('inf')
    beta = float('inf')

    for m in _order_moves(env.validinputs):
        env_copy = env.copy()
        _, reward, done, _ = env_copy.step(m)

        # Thắng ngay → đi ngay
        if done and reward != 0:
            return m

        score = -_negamax(env_copy, depth - 1, -beta, -alpha, use_eval)
        if score > best_score:
            best_score = score
            best_move = m
        alpha = max(alpha, best_score)

    return best_move


def minimax_agent(env: ConnectFourEnv, depth: int = 4) -> int:
    """MiniMax không dùng hàm đánh giá (Ch. 5)."""
    return alphabeta_agent(env, depth=depth, use_eval=False)


def minimax_eval_agent(env: ConnectFourEnv, depth: int = 5) -> int:
    """MiniMax với hàm đánh giá vị trí (Ch. 7)."""
    return alphabeta_agent(env, depth=depth, use_eval=True)

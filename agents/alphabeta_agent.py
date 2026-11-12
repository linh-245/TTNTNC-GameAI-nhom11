import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
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

Các công tắc để so sánh/benchmark:
  - use_pruning=False : Negamax thuần (không cập nhật alpha, không cắt nhánh)
  - use_ordering=False: giữ nguyên thứ tự env.validinputs, không sắp xếp từ giữa ra
  - budget_ms=<số>    : iterative deepening (độ sâu 1..depth), dừng khi hết thời gian
  - use_tt=True       : bảng chuyển vị Zobrist (chỉ có tác dụng khi use_pruning=True)
  - alphabeta_agent.last_stats: {"nodes", "depth", "time_ms", "timed_out"}

Tối ưu hàm đánh giá: 69 cửa sổ 4 ô (24 ngang, 21 dọc, 24 chéo) được tính trước
thành mảng chỉ số; mỗi lần đánh giá chỉ cần 1 phép lấy chỉ số trên board.ravel().
"""
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


def _build_windows() -> np.ndarray:
    """Chỉ số phẳng (c*6 + r) của 4 ô trong từng cửa sổ; tổng cộng 69 cửa sổ."""
    def idx(c, r):
        return c * 6 + r

    wins = []
    for r in range(6):                       # ngang: 6 hàng x 4 vị trí = 24
        for c in range(4):
            wins.append([idx(c + i, r) for i in range(4)])
    for c in range(7):                       # dọc: 7 cột x 3 vị trí = 21
        for r in range(3):
            wins.append([idx(c, r + i) for i in range(4)])
    for c in range(4):                       # chéo /: 12
        for r in range(3):
            wins.append([idx(c + i, r + i) for i in range(4)])
    for c in range(4):                       # chéo \: 12
        for r in range(3, 6):
            wins.append([idx(c + i, r - i) for i in range(4)])
    assert len(wins) == 69
    return np.array(wins, dtype=np.intp)


_WINDOWS = _build_windows()   # shape (69, 4)


def _count_windows(board: np.ndarray, piece: int, length: int) -> int:
    """Đếm số cửa sổ 4 ô có thể giúp piece thắng với `length` quân đã có."""
    v = np.asarray(board).ravel()[_WINDOWS]
    return int(np.count_nonzero(((v == piece).sum(axis=1) == length) &
                                ((v == 0).sum(axis=1) == 4 - length)))


def eval_payoff_conn(env: ConnectFourEnv) -> float:
    """
    Hàm đánh giá vị trí (position evaluation function).
    Theo sách Ch. 7, eval_payoff_conn().
    
    Tính điểm từ góc nhìn của người chơi hiện tại (env.turn).
    Điểm dương = có lợi, điểm âm = bất lợi.
    """
    board = env.state
    piece = 1 if env.turn == 'red' else -1

    # Điểm vị trí
    score = float(np.sum(board * piece * POSITION_WEIGHTS))

    # Một lần lấy chỉ số cho cả 69 cửa sổ
    v = board.ravel()[_WINDOWS]
    s = v.sum(axis=1) * piece            # quân mình - quân đối thủ
    a = np.abs(v).sum(axis=1)            # số ô đã có quân
    # Mã (a, s) duy nhất vì |s| <= a <= 4 và s nằm trong [-4, 4]
    c = np.bincount((a * 9 + s).astype(np.intp), minlength=37)
    # Cửa sổ "thuần" (chỉ có một phe + ô trống):
    #   3 quân mình: a=3,s=3 -> 30 | 3 quân đối thủ: a=3,s=-3 -> 24
    #   2 quân mình: a=2,s=2 -> 20 | 2 quân đối thủ: a=2,s=-2 -> 16
    score += 50 * int(c[30]) - 80 * int(c[24])
    score += 10 * int(c[20]) - 15 * int(c[16])
    return score


def _order_moves(moves):
    """Sắp xếp nước đi từ cột giữa (4) ra hai bên để alpha-beta hiệu quả hơn."""
    center = 4
    return sorted(moves, key=lambda m: abs(m - center))


# ─────────────────────────────────────────────────────────────────────────────
# Bảng chuyển vị Zobrist (tùy chọn)
# ─────────────────────────────────────────────────────────────────────────────
_rng = np.random.default_rng(20241112)
_ZOBRIST = _rng.integers(1, 2**63 - 1, size=(42, 3), dtype=np.int64)  # [ô][-1/0/1 -> 0/1/2]
_ZOBRIST_TURN = int(_rng.integers(1, 2**63 - 1, dtype=np.int64))
_CELLS = np.arange(42)
_EXACT, _LOWER, _UPPER = 0, 1, 2


def _zobrist_key(env: ConnectFourEnv) -> int:
    idx = env.state.ravel().astype(np.intp) + 1
    h = int(np.bitwise_xor.reduce(_ZOBRIST[_CELLS, idx]))
    return h ^ _ZOBRIST_TURN if env.turn == 'red' else h


class _SearchTimeout(Exception):
    """Hết ngân sách thời gian giữa chừng một lần tìm kiếm."""


def _negamax(env: ConnectFourEnv, depth: int, alpha: float, beta: float,
             use_eval: bool = True, use_pruning: bool = True,
             use_ordering: bool = True, stats: dict = None,
             deadline: float = None, tt: dict = None) -> float:
    """
    Negamax, có hoặc không có alpha-beta pruning.
    
    use_pruning=True : alpha-beta (cập nhật alpha, cắt khi alpha >= beta)
    use_pruning=False: negamax thuần, duyệt toàn bộ cây tới độ sâu `depth`
    stats            : dict có khóa "nodes"; mỗi lần vào hàm tính là 1 nút
    deadline         : mốc time.perf_counter(); quá mốc thì ném _SearchTimeout
    tt               : bảng chuyển vị (chỉ dùng khi use_pruning=True)
    
    Returns: điểm tốt nhất từ góc nhìn của người chơi HIỆN TẠI.
    """
    if stats is not None:
        stats["nodes"] += 1
    if deadline is not None and time.perf_counter() > deadline:
        raise _SearchTimeout

    if env.done:
        # Người chơi vừa đi đã thắng → người hiện tại thua
        if env.reward != 0:
            return -1000 - depth  # thua sớm hơn = tệ hơn
        return 0  # hòa

    if depth == 0:
        if use_eval:
            return eval_payoff_conn(env) / 1000.0
        return 0

    moves = _order_moves(env.validinputs) if use_ordering else list(env.validinputs)

    key = None
    if tt is not None and use_pruning:
        key = _zobrist_key(env)
        entry = tt.get(key)
        if entry is not None:
            e_depth, e_flag, e_val, e_move = entry
            if e_depth >= depth:
                if e_flag == _EXACT:
                    return e_val
                if e_flag == _LOWER:
                    alpha = max(alpha, e_val)
                else:
                    beta = min(beta, e_val)
                if alpha >= beta:
                    return e_val
            if use_ordering and e_move in moves:   # nước tốt nhất lần trước đi trước
                moves.remove(e_move)
                moves.insert(0, e_move)
    alpha_orig = alpha

    best = -float('inf')
    best_move = None
    for m in moves:
        env_copy = env.copy()
        env_copy.step(m)
        score = -_negamax(env_copy, depth - 1, -beta, -alpha,
                          use_eval, use_pruning, use_ordering, stats,
                          deadline, tt)
        if score > best:
            best = score
            best_move = m
        if use_pruning:
            alpha = max(alpha, best)
            if alpha >= beta:
                break  # beta cutoff

    if key is not None:
        flag = _UPPER if best <= alpha_orig else (_LOWER if best >= beta else _EXACT)
        tt[key] = (depth, flag, best, best_move)
    return best


def _search_root(children, depth, use_eval, use_pruning, use_ordering,
                 stats, deadline, tt, first=None):
    """Một lần tìm kiếm ở độ sâu `depth`. children = [(nước đi, env sau nước đi)]."""
    order = list(children)
    if first is not None:                       # nước tốt nhất của độ sâu trước đi trước
        order.sort(key=lambda mc: mc[0] != first)
    best_move, best_score = None, -float('inf')
    alpha, beta = -float('inf'), float('inf')
    for m, child in order:
        score = -_negamax(child, depth - 1, -beta, -alpha, use_eval,
                          use_pruning, use_ordering, stats, deadline, tt)
        if score > best_score:
            best_score = score
            best_move = m
        if use_pruning:
            alpha = max(alpha, best_score)
    return best_move


def alphabeta_agent(env: ConnectFourEnv, depth: int = 5,
                    use_eval: bool = True, use_pruning: bool = True,
                    use_ordering: bool = True, budget_ms: float = None,
                    use_tt: bool = False) -> int:
    """
    Cấp 5 - Alpha-Beta Pruning MiniMax agent.
    
    Theo sách Liu (2024), Ch. 6-7.
    
    Parameters
    ----------
    env : ConnectFourEnv
    depth : int
        Độ sâu tìm kiếm (mặc định 5). Khi có budget_ms: độ sâu tối đa.
    use_eval : bool
        True = dùng hàm đánh giá vị trí (Ch. 7), False = chỉ thắng/thua/hòa
    use_pruning : bool
        True = alpha-beta pruning; False = negamax thuần (không cắt nhánh)
    use_ordering : bool
        True = sắp xếp nước đi từ cột giữa ra; False = giữ thứ tự validinputs
    budget_ms : float | None
        None = tìm đúng một lần ở độ sâu `depth` (hành vi cũ).
        Có giá trị = iterative deepening từ độ sâu 1 đến `depth`, kiểm tra
        thời gian ở mỗi nút, trả nước của độ sâu đã hoàn thành gần nhất.
    use_tt : bool
        True = bảng chuyển vị Zobrist (cần use_pruning=True). Mặc định tắt vì
        điểm "thua sớm/muộn" phụ thuộc độ sâu còn lại nên hiếm khi có thể đổi
        nước chọn so với khi không dùng bảng.
    
    Returns
    -------
    int : cột tốt nhất (1-7)
    
    Side effect
    -----------
    alphabeta_agent.last_stats = {
        "nodes": số nút đã duyệt (cộng dồn qua các độ sâu; gồm nút gốc),
        "depth": độ sâu hoàn thành sâu nhất (0 nếu chưa xong độ sâu nào),
        "time_ms": thời gian của lần gọi,
        "timed_out": True nếu bị dừng vì hết ngân sách,
    }
    """
    t0 = time.perf_counter()
    stats = {"nodes": 1, "depth": 0, "time_ms": 0.0, "timed_out": False}  # nút gốc
    alphabeta_agent.last_stats = stats
    deadline = None if budget_ms is None else t0 + budget_ms / 1000.0

    moves = _order_moves(env.validinputs) if use_ordering else list(env.validinputs)

    # Tạo các nút con một lần; nếu có nước thắng ngay thì đi luôn
    children = []
    for m in moves:
        env_copy = env.copy()
        _, reward, done, _ = env_copy.step(m)
        if done and reward != 0:
            stats["nodes"] += 1
            stats["time_ms"] = (time.perf_counter() - t0) * 1000.0
            return m
        children.append((m, env_copy))

    best_move = moves[0]   # dự phòng hợp lệ nếu hết giờ trước khi xong độ sâu 1
    tt = {} if (use_tt and use_pruning) else None
    depths = [depth] if budget_ms is None else range(1, depth + 1)

    for d in depths:
        try:
            m = _search_root(children, d, use_eval, use_pruning, use_ordering,
                             stats, deadline, tt,
                             first=best_move if budget_ms is not None and stats["depth"] else None)
        except _SearchTimeout:
            stats["timed_out"] = True
            break
        best_move = m
        stats["depth"] = d

    stats["time_ms"] = (time.perf_counter() - t0) * 1000.0
    return best_move


# Khởi tạo sẵn để đọc last_stats trước lần gọi đầu tiên không bị lỗi
alphabeta_agent.last_stats = {"nodes": 0, "depth": 0, "time_ms": 0.0, "timed_out": False}


def minimax_agent(env: ConnectFourEnv, depth: int = 4) -> int:
    """MiniMax không dùng hàm đánh giá (Ch. 5)."""
    return alphabeta_agent(env, depth=depth, use_eval=False)


def minimax_eval_agent(env: ConnectFourEnv, depth: int = 5) -> int:
    """MiniMax với hàm đánh giá vị trí (Ch. 7)."""
    return alphabeta_agent(env, depth=depth, use_eval=True)
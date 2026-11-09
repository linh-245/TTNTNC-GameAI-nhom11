"""
Mã hóa trạng thái Connect Four thành tensor cho mạng nơ-ron.
==========================================================
Đầu vào: ConnectFourEnv (state shape (7, 6) theo (cột, hàng); 1 = đỏ, -1 = vàng).
Đầu ra:  tensor float32 shape (3, 6, 7) = (kênh, hàng, cột), hàng 0 là hàng dưới cùng
         (giống chỉ số hàng của engine, khác với get_board_2d dùng để hiển thị).

Góc nhìn của người sắp đi (env.turn), nên mạng không cần biết đỏ hay vàng:
  kênh 0: quân của người sắp đi
  kênh 1: quân của đối thủ
  kênh 2: ô sẽ rơi xuống nếu đánh vào từng cột còn hợp lệ (1 ô mỗi cột)
"""
import numpy as np

from game.connect4_env import ConnectFourEnv

ROWS = ConnectFourEnv.ROWS
COLS = ConnectFourEnv.COLS
N_CHANNELS = 3
ACTION_SIZE = COLS


def encode_state(env: ConnectFourEnv) -> np.ndarray:
    """Trả về tensor (3, 6, 7), không thay đổi env."""
    me = 1 if env.turn == 'red' else -1
    board = env.state.T  # (ROWS, COLS)
    x = np.zeros((N_CHANNELS, ROWS, COLS), dtype=np.float32)
    x[0] = (board == me)
    x[1] = (board == -me)
    for action in env.validinputs:
        col = action - 1
        row = int(np.count_nonzero(env.state[col]))  # quân xếp từ dưới lên
        x[2, row, col] = 1.0
    return x


def legal_mask(env: ConnectFourEnv) -> np.ndarray:
    """Vector 7 phần tử: 1.0 nếu cột (index 0..6, tức action - 1) còn đánh được."""
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    for action in env.validinputs:
        mask[action - 1] = 1.0
    return mask
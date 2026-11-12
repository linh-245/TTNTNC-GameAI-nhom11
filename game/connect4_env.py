import numpy as np
import random
from typing import Tuple, List, Optional


"""
Connect Four Game Environment
==============================
Theo sách: Mark Liu, "AlphaGo Simplified", Ch. 3-8.
API tương tự OpenAI Gym: reset(), step(), render(), sample(), close().

Bàn cờ: 7 cột × 6 hàng
- State: numpy array shape (7, 6)
  - 0: ô trống
  - 1: quân đỏ (red)
  - -1: quân vàng (yellow)
- Action: số cột từ 1 đến 7
- Reward: 1 nếu người hiện tại thắng, -1 nếu thua, 0 nếu hòa hoặc đang chơi
- Turn: 'red' hoặc 'yellow' (đi trước là red)
"""


class ActionSpace:
    """Không gian hành động: 7 cột (1 đến 7)."""
    def __init__(self):
        self.n = 7
        self.actions = list(range(1, 8))


class ObservationSpace:
    """Không gian trạng thái: ma trận 7×6."""
    def __init__(self):
        self.shape = (7, 6)


class ConnectFourEnv:
    """
    Môi trường game Connect Four 7×6.
    
    Attributes
    ----------
    action_space : ActionSpace
        Không gian hành động (7 cột)
    observation_space : ObservationSpace
        Không gian quan sát (7×6)
    state : np.ndarray
        Trạng thái bàn cờ hiện tại, shape (7, 6)
    turn : str
        Lượt hiện tại ('red' hoặc 'yellow')
    validinputs : list
        Danh sách các cột hợp lệ có thể đánh
    reward : float
        Phần thưởng sau mỗi bước
    done : bool
        True nếu game kết thúc
    """

    ROWS = 6
    COLS = 7
    WIN_LENGTH = 4

    def __init__(self):
        self.action_space = ActionSpace()
        self.observation_space = ObservationSpace()
        self.state: np.ndarray = np.zeros((self.COLS, self.ROWS), dtype=np.int8)
        self.turn: str = 'red'
        self.reward: float = 0
        self.done: bool = False
        self.info: str = ''
        self.validinputs: List[int] = list(range(1, self.COLS + 1))
        self._history: List[Tuple[int, int]] = []  # (col, row) từng nước, phục vụ undo()

    # ------------------------------------------------------------------
    # Core API (theo sách)
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Khởi tạo lại bàn cờ về trạng thái ban đầu."""
        self.state = np.zeros((self.COLS, self.ROWS), dtype=np.int8)
        self.turn = 'red'
        self.reward = 0
        self.done = False
        self.info = ''
        self.validinputs = list(range(1, self.COLS + 1))
        self._history = []
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, str]:
        """
        Thực hiện nước đi.

        Parameters
        ----------
        action : int
            Số cột (1-7)

        Returns
        -------
        state : np.ndarray, shape (7,6)
        reward : float  (1 thắng, -1 thua, 0 hòa/đang chơi)
        done : bool
        info : str
        """
        if self.done:
            raise ValueError("Game đã kết thúc. Gọi reset() để bắt đầu lại.")
        if action not in self.validinputs:
            raise ValueError(f"Hành động {action} không hợp lệ. Hợp lệ: {self.validinputs}")

        col = action - 1  # chuyển sang index 0-based
        piece = 1 if self.turn == 'red' else -1

        # Tìm hàng thấp nhất còn trống trong cột
        row = self._find_empty_row(col)
        self.state[col][row] = piece
        self._history.append((col, row))

        # Cập nhật danh sách cột hợp lệ
        if row == self.ROWS - 1:
            self.validinputs.remove(action)

        # Kiểm tra thắng
        if self._check_win(col, row, piece):
            self.reward = 1
            self.done = True
            self.info = f"{self.turn} wins"
        elif len(self.validinputs) == 0 and not any(
            self.state[c][self.ROWS - 1] == 0 for c in range(self.COLS)
        ):
            # Hòa: bàn đầy
            self.reward = 0
            self.done = True
            self.info = "tie"
        else:
            # Kiểm tra hòa: bàn đầy
            full = all(self.state[c][self.ROWS - 1] != 0 for c in range(self.COLS))
            if full:
                self.reward = 0
                self.done = True
                self.info = "tie"
            else:
                self.reward = 0

        # Đổi lượt
        if not self.done:
            self.turn = 'yellow' if self.turn == 'red' else 'red'

        return self.state.copy(), self.reward, self.done, self.info

    def sample(self) -> int:
        """Chọn ngẫu nhiên một hành động hợp lệ."""
        return random.choice(self.validinputs)

    def render(self) -> None:
        """In bàn cờ ra console (text)."""
        print(self._board_str())

    def close(self) -> None:
        """Đóng môi trường (placeholder)."""
        pass

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _find_empty_row(self, col: int) -> int:
        """Tìm hàng thấp nhất còn trống trong cột (index 0-based từ dưới)."""
        for row in range(self.ROWS):
            if self.state[col][row] == 0:
                return row
        raise ValueError(f"Cột {col+1} đã đầy.")

    def _check_win(self, col: int, row: int, piece: int) -> bool:
        """Kiểm tra nước đi tại (col, row) có tạo 4 liên tiếp không."""
        directions = [
            (1, 0),   # ngang
            (0, 1),   # dọc
            (1, 1),   # chéo /
            (1, -1),  # chéo \
        ]
        for dc, dr in directions:
            count = 1
            # Đếm theo chiều dương
            c, r = col + dc, row + dr
            while 0 <= c < self.COLS and 0 <= r < self.ROWS and self.state[c][r] == piece:
                count += 1
                c += dc
                r += dr
            # Đếm theo chiều âm
            c, r = col - dc, row - dr
            while 0 <= c < self.COLS and 0 <= r < self.ROWS and self.state[c][r] == piece:
                count += 1
                c -= dc
                r -= dr
            if count >= self.WIN_LENGTH:
                return True
        return False

    def _board_str(self) -> str:
        """Tạo chuỗi hiển thị bàn cờ (từ trên xuống dưới)."""
        lines = []
        lines.append("  " + "  ".join(str(i) for i in range(1, self.COLS + 1)))
        lines.append("  " + "---" * self.COLS)
        for row in range(self.ROWS - 1, -1, -1):
            cells = []
            for col in range(self.COLS):
                v = self.state[col][row]
                if v == 1:
                    cells.append(" R")
                elif v == -1:
                    cells.append(" Y")
                else:
                    cells.append(" .")
            lines.append(" |" + " ".join(cells) + " |")
        lines.append("  " + "---" * self.COLS)
        lines.append(f"  Turn: {self.turn.upper()}")
        return "\n".join(lines)

    def get_board_2d(self) -> np.ndarray:
        """Trả về bàn cờ dạng 2D shape (ROWS, COLS) để hiển thị (hàng trên cùng = hàng cao nhất)."""
        return self.state.T[::-1].copy()

    def copy(self) -> 'ConnectFourEnv':
        """Tạo bản sao sâu của môi trường (dùng cho look-ahead)."""
        new_env = ConnectFourEnv()
        new_env.state = self.state.copy()
        new_env.turn = self.turn
        new_env.reward = self.reward
        new_env.done = self.done
        new_env.info = self.info
        new_env.validinputs = self.validinputs.copy()
        new_env._history = self._history.copy()
        return new_env

    def is_valid_action(self, action: int) -> bool:
        """Kiểm tra action có hợp lệ không."""
        return action in self.validinputs

    def get_valid_actions(self) -> List[int]:
        """Trả về danh sách cột hợp lệ."""
        return self.validinputs.copy()

    # ------------------------------------------------------------------
    # API bổ sung theo đề bài (không đổi hành vi các hàm cũ)
    # ------------------------------------------------------------------

    def legal_actions(self) -> List[int]:
        """Danh sách cột hợp lệ (1..7). Giống validinputs nhưng là bản sao."""
        return self.validinputs.copy()

    def is_terminal(self) -> bool:
        """True nếu ván đã kết thúc (có người thắng hoặc hòa)."""
        return self.done

    def result(self, player: str) -> int:
        """
        Kết quả của ván theo góc nhìn `player` ('red' hoặc 'yellow').

        +1 nếu player thắng, -1 nếu player thua, 0 nếu hòa hoặc ván đang chơi.
        Khi ván kết thúc do có người thắng, `turn` giữ nguyên là người vừa đi,
        tức người thắng.
        """
        if player not in ('red', 'yellow'):
            raise ValueError(f"player phải là 'red' hoặc 'yellow', nhận {player!r}")
        if not self.done or self.reward == 0:
            return 0
        return 1 if self.turn == player else -1

    def clone(self) -> 'ConnectFourEnv':
        """Bản sao độc lập của môi trường (giống copy())."""
        return self.copy()

    def undo(self) -> None:
        """
        Hoàn tác nước đi cuối cùng, khôi phục ô, lượt, done, reward, info và
        danh sách cột hợp lệ (giữ thứ tự tăng dần).
        """
        if not self._history:
            raise ValueError("Không có nước nào để hoàn tác.")
        col, row = self._history.pop()
        piece = int(self.state[col][row])
        self.state[col][row] = 0
        # Người vừa đi là người sở hữu quân này, nên đến lượt họ đi lại
        self.turn = 'red' if piece == 1 else 'yellow'
        self.done = False
        self.reward = 0
        self.info = ''
        action = col + 1
        if action not in self.validinputs:
            self.validinputs.append(action)
            self.validinputs.sort()

    def __repr__(self) -> str:
        return self._board_str()


conn = ConnectFourEnv

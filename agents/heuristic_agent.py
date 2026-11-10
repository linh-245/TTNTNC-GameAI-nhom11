import sys
import os


from game.connect4_env import ConnectFourEnv


"""
Cấp 2, 3 & 4: Heuristic Rule-Based Agent
=========================================
Kế thừa khung sách Liu (2024), Ch. 3 (conn_think1, conn_think2, conn_think3);
nhóm đã sửa lỗi và chỉnh mô tả cho khớp với code hiện tại.

Cấp 2 (Think1):  Thắng ngay nếu có nước thắng, ngược lại đi ngẫu nhiên.
Cấp 3 (Think2):  Thắng ngay; nếu không, đi nước đầu tiên (cột nhỏ nhất)
                 không để đối thủ thắng ngay ở lượt sau.
Cấp 4 (Think3):  Thắng ngay; chặn nước thắng của đối thủ; nếu không, đi nước
                 không để đối thủ thắng ngay theo thứ tự ưu tiên cột giữa.
                 Không có bước tìm "fork" hay chuỗi tấn công nhiều nước.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def conn_think1(env: ConnectFourEnv) -> int:
    """
    Cấp 2 - Heuristic (Look 1 step ahead):
    - Nếu có nước thắng ngay, đánh ngay.
    - Ngược lại, chọn ngẫu nhiên.
    
    Theo sách Liu (2024), Ch. 3, conn_think1().
    """
    for m in env.validinputs:
        env_copy = env.copy()
        _, reward, done, _ = env_copy.step(m)
        if done and reward != 0:
            return m
    return env.sample()


def conn_think2(env: ConnectFourEnv) -> int:
    """
        Cấp 3 - Heuristic (nhìn 2 nước: nước của ta và phản hồi của đối thủ):
    1. Nếu có nước thắng ngay → đánh ngay.
    2. Với mỗi nước còn lại, thử xem đối thủ có thắng ngay ở lượt sau không;
       nước nào để đối thủ thắng thì xếp vào nhóm "xấu". Nếu đối thủ sắp
       thắng, các nước khác đều là "xấu" nên ta sẽ chọn đúng nước chặn.
    3. Chọn nước đầu tiên (cột nhỏ nhất) trong nhóm không để đối thủ thắng.
    4. Nếu mọi nước đều "xấu": chọn cột 4 nếu có, không thì nước "xấu" đầu tiên.
       (env.sample() chỉ là dự phòng khi không còn nước hợp lệ, không phải
       bước chọn ngẫu nhiên.)
    
    Theo sách Liu (2024), Ch. 3, conn_think2().
    """
    wins = []
    neutral = []
    losing = []

    for m in env.validinputs:
        env_copy = env.copy()
        _, reward, done, _ = env_copy.step(m)

        # Bước 1: thắng ngay
        if done and reward != 0:
            return m

        # Bước 2: chặn đối thủ thắng ngay
        opponent_can_win = False
        for m2 in env_copy.validinputs:
            env_copy2 = env_copy.copy()
            _, r2, d2, _ = env_copy2.step(m2)
            if d2 and r2 != 0:
                opponent_can_win = True
                break

        if opponent_can_win:
            losing.append(m)
        else:
            neutral.append(m)

    if neutral:
        return neutral[0]
    if losing:
        # Ưu tiên cột giữa nếu buộc phải chơi nước xấu
        center = [m for m in losing if m == 4]
        return center[0] if center else losing[0]
    return env.sample()


def _opponent_can_win_immediately(env: ConnectFourEnv) -> bool:
    """Kiểm tra đối thủ có nước thắng ngay không."""
    for m in env.validinputs:
        env_copy = env.copy()
        _, reward, done, _ = env_copy.step(m)
        if done and reward != 0:
            return True
    return False


def conn_think3(env: ConnectFourEnv) -> int:
    """
    Cấp 4 - Heuristic (nhìn 2 nước, chặn đối thủ trước, ưu tiên cột giữa):
    1. Thắng ngay nếu có thể.
    2. Chặn: nếu đối thủ có nước thắng ngay thì chơi vào cột đó (nếu có nhiều
       cột thì lấy cột nhỏ nhất).
    3. Lọc các nước an toàn (không để đối thủ thắng ngay ở lượt sau), rồi chọn
       theo thứ tự ưu tiên 4, 3, 5, 2, 6, 1, 7.
    4. Nếu không có nước an toàn, chọn cột hợp lệ theo cùng thứ tự ưu tiên.
    Lưu ý: không có bước tìm "fork" (hai mối đe dọa cùng lúc) và không có
    chuỗi tấn công nhiều nước.
    
    Theo sách Liu (2024), Ch. 3, conn_think3().
    """
    # Bước 1: thắng ngay
    for m in env.validinputs:
        env_copy = env.copy()
        _, reward, done, _ = env_copy.step(m)
        if done and reward != 0:
            return m

    # Bước 2: chặn đối thủ thắng ngay (đặt quân vào đúng ô mà đối thủ sẽ thắng)
    opp_wins = []
    for m in env.validinputs:
        e2 = env.copy()
        e2.turn = 'yellow' if env.turn == 'red' else 'red'   # giả sử đối thủ đi
        _, r2, d2, _ = e2.step(m)
        if d2 and r2 != 0:
            opp_wins.append(m)
    if opp_wins:
        return opp_wins[0]

    # Bước 3: tìm nước không giúp đối thủ thắng sau 2 bước
    safe_moves = []
    for m in env.validinputs:
        env_copy = env.copy()
        _, reward, done, _ = env_copy.step(m)
        if done:
            continue
        # Sau khi ta đánh m, đối thủ đánh m2
        opponent_can_win_next = _opponent_can_win_immediately(env_copy)
        if not opponent_can_win_next:
            safe_moves.append(m)

    if safe_moves:
        # Ưu tiên cột giữa
        center_priority = [4, 3, 5, 2, 6, 1, 7]
        for c in center_priority:
            if c in safe_moves:
                return c

    # Bước 4: nếu không có nước an toàn, ưu tiên cột giữa
    center_priority = [4, 3, 5, 2, 6, 1, 7]
    for c in center_priority:
        if c in env.validinputs:
            return c

    return env.sample()

"""
Sinh dữ liệu huấn luyện từ các ván tự đấu của agent giáo viên.
==============================================================
Giáo viên mặc định là conn_think3 (agent Rule-based đã sửa lỗi).

Với mỗi thế cờ (trước khi đi) ta ghi lại:
  x      : tensor (3, 6, 7) góc nhìn người sắp đi (training/encode.py)
  policy : cột (0..6) mà giáo viên chọn -> nhãn cho đầu policy
  value  : kết quả ván theo góc nhìn người sắp đi: +1 thắng, -1 thua, 0 hòa
           -> nhãn cho đầu value
  game   : số thứ tự ván (để dataset.py chia train/val/test theo ván)
  ply    : nước thứ mấy trong ván (0 là nước đầu tiên)

conn_think3 luôn đi giống nhau khi gặp cùng một thế cờ, nên nếu cho nó đấu với
chính nó thì mọi ván sẽ giống hệt nhau. Để có nhiều ván khác nhau, ta cho nước ĐI
THẬT là ngẫu nhiên trong 2 trường hợp: vài nước đầu (opening) và với xác suất nhỏ
ở mỗi nước (explore). Nhãn policy vẫn luôn là nước mà giáo viên sẽ chọn ở thế đó.

Chạy (từ thư mục gốc repo):
    python -m training.generate_data --games 2000 --seed 0 --out data/teacher_think3.npz

File .npz nằm trong data/ và đã bị .gitignore chặn, không đưa lên GitHub.
"""
import argparse
import os
import random
import time

import numpy as np

from game.connect4_env import ConnectFourEnv
from agents.heuristic_agent import conn_think3
from training.encode import encode_state


def play_one_game(teacher, rng, explore=0.2, opening=2):
    """Chơi 1 ván tự đấu. Trả về (danh sách x, danh sách policy, danh sách value)."""
    env = ConnectFourEnv()
    env.reset()
    xs, labels, movers = [], [], []
    ply = 0
    done = False
    reward = 0
    while not done:
        xs.append(encode_state(env))
        teacher_move = teacher(env)            # cột 1..7
        labels.append(teacher_move - 1)        # nhãn 0..6
        movers.append(env.turn)
        if ply < opening or rng.random() < explore:
            move = rng.choice(env.validinputs)  # đi thật: ngẫu nhiên
        else:
            move = teacher_move
        _, reward, done, _ = env.step(move)
        ply += 1
    # Khi ván kết thúc engine không đổi lượt, nên env.turn là người vừa đi.
    winner = env.turn if reward != 0 else None
    values = [0.0 if winner is None else (1.0 if m == winner else -1.0) for m in movers]
    return xs, labels, values


def generate(n_games, seed=0, explore=0.2, opening=2, teacher=conn_think3):
    """Sinh dữ liệu từ n_games ván. Cùng seed thì ra cùng dữ liệu."""
    rng = random.Random(seed)
    X, P, V, G, T = [], [], [], [], []
    for g in range(n_games):
        xs, labels, values = play_one_game(teacher, rng, explore, opening)
        for t, (x, p, v) in enumerate(zip(xs, labels, values)):
            X.append(x)
            P.append(p)
            V.append(v)
            G.append(g)
            T.append(t)
    return {
        "x": np.stack(X).astype(np.float32),
        "policy": np.array(P, dtype=np.int64),
        "value": np.array(V, dtype=np.float32),
        "game": np.array(G, dtype=np.int32),
        "ply": np.array(T, dtype=np.int16),
    }


def main():
    ap = argparse.ArgumentParser(description="Sinh dữ liệu từ ván của agent giáo viên")
    ap.add_argument("--games", type=int, default=2000, help="số ván")
    ap.add_argument("--seed", type=int, default=0, help="hạt giống ngẫu nhiên")
    ap.add_argument("--explore", type=float, default=0.2,
                    help="xác suất đi ngẫu nhiên ở mỗi nước")
    ap.add_argument("--opening", type=int, default=2,
                    help="số nước đầu luôn đi ngẫu nhiên")
    ap.add_argument("--out", default="data/teacher_think3.npz", help="file .npz đầu ra")
    args = ap.parse_args()

    t0 = time.time()
    data = generate(args.games, args.seed, args.explore, args.opening)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(args.out, **data)
    n = len(data["policy"])
    print(f"{args.games} ván, {n} thế cờ -> {args.out} "
          f"(seed={args.seed}, explore={args.explore}, opening={args.opening}, "
          f"{time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
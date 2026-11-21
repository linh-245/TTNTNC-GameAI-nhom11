"""
Đo sức mạnh của checkpoint mạng (SL hoặc RL) bằng đối thủ mốc
==============================================================
Cho mỗi checkpoint, cho tác tử mạng (nn_agent) đấu với các đối thủ mốc cố định
(Random, Think1, Think2, Think3) và ghi tỷ lệ thắng ra CSV. Dùng để vẽ đường cong
sức mạnh của RL qua các vòng huấn luyện (rl_curve.csv) và so với bản SL.

Cách đo giống giải đấu của nhóm (eval/tournament.py):
  - mỗi cặp chơi --games ván, chia đều lượt đi trước / đi sau;
  - hai ván liền nhau dùng chung khai cuộc (--openers nước đầu ngẫu nhiên theo seed),
    vì nn_agent chọn argmax nên không có khai cuộc ngẫu nhiên thì các ván giống hệt nhau;
  - Win Score = (thắng + 0,5 * hòa) / số ván, khoảng tin cậy Wilson 95%.
Một ván mà tác tử mạng báo lỗi hoặc đi sai luật bị xử thua VÀ được đếm vào cột "loi";
nếu cột này khác 0 thì kết quả không dùng được (đó là lỗi, không phải tác tử yếu).

Chạy (từ thư mục gốc repo):
    python training/eval_anchors.py --ckpt weights/sl.pt --label SL --games 100
    python training/eval_anchors.py --ckpt weights/rl.pt --label RL --games 100 --append
    python training/eval_anchors.py --ckpt weights/rl_ckpt/rl_it050.pt weights/rl_ckpt/rl_it100.pt \
        --label it050 it100 --games 100 --out logs/rl_curve.csv

Nhãn --label đi theo thứ tự với --ckpt. Mặc định ghi logs/anchors.csv (ghi đè);
thêm --append để ghi nối vào file có sẵn (dùng khi đo từng checkpoint một).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import random
import time

import numpy as np

from eval.tournament import wilson_ci
from game.connect4_env import ConnectFourEnv

ANCHORS = ["Random", "Think1", "Think2", "Think3"]
FIELDS = ["label", "ckpt", "anchor", "games", "thang", "hoa", "thua", "loi",
          "win_score", "ci_lo", "ci_hi", "openers", "seed"]


def anchor_fn(name):
    """Trả về hàm fn(env) -> cột 1..7 của đối thủ mốc."""
    if name == "Random":
        from agents.random_agent import conn_random
        return conn_random
    from agents import heuristic_agent
    return getattr(heuristic_agent, "conn_" + name.lower())


def nn_fn(ckpt):
    """Tác tử mạng đọc checkpoint `ckpt` (policy argmax, cột đầy bị che)."""
    from agents.nn_agent import nn_agent
    return lambda env: nn_agent(env, ckpt=ckpt)


def play_one(fn_nn, fn_other, nn_first, opener_seed, n_openers, seed):
    """Một ván. Trả về (kết quả theo góc nhìn tác tử mạng: 'thang'|'hoa'|'thua', lỗi hoặc None)."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    env = ConnectFourEnv()
    rng = random.Random(opener_seed)
    ply = 0
    for _ in range(n_openers):
        if env.done:
            break
        env.step(int(rng.choice(env.validinputs)))
        ply += 1
    # nn_first: tác tử mạng là bên đi nước đầu (ply chẵn); ngược lại đi ở ply lẻ
    nn_mover = 0 if nn_first else 1
    last_mover_is_nn = False
    while not env.done:
        is_nn = (ply % 2 == nn_mover)
        fn = fn_nn if is_nn else fn_other
        try:
            col = int(fn(env.copy()))
        except Exception as exc:
            if is_nn:
                return "thua", f"{type(exc).__name__}: {exc}"
            raise                                  # lỗi ở đối thủ mốc là lỗi code, không giấu
        if col not in env.validinputs:
            if is_nn:
                return "thua", f"đi sai luật cột {col}"
            raise ValueError(f"đối thủ mốc đi sai luật cột {col}")
        env.step(col)
        last_mover_is_nn = is_nn
        ply += 1
    if env.reward == 0:
        return "hoa", None
    return ("thang" if last_mover_is_nn else "thua"), None


def eval_checkpoint(ckpt, label, games, n_openers, seed):
    fn_nn = nn_fn(ckpt)
    rows = []
    first_error = None
    for anchor in ANCHORS:
        fn_a = anchor_fn(anchor)
        c = {"thang": 0, "hoa": 0, "thua": 0}
        loi = 0
        t0 = time.time()
        for g in range(games):
            res, err = play_one(fn_nn, fn_a, nn_first=(g % 2 == 0),
                                opener_seed=seed * 1000 + g // 2,     # 2 ván liền nhau chung khai cuộc
                                n_openers=n_openers, seed=seed * 131 + g)
            c[res] += 1
            if err:
                loi += 1
                first_error = first_error or err
        score = (c["thang"] + 0.5 * c["hoa"]) / games
        lo, hi = wilson_ci(c["thang"] + 0.5 * c["hoa"], games)
        rows.append({"label": label, "ckpt": ckpt, "anchor": anchor, "games": games,
                     "thang": c["thang"], "hoa": c["hoa"], "thua": c["thua"], "loi": loi,
                     "win_score": round(score, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                     "openers": n_openers, "seed": seed})
        print(f"{label:>8} vs {anchor:<7} thắng/hòa/thua {c['thang']:>3}/{c['hoa']:>3}/{c['thua']:>3} "
              f"| Win Score {score:.3f} [{lo:.3f}, {hi:.3f}] | lỗi {loi} | {time.time() - t0:.1f}s")
    if first_error:
        print(f"!! {label}: có ván lỗi, ví dụ: {first_error}. Dòng này KHÔNG dùng được.")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Đo checkpoint mạng với đối thủ mốc")
    ap.add_argument("--ckpt", nargs="+", default=["weights/sl.pt"], help="một hoặc nhiều checkpoint")
    ap.add_argument("--label", nargs="+", default=None, help="nhãn cho từng checkpoint (cùng thứ tự)")
    ap.add_argument("--games", type=int, default=100, help="số ván mỗi cặp (số chẵn)")
    ap.add_argument("--openers", type=int, default=2, help="số nước đầu ngẫu nhiên")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="logs/anchors.csv")
    ap.add_argument("--append", action="store_true", help="ghi nối vào CSV có sẵn")
    args = ap.parse_args()

    labels = args.label or [os.path.splitext(os.path.basename(c))[0] for c in args.ckpt]
    if len(labels) != len(args.ckpt):
        ap.error("--label phải có số phần tử bằng --ckpt")
    if args.games % 2:
        ap.error("--games phải là số chẵn để chia đều đi trước/đi sau")

    rows = []
    for ck, lb in zip(args.ckpt, labels):
        if not os.path.exists(ck):
            raise SystemExit(f"Không thấy checkpoint: {ck}")
        rows += eval_checkpoint(ck, lb, args.games, args.openers, args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new_file = not (args.append and os.path.exists(args.out))
    with open(args.out, "w" if new_file else "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerows(rows)
    print(f"Đã ghi {args.out}")


if __name__ == "__main__":
    main()

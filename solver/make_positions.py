"""
Sinh bộ thế chuẩn cố định theo seed và gán nhãn nước đi tối ưu.
==============================================================
Mỗi thế: đã đi ít nhất `--min-moves` nước, chưa kết thúc, không trùng nhau.
Thế được tạo bằng các nước hợp lệ ngẫu nhiên (random.Random(seed) riêng,
không dùng random toàn cục), nên cùng seed luôn ra cùng bộ thế.

Gán nhãn (chọn một cách):
  1) Connect4 Solver của Pascal Pons (khuyến nghị, nhãn tối ưu thật):
       python solver/make_positions.py --solver /duong/dan/c4solver
     Mã nguồn: https://github.com/PascalPons/connect4 (giấy phép AGPL-3.0).
     Chỉ chạy như công cụ ngoài, không chép mã vào repo. Build: `make` (cần g++).
     Nếu có file 7x6.book, đặt cạnh solver hoặc dùng --solver-cwd để solver nạp.
  2) Dự phòng, nhãn Alpha-Beta sâu (không phải nhãn tối ưu tuyệt đối):
       python solver/make_positions.py --labeler alphabeta --depth 6
     Trong báo cáo phải gọi đúng tên "nhãn Alpha-Beta sâu".

Quy ước điểm của solver (analyze mode): điểm dương = người đi ở thế đó thắng
(càng lớn thì thắng càng nhanh), 0 = hòa, âm = thua (càng âm thì thua càng sớm).
Nước tối ưu = các cột có điểm lớn nhất (có thể nhiều cột bằng điểm).

Kết quả: solver/positions.json
"""
import argparse
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.connect4_env import ConnectFourEnv

FULL_COLUMN = -1000   # solver in -1000 cho cột đã đầy


# ---------------------------------------------------------------------------
# Sinh thế
# ---------------------------------------------------------------------------

def generate_positions(n: int, seed: int, min_moves: int, max_moves: int):
    """Trả về list các chuỗi nước đi (cột 1..7 liền nhau), không trùng thế."""
    if min_moves < 1 or max_moves < min_moves:
        raise ValueError("Yêu cầu 1 <= min_moves <= max_moves")
    rng = random.Random(seed)
    seen = set()
    positions = []
    attempts = 0
    while len(positions) < n:
        attempts += 1
        if attempts > n * 200:
            raise RuntimeError("Không sinh đủ thế khác nhau; nới khoảng min/max moves.")
        target = rng.randint(min_moves, max_moves)
        env = ConnectFourEnv()
        env.reset()
        moves = []
        while len(moves) < target and not env.done:
            a = rng.choice(env.legal_actions())
            env.step(a)
            moves.append(a)
        if env.done:
            continue                       # thế đã kết thúc: bỏ
        key = (env.state.tobytes(), env.turn)
        if key in seen:
            continue                       # trùng thế (khác thứ tự nước): bỏ
        seen.add(key)
        positions.append("".join(str(m) for m in moves))
    return positions


def env_from_moves(moves: str) -> ConnectFourEnv:
    env = ConnectFourEnv()
    env.reset()
    for ch in moves:
        env.step(int(ch))
    return env


# ---------------------------------------------------------------------------
# Gán nhãn bằng Connect4 Solver (Pons)
# ---------------------------------------------------------------------------

def label_with_solver(sequences, solver_path: str, cwd: str = None):
    """Chạy `c4solver -a` một lần cho tất cả thế. Trả về {chuỗi: [7 điểm]}."""
    inp = "\n".join(sequences) + "\n"
    proc = subprocess.run([solver_path, "-a"], input=inp, text=True,
                          capture_output=True, cwd=cwd)
    if proc.returncode not in (0,):
        raise RuntimeError(f"Solver lỗi (mã {proc.returncode}): {proc.stderr[-300:]}")
    result = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) != 8:
            continue
        result[parts[0]] = [int(x) for x in parts[1:]]
    missing = [s for s in sequences if s not in result]
    if missing:
        raise RuntimeError(f"Solver không trả điểm cho {len(missing)} thế, ví dụ {missing[0]}")
    return result


# ---------------------------------------------------------------------------
# Dự phòng: nhãn Alpha-Beta sâu (win/draw/loss, không dùng hàm đánh giá)
# ---------------------------------------------------------------------------

def _negamax(env: ConnectFourEnv, depth: int, alpha: float, beta: float) -> float:
    """Điểm từ góc nhìn người sắp đi. Thắng sớm hơn điểm cao hơn."""
    if env.done:
        # Người vừa đi đã thắng -> người sắp đi thua; thua càng sớm càng tệ
        return (-100 - depth) if env.reward != 0 else 0
    if depth == 0:
        return 0
    best = -float("inf")
    for m in sorted(env.legal_actions(), key=lambda c: abs(c - 4)):
        e = env.clone()
        e.step(m)
        score = -_negamax(e, depth - 1, -beta, -alpha)
        best = max(best, score)
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    return best


def label_with_alphabeta(moves: str, depth: int):
    """Điểm 7 cột theo Alpha-Beta độ sâu `depth` (cửa sổ đầy đủ cho từng nước gốc)."""
    env = env_from_moves(moves)
    scores = []
    for col in range(1, 8):
        if col not in env.legal_actions():
            scores.append(FULL_COLUMN)
            continue
        e = env.clone()
        e.step(col)
        scores.append(-_negamax(e, depth - 1, -float("inf"), float("inf")))
    return scores


# ---------------------------------------------------------------------------

def optimal_columns(scores):
    legal = [c + 1 for c, s in enumerate(scores) if s != FULL_COLUMN]
    best = max(scores[c - 1] for c in legal)
    return [c for c in legal if scores[c - 1] == best]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500, help="số thế (mặc định 500)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-moves", type=int, default=8, help="số nước đã đi tối thiểu")
    ap.add_argument("--max-moves", type=int, default=20)
    ap.add_argument("--solver", default=None, help="đường dẫn c4solver (Pons)")
    ap.add_argument("--solver-cwd", default=None, help="thư mục chạy solver (chứa 7x6.book)")
    ap.add_argument("--labeler", choices=["solver", "alphabeta"], default=None)
    ap.add_argument("--depth", type=int, default=6, help="độ sâu cho nhãn Alpha-Beta")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "positions.json"))
    args = ap.parse_args()

    labeler = args.labeler or ("solver" if args.solver else "alphabeta")
    if labeler == "solver" and not args.solver:
        ap.error("--labeler solver cần --solver /duong/dan/c4solver")

    seqs = generate_positions(args.n, args.seed, args.min_moves, args.max_moves)
    print(f"Đã sinh {len(seqs)} thế (seed={args.seed}, "
          f"{args.min_moves}-{args.max_moves} nước).")

    if labeler == "solver":
        source = "Connect4 Solver (Pascal Pons, AGPL-3.0)"
        scored = label_with_solver(seqs, args.solver, args.solver_cwd)
        get_scores = lambda s: scored[s]
    else:
        source = f"nhãn Alpha-Beta sâu (độ sâu {args.depth}, thắng/hòa/thua)"
        get_scores = lambda s: label_with_alphabeta(s, args.depth)

    positions = []
    for i, s in enumerate(seqs):
        scores = get_scores(s)
        env = env_from_moves(s)
        positions.append({
            "id": i,
            "moves": s,
            "n_moves": len(s),
            "turn": env.turn,
            "scores": scores,
            "optimal": optimal_columns(scores),
        })
        if (i + 1) % 50 == 0:
            print(f"  gán nhãn {i + 1}/{len(seqs)}")

    data = {
        "meta": {
            "seed": args.seed, "n": len(positions),
            "min_moves": args.min_moves, "max_moves": args.max_moves,
            "generator": "nước hợp lệ ngẫu nhiên (random.Random(seed)), bỏ thế đã kết thúc và thế trùng",
            "label_source": source,
            "score_convention": "dương: người đi thắng (lớn = nhanh hơn), 0: hòa, âm: thua; -1000: cột đầy",
        },
        "positions": positions,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    n_multi = sum(len(p["optimal"]) > 1 for p in positions)
    print(f"Nguồn nhãn: {source}")
    print(f"Đã ghi {args.out} ({n_multi} thế có nhiều hơn một nước tối ưu).")


if __name__ == "__main__":
    main()

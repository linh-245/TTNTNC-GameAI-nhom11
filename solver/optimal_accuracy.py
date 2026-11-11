"""
Đo độ chính xác nước đi tối ưu của từng tác tử trên bộ thế chuẩn.
================================================================
Đọc solver/positions.json (do make_positions.py sinh), cho mỗi tác tử chọn
nước ở từng thế, tính % nước chọn nằm trong tập nước tối ưu của thế đó.

Chạy:
  python solver/optimal_accuracy.py
  python solver/optimal_accuracy.py --agents Random Think3 "AlphaBeta d5" --seed 0
  python solver/optimal_accuracy.py --budget-ms 50 --out results/optimal_accuracy.csv

Tái lập: trước mỗi lần gọi tác tử đều đặt random.seed(seed + id thế), nên cùng
seed thì Random/Think1/MCTS cho cùng kết quả. Tác tử trả nước bất hợp lệ bị
tính là sai và đếm riêng. Độ chính xác kèm khoảng tin cậy Wilson 95%.
"""
import argparse
import csv
import inspect
import json
import math
import os
import random
import sys
import time
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.connect4_env import ConnectFourEnv

HERE = os.path.dirname(os.path.abspath(__file__))


def wilson_ci(hits: int, n: int, z: float = 1.96):
    """Khoảng tin cậy Wilson cho tỷ lệ hits/n."""
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def build_agents(budget_ms=None):
    """Trả về {tên: hàm agent}. Tác tử chưa có (NN, RL) bị bỏ qua nếu import lỗi."""
    from agents.random_agent import conn_random
    from agents.heuristic_agent import conn_think1, conn_think2, conn_think3
    from agents.alphabeta_agent import alphabeta_agent
    from agents.mcts_agent import mcts_agent

    def with_budget(fn, **kw):
        """Thêm budget_ms nếu agent hỗ trợ."""
        if budget_ms is not None and 'budget_ms' in inspect.signature(fn).parameters:
            kw['budget_ms'] = budget_ms
        return partial(fn, **kw)

    agents = {
        'Random': conn_random,
        'Think1': conn_think1,
        'Think2': conn_think2,
        'Think3': conn_think3,
        'AlphaBeta d5': with_budget(alphabeta_agent, depth=5),
        'MCTS-200': with_budget(mcts_agent, n_simulations=200),
    }
    try:
        from agents.nn_agent import nn_agent
        agents['NN (SL)'] = partial(nn_agent, ckpt='weights/sl.pt')
        agents['RL'] = partial(nn_agent, ckpt='weights/rl.pt')
    except Exception:
        pass
    return agents


def env_from_moves(moves: str) -> ConnectFourEnv:
    env = ConnectFourEnv()
    env.reset()
    for ch in moves:
        env.step(int(ch))
    return env


def evaluate(agent, positions, seed: int):
    """Trả về dict thống kê của một tác tử trên danh sách thế."""
    hits = invalid = 0
    latencies = []
    errors = 0
    for p in positions:
        env = env_from_moves(p['moves'])
        random.seed(seed + p['id'])
        t0 = time.perf_counter()
        try:
            move = agent(env)
        except Exception:
            errors += 1
            continue
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if move not in env.legal_actions():
            invalid += 1
        elif move in p['optimal']:
            hits += 1
    n = len(positions)
    lo, hi = wilson_ci(hits, n)
    lat_sorted = sorted(latencies)
    p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))] if lat_sorted else float('nan')
    return {
        'n': n, 'hits': hits, 'accuracy': hits / n if n else 0.0,
        'ci_low': lo, 'ci_high': hi, 'invalid': invalid, 'errors': errors,
        'mean_ms': sum(latencies) / len(latencies) if latencies else float('nan'),
        'p95_ms': p95,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--positions', default=os.path.join(HERE, 'positions.json'))
    ap.add_argument('--agents', nargs='*', default=None, help='tên tác tử (mặc định: tất cả)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--budget-ms', type=int, default=None)
    ap.add_argument('--limit', type=int, default=None, help='chỉ dùng N thế đầu (chạy thử)')
    ap.add_argument('--out', default=None, help='ghi CSV kết quả')
    args = ap.parse_args()

    with open(args.positions, encoding='utf-8') as f:
        data = json.load(f)
    positions = data['positions'][:args.limit] if args.limit else data['positions']
    meta = data['meta']
    print(f"{len(positions)} thế, nguồn nhãn: {meta['label_source']}, seed thế = {meta['seed']}")

    agents = build_agents(args.budget_ms)
    names = args.agents if args.agents else list(agents)
    unknown = [n for n in names if n not in agents]
    if unknown:
        sys.exit(f"Không có tác tử: {unknown}. Có: {list(agents)}")

    rows = []
    print(f"{'Tác tử':<14}{'Chính xác':>10}{'CI 95%':>18}{'Sai nước':>10}{'ms TB':>9}{'ms p95':>9}")
    for name in names:
        r = evaluate(agents[name], positions, args.seed)
        r['agent'] = name
        rows.append(r)
        ci = f"[{100 * r['ci_low']:.1f}, {100 * r['ci_high']:.1f}]"
        print(f"{name:<14}{100 * r['accuracy']:>9.1f}%{ci:>18}{r['invalid'] + r['errors']:>10}"
              f"{r['mean_ms']:>9.1f}{r['p95_ms']:>9.1f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['agent', 'n', 'hits', 'accuracy', 'ci_low',
                                              'ci_high', 'invalid', 'errors', 'mean_ms', 'p95_ms'])
            w.writeheader()
            w.writerows(rows)
        print(f"Đã ghi {args.out}")


if __name__ == '__main__':
    main()

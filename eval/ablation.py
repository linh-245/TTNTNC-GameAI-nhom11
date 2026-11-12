"""
Ablation
=============
A1: Alpha-Beta bật/tắt cắt tỉa và bật/tắt sắp xếp nước ở CÙNG độ sâu:
      - số nút và thời gian mỗi nước (trên cùng một tập thế cờ)
      - độ sâu đạt được trong 50 ms và 200 ms (deepening lặp)
      - tỷ lệ thắng trước một đối thủ cố định
A2: MCTS với n_simulations in {50, 100, 200, 500, 1000} và c in {0.5, 1.0, 1.414, 2.0},
    tỷ lệ thắng trước một đối thủ cố định.

Ghi JSONL (mỗi dòng một bản ghi, trường "kind"):
    meta | A1_cost | A1_budget | A1_match | A2_match
và log từng ván ở <out>_games.jsonl.

Số ván mỗi cấu hình mặc định là FULL_GAMES = 100. Có thể giảm bằng --games; khi đó
mọi bản ghi match và dòng meta đều có reduced=true cùng ghi chú số ván thực tế.

Chạy thử cấu hình nhỏ:
  python eval/ablation.py --part all --games 4 --positions 5 --a2-sims 50,100 \
         --a2-c 1.0,1.414 --out logs/ablation_smoke.jsonl
Chạy đầy đủ:
  python eval/ablation.py --part all --games 100 --workers 4 --out logs/ablation.jsonl
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import platform
import random
import time
from typing import List, Optional

import numpy as np

from game.connect4_env import ConnectFourEnv
from agents.alphabeta_agent import alphabeta_agent
from eval.tournament import (make_match_tasks, run_tasks, spec_from_name,
                             summarize_match)

FULL_GAMES = 100
A1_CONFIGS = [(True, True), (True, False), (False, True), (False, False)]  # (pruning, ordering)
A2_SIMS = [50, 100, 200, 500, 1000]
A2_C = [0.5, 1.0, 1.414, 2.0]
A1_BUDGETS_MS = [50, 200]


def _emit(fh, row: dict):
    fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    fh.flush()


def _floats(arg: str) -> List[float]:
    return [float(x) for x in arg.split(',') if x.strip()]


def _ints(arg: str) -> List[int]:
    return [int(x) for x in arg.split(',') if x.strip()]


def _has_immediate_win(env: ConnectFourEnv) -> bool:
    for m in env.validinputs:
        e = env.copy()
        _, reward, done, _ = e.step(m)
        if done and reward != 0:
            return True
    return False


def sample_positions(n: int, seed: int, lo: int = 6, hi: int = 26) -> List[ConnectFourEnv]:
    """
    n thế cờ chưa kết thúc (lo..hi nước đã đi), cố định theo seed. Bỏ các thế mà người
    sắp đi có nước thắng ngay, vì agent trả về tức thì và làm lệch số nút / độ sâu.
    """
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        env = ConnectFourEnv()
        for _ in range(rng.randint(lo, hi)):
            env.step(rng.choice(env.validinputs))
            if env.done:
                break
        if not env.done and not _has_immediate_win(env):
            out.append(env)
    return out


def _summary(values: List[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {'mean': round(float(a.mean()), 3), 'median': round(float(np.median(a)), 3),
            'min': round(float(a.min()), 3), 'max': round(float(a.max()), 3),
            'p95': round(float(np.percentile(a, 95)), 3)}


def _match_row(kind: str, config: dict, records: List[dict], opponent: str,
               games_requested: int) -> dict:
    s = summarize_match(records)
    reduced = len(records) < FULL_GAMES
    return {
        'kind': kind, 'config': config, 'opponent': opponent,
        'games': s['n_games'], 'games_requested': games_requested,
        'wins': s['wins'], 'losses': s['losses'], 'draws': s['draws'],
        'score': s['score'], 'wilson_ci': list(s['wilson_ci']),
        'forfeits': s['forfeits'],
        'latency_ms': s['latency'][s['agent1']],
        'reduced': reduced,
        'note': (f"giảm số ván: {len(records)}/{FULL_GAMES} ván mỗi cấu hình"
                 if reduced else None),
    }


# ─────────────────────────────────────────────────────────────────────────────
# A1
# ─────────────────────────────────────────────────────────────────────────────

def a1_cost_and_budget(fh, depth: int, n_positions: int, seed: int,
                       budgets: List[int], max_depth: int):
    positions = sample_positions(n_positions, seed)
    print(f"\n[A1] số nút / thời gian / độ sâu trong budget — {n_positions} thế cờ, depth={depth}")
    print(f"{'pruning':>8}{'order':>7}{'nút (tb)':>11}{'ms/nước (tb)':>14}  độ sâu đạt được " +
          " | ".join(f"{b}ms" for b in budgets))
    for pruning, ordering in A1_CONFIGS:
        cfg = {'use_pruning': pruning, 'use_ordering': ordering, 'depth': depth}
        nodes, times, per_pos = [], [], []
        for env in positions:
            t0 = time.perf_counter()
            alphabeta_agent(env, depth=depth, use_pruning=pruning, use_ordering=ordering)
            ms = (time.perf_counter() - t0) * 1000.0
            n = alphabeta_agent.last_stats['nodes']
            nodes.append(n); times.append(ms)
            per_pos.append({'nodes': n, 'ms': round(ms, 3)})
        _emit(fh, {'kind': 'A1_cost', 'config': cfg, 'n_positions': n_positions,
                   'nodes': _summary(nodes), 'ms_per_move': _summary(times),
                   'per_position': per_pos})

        depth_txt = []
        for b in budgets:
            reached, timed_out = [], 0
            for env in positions:
                alphabeta_agent(env, depth=max_depth, budget_ms=b,
                                use_pruning=pruning, use_ordering=ordering)
                st = alphabeta_agent.last_stats
                reached.append(st['depth']); timed_out += st['timed_out']
            _emit(fh, {'kind': 'A1_budget',
                       'config': {'use_pruning': pruning, 'use_ordering': ordering,
                                  'budget_ms': b, 'max_depth': max_depth},
                       'n_positions': n_positions, 'depth_reached': _summary(reached),
                       'timed_out_rate': round(timed_out / n_positions, 3)})
            depth_txt.append(f"{np.mean(reached):.1f}")
        print(f"{str(pruning):>8}{str(ordering):>7}{np.mean(nodes):>11.0f}"
              f"{np.mean(times):>14.2f}  {' | '.join(depth_txt)}")


def a1_matches(fh, games_fh, depth: int, games: int, opponent: str, seed: int,
               openers: int, workers: int):
    print(f"\n[A1] tỷ lệ thắng trước {opponent} — {games} ván/cấu hình, depth={depth}")
    opp_spec = spec_from_name(opponent)
    for idx, (pruning, ordering) in enumerate(A1_CONFIGS):
        cfg = {'use_pruning': pruning, 'use_ordering': ordering, 'depth': depth}
        name = f"AB(prune={int(pruning)},order={int(ordering)},d={depth})"
        spec = {'kind': 'alphabeta', 'kwargs': dict(cfg)}
        tasks = make_match_tasks(name, spec, opponent, opp_spec, games, seed,
                                 pair_index=idx, n_random_openers=openers,
                                 tag={'exp': 'A1', 'config': cfg})
        recs = run_tasks(tasks, workers, games_fh, verbose=False)
        row = _match_row('A1_match', cfg, recs, opponent, games)
        _emit(fh, row)
        print(f"  {name}: score={row['score']:.3f} W={row['wins']} L={row['losses']} "
              f"D={row['draws']} CI={tuple(row['wilson_ci'])}")


# ─────────────────────────────────────────────────────────────────────────────
# A2
# ─────────────────────────────────────────────────────────────────────────────

def a2_matches(fh, games_fh, sims_list: List[int], c_list: List[float], games: int,
               opponent: str, seed: int, openers: int, workers: int):
    print(f"\n[A2] MCTS n_simulations x c trước {opponent} — {games} ván/cấu hình")
    opp_spec = spec_from_name(opponent)
    print(f"{'sims':>6}" + "".join(f"{'c=' + str(c):>10}" for c in c_list))
    idx = 100  # tách dải seed khỏi A1
    for n in sims_list:
        cells = []
        for c in c_list:
            cfg = {'n_simulations': n, 'c': c}
            name = f"MCTS(n={n},c={c})"
            spec = {'kind': 'mcts', 'kwargs': dict(cfg)}
            tasks = make_match_tasks(name, spec, opponent, opp_spec, games, seed,
                                     pair_index=idx, n_random_openers=openers,
                                     tag={'exp': 'A2', 'config': cfg})
            idx += 1
            recs = run_tasks(tasks, workers, games_fh, verbose=False)
            row = _match_row('A2_match', cfg, recs, opponent, games)
            _emit(fh, row)
            cells.append(f"{row['score']:.2f}")
        print(f"{n:>6}" + "".join(f"{x:>10}" for x in cells))


# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description="Ablation A1 (Alpha-Beta) và A2 (MCTS)")
    ap.add_argument('--part', choices=['A1', 'A2', 'all'], default='all')
    ap.add_argument('--games', type=int, default=FULL_GAMES,
                    help=f"số ván mỗi cấu hình (đầy đủ = {FULL_GAMES}; ít hơn sẽ ghi reduced)")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--opponent', default='Think2',
                    help="đối thủ cố định: Random, Think1, Think2, Think3, ...")
    ap.add_argument('--depth', type=int, default=4, help="độ sâu cố định cho A1")
    ap.add_argument('--positions', type=int, default=20, help="số thế cờ đo nút/thời gian (A1)")
    ap.add_argument('--budgets', default=','.join(map(str, A1_BUDGETS_MS)),
                    help="budget_ms đo độ sâu đạt được, ví dụ 50,200")
    ap.add_argument('--max-depth', type=int, default=12, help="trần độ sâu khi đo budget")
    ap.add_argument('--a2-sims', default=','.join(map(str, A2_SIMS)))
    ap.add_argument('--a2-c', default=','.join(map(str, A2_C)))
    ap.add_argument('--openers', type=int, default=2, help="số nước đầu ngẫu nhiên theo seed")
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--out', default=os.path.join('logs', 'ablation.jsonl'))
    args = ap.parse_args(argv)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stem, _ = os.path.splitext(out_path)
    reduced = args.games < FULL_GAMES

    with open(out_path, 'w', encoding='utf-8') as fh, \
         open(stem + '_games.jsonl', 'w', encoding='utf-8') as games_fh:
        _emit(fh, {
            'kind': 'meta', 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'part': args.part, 'seed': args.seed, 'opponent': args.opponent,
            'games': args.games, 'full_games': FULL_GAMES, 'reduced': reduced,
            'note': (f"giảm số ván: {args.games}/{FULL_GAMES} ván mỗi cấu hình"
                     if reduced else None),
            'a1_depth': args.depth, 'positions': args.positions,
            'budgets_ms': _ints(args.budgets), 'openers': args.openers,
            'workers': args.workers, 'python': platform.python_version(),
            'machine': platform.platform(),
        })
        if reduced:
            print(f"Lưu ý: chạy giảm số ván ({args.games}/{FULL_GAMES}); đã ghi rõ trong log.")
        if args.part in ('A1', 'all'):
            a1_cost_and_budget(fh, args.depth, args.positions, args.seed,
                               _ints(args.budgets), args.max_depth)
            a1_matches(fh, games_fh, args.depth, args.games, args.opponent,
                       args.seed, args.openers, args.workers)
        if args.part in ('A2', 'all'):
            a2_matches(fh, games_fh, _ints(args.a2_sims), _floats(args.a2_c),
                       args.games, args.opponent, args.seed, args.openers, args.workers)

    print(f"\nLog tổng hợp: {out_path}\nLog từng ván: {stem}_games.jsonl")


if __name__ == '__main__':
    main()

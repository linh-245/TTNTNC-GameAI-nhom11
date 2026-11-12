"""
Round-Robin Tournament Evaluation (M5)
======================================
Thay cho notebook tournament.ipynb. Tái dùng wilson_ci / play_match / round_robin,
bổ sung:
  - đa tiến trình (--workers)
  - chia đều lượt đi trước/đi sau; hai ván liền nhau của một cặp dùng chung khai cuộc
  - khai cuộc k nước ngẫu nhiên theo seed (--openers = n_random_openers)
  - log JSONL mỗi ván: cặp agent, ai đi trước, kết quả theo góc nhìn agent 1, các nước đi,
    độ trễ từng nước, số nút hoặc số mô phỏng, seed
  - Win Score, khoảng tin cậy Wilson, độ trễ trung bình và p95 TÍNH THEO TỪNG NƯỚC
    (gom từ log, không phải theo ván)
  - xuất ma trận kết quả CSV

Chạy:
  python eval/tournament.py --games 100 --seed 42 --out logs/tournament_games.jsonl
  python eval/tournament.py --games 10 --agents Think2 Random --budget-ms 50

Quy ước gọi agent: fn(env) -> cột 1-7 (giống alphabeta_agent, mcts_agent).
Agent chạy song song nên độ trễ đo được chịu ảnh hưởng của tải CPU; muốn đo độ trễ
sạch hơn thì dùng --workers 1.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

from game.connect4_env import ConnectFourEnv

DEFAULT_AGENTS = ['Random', 'Think1', 'Think2', 'Think3', 'AlphaBeta', 'MCTS']


# ─────────────────────────────────────────────────────────────────────────────
# Thống kê
# ─────────────────────────────────────────────────────────────────────────────

def wilson_ci(wins: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score confidence interval cho tỷ lệ thắng.

    Parameters
    ----------
    wins : số ván thắng (có thể là wins + 0.5 * draws khi tính theo Win Score)
    n : tổng số ván
    z : 1.96 = 95% CI

    Returns
    -------
    (lower, upper)
    """
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


# ─────────────────────────────────────────────────────────────────────────────
# Mô tả agent bằng dict (spec) để truyền qua tiến trình con
# ─────────────────────────────────────────────────────────────────────────────

def spec_from_name(name: str, budget_ms: Optional[float] = None,
                   depth: int = 5, n_sims: int = 200, max_depth: int = 12) -> dict:
    """
    'Random', 'Think1..3', 'AlphaBeta' / 'AlphaBeta-<depth>', 'MCTS' / 'MCTS-<n_sims>'.
    Có budget_ms: AlphaBeta dùng deepening lặp (tới max_depth), MCTS lặp đến hết giờ.
    """
    base, _, arg = name.partition('-')
    key = base.lower()
    if key == 'random':
        return {'kind': 'random'}
    if key in ('think1', 'think2', 'think3'):
        return {'kind': key}
    if key == 'alphabeta':
        d = int(arg) if arg else depth
        if budget_ms is not None:
            return {'kind': 'alphabeta',
                    'kwargs': {'depth': max_depth, 'budget_ms': budget_ms}}
        return {'kind': 'alphabeta', 'kwargs': {'depth': d}}
    if key == 'mcts':
        n = int(arg) if arg else n_sims
        if budget_ms is not None:
            return {'kind': 'mcts', 'kwargs': {'budget_ms': budget_ms}}
        return {'kind': 'mcts', 'kwargs': {'n_simulations': n}}
    raise ValueError(f"Agent không hợp lệ: {name}")


def build_agent(spec: dict):
    """Trả về (fn(env, seed) -> cột, owner) ; owner có .last_stats (hoặc None)."""
    kind = spec['kind']
    kw = dict(spec.get('kwargs', {}))
    if kind == 'random':
        from agents.random_agent import conn_random
        return (lambda env, seed: conn_random(env)), None
    if kind in ('think1', 'think2', 'think3'):
        from agents import heuristic_agent
        fn = getattr(heuristic_agent, 'conn_' + kind)
        return (lambda env, seed: fn(env)), None
    if kind == 'alphabeta':
        from agents.alphabeta_agent import alphabeta_agent
        return (lambda env, seed: alphabeta_agent(env, **kw)), alphabeta_agent
    if kind == 'mcts':
        from agents.mcts_agent import mcts_agent
        return (lambda env, seed: mcts_agent(env, seed=seed, **kw)), mcts_agent
    raise ValueError(f"Loại agent không hợp lệ: {kind}")


def _work(owner) -> dict:
    """Số nút / độ sâu (Alpha-Beta) hoặc số mô phỏng (MCTS) của nước vừa đi."""
    stats = getattr(owner, 'last_stats', None) if owner is not None else None
    if not stats:
        return {}
    return {k: int(stats[k]) for k in ('nodes', 'depth', 'sims') if k in stats}


# ─────────────────────────────────────────────────────────────────────────────
# Một ván cờ (chạy trong tiến trình con)
# ─────────────────────────────────────────────────────────────────────────────

def play_game(task: dict) -> dict:
    """
    Chơi một ván theo `task` và trả về bản ghi JSON-able.
    Kết quả tính theo góc nhìn agent 1 (pair[0]). Nước sai luật hoặc agent báo lỗi
    thì agent đó bị xử thua và ghi vào 'forfeit' / 'error'.
    """
    name1, name2 = task['pair']
    names = [name1, name2]
    a_first = task['a_first']
    seed = task['seed']

    random.seed(seed)                       # cho agent dùng random toàn cục
    np.random.seed(seed % (2**32))
    built = [build_agent(task['spec1']), build_agent(task['spec2'])]
    order = [0, 1] if a_first else [1, 0]   # order[ply % 2] = agent đi ở lượt đó

    env = ConnectFourEnv()
    openers = []
    opener_rng = random.Random(task['opener_seed'])
    for _ in range(task['n_random_openers']):
        if env.done:
            break
        m = int(opener_rng.choice(env.validinputs))
        env.step(m)
        openers.append(m)

    ply = len(openers)
    moves = []
    winner = None        # chỉ số agent thắng (0/1) hoặc None
    forfeit = None
    error = None

    while not env.done:
        idx = order[ply % 2]
        fn, owner = built[idx]
        t0 = time.perf_counter()
        try:
            col = int(fn(env.copy(), seed * 131 + ply))
        except Exception as exc:            # agent lỗi -> xử thua, ghi lại nguyên nhân
            forfeit, error = idx, f"{type(exc).__name__}: {exc}"
            break
        ms = (time.perf_counter() - t0) * 1000.0
        if col not in env.validinputs:
            forfeit, error = idx, f"illegal move {col}"
            break
        entry = {'agent': names[idx], 'col': col, 'ms': round(ms, 3)}
        entry.update(_work(owner))
        moves.append(entry)
        env.step(col)
        ply += 1
        if env.done and env.reward != 0:
            winner = idx

    if forfeit is not None:
        winner = 1 - forfeit

    if winner is None:
        result, score1 = 'draw', 0.5
    elif winner == 0:
        result, score1 = 'win', 1.0
    else:
        result, score1 = 'loss', 0.0

    return {
        'game_id': task['game_id'],
        'pair': [name1, name2],
        'first': names[order[0]],
        'a_first': a_first,
        'result': result,                    # theo góc nhìn agent 1
        'score1': score1,
        'winner': None if winner is None else names[winner],
        'forfeit': None if forfeit is None else names[forfeit],
        'error': error,
        'seed': seed,
        'opener_seed': task['opener_seed'],
        'n_random_openers': task['n_random_openers'],
        'openers': openers,
        'plies': len(openers) + len(moves),
        'moves': moves,
        'spec1': task['spec1'],
        'spec2': task['spec2'],
        'tag': task.get('tag', {}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tạo và chạy các ván
# ─────────────────────────────────────────────────────────────────────────────

def _derive(seed: int, pair_idx: int, game_idx: int, salt: int = 0) -> int:
    return seed * 1_000_003 + pair_idx * 10_007 + game_idx + salt


def make_match_tasks(name1: str, spec1: dict, name2: str, spec2: dict,
                     n_games: int, seed: int = 42, pair_index: int = 0,
                     n_random_openers: int = 0, tag: Optional[dict] = None) -> List[dict]:
    """
    n_games ván giữa hai agent. Ván chẵn: agent 1 đi trước; ván lẻ: agent 2 đi trước.
    Ván 2j và 2j+1 dùng chung khai cuộc (cùng opener_seed) nên mỗi agent đều chơi
    cả hai phía của cùng một khai cuộc.
    """
    tasks = []
    for g in range(n_games):
        tasks.append({
            'game_id': f"{name1}_vs_{name2}#{g}",
            'pair': [name1, name2],
            'spec1': spec1,
            'spec2': spec2,
            'a_first': g % 2 == 0,
            'seed': _derive(seed, pair_index, g),
            'opener_seed': _derive(seed, pair_index, g // 2, salt=500_000),
            'n_random_openers': n_random_openers,
            'tag': tag or {},
        })
    return tasks


def run_tasks(tasks: List[dict], workers: int = 1, log_fh=None,
              verbose: bool = True) -> List[dict]:
    """Chạy các ván (song song nếu workers > 1), ghi JSONL ngay khi có kết quả."""
    records = []
    total = len(tasks)

    def handle(rec):
        records.append(rec)
        if log_fh is not None:
            log_fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            log_fh.flush()
        if verbose and (len(records) % 20 == 0 or len(records) == total):
            print(f"    [{len(records)}/{total}] ván đã xong", flush=True)

    if workers <= 1:
        for t in tasks:
            handle(play_game(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for rec in pool.map(play_game, tasks, chunksize=1):
                handle(rec)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Tổng hợp kết quả từ log
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def move_latency_stats(records: List[dict], agent_name: str) -> dict:
    """Độ trễ TỪNG NƯỚC của agent (gom mọi nước đi của agent trong các ván)."""
    ms = [m['ms'] for r in records for m in r['moves'] if m['agent'] == agent_name]
    if not ms:
        return {'moves': 0, 'avg_ms': None, 'p95_ms': None}
    return {'moves': len(ms),
            'avg_ms': round(float(np.mean(ms)), 3),
            'p95_ms': round(float(np.percentile(ms, 95)), 3)}


def summarize_match(records: List[dict]) -> dict:
    """Tổng hợp các ván của MỘT cặp (theo góc nhìn agent 1)."""
    n1, n2 = records[0]['pair']
    n = len(records)
    wins = sum(r['result'] == 'win' for r in records)
    losses = sum(r['result'] == 'loss' for r in records)
    draws = sum(r['result'] == 'draw' for r in records)
    score = (wins + 0.5 * draws) / n
    lo, hi = wilson_ci(wins + 0.5 * draws, n)
    return {
        'agent1': n1, 'agent2': n2,
        'wins': wins, 'losses': losses, 'draws': draws, 'n_games': n,
        'score': round(score, 4),
        'wilson_ci': (round(lo, 4), round(hi, 4)),
        'forfeits': sum(r['forfeit'] is not None for r in records),
        'latency': {n1: move_latency_stats(records, n1),
                    n2: move_latency_stats(records, n2)},
    }


def play_match(name1: str, spec1: dict, name2: str, spec2: dict,
               n_games: int = 200, seed: int = 42, n_random_openers: int = 0,
               workers: int = 1, log_fh=None, verbose: bool = False) -> dict:
    """Chạy một cặp và trả về summarize_match(...)."""
    tasks = make_match_tasks(name1, spec1, name2, spec2, n_games, seed,
                             n_random_openers=n_random_openers)
    return summarize_match(run_tasks(tasks, workers, log_fh, verbose))


def write_matrix_csv(path: str, names: List[str], results: Dict[Tuple[str, str], dict]):
    """Ma trận Win Score: ô (hàng, cột) = điểm của agent ở hàng khi gặp agent ở cột."""
    cell = {}
    for (a, b), res in results.items():
        cell[(a, b)] = res['score']
        cell[(b, a)] = round(1.0 - res['score'], 4)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['agent'] + names + ['total'])
        for a in names:
            row = [cell.get((a, b), '') if a != b else '' for b in names]
            total = round(sum(v for v in row if v != ''), 4)
            w.writerow([a] + row + [total])


def round_robin(agents: Dict[str, dict], n_games_per_pair: int = 200, seed: int = 42,
                n_random_openers: int = 2, workers: int = 1,
                out_file: Optional[str] = None, csv_file: Optional[str] = None,
                verbose: bool = True) -> Dict:
    """
    Round-robin giữa các agent. agents = {tên: spec}. Mọi ván của mọi cặp được dồn
    vào một pool tiến trình chung.
    """
    names = list(agents.keys())
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
    if n_games_per_pair % 2:
        print("Cảnh báo: số ván mỗi cặp là số lẻ, lượt đi trước/sau sẽ lệch 1 ván.")

    print(f"\n{'='*60}\nROUND-ROBIN TOURNAMENT\nAgents: {names}\n"
          f"Ván mỗi cặp: {n_games_per_pair} | workers: {workers} | "
          f"openers: {n_random_openers} | seed: {seed}\n{'='*60}\n")

    tasks = []
    for p, (i, j) in enumerate(pairs):
        tasks += make_match_tasks(names[i], agents[names[i]], names[j], agents[names[j]],
                                  n_games_per_pair, seed, pair_index=p,
                                  n_random_openers=n_random_openers)

    log_fh = None
    if out_file:
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        log_fh = open(out_file, 'w', encoding='utf-8')
    t0 = time.perf_counter()
    try:
        records = run_tasks(tasks, workers, log_fh, verbose)
    finally:
        if log_fh:
            log_fh.close()
    elapsed = time.perf_counter() - t0

    # Tổng hợp lại từ log (đúng yêu cầu: số liệu lấy từ log)
    by_pair: Dict[Tuple[str, str], List[dict]] = {}
    for r in records:
        by_pair.setdefault(tuple(r['pair']), []).append(r)
    results = {}
    scores = {n: 0.0 for n in names}
    for (a, b), recs in by_pair.items():
        res = summarize_match(recs)
        results[(a, b)] = res
        scores[a] += res['score']
        scores[b] += 1.0 - res['score']
        la, lb = res['latency'][a], res['latency'][b]
        print(f"{a} vs {b}: score={res['score']:.3f} W={res['wins']} L={res['losses']} "
              f"D={res['draws']} CI={res['wilson_ci']} | "
              f"p95/nước: {a}={la['p95_ms']}ms {b}={lb['p95_ms']}ms")

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print("\n" + "=" * 60 + "\nFINAL RANKING\n" + "=" * 60)
    for rank, (name, sc) in enumerate(ranking, 1):
        lat = move_latency_stats(records, name)
        print(f"  {rank}. {name}: {sc:.3f}  (avg {lat['avg_ms']} ms, p95 {lat['p95_ms']} ms/nước)")
    print(f"\nTổng thời gian: {elapsed:.1f}s")

    if csv_file:
        os.makedirs(os.path.dirname(os.path.abspath(csv_file)), exist_ok=True)
        write_matrix_csv(csv_file, names, results)
        print(f"Ma trận CSV: {csv_file}")
    if out_file:
        print(f"Log JSONL : {out_file}")

    return {'agents': names, 'n_games_per_pair': n_games_per_pair, 'seed': seed,
            'results': {f"{a} vs {b}": r for (a, b), r in results.items()},
            'ranking': ranking, 'scores': scores}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description="Round-robin tournament cho Connect 4")
    ap.add_argument('--games', type=int, default=100, help="số ván mỗi cặp (nên chẵn, >= 100)")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--budget-ms', type=float, default=None,
                    help="ngân sách thời gian mỗi nước cho AlphaBeta/MCTS (ms)")
    ap.add_argument('--out', default=os.path.join('logs', 'tournament_games.jsonl'),
                    help="file log JSONL; ma trận CSV ghi cạnh đó (<out>_matrix.csv)")
    ap.add_argument('--agents', nargs='+', default=DEFAULT_AGENTS,
                    help="Random Think1 Think2 Think3 AlphaBeta[-d] MCTS[-n]")
    ap.add_argument('--openers', type=int, default=2,
                    help="số nước đầu ngẫu nhiên theo seed (n_random_openers, 0-4)")
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument('--depth', type=int, default=5, help="độ sâu AlphaBeta khi không có budget")
    ap.add_argument('--sims', type=int, default=200, help="n_simulations MCTS khi không có budget")
    args = ap.parse_args(argv)

    agents = {n: spec_from_name(n, args.budget_ms, args.depth, args.sims)
              for n in args.agents}
    stem, _ = os.path.splitext(args.out)
    round_robin(agents, n_games_per_pair=args.games, seed=args.seed,
                n_random_openers=args.openers, workers=args.workers,
                out_file=args.out, csv_file=stem + '_matrix.csv')


if __name__ == '__main__':
    main()

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import random
from pathlib import Path
from typing import Dict, Any

from game.connect4_env import ConnectFourEnv
from agents.alphabeta_agent import alphabeta_agent
from agents.mcts_agent import mcts_agent

ROOT = Path.cwd()
LOG_DIR = ROOT / "eval" / "logs" if ROOT.name != "eval" else ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def play_single_game(game_id: int, time_limit_ms: int, seed: int = 42) -> Dict[str, Any]:
    env = ConnectFourEnv()
    game_seed = seed + game_id
    is_ab_first = (game_id % 2 == 0)

    # Định nghĩa tên và spec khớp với cấu trúc của tournament/notebook
    ab_spec = {"name": "alphabeta", "kwargs": {"depth": 6, "budget_ms": time_limit_ms}}
    mcts_spec = {"name": "mcts", "kwargs": {"budget_ms": time_limit_ms}}

    if is_ab_first:
        pair = ["alphabeta", "mcts"]
        spec1, spec2 = ab_spec, mcts_spec
    else:
        pair = ["mcts", "alphabeta"]
        spec1, spec2 = mcts_spec, ab_spec

    moves_log = []
    ply = 0

    while not env.done:
        ply += 1
        current_turn = env.turn
        is_ab_turn = (current_turn == 'red' and is_ab_first) or (current_turn == 'yellow' and not is_ab_first)

        if is_ab_turn:
            t0 = time.perf_counter()
            move = alphabeta_agent(
                env,
                depth=6,
                budget_ms=time_limit_ms,
                use_eval=True,
                use_pruning=True,
                use_ordering=True
            )
            dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            stats = getattr(alphabeta_agent, "last_stats", {})
            moves_log.append({
                "ply": ply,
                "agent": "alphabeta",
                "move": int(move),
                "ms": stats.get("time_ms", dt_ms),
                "depth": stats.get("depth", 0),
                "nodes": stats.get("nodes", 0),
            })
        else:
            t0 = time.perf_counter()
            move = mcts_agent(
                env,
                budget_ms=time_limit_ms,
                seed=game_seed + ply
            )
            dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            stats = getattr(mcts_agent, "last_stats", {})
            moves_log.append({
                "ply": ply,
                "agent": "mcts",
                "move": int(move),
                "ms": stats.get("latency_ms", dt_ms),
                "sims": stats.get("sims", 0),
            })

        _, reward, done, _ = env.step(move)
        if done:
            break

    # env.reward: >0 là red (người đi 1) thắng, <0 là yellow (người đi 2) thắng, 0 là hòa
    if env.reward > 0:
        score1 = 1.0
    elif env.reward < 0:
        score1 = 0.0
    else:
        score1 = 0.5

    return {
        "game_id": game_id,
        "seed": game_seed,
        "pair": pair,
        "spec1": spec1,
        "spec2": spec2,
        "score1": score1,
        "moves": moves_log,
    }


def run_rq1_experiment(time_budgets_ms=(200, 1000), num_games: int = 100, seed: int = 42):
    print("=" * 60)
    print(" CHẠY THỰC NGHIỆM RQ1 (Alpha-Beta vs MCTS)")
    print(f" Thông số: {num_games} ván/mức | Ngân sách: {list(time_budgets_ms)} ms")
    print("=" * 60)

    for budget in time_budgets_ms:
        out_path = LOG_DIR / f"rq1_{budget}ms_games.jsonl"
        ab_wins, mcts_wins, draws = 0, 0, 0
        print(f"\n[Đang chạy] Budget = {budget} ms...")

        with open(out_path, "w", encoding="utf-8") as f:
            for game_id in range(1, num_games + 1):
                rec = play_single_game(game_id=game_id, time_limit_ms=budget, seed=seed)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                ab_score = rec["score1"] if rec["pair"][0] == "alphabeta" else (1.0 - rec["score1"])
                if ab_score == 1.0:
                    ab_wins += 1
                elif ab_score == 0.0:
                    mcts_wins += 1
                else:
                    draws += 1

                if game_id % 10 == 0 or game_id == num_games:
                    print(f"  -> Xong {game_id}/{num_games} ván (AB thắng: {ab_wins}, MCTS thắng: {mcts_wins}, Hòa: {draws})")

        print(f" Đã lưu: {out_path}")


if __name__ == "__main__":
    run_rq1_experiment(time_budgets_ms=(200, 1000), num_games=100, seed=42)
"""
Vẽ đường cong học của RL tự chơi từ log CSV
===========================================
Đọc hai file do nhóm tạo ra:
  - logs/training/rl_log.csv    : log từng vòng của training/train_rl.py
  - logs/training/rl_curve.csv  : kết quả training/eval_anchors.py cho SL (vòng 0) và các mốc RL

Ba hình cạnh nhau:
  (a) kết quả tự chơi mỗi vòng (đỏ thắng, vàng thắng, hòa trên 32 ván), trung bình trượt;
  (b) độ dài ván trung bình (số thế / số ván), cho thấy ván ngắn dần hay dài dần;
  (c) Win Score trước các đối thủ mốc theo số vòng huấn luyện, kèm khoảng tin cậy Wilson 95%.

Chạy (từ thư mục gốc repo):
    python -m training.plot_rl_curves --out results/figures/fig_learning_curve.png
"""
import argparse
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")   # không cần cửa sổ, chỉ lưu ra file ảnh
import matplotlib.pyplot as plt

ANCHORS = ["Random", "Think1", "Think2", "Think3"]
COLORS = {"Random": "#7f7f7f", "Think1": "#1f77b4", "Think2": "#d95f02", "Think3": "#2ca02c"}


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def moving_avg(values, window):
    """Trung bình trượt (cửa sổ lùi về trước, đầu chuỗi dùng số phần tử có sẵn)."""
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def iteration_of(label):
    """'it0050' -> 50, 'it0000_SL' -> 0 (bản SL là điểm xuất phát, vòng 0)."""
    m = re.match(r"it(\d+)", label)
    if not m:
        raise ValueError(f"nhãn không đúng dạng itNNNN: {label}")
    return int(m.group(1))


def load_selfplay(path):
    rows = read_rows(path)
    games = [int(r["red_wins"]) + int(r["yellow_wins"]) + int(r["draws"]) for r in rows]
    return {
        "iter": [int(r["iter"]) for r in rows],
        "red": [int(r["red_wins"]) for r in rows],
        "yellow": [int(r["yellow_wins"]) for r in rows],
        "draw": [int(r["draws"]) for r in rows],
        "moves": [int(r["positions"]) / g for r, g in zip(rows, games)],
        "games": games,
    }


def load_anchors(path):
    """-> {anchor: [(vòng, win_score, ci_lo, ci_hi, số lỗi), ...]} sắp theo vòng."""
    out = {a: [] for a in ANCHORS}
    for r in read_rows(path):
        out[r["anchor"]].append((iteration_of(r["label"]), float(r["win_score"]),
                                 float(r["ci_lo"]), float(r["ci_hi"]), int(r["loi"])))
    for a in out:
        out[a].sort()
    return out


def draw(selfplay, anchors, window=10):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    it = selfplay["iter"]

    ax = axes[0]
    for key, name, color in (("red", "Đỏ (đi trước) thắng", "#c0392b"),
                             ("yellow", "Vàng thắng", "#e0a800"),
                             ("draw", "Hòa", "#34495e")):
        ax.plot(it, moving_avg(selfplay[key], window), label=name, color=color, lw=1.8)
    ax.set_title(f"(a) Kết quả tự chơi mỗi vòng (32 ván, trung bình trượt {window} vòng)", fontsize=10)
    ax.set_xlabel("Vòng huấn luyện")
    ax.set_ylabel("Số ván")
    ax.set_ylim(0, 33)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(it, selfplay["moves"], color="#999999", lw=0.8, label="từng vòng")
    ax.plot(it, moving_avg(selfplay["moves"], window), color="#2c3e50", lw=2, label=f"trung bình trượt {window} vòng")
    ax.axhline(42, color="#c0392b", ls="--", lw=1, label="đầy bàn (42 nước)")
    ax.set_title("(b) Độ dài ván tự chơi (số nước trung bình)", fontsize=10)
    ax.set_xlabel("Vòng huấn luyện")
    ax.set_ylabel("Số nước mỗi ván")
    ax.set_ylim(0, 45)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)

    ax = axes[2]
    for a in ANCHORS:
        pts = anchors[a]
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        lo = [p[1] - p[2] for p in pts]
        hi = [p[3] - p[1] for p in pts]
        ax.errorbar(x, y, yerr=[lo, hi], marker="o", ms=4, lw=1.6, capsize=2.5,
                    label=f"vs {a}", color=COLORS[a])
    ax.axhline(0.5, color="#888888", ls=":", lw=1)
    ax.set_title("(c) Win Score của mạng trước đối thủ mốc (vòng 0 = bản SL)", fontsize=10)
    ax.set_xlabel("Vòng huấn luyện")
    ax.set_ylabel("Win Score (thắng + 0,5 × hòa)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Vẽ đường cong học của RL từ log CSV")
    ap.add_argument("--rl-log", default="logs/training/rl_log.csv")
    ap.add_argument("--curve", default="logs/training/rl_curve.csv")
    ap.add_argument("--out", default="results/figures/fig_learning_curve.png")
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()

    selfplay = load_selfplay(args.rl_log)
    anchors = load_anchors(args.curve)
    if any(p[4] for pts in anchors.values() for p in pts):
        print("CẢNH BÁO: có ván lỗi trong rl_curve.csv (cột loi khác 0); các điểm đó không dùng được.")
    fig = draw(selfplay, anchors, args.window)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Đã lưu {args.out}")


if __name__ == "__main__":
    main()
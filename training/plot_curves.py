"""
Vẽ đường cong học từ log CSV của training/train_sl.py.
======================================================
Ba hình cạnh nhau: loss policy, loss value, độ chính xác policy (train và val theo epoch).

Chạy (từ thư mục gốc repo):
    python -m training.plot_curves --csv logs/training/sl_log.csv --out docs/images/sl_curves.png
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")   # không cần cửa sổ, chỉ lưu ra file ảnh
import matplotlib.pyplot as plt


def read_csv(path):
    """Đọc CSV thành dict: tên cột -> danh sách số."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def main():
    ap = argparse.ArgumentParser(description="Vẽ đường cong học từ log CSV")
    ap.add_argument("--csv", default="logs/training/sl_log.csv")
    ap.add_argument("--out", default="docs/images/sl_curves.png")
    ap.add_argument("--title", default="Học có giám sát (bắt chước Think3)")
    args = ap.parse_args()

    d = read_csv(args.csv)
    epochs = d["epoch"]
    panels = [
        ("Loss policy", "train_policy_loss", "val_policy_loss"),
        ("Loss value", "train_value_loss", "val_value_loss"),
        ("Độ chính xác policy so với giáo viên", "train_acc", "val_policy_acc"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (name, tr, va) in zip(axes, panels):
        ax.plot(epochs, d[tr], marker="o", markersize=3, label="train")
        ax.plot(epochs, d[va], marker="o", markersize=3, label="val")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(args.title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Đã lưu {args.out}")


if __name__ == "__main__":
    main()
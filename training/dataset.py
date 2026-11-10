"""
Xử lý dữ liệu từ training/generate_data.py: loại thế cờ trùng, chia train/val/test.
===================================================================================
Hai điểm quan trọng:
  1. Loại trùng thế: nhiều ván có thể đi qua cùng một thế cờ (nhất là các nước đầu).
     Ta giữ mỗi thế một lần; nhãn value của thế trùng được lấy trung bình qua các ván.
  2. Chia THEO VÁN, không chia theo từng thế: mọi thế của một ván nằm cùng một tập.
     Vì đã loại trùng trước khi chia nên không có thế nào xuất hiện ở 2 tập khác nhau.

Chạy thử (từ thư mục gốc repo):
    python -m training.dataset --in data/teacher_think3.npz
"""
import argparse

import numpy as np


def load_npz(path):
    """Đọc file .npz thành dict các mảng numpy."""
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def dedup_positions(data):
    """Loại thế cờ trùng (cùng tensor x). Giữ lần xuất hiện đầu tiên theo thứ tự ván.

    policy, game, ply lấy từ lần xuất hiện đầu; value là trung bình qua mọi lần trùng.
    """
    x = data["x"]
    flat = x.reshape(len(x), -1)
    _, first_idx, inverse = np.unique(
        flat, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    counts = np.bincount(inverse)
    value_mean = np.bincount(inverse, weights=data["value"]) / counts
    order = np.argsort(first_idx)          # trả về đúng thứ tự xuất hiện ban đầu
    keep = first_idx[order]
    out = {k: v[keep] for k, v in data.items()}
    out["value"] = value_mean[order].astype(np.float32)
    return out


def split_by_game(data, val_frac=0.1, test_frac=0.1, seed=0):
    """Chia (train, val, test) theo ván. Cùng seed thì chia giống nhau."""
    games = np.unique(data["game"])
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n = len(games)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_g = games[:n_test]
    val_g = games[n_test:n_test + n_val]
    train_g = games[n_test + n_val:]

    def pick(gs):
        mask = np.isin(data["game"], gs)
        return {k: v[mask] for k, v in data.items()}

    return pick(train_g), pick(val_g), pick(test_g)


def build_splits(path, val_frac=0.1, test_frac=0.1, seed=0):
    """Đọc file, loại trùng, chia theo ván. Trả về (train, val, test)."""
    data = dedup_positions(load_npz(path))
    return split_by_game(data, val_frac, test_frac, seed)


def main():
    ap = argparse.ArgumentParser(description="Loại trùng và chia train/val/test")
    ap.add_argument("--in", dest="inp", required=True, help="file .npz từ generate_data")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = load_npz(args.inp)
    print(f"Ban đầu : {len(raw['policy'])} thế cờ, {len(np.unique(raw['game']))} ván")
    train, val, test = build_splits(args.inp, args.val, args.test, args.seed)
    total = sum(len(s["policy"]) for s in (train, val, test))
    print(f"Sau khi loại trùng: {total} thế cờ")
    for name, s in (("train", train), ("val", val), ("test", test)):
        print(f"  {name:5s}: {len(s['policy']):6d} thế cờ, {len(np.unique(s['game'])):4d} ván")


if __name__ == "__main__":
    main()
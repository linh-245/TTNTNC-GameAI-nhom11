"""
Huấn luyện có giám sát (SL) mạng policy-value cho Connect Four.
================================================================
Học bắt chước agent giáo viên (conn_think3) từ dữ liệu của training/generate_data.py.

  - Dữ liệu : loại trùng và chia train/val/test THEO VÁN (training/dataset.py)
  - Loss    : cross-entropy cho policy (che cột đầy) + MSE cho value
  - Chọn mô hình: theo loss trên tập VAL. Tập TEST không dùng ở đây, chỉ chạy một lần
    sau khi đã chốt siêu tham số (đề yêu cầu).
  - Lưu lại: checkpoint tốt nhất (.pt) và log huấn luyện (.json) gồm seed, thiết bị,
    phiên bản torch, thời gian huấn luyện.

Chạy thử (từ thư mục gốc repo):
    python -m training.train_sl --data data/teacher_think3.npz --epochs 5
"""
import argparse
import csv
import json
import os
import platform
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from training.dataset import build_splits
from training.model import PolicyValueNet


def set_seed(seed):
    """Cố định seed để chạy lại cho kết quả giống nhau."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensors(split, device):
    """Đổi một tập (dict numpy) sang tensor đặt trên thiết bị."""
    x = torch.from_numpy(split["x"]).float().to(device)
    policy = torch.from_numpy(split["policy"]).long().to(device)
    value = torch.from_numpy(split["value"]).float().to(device)
    return x, policy, value


def compute_losses(model, x, policy, value):
    """Trả về (loss policy, loss value, độ chính xác policy so với giáo viên)."""
    logits, v = model(x)
    # Kênh 2 của đầu vào đánh dấu ô rơi xuống ở mỗi cột còn đánh được -> cột nào có số 1 là hợp lệ
    legal = x[:, 2].sum(dim=1) > 0                      # (B, 7)
    logits = logits.masked_fill(~legal, float("-inf"))
    p_loss = F.cross_entropy(logits, policy)
    v_loss = F.mse_loss(v, value)
    acc = (logits.argmax(dim=1) == policy).float().mean()
    return p_loss, v_loss, acc


@torch.no_grad()
def evaluate(model, tensors, value_weight, batch_size=1024):
    """Tính loss và độ chính xác trên một tập, trung bình theo số thế cờ."""
    model.eval()
    x, policy, value = tensors
    n = len(x)
    sums = np.zeros(3)  # policy loss, value loss, acc
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        p_loss, v_loss, acc = compute_losses(model, x[sl], policy[sl], value[sl])
        m = len(x[sl])
        sums += np.array([p_loss.item(), v_loss.item(), acc.item()]) * m
    p_loss, v_loss, acc = sums / n
    return {"policy_loss": p_loss, "value_loss": v_loss,
            "loss": p_loss + value_weight * v_loss, "policy_acc": acc}


def main():
    ap = argparse.ArgumentParser(description="Huấn luyện SL cho mạng policy-value")
    ap.add_argument("--data", required=True, help="file .npz từ generate_data")
    ap.add_argument("--out", default="weights/sl.pt", help="đường dẫn checkpoint")
    ap.add_argument("--log-csv", default="logs/training/sl_log.csv", help="log từng epoch dạng CSV")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, val, test = build_splits(args.data, args.val, args.test, args.seed)
    print(f"Thiết bị: {device} | train {len(train['policy'])} | "
          f"val {len(val['policy'])} | test {len(test['policy'])} (không dùng)")
    tr, va = to_tensors(train, device), to_tensors(val, device)

    model = PolicyValueNet(args.channels, args.blocks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    history, best_val, best_epoch = [], float("inf"), 0
    t0 = time.time()
    n = len(tr[0])
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        run = np.zeros(3)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            p_loss, v_loss, acc = compute_losses(model, tr[0][idx], tr[1][idx], tr[2][idx])
            loss = p_loss + args.value_weight * v_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += np.array([p_loss.item(), v_loss.item(), acc.item()]) * len(idx)
        run /= n
        v = evaluate(model, va, args.value_weight)
        row = {"epoch": epoch,
               "train_policy_loss": run[0], "train_value_loss": run[1], "train_acc": run[2],
               **{"val_" + k: x for k, x in v.items()}}
        history.append(row)
        print(f"epoch {epoch:3d} | train p {run[0]:.3f} v {run[1]:.3f} acc {run[2]:.3f} | "
              f"val p {v['policy_loss']:.3f} v {v['value_loss']:.3f} acc {v['policy_acc']:.3f}")
        if v["loss"] < best_val:
            best_val, best_epoch = v["loss"], epoch
            torch.save({"model": model.state_dict(),
                        "config": {"channels": args.channels, "blocks": args.blocks},
                        "epoch": epoch, "val": v}, args.out)

    seconds = time.time() - t0
    log = {"args": vars(args), "device": str(device), "torch": torch.__version__,
           "python": platform.python_version(), "train_seconds": round(seconds, 1),
           "n_train": len(train["policy"]), "n_val": len(val["policy"]),
           "best_epoch": best_epoch, "best_val_loss": best_val, "history": history}
    log_path = os.path.splitext(args.out)[0] + ".json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    if history:
        os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)
        with open(args.log_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)
    print(f"Xong sau {seconds:.1f}s. Tốt nhất ở epoch {best_epoch}. "
          f"Đã lưu {args.out} và {log_path}")


if __name__ == "__main__":
    main()
"""
Khung huấn luyện tăng cường (RL) bằng tự chơi - CHƯA HUẤN LUYỆN THẬT.
=====================================================================
Trạng thái: đây chỉ là bộ khung chạy được, nhóm chưa chạy huấn luyện RL đầy đủ nên
chưa có weights/rl.pt và KHÔNG có kết quả RL nào để báo cáo.

Ý tưởng (policy gradient có baseline, tự chơi):
  1. Khởi tạo mạng từ checkpoint SL (weights/sl.pt).
  2. Mỗi vòng: mạng tự chơi N ván với chính nó, chọn nước đi bằng cách lấy mẫu từ policy
     (cột đầy bị che). Ghi lại từng thế cờ cùng người đi và nước đã chọn.
  3. Khi ván kết thúc, mỗi thế cờ có đích z = +1 (người đi thế đó thắng), -1 (thua), 0 (hòa).
  4. Cập nhật: loss = -(z - v) * log pi(a|s) + value_weight * (v - z)^2
     trong đó v là value của mạng (dùng làm baseline, không truyền gradient qua phần advantage).

Chạy thử nhỏ (từ thư mục gốc repo):
    python -m training.train_rl --iterations 2 --games 4 --out weights/rl_test.pt
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

from game.connect4_env import ConnectFourEnv
from training.encode import encode_state, legal_mask
from training.model import PolicyValueNet


def set_seed(seed):
    """Cố định seed để chạy lại cho kết quả giống nhau."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_init(path, device):
    """Nạp mạng từ checkpoint SL (cùng định dạng train_sl.py lưu). Không có file thì báo lỗi rõ."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không thấy checkpoint khởi tạo: {path}. Chạy training/train_sl.py trước.")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = PolicyValueNet(cfg["channels"], cfg["blocks"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model, cfg


@torch.no_grad()
def self_play_game(model, device, temperature=1.0):
    """Cho mạng tự chơi một ván. Trả về (danh sách thế cờ, môi trường sau khi kết thúc).

    Mỗi thế cờ là (x, mask, action_index, người đi); người đi là 'red' hoặc 'yellow'.
    """
    model.eval()
    env = ConnectFourEnv()
    env.reset()
    steps = []
    while not env.is_terminal():
        x = encode_state(env)
        mask = legal_mask(env)
        xt = torch.from_numpy(x).unsqueeze(0).to(device)
        logits, _ = model(xt)
        logits = logits[0] / temperature
        logits = logits.masked_fill(torch.from_numpy(mask).to(device) == 0, float("-inf"))
        probs = torch.softmax(logits, dim=0)
        a = int(torch.multinomial(probs, 1).item())
        steps.append((x, mask, a, env.turn))
        env.step(a + 1)
    return steps, env


def build_batch(games, device):
    """Gộp các ván thành tensor huấn luyện: x, mask, action, z (đích theo góc nhìn người đi)."""
    xs, masks, acts, zs = [], [], [], []
    for steps, env in games:
        for x, mask, a, player in steps:
            xs.append(x)
            masks.append(mask)
            acts.append(a)
            zs.append(env.result(player))
    x = torch.from_numpy(np.stack(xs)).float().to(device)
    mask = torch.from_numpy(np.stack(masks)).float().to(device)
    act = torch.tensor(acts, dtype=torch.long, device=device)
    z = torch.tensor(zs, dtype=torch.float32, device=device)
    return x, mask, act, z


def update(model, opt, batch, value_weight):
    """Một bước cập nhật policy gradient có baseline. Trả về (policy loss, value loss)."""
    model.train()
    x, mask, act, z = batch
    logits, v = model(x)
    logits = logits.masked_fill(mask == 0, float("-inf"))
    logp = F.log_softmax(logits, dim=1).gather(1, act.unsqueeze(1)).squeeze(1)
    advantage = (z - v).detach()
    p_loss = -(advantage * logp).mean()
    v_loss = F.mse_loss(v, z)
    loss = p_loss + value_weight * v_loss
    opt.zero_grad()
    loss.backward()
    opt.step()
    return p_loss.item(), v_loss.item()


def main():
    ap = argparse.ArgumentParser(description="Khung RL tự chơi (chưa huấn luyện thật)")
    ap.add_argument("--init", default="weights/sl.pt", help="checkpoint SL để khởi tạo")
    ap.add_argument("--out", default="weights/rl.pt", help="đường dẫn checkpoint RL")
    ap.add_argument("--log-csv", default="logs/training/rl_log.csv", help="log từng vòng dạng CSV")
    ap.add_argument("--iterations", type=int, default=100, help="số vòng (mỗi vòng: tự chơi rồi cập nhật)")
    ap.add_argument("--games", type=int, default=32, help="số ván tự chơi mỗi vòng")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_init(args.init, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Thiết bị: {device} | khởi tạo từ {args.init} | {args.iterations} vòng x {args.games} ván")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)
    history = []
    t0 = time.time()
    for it in range(1, args.iterations + 1):
        games = [self_play_game(model, device, args.temperature) for _ in range(args.games)]
        batch = build_batch(games, device)
        p_loss, v_loss = update(model, opt, batch, args.value_weight)
        n_red = sum(1 for _, e in games if e.result("red") == 1)
        n_yellow = sum(1 for _, e in games if e.result("yellow") == 1)
        n_draw = len(games) - n_red - n_yellow
        row = {"iter": it, "positions": len(batch[3]), "policy_loss": p_loss, "value_loss": v_loss,
               "red_wins": n_red, "yellow_wins": n_yellow, "draws": n_draw}
        history.append(row)
        print(f"vòng {it}/{args.iterations} | thế {row['positions']} | "
              f"p {p_loss:.4f} | v {v_loss:.4f} | đỏ/vàng/hòa {n_red}/{n_yellow}/{n_draw}")

    seconds = time.time() - t0
    torch.save({"model": model.state_dict(), "config": cfg, "iterations": args.iterations}, args.out)
    with open(args.log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    meta = {"args": vars(args), "device": str(device), "torch": torch.__version__,
            "python": platform.python_version(), "train_seconds": round(seconds, 1)}
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {args.out} sau {seconds:.1f} giây")


if __name__ == "__main__":
    main()
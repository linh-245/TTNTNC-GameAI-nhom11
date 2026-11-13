"""
Cấp 5: Neural Network Agent (policy-only)
=========================================
Đi theo mạng policy-value đã huấn luyện (training/train_sl.py): chọn cột có xác suất cao
nhất trong các cột còn đánh được. Chưa kết hợp tìm kiếm MCTS.

Giao diện giống các agent khác: hàm nhận ConnectFourEnv, trả về cột 1..7.

    from functools import partial
    agent = partial(nn_agent, ckpt="weights/sl.pt")

Lưu ý: torch được import trễ (bên trong hàm) để máy không có torch vẫn import được
module này, ví dụ khi chạy CI.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.connect4_env import ConnectFourEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHECKPOINT = os.path.join(ROOT, "weights", "sl.pt")

_CACHE = {}   # đường dẫn checkpoint -> mô hình đã nạp (nạp một lần)


def load_model(path=DEFAULT_CHECKPOINT):
    """Nạp mô hình từ checkpoint do train_sl.py lưu. Có nhớ đệm theo đường dẫn."""
    import torch
    from training.model import PolicyValueNet

    path = os.path.abspath(path)
    if path in _CACHE:
        return _CACHE[path]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không thấy checkpoint: {path}. Chạy training/train_sl.py trước "
            f"hoặc truyền ckpt=... cho nn_agent.")
    # File do nhóm tự lưu nên tin cậy; weights_only=False vì checkpoint có dict số liệu
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = PolicyValueNet(cfg["channels"], cfg["blocks"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    _CACHE[path] = model
    return model


def nn_agent(env: ConnectFourEnv, ckpt: str = DEFAULT_CHECKPOINT) -> int:
    """
    Cấp 5 - chọn cột có xác suất policy cao nhất (cột đầy đã bị che).

    Returns
    -------
    int : cột 1..7
    Side effect
    -----------
    nn_agent.last_stats = {"latency_ms": ..., "value": ...}
    value là ước lượng của mạng cho thế cờ hiện tại (góc nhìn người sắp đi).
    """
    from training.model import predict

    model = load_model(ckpt)
    start = time.time()
    probs, value = predict(model, env)
    nn_agent.last_stats = {
        "latency_ms": round((time.time() - start) * 1000.0, 2),
        "value": round(value, 3),
    }
    return int(np.argmax(probs)) + 1


nn_agent.last_stats = {"latency_ms": 0.0, "value": 0.0}
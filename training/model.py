"""
Mạng CNN policy-value cho Connect Four.
=======================================
Đầu vào : tensor (B, 3, 6, 7) từ training/encode.py (góc nhìn người sắp đi).
Đầu ra  : (logits, value)
  - logits: (B, 7) điểm cho 7 cột (chưa softmax, chưa che cột đầy)
  - value : (B,) trong [-1, 1], theo góc nhìn người sắp đi
            (+1 = người sắp đi sẽ thắng, -1 = sẽ thua, 0 = hòa/chưa rõ)

Cấu trúc: 1 lớp conv đầu + vài khối residual (conv 3x3 giữ nguyên kích thước 6x7)
rồi tách 2 đầu: policy (chọn cột) và value (ước lượng thế cờ).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from game.connect4_env import ConnectFourEnv
from training.encode import encode_state, legal_mask, N_CHANNELS, ROWS, COLS, ACTION_SIZE


class ResBlock(nn.Module):
    """Khối residual: 2 lớp conv 3x3, cộng đường tắt, rồi ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 64, n_blocks: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(N_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])
        # Đầu policy: conv 1x1 -> flatten -> linear ra 7 cột
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU())
        self.policy_fc = nn.Linear(2 * ROWS * COLS, ACTION_SIZE)
        # Đầu value: conv 1x1 -> flatten -> linear -> tanh
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU())
        self.value_fc1 = nn.Linear(ROWS * COLS, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.blocks(self.stem(x))
        logits = self.policy_fc(self.policy_conv(h).flatten(1))
        v = self.value_conv(h).flatten(1)
        v = torch.tanh(self.value_fc2(F.relu(self.value_fc1(v)))).squeeze(1)
        return logits, v


@torch.no_grad()
def predict(model: PolicyValueNet, env: ConnectFourEnv):
    """Dự đoán cho 1 thế cờ: trả về (probs, value).

    probs: mảng numpy 7 phần tử, cột đầy có xác suất 0 (index 0..6 = cột 1..7).
    Lưu ý: hàm này gọi model.eval(); khi huấn luyện nhớ gọi lại model.train().
    """
    model.eval()
    device = next(model.parameters()).device
    x = torch.from_numpy(encode_state(env)).unsqueeze(0).to(device)
    logits, value = model(x)
    mask = torch.from_numpy(legal_mask(env)).to(device)
    logits = logits[0].masked_fill(mask == 0, float("-inf"))
    probs = torch.softmax(logits, dim=0).cpu().numpy()
    return probs, float(value[0])
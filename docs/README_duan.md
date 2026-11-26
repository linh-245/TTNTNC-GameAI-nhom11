# Connect Four 7×6 – Game AI Đa Cấp Độ

Dự án thực hiện theo **đề tài cuối kỳ TTNTNC** và sách **Mark Liu, "AlphaGo Simplified", CRC Press (2024)**.

## Cấu trúc thư mục

```
TTNTNC-GameAI-nhom11/
├── game/               # Engine và API (theo sách)
│   ├── connect4_env.py # ConnectFourEnv (class conn), có legal_actions, is_terminal, result, clone, undo
│   └── game_utils.py   # one_conn_game(), run_games(), conn_manual()
├── agents/             # Các tác tử AI
│   ├── random_agent.py     # Cấp 1: Random
│   ├── heuristic_agent.py  # Cấp 2: Think1/2/3
│   ├── alphabeta_agent.py  # Cấp 3: Alpha-Beta
│   ├── mcts_agent.py       # Cấp 4: MCTS/UCT
│   ├── nn_agent.py         # Cấp 5 và 6: mạng policy-value (sl.pt / rl.pt)
│   └── az_mcts_agent.py    # Cấp 7: AZ-MCTS (PUCT + mạng); cấp 8 là cùng tác tử với weights/az_iter_N.pt
├── training/           # Sinh dữ liệu, huấn luyện SL, RL, tự chơi AlphaZero, đo checkpoint
├── eval/               # Giải đấu round-robin, Wilson CI, RQ1, ablation
├── solver/             # 500 thế chuẩn, độ chính xác nước tối ưu
├── ui/pygame_ui.py     # Giao diện Pygame
├── tests/              # Kiểm thử tự động
├── notebooks/          # Notebook đường cong học
├── results/figures/    # Hình sinh từ log
├── weights/            # Checkpoint mô hình
├── logs/               # Log thi đấu, huấn luyện, độ chính xác
└── docs/               # README, nhật ký đóng góp
```

## Cài đặt

```bash
# Clone hoặc copy thư mục
cd connect4-ai

# Cài thư viện
pip install -r requirements.txt

# Hoặc dùng conda
conda install numpy pygame scipy matplotlib
pip install torch   # cần cho cấp 5 đến 8
```

## Chạy Demo

### 1. Giao diện Pygame (Demo chính)

```bash
python ui/pygame_ui.py
```

- Chọn agent cho đỏ và vàng từ menu
- Nhấn **START GAME**
- Điều khiển: nhấn **số 1-7** hoặc **click chuột** vào cột (khi là Human)
- Nhấn **R** để chơi lại, **M** để về menu

### 2. Chạy Tests

```bash
python -m pytest tests -q
```

### 3. Giải đấu Round-Robin

```bash
python -m eval.tournament --games 100 --seed 42 --workers 4
# chi tiết tham số và tên tác tử: docs/README_eval.md
```

### 4. Chạy game từ command line

```python
from game.connect4_env import ConnectFourEnv
from game.game_utils import one_conn_game
from agents.random_agent import conn_random
from agents.heuristic_agent import conn_think1
from agents.alphabeta_agent import alphabeta_agent
from agents.mcts_agent import mcts_agent
from functools import partial

# MCTS vs AlphaBeta
reward, moves = one_conn_game(
    partial(mcts_agent, n_simulations=200),
    partial(alphabeta_agent, depth=5),
    render=True
)
print(f"Kết quả: {reward}")
```

## Cấp độ AI

| Cấp | Tên | Mô tả | File |
|-----|-----|-------|------|
| 1 | Random | Chọn ngẫu nhiên | `random_agent.py` |
| 2 | Think1, Think2, Think3 | Luật: nhìn 1, 2, 3 bước (thắng ngay, chặn, ưu tiên giữa) | `heuristic_agent.py` |
| 3 | Alpha-Beta | Negamax + cắt tỉa + hàm đánh giá, độ sâu hoặc ngân sách thời gian | `alphabeta_agent.py` |
| 4 | MCTS | UCT + rollout ngẫu nhiên, số mô phỏng hoặc ngân sách thời gian | `mcts_agent.py` |
| 5 | Neural Network | Mạng CNN policy-value học bắt chước Think3 (`weights/sl.pt`) | `nn_agent.py` |
| 6 | Reinforcement Learning | Khởi tạo từ SL, tự chơi policy gradient (`weights/rl.pt`) | `nn_agent.py`, `training/train_rl.py` |
| 7 | AZ-MCTS | MCTS dùng PUCT, prior và value từ mạng | `az_mcts_agent.py` |
| 8 | Tự chơi AlphaZero | Cải thiện mạng bằng tự chơi AZ-MCTS (`weights/az_iter_N.pt`, tác tử `AZ8-N`) | `training/selfplay.py` |

Phần học máy (cấp 5 đến 8): `docs/README_learning.md`.

## API (theo sách)

```python
env = conn()         # Khởi tạo
state = env.reset()  # Reset
env.render()         # Hiển thị
action = env.sample()  # Random move
state, reward, done, info = env.step(action)  # Thực hiện nước đi
env.close()
```

## Tham khảo

- M. H. Liu, *AlphaGo Simplified*, CRC Press, 2024. GitHub: https://github.com/markhliu/AlphaGoSimplified
- S. Russell, P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed., 2021
- R. Sutton, A. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018
- L. Kocsis, C. Szepesvári, "Bandit based Monte-Carlo planning", ECML 2006 (UCT)
- V. Allis, *A Knowledge-based Approach of Connect-Four*, 1988
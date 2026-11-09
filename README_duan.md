# Connect Four 7×6 – Game AI Đa Cấp Độ

Dự án thực hiện theo **đề tài cuối kỳ TTNTNC** và sách **Mark Liu, "AlphaGo Simplified", CRC Press (2024)**.

## Cấu trúc thư mục

```
connect4-ai/
├── game/               # Game engine và API chuẩn (theo sách)
│   ├── connect4_env.py # ConnectFourEnv (class conn) – OpenAI Gym-style
│   └── game_utils.py   # one_conn_game(), run_games(), conn_manual()
├── agents/             # Các tác tử AI
│   ├── random_agent.py     # Cấp 1: Random (Ch. 3)
│   ├── heuristic_agent.py  # Cấp 2-4: Think1/2/3 (Ch. 3)
│   ├── alphabeta_agent.py  # Cấp 5: Alpha-Beta + Eval (Ch. 5-7)
│   └── mcts_agent.py       # Cấp 6: MCTS/UCT (Ch. 8)
├── training/           # Sinh dữ liệu, huấn luyện NN, self-play
├── eval/               # Round-robin, Wilson CI, log
│   └── tournament.py
├── ui/                 # Giao diện Pygame
│   └── pygame_ui.py
├── tests/              # Kiểm thử tự động
│   └── test_engine.py
├── notebooks/          # Tạo bảng/hình từ log
├── weights/            # Checkpoint mô hình
├── logs/               # Log thi đấu thô
└── README.md
```

## Cài đặt

```bash
# Clone hoặc copy thư mục
cd connect4-ai

# Cài thư viện
pip install numpy pygame torch scipy matplotlib

# Hoặc dùng conda
conda install numpy pygame scipy matplotlib
pip install torch
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
python tests/test_engine.py
# hoặc
python -m pytest tests/ -v
```

### 3. Giải đấu Round-Robin

```bash
python eval/tournament.py
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

| Cấp | Tên | Mô tả | Chương sách |
|-----|-----|-------|-------------|
| 1 | Random | Chọn ngẫu nhiên | Ch. 3 |
| 2 | Think1 | Nhìn 1 bước – thắng ngay | Ch. 3 |
| 3 | Think2 | Nhìn 2 bước – thắng/chặn | Ch. 3 |
| 4 | Think3 | Nhìn 3 bước + ưu tiên giữa | Ch. 3 |
| 5 | Alpha-Beta | Negamax + alpha-beta + eval | Ch. 5-7 |
| 6 | MCTS | UCT + random rollout | Ch. 8 |

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

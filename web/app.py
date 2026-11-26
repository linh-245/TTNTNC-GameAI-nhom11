"""Demo web Connect Four 7x6 (Gradio) - Nhóm 11.

Chạy local (từ thư mục gốc repo):   python web/app.py
Hugging Face Spaces: đặt file này ở gốc Space với tên app.py
(cùng các thư mục game/, agents/, weights/).

File này KHÔNG import pygame. Registry tác tử được sao từ ui/pygame_ui.py
để không động vào code đã tag v1.0-cuoiki.
"""
import os
import sys
import time
from functools import partial

# --- Tìm thư mục gốc repo (chạy được cả ở web/app.py lẫn app.py ở gốc Space) ---
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.isdir(os.path.join(HERE, "game")) else os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # để 'weights/sl.pt' (đường dẫn tương đối) luôn tìm thấy

import gradio as gr

from game.connect4_env import ConnectFourEnv
from agents.random_agent import conn_random
from agents.heuristic_agent import conn_think1, conn_think2, conn_think3
from agents.alphabeta_agent import alphabeta_agent
from agents.mcts_agent import mcts_agent

ROWS, COLS = 6, 7


# =====================================================================
# REGISTRY TÁC TỬ  (sao từ ui/pygame_ui.py)
# name -> hàm nhận budget_ms (từ thanh trượt) và trả về agent fn(env) -> cột 1..7
# =====================================================================
AGENT_FACTORIES = {
    "Random":       lambda ms: conn_random,
    "Think1":       lambda ms: conn_think1,
    "Think2":       lambda ms: conn_think2,
    "Think3":       lambda ms: conn_think3,
    "Minimax d4":   lambda ms: partial(alphabeta_agent, depth=4, use_eval=False),
    "AlphaBeta d5": lambda ms: partial(alphabeta_agent, depth=5, use_eval=True),
    "MCTS-200":     lambda ms: partial(mcts_agent, n_simulations=200),
    "MCTS-500":     lambda ms: partial(mcts_agent, n_simulations=500),
}

# MCTS theo ngân sách thời gian: dùng thanh trượt (nếu mcts_agent có budget_ms)
try:
    import inspect
    if "budget_ms" in inspect.signature(mcts_agent).parameters:
        AGENT_FACTORIES["MCTS (theo ms)"] = lambda ms: partial(
            mcts_agent, n_simulations=100000, budget_ms=ms)
except Exception:
    pass

# Cấp 5 (SL) và Cấp 6 (RL): cùng nn_agent, khác checkpoint
try:
    from agents.nn_agent import nn_agent
    AGENT_FACTORIES["NN (SL)"] = lambda ms: partial(nn_agent, ckpt="weights/sl.pt")
    AGENT_FACTORIES["RL"] = lambda ms: partial(nn_agent, ckpt="weights/rl.pt")
except Exception:
    pass

# AZ-MCTS (cấp 7) và tự chơi AlphaZero (cấp 8): MCTS dẫn đường bằng mạng chính sách-giá trị.
# Theo tests/test_az_mcts.py, hàm là az_mcts_agent(env, n_simulations=...) và _get_model(ckpt).
# Đoạn dưới tự dò xem hàm có tham số `ckpt` không; chỉ thêm vào menu nếu file .pt tồn tại.
try:
    from agents.az_mcts_agent import az_mcts_agent
    import inspect as _inspect
    _az_params = _inspect.signature(az_mcts_agent).parameters

    def _make_az(ckpt_path, sims):
        kw = {"n_simulations": sims}
        if "ckpt" in _az_params:
            kw["ckpt"] = ckpt_path
        return partial(az_mcts_agent, **kw)

    # (tên trong menu, file trọng số, số mô phỏng) - sửa cho khớp thư mục weights/ của bạn
    for _label, _path, _sims in [
        ("AZ-MCTS iter1",  "weights/az_iter_1.pt",  200),
        ("AZ-MCTS iter10", "weights/az_iter_10.pt", 200),
    ]:
        if os.path.exists(_path):
            AGENT_FACTORIES[_label] = (
                lambda ms, p=_path, s=_sims: _make_az(p, s))
except Exception:
    pass

AGENT_NAMES = list(AGENT_FACTORIES.keys())


# =====================================================================
# ADAPTER  (đã nối với ConnectFourEnv; env.turn là 'red'/'yellow',
# cột đi là 1..7, env.state[cột][hàng], hàng 0 = dưới cùng, 1=đỏ, -1=vàng)
# =====================================================================
def new_env():
    env = ConnectFourEnv()
    env.reset()
    return env


def board_array(env):
    """6 hàng x 7 cột, hàng đầu = trên cùng. Giá trị: 1 đỏ, -1 vàng, 0 trống."""
    st = env.state
    return [[int(st[c][r]) for c in range(COLS)] for r in range(ROWS - 1, -1, -1)]


def legal_moves(env):
    return list(env.validinputs)  # các cột 1..7 còn đi được


def play(env, col):
    """Đi nước `col`. Trả về None (đang chơi) | 'draw' | 'red' | 'yellow' (người thắng)."""
    mover = env.turn
    _, reward, done, _ = env.step(col)
    if not done:
        return None
    return mover if reward != 0 else "draw"


def make_agent(name, budget_ms):
    return AGENT_FACTORIES[name](int(budget_ms))


def _get_stats(fn):
    base = fn.func if isinstance(fn, partial) else fn
    return getattr(base, "last_stats", None)


def agent_choose(fn, env):
    """Trả về (cột, chuỗi thống kê: thời gian | số nút | số mô phỏng)."""
    snapshot = env.copy()  # AI nghĩ trên bản sao, giống pygame_ui
    t0 = time.perf_counter()
    try:
        col = fn(snapshot)
        stats = _get_stats(fn)
        err = ""
    except Exception as e:  # ví dụ thiếu file .pt -> đi ngẫu nhiên, không treo
        col, stats, err = snapshot.sample(), None, f"lỗi tác tử: {e}"
    parts = [f"{(time.perf_counter() - t0) * 1000:.0f}ms"]
    if stats:
        if stats.get("nodes"):
            parts.append(f"{int(stats['nodes'])} nút")
        if stats.get("sims"):
            parts.append(f"{int(stats['sims'])} mô phỏng")
    if err:
        parts.append(err)
    return col, " | ".join(parts)


# =====================================================================
# GIAO DIỆN
# =====================================================================
PIECE_COLOR = {0: "#0a0a1c", 1: "#dc3232", -1: "#f0c81e"}


def render_board(env):
    rows = []
    for row in board_array(env):
        tds = "".join(
            f'<td><div style="width:46px;height:46px;border-radius:50%;'
            f'background:{PIECE_COLOR[v]};"></div></td>' for v in row)
        rows.append(f"<tr>{tds}</tr>")
    nums = "".join(
        f'<td style="text-align:center;color:#999;font-weight:bold">{c}</td>'
        for c in range(1, COLS + 1))
    return ('<table style="margin:auto;background:#143c8c;padding:10px;'
            'border-radius:12px;border-spacing:6px;">'
            + "".join(rows) + f"<tr>{nums}</tr></table>")


def _status(state, extra=""):
    res = state["result"]
    if res is None:
        msg = "Đến lượt bạn."
    elif res == "draw":
        msg = "HÒA!"
    elif res == state["human"]:
        msg = "BẠN THẮNG!"
    else:
        msg = "AI THẮNG."
    return msg + (f"  |  {extra}" if extra else "")


def _ai_turn(state):
    col, info = agent_choose(state["agent"], state["env"])
    if col not in legal_moves(state["env"]):
        col = legal_moves(state["env"])[0]
    state["result"] = play(state["env"], col)
    return f"AI đi cột {col} ({info})"


def start(agent_name, budget_ms, ai_first):
    state = {
        "env": new_env(),
        "agent": make_agent(agent_name, budget_ms),
        "human": "yellow" if ai_first else "red",  # đỏ luôn đi trước
        "result": None,
    }
    extra = _ai_turn(state) if ai_first else ""
    return state, render_board(state["env"]), _status(state, extra)


def make_click(col):
    def on_click(state):
        if state is None:
            return state, gr.update(), "Hãy bấm 'Ván mới' trước."
        if state["result"] is not None:
            return state, render_board(state["env"]), _status(state)
        if col not in legal_moves(state["env"]):
            return state, render_board(state["env"]), "Cột đã đầy, chọn cột khác."
        state["result"] = play(state["env"], col)
        extra = _ai_turn(state) if state["result"] is None else ""
        return state, render_board(state["env"]), _status(state, extra)
    return on_click


with gr.Blocks(title="Connect Four 7x6 - Nhóm 11") as demo:
    gr.Markdown("# Connect Four 7×6 – Game AI đa cấp độ\n"
                "Nhóm 11 – Trí tuệ nhân tạo nâng cao. Bạn là quân **đỏ** "
                "(đi trước), trừ khi tick 'AI đi trước'.")
    with gr.Row():
        agent_dd = gr.Dropdown(AGENT_NAMES, value=AGENT_NAMES[min(7, len(AGENT_NAMES) - 1)],
                               label="Đối thủ AI")
        budget = gr.Slider(50, 2000, value=300, step=50,
                           label="Ngân sách (ms) - chỉ áp dụng cho 'MCTS (theo ms)'")
        ai_first = gr.Checkbox(False, label="AI đi trước")
        new_btn = gr.Button("Ván mới", variant="primary")

    board_html = gr.HTML()
    with gr.Row():
        col_btns = [gr.Button(str(c)) for c in range(1, COLS + 1)]
    status = gr.Textbox(label="Trạng thái", interactive=False)
    state = gr.State(None)

    new_btn.click(start, [agent_dd, budget, ai_first], [state, board_html, status])
    for c, b in zip(range(1, COLS + 1), col_btns):
        b.click(make_click(c), [state], [state, board_html, status])
    demo.load(start, [agent_dd, budget, ai_first], [state, board_html, status])

if __name__ == "__main__":
    demo.launch()

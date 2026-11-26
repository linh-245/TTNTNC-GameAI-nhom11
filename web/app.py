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
# GIAO DIỆN  (chọn được cả hai vai: Human hoặc bất kỳ tác tử nào, như pygame)
# =====================================================================
PIECE_COLOR = {0: "#0a0a1c", 1: "#dc3232", -1: "#f0c81e"}
HUMAN = "Human"
VN = {"red": "ĐỎ", "yellow": "VÀNG"}
PLAYER_CHOICES = [HUMAN] + AGENT_NAMES


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


def _is_human(state, color):
    return state["agents"][color] is None


def _status(state, thinking=False):
    res, names = state["result"], state["names"]
    if res is None:
        turn = state["env"].turn
        if _is_human(state, turn):
            return f"Lượt {VN[turn]} ({names[turn]}): bấm số cột 1–7."
        return (f"Lượt {VN[turn]} ({names[turn]}): "
                + ("đang suy nghĩ…" if thinking else "AI sắp đi."))
    if res == "draw":
        return "HÒA!"
    return f"{VN[res]} ({names[res]}) THẮNG!"


def _out(state, thinking=False):
    """Bộ 4 đầu ra: state, bàn cờ, trạng thái, lịch sử."""
    return (state, render_board(state["env"]), _status(state, thinking),
            "\n".join(state["log"]))


def _run_ai(state):
    """Cho AI đi liên tục tới khi đến lượt người hoặc hết ván (yield sau mỗi nước)."""
    state["busy"] = True
    try:
        has_human = any(_is_human(state, c) for c in ("red", "yellow"))
        while state["result"] is None and not _is_human(state, state["env"].turn):
            env = state["env"]
            turn = env.turn
            yield _out(state, thinking=True)
            col, info = agent_choose(state["agents"][turn], env)
            if col not in legal_moves(env):
                col = legal_moves(env)[0]
            state["result"] = play(env, col)
            state["log"].append(f"{VN[turn]} ({state['names'][turn]}) → cột {col} ({info})")
            yield _out(state)
            if not has_human:
                time.sleep(0.4)  # AI vs AI: dừng chút để xem được từng nước
    finally:
        state["busy"] = False


def start(red_name, yellow_name, budget_ms):
    names = {"red": red_name, "yellow": yellow_name}
    agents = {c: (None if n == HUMAN else make_agent(n, budget_ms))
              for c, n in names.items()}
    state = {"env": new_env(), "agents": agents, "names": names,
             "result": None, "log": [], "busy": False}
    yield _out(state)
    yield from _run_ai(state)


def make_click(col):
    def on_click(state):
        if state is None:
            yield None, gr.update(), "Hãy bấm 'Ván mới / Bắt đầu' trước.", gr.update()
            return
        env = state["env"]
        if state["busy"]:
            yield state, gr.update(), "AI đang suy nghĩ, chờ một chút…", gr.update()
            return
        if state["result"] is not None:
            yield _out(state)
            return
        turn = env.turn
        if not _is_human(state, turn):
            yield state, gr.update(), f"Không phải lượt người chơi ({VN[turn]} là AI).", gr.update()
            return
        if col not in legal_moves(env):
            yield state, gr.update(), "Cột đã đầy, chọn cột khác.", gr.update()
            return
        state["result"] = play(env, col)
        state["log"].append(f"{VN[turn]} ({state['names'][turn]}) → cột {col}")
        yield _out(state)
        yield from _run_ai(state)
    return on_click


with gr.Blocks(title="Connect Four 7x6 - Nhóm 11") as demo:
    gr.Markdown("# Connect Four 7×6 – Game AI đa cấp độ\n"
                "Nhóm 11 – Trí tuệ nhân tạo nâng cao. Chọn **Human** hoặc một tác tử AI "
                "cho từng bên (ĐỎ đi trước). Human vs AI, AI vs AI, hoặc Human vs Human.")
    with gr.Row():
        red_dd = gr.Dropdown(PLAYER_CHOICES, value=HUMAN,
                             label="Người chơi ĐỎ (đi trước)")
        yellow_default = "MCTS-500" if "MCTS-500" in AGENT_NAMES else AGENT_NAMES[0]
        yellow_dd = gr.Dropdown(PLAYER_CHOICES, value=yellow_default,
                                label="Người chơi VÀNG")
        budget = gr.Slider(50, 2000, value=300, step=50,
                           label="Ngân sách (ms) - chỉ áp dụng cho 'MCTS (theo ms)'")
        new_btn = gr.Button("Ván mới / Bắt đầu", variant="primary")

    status = gr.Textbox(label="Trạng thái", interactive=False)
    board_html = gr.HTML()
    with gr.Row():
        col_btns = [gr.Button(str(c)) for c in range(1, COLS + 1)]
    log_box = gr.Textbox(label="Lịch sử nước đi", lines=8, interactive=False)
    state = gr.State(None)

    outs = [state, board_html, status, log_box]
    new_btn.click(start, [red_dd, yellow_dd, budget], outs)
    for c, b in zip(range(1, COLS + 1), col_btns):
        b.click(make_click(c), [state], outs)
    demo.load(start, [red_dd, yellow_dd, budget], outs)

if __name__ == "__main__":
    demo.launch()
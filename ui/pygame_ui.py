import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import math
import time
import threading
from functools import partial
from typing import Optional, Callable

from game.connect4_env import ConnectFourEnv
from agents.random_agent import conn_random
from agents.heuristic_agent import conn_think1, conn_think2, conn_think3
from agents.alphabeta_agent import alphabeta_agent, minimax_eval_agent
from agents.mcts_agent import mcts_agent


"""
Connect Four - Giao diện Pygame
================================
Demo ứng dụng có thể chơi, chọn cấp độ AI.
Hỗ trợ: Người vs AI, AI vs AI.

Yêu cầu: pip install pygame
Chạy: python ui/pygame_ui.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import math
import time
import threading
from functools import partial
from typing import Optional, Callable

from game.connect4_env import ConnectFourEnv
from agents.random_agent import conn_random
from agents.heuristic_agent import conn_think1, conn_think2, conn_think3
from agents.alphabeta_agent import alphabeta_agent, minimax_eval_agent
from agents.mcts_agent import mcts_agent

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CELL_SIZE = 90
COLS = 7
ROWS = 6
BOARD_WIDTH = COLS * CELL_SIZE
BOARD_HEIGHT = ROWS * CELL_SIZE
HEADER_H = 120
SIDEBAR_W = 280
WINDOW_W = BOARD_WIDTH + SIDEBAR_W
WINDOW_H = BOARD_HEIGHT + HEADER_H
RADIUS = CELL_SIZE // 2 - 6
FPS = 60

# Màu sắc
BG_COLOR      = (15, 15, 35)
BOARD_COLOR   = (20, 60, 140)
EMPTY_COLOR   = (10, 10, 28)
RED_COLOR     = (220, 50, 50)
YELLOW_COLOR  = (240, 200, 30)
WHITE         = (255, 255, 255)
GRAY          = (150, 150, 170)
DARK_GRAY     = (40, 40, 60)
ACCENT        = (80, 120, 220)
GREEN         = (50, 200, 80)
HIGHLIGHT     = (60, 160, 255)
WIN_GLOW      = (255, 220, 60)

# Tên các agent
AGENTS = {
    'Human':       None,
    'Random':      conn_random,
    'Think1':      conn_think1,
    'Think2':      conn_think2,
    'Think3':      conn_think3,
    'Minimax d4':  partial(alphabeta_agent, depth=4, use_eval=False),
    'AlphaBeta d5': partial(alphabeta_agent, depth=5, use_eval=True),
    'MCTS-200':    partial(mcts_agent, n_simulations=200),
    'MCTS-500':    partial(mcts_agent, n_simulations=500),
}
AGENT_NAMES = list(AGENTS.keys())


def draw_gradient_rect(surface, rect, color1, color2, vertical=True):
    """Vẽ hình chữ nhật gradient."""
    x, y, w, h = rect
    for i in range(h if vertical else w):
        t = i / (h if vertical else w)
        r = int(color1[0] + (color2[0] - color1[0]) * t)
        g = int(color1[1] + (color2[1] - color1[1]) * t)
        b = int(color1[2] + (color2[2] - color1[2]) * t)
        if vertical:
            pygame.draw.line(surface, (r, g, b), (x, y+i), (x+w, y+i))
        else:
            pygame.draw.line(surface, (r, g, b), (x+i, y), (x+i, y+h))


def draw_glow_circle(surface, color, center, radius, alpha=80):
    """Vẽ vòng tròn có hiệu ứng glow."""
    glow_surf = pygame.Surface((radius*4, radius*4), pygame.SRCALPHA)
    for r in range(radius + 20, radius - 1, -1):
        a = max(0, int(alpha * (1 - (r - radius) / 20)))
        col = (*color, a)
        pygame.draw.circle(glow_surf, col, (radius*2, radius*2), r)
    surface.blit(glow_surf, (center[0] - radius*2, center[1] - radius*2))
    pygame.draw.circle(surface, color, center, radius)


class Button:
    def __init__(self, rect, text, font, color=ACCENT, text_color=WHITE,
                 radius=10):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.color = color
        self.text_color = text_color
        self.radius = radius
        self.hovered = False

    def draw(self, surface):
        col = tuple(min(255, c + 30) for c in self.color) if self.hovered else self.color
        pygame.draw.rect(surface, col, self.rect, border_radius=self.radius)
        pygame.draw.rect(surface, WHITE, self.rect, 1, border_radius=self.radius)
        text_surf = self.font.render(self.text, True, self.text_color)
        surface.blit(text_surf, text_surf.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class Connect4UI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Connect Four 7×6 – AI Demo")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title  = pygame.font.SysFont('segoeui', 28, bold=True)
        self.font_medium = pygame.font.SysFont('segoeui', 18)
        self.font_small  = pygame.font.SysFont('segoeui', 14)
        self.font_big    = pygame.font.SysFont('segoeui', 36, bold=True)

        # State
        self.env = ConnectFourEnv()
        self.env.reset()
        self.red_agent_idx   = 0   # Human
        self.yellow_agent_idx = 1  # Random
        self.game_over = False
        self.winner_msg = ''
        self.ai_thinking = False
        self.ai_move: Optional[int] = None
        self.drop_animation = None   # (col, start_row, target_row, piece, t)
        self.hover_col = -1
        self.move_history = []
        self.thinking_start = 0.0
        self.thinking_time_ms = 0.0
        self.win_cells = []
        self.mode_screen = True  # True = menu, False = game

        # Buttons (mode screen)
        bx, by = WINDOW_W//2 - 120, WINDOW_H//2 - 80
        self.btn_start = Button((bx, by, 240, 50), 'START GAME',
                                 self.font_medium, ACCENT)
        self.btn_restart = Button((BOARD_WIDTH + 20, WINDOW_H - 70, 240, 44),
                                   'NEW GAME', self.font_medium, (60, 140, 80))
        self.btn_menu = Button((BOARD_WIDTH + 20, WINDOW_H - 120, 240, 44),
                                'MAIN MENU', self.font_medium, DARK_GRAY)

        # Selector arrows
        self._build_selector_rects()

    def _build_selector_rects(self):
        sx = BOARD_WIDTH + 20
        self.sel_left_red  = pygame.Rect(sx, 70, 28, 28)
        self.sel_right_red = pygame.Rect(sx + 220, 70, 28, 28)
        self.sel_left_yel  = pygame.Rect(sx, 120, 28, 28)
        self.sel_right_yel = pygame.Rect(sx + 220, 120, 28, 28)

    # ── Drawing ──────────────────────────────────────────────────────────────

    def draw_menu(self):
        s = self.screen
        draw_gradient_rect(s, (0, 0, WINDOW_W, WINDOW_H),
                           (10, 10, 40), (30, 20, 80))
        # Title
        t = self.font_big.render("CONNECT FOUR 7×6", True, WHITE)
        s.blit(t, t.get_rect(center=(WINDOW_W//2, 80)))
        sub = self.font_medium.render("AI Multi-Level Demo • Liu (2024)", True, GRAY)
        s.blit(sub, sub.get_rect(center=(WINDOW_W//2, 120)))

        # Agent selectors
        self._draw_agent_selector(s, 200, "RED Player", self.red_agent_idx, RED_COLOR)
        self._draw_agent_selector(s, 280, "YELLOW Player", self.yellow_agent_idx, YELLOW_COLOR)

        # Info
        info_lines = [
            "Cấp độ AI:",
            "• Human: Bạn chơi",
            "• Random: Ngẫu nhiên (Ch.3)",
            "• Think1: Nhìn 1 bước (Ch.3)",
            "• Think2: Nhìn 2 bước (Ch.3)",
            "• Think3: Nhìn 3 bước (Ch.3)",
            "• Minimax: Alpha-Beta (Ch.5-7)",
            "• MCTS: Monte Carlo (Ch.8)",
        ]
        for i, line in enumerate(info_lines):
            col = ACCENT if i == 0 else GRAY
            t = self.font_small.render(line, True, col)
            s.blit(t, (WINDOW_W//2 - 100, 350 + i*22))

        self.btn_start.draw(s)

    def _draw_agent_selector(self, surface, y, label, idx, color):
        cx = WINDOW_W // 2
        # Label
        t = self.font_medium.render(label, True, color)
        surface.blit(t, t.get_rect(center=(cx, y)))
        # Agent name
        name = AGENT_NAMES[idx]
        nt = self.font_title.render(name, True, WHITE)
        surface.blit(nt, nt.get_rect(center=(cx, y + 30)))
        # Arrows
        pygame.draw.polygon(surface, GRAY,
                             [(cx-130, y+30), (cx-110, y+20), (cx-110, y+40)])
        pygame.draw.polygon(surface, GRAY,
                             [(cx+130, y+30), (cx+110, y+20), (cx+110, y+40)])

    def draw_game(self):
        s = self.screen
        s.fill(BG_COLOR)
        self._draw_header()
        self._draw_board()
        self._draw_sidebar()

    def _draw_header(self):
        s = self.screen
        draw_gradient_rect(s, (0, 0, BOARD_WIDTH, HEADER_H),
                           (20, 20, 50), (10, 10, 35))
        # Turn indicator
        if not self.game_over:
            turn = self.env.turn
            color = RED_COLOR if turn == 'red' else YELLOW_COLOR
            agent_name = AGENT_NAMES[self.red_agent_idx if turn == 'red'
                                     else self.yellow_agent_idx]
            txt = f"{'🔴' if turn == 'red' else '🟡'} {turn.upper()} — {agent_name}"
            t = self.font_title.render(txt, True, color)
            s.blit(t, t.get_rect(center=(BOARD_WIDTH//2, 35)))
            # Thinking
            if self.ai_thinking:
                dots = '.' * (int(time.time() * 3) % 4)
                th = self.font_medium.render(f"Đang suy nghĩ{dots}", True, GRAY)
                s.blit(th, th.get_rect(center=(BOARD_WIDTH//2, 70)))
            else:
                t2 = self.font_small.render(
                    f"Nhấn 1-7 để đặt quân | {self.thinking_time_ms:.0f}ms", True, GRAY)
                s.blit(t2, t2.get_rect(center=(BOARD_WIDTH//2, 70)))
        else:
            wt = self.font_big.render(self.winner_msg, True, WIN_GLOW)
            s.blit(wt, wt.get_rect(center=(BOARD_WIDTH//2, 50)))
            rt = self.font_small.render("Nhấn R để chơi lại | M để về menu", True, GRAY)
            s.blit(rt, rt.get_rect(center=(BOARD_WIDTH//2, 90)))

        # Hover column indicator
        if self.hover_col >= 0 and not self.game_over and not self.ai_thinking:
            cx = self.hover_col * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.polygon(s, WHITE,
                                 [(cx, HEADER_H - 10), (cx-12, HEADER_H - 22),
                                  (cx+12, HEADER_H - 22)])

    def _draw_board(self):
        s = self.screen
        # Board background
        pygame.draw.rect(s, BOARD_COLOR,
                          (0, HEADER_H, BOARD_WIDTH, BOARD_HEIGHT),
                          border_radius=8)

        board = self.env.state  # (7, 6)

        for col in range(COLS):
            for row_idx in range(ROWS):
                # row_idx=0 ở dưới cùng, vẽ từ trên xuống
                screen_row = ROWS - 1 - row_idx
                cx = col * CELL_SIZE + CELL_SIZE // 2
                cy = HEADER_H + screen_row * CELL_SIZE + CELL_SIZE // 2

                piece = board[col][row_idx]

                # Nền ô
                pygame.draw.circle(s, EMPTY_COLOR, (cx, cy), RADIUS + 4)

                # Glow cho winning cells
                is_win_cell = (col, row_idx) in self.win_cells
                if is_win_cell:
                    draw_glow_circle(s, WIN_GLOW, (cx, cy), RADIUS, alpha=60)

                if piece == 1:
                    color = RED_COLOR
                    pygame.draw.circle(s, color, (cx, cy), RADIUS)
                    pygame.draw.circle(s, (255, 120, 120), (cx-RADIUS//4, cy-RADIUS//4), RADIUS//4)
                elif piece == -1:
                    color = YELLOW_COLOR
                    pygame.draw.circle(s, color, (cx, cy), RADIUS)
                    pygame.draw.circle(s, (255, 240, 120), (cx-RADIUS//4, cy-RADIUS//4), RADIUS//4)
                else:
                    # Column highlight on hover
                    if col == self.hover_col and not self.ai_thinking:
                        pygame.draw.circle(s, (35, 70, 160), (cx, cy), RADIUS + 4)
                    pygame.draw.circle(s, EMPTY_COLOR, (cx, cy), RADIUS)

                # Cell border
                pygame.draw.circle(s, BOARD_COLOR, (cx, cy), RADIUS, 2)

        # Column numbers
        for col in range(COLS):
            cx = col * CELL_SIZE + CELL_SIZE // 2
            nt = self.font_small.render(str(col+1), True, GRAY)
            s.blit(nt, nt.get_rect(center=(cx, HEADER_H + BOARD_HEIGHT - 10)))

    def _draw_sidebar(self):
        s = self.screen
        sx = BOARD_WIDTH
        draw_gradient_rect(s, (sx, 0, SIDEBAR_W, WINDOW_H),
                           (18, 18, 40), (12, 12, 30))
        pygame.draw.line(s, DARK_GRAY, (sx, 0), (sx, WINDOW_H), 2)

        y = 15
        title = self.font_title.render("THÔNG TIN", True, ACCENT)
        s.blit(title, (sx + 20, y)); y += 40

        # Players
        red_name = AGENT_NAMES[self.red_agent_idx]
        yel_name = AGENT_NAMES[self.yellow_agent_idx]
        self._sidebar_label(s, sx, y, "🔴 ĐỎ:", red_name, RED_COLOR); y += 30
        self._sidebar_label(s, sx, y, "🟡 VÀNG:", yel_name, YELLOW_COLOR); y += 40

        # Move history
        hist_label = self.font_medium.render("Lịch sử nước đi:", True, GRAY)
        s.blit(hist_label, (sx + 20, y)); y += 25

        recent_moves = self.move_history[-12:]
        for i, (turn, col, t_ms) in enumerate(recent_moves):
            color = RED_COLOR if turn == 'red' else YELLOW_COLOR
            mv_t = self.font_small.render(f"  {turn[:3].upper()} → cột {col}  ({t_ms:.0f}ms)",
                                           True, color)
            s.blit(mv_t, (sx + 20, y)); y += 18

        y = WINDOW_H - 160
        pygame.draw.line(s, DARK_GRAY, (sx+10, y), (WINDOW_W-10, y))
        y += 10

        self.btn_menu.rect.topleft = (sx + 20, y)
        self.btn_restart.rect.topleft = (sx + 20, y + 54)
        self.btn_menu.draw(s)
        self.btn_restart.draw(s)

    def _sidebar_label(self, s, sx, y, label, value, color):
        lt = self.font_small.render(label, True, GRAY)
        s.blit(lt, (sx + 20, y))
        vt = self.font_medium.render(value, True, color)
        s.blit(vt, (sx + 90, y - 1))

    # ── Logic ─────────────────────────────────────────────────────────────────

    def get_current_agent_fn(self):
        turn = self.env.turn
        idx = self.red_agent_idx if turn == 'red' else self.yellow_agent_idx
        return AGENTS[AGENT_NAMES[idx]]

    def is_human_turn(self) -> bool:
        fn = self.get_current_agent_fn()
        return fn is None

    def trigger_ai_move(self):
        fn = self.get_current_agent_fn()
        if fn is None:
            return
        self.ai_thinking = True
        self.thinking_start = time.perf_counter()

        def run_ai():
            self.ai_move = fn(self.env)

        t = threading.Thread(target=run_ai, daemon=True)
        t.start()

    def apply_move(self, col: int):
        t0 = time.perf_counter()
        turn = self.env.turn
        _, reward, done, _ = self.env.step(col)
        elapsed = (time.perf_counter() - t0) * 1000
        self.thinking_time_ms = elapsed
        self.move_history.append((turn, col, elapsed))

        if done:
            self.game_over = True
            if reward != 0:
                winner = turn.upper()
                self.winner_msg = f"{'🔴' if turn == 'red' else '🟡'} {winner} THẮNG!"
                self._find_win_cells(col, turn)
            else:
                self.winner_msg = "HÒA!"

    def _find_win_cells(self, last_col, last_turn):
        """Tìm 4 ô thắng để highlight."""
        board = self.env.state
        piece = 1 if last_turn == 'red' else -1
        directions = [(1,0),(0,1),(1,1),(1,-1)]
        for col in range(COLS):
            for row in range(ROWS):
                if board[col][row] != piece:
                    continue
                for dc, dr in directions:
                    cells = [(col + dc*i, row + dr*i) for i in range(4)]
                    if all(0 <= c < COLS and 0 <= r < ROWS and
                           board[c][r] == piece for c, r in cells):
                        self.win_cells = cells
                        return

    def reset_game(self):
        self.env.reset()
        self.game_over = False
        self.winner_msg = ''
        self.ai_thinking = False
        self.ai_move = None
        self.move_history = []
        self.win_cells = []
        self.thinking_time_ms = 0.0

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self):
        running = True
        ai_triggered = False

        while running:
            dt = self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if self.mode_screen:
                    # Menu events
                    if self.btn_start.handle_event(event):
                        self.mode_screen = False
                        self.reset_game()
                        ai_triggered = False

                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        mx, my = event.pos
                        cx = WINDOW_W // 2
                        # Red selectors
                        if cx - 130 <= mx <= cx - 110 and 220 <= my <= 240:
                            self.red_agent_idx = (self.red_agent_idx - 1) % len(AGENT_NAMES)
                        elif cx + 110 <= mx <= cx + 130 and 220 <= my <= 240:
                            self.red_agent_idx = (self.red_agent_idx + 1) % len(AGENT_NAMES)
                        # Yellow selectors
                        if cx - 130 <= mx <= cx - 110 and 300 <= my <= 320:
                            self.yellow_agent_idx = (self.yellow_agent_idx - 1) % len(AGENT_NAMES)
                        elif cx + 110 <= mx <= cx + 130 and 300 <= my <= 320:
                            self.yellow_agent_idx = (self.yellow_agent_idx + 1) % len(AGENT_NAMES)

                else:
                    # Game events
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            self.reset_game(); ai_triggered = False
                        elif event.key == pygame.K_m:
                            self.mode_screen = True; self.reset_game()
                        elif event.key == pygame.K_ESCAPE:
                            self.mode_screen = True

                        # Human keyboard input (1-7)
                        if not self.game_over and self.is_human_turn() and not self.ai_thinking:
                            key_col = None
                            if event.key == pygame.K_1: key_col = 1
                            elif event.key == pygame.K_2: key_col = 2
                            elif event.key == pygame.K_3: key_col = 3
                            elif event.key == pygame.K_4: key_col = 4
                            elif event.key == pygame.K_5: key_col = 5
                            elif event.key == pygame.K_6: key_col = 6
                            elif event.key == pygame.K_7: key_col = 7
                            if key_col and key_col in self.env.validinputs:
                                self.apply_move(key_col)
                                ai_triggered = False

                    if event.type == pygame.MOUSEMOTION:
                        mx, my = event.pos
                        if my > HEADER_H and mx < BOARD_WIDTH:
                            self.hover_col = mx // CELL_SIZE
                        else:
                            self.hover_col = -1

                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        mx, my = event.pos
                        # Click on board
                        if (my > HEADER_H and mx < BOARD_WIDTH and
                                not self.game_over and self.is_human_turn() and
                                not self.ai_thinking):
                            col = mx // CELL_SIZE + 1
                            if col in self.env.validinputs:
                                self.apply_move(col)
                                ai_triggered = False

                        # Sidebar buttons
                        if self.btn_restart.handle_event(event):
                            self.reset_game(); ai_triggered = False
                        if self.btn_menu.handle_event(event):
                            self.mode_screen = True; self.reset_game()

                    self.btn_restart.handle_event(event)
                    self.btn_menu.handle_event(event)

            # AI turn logic
            if not self.mode_screen and not self.game_over:
                if not self.is_human_turn() and not self.ai_thinking and not ai_triggered:
                    self.trigger_ai_move()
                    ai_triggered = True

                if self.ai_thinking and self.ai_move is not None:
                    col = self.ai_move
                    self.ai_move = None
                    self.ai_thinking = False
                    ai_triggered = False
                    self.thinking_time_ms = (time.perf_counter() - self.thinking_start) * 1000
                    if col in self.env.validinputs:
                        self.apply_move(col)
                elif not self.ai_thinking:
                    ai_triggered = False

            # Draw
            if self.mode_screen:
                self.draw_menu()
            else:
                self.draw_game()

            pygame.display.flip()

        pygame.quit()


if __name__ == '__main__':
    app = Connect4UI()
    app.run()

"""
Test engine và các tác tử cơ bản (chuyển từ test_engine.ipynb sang .py).
Chạy: python -m pytest tests/test_engine.py -v
      python -m unittest tests.test_engine -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.connect4_env import ConnectFourEnv

# Ván hòa thật: random với seed 2026, ván thứ 33 (42 nước, bàn đầy, không ai thắng)
TIE_MOVES = [7, 7, 7, 5, 2, 6, 5, 1, 2, 3, 5, 2, 2, 1, 1, 5, 6, 6, 6, 7, 1, 7,
             3, 1, 3, 3, 2, 5, 2, 6, 6, 3, 3, 1, 5, 7, 4, 4, 4, 4, 4, 4]
# Ván kết thúc bằng đường chéo (người thắng là vàng, nước cuối do vàng đi)
DIAG_BACKSLASH = [2, 4, 2, 2, 1, 1, 3, 3, 1, 1]   # chéo \
DIAG_SLASH = [2, 4, 3, 1, 4, 2, 3, 3, 4, 4]       # chéo /
# Vàng có 3 quân ngang ở hàng dưới (cột 1-3), đến lượt đỏ -> phải chặn cột 4
BLOCK_HORIZONTAL = [6, 1, 6, 2, 7, 3]
# Đỏ có 3 quân dọc ở cột 4, đến lượt đỏ -> thắng ngay ở cột 4
WIN_VERTICAL = [4, 1, 4, 1, 4, 2]


def play(moves):
    env = ConnectFourEnv()
    env.reset()
    for m in moves:
        env.step(m)
    return env


class TestConnectFourEnv(unittest.TestCase):
    """Kiểm thử cơ bản của game engine."""

    def setUp(self):
        self.env = ConnectFourEnv()
        self.env.reset()

    def test_initial_state_zeros(self):
        self.assertTrue(np.all(self.env.state == 0))

    def test_initial_turn_red(self):
        self.assertEqual(self.env.turn, 'red')

    def test_initial_valid_inputs(self):
        self.assertEqual(sorted(self.env.validinputs), list(range(1, 8)))

    def test_initial_not_done(self):
        self.assertFalse(self.env.done)

    def test_action_space_n(self):
        self.assertEqual(self.env.action_space.n, 7)

    def test_observation_space_shape(self):
        self.assertEqual(self.env.observation_space.shape, (7, 6))

    def test_step_returns_correct_shape(self):
        state, reward, done, info = self.env.step(4)
        self.assertEqual(state.shape, (7, 6))

    def test_step_places_piece_in_correct_column(self):
        self.env.step(1)
        self.assertEqual(self.env.state[0][0], 1)

    def test_step_alternates_turns(self):
        self.env.step(1)
        self.assertEqual(self.env.turn, 'yellow')
        self.env.step(2)
        self.assertEqual(self.env.turn, 'red')

    def test_step_pieces_stack(self):
        self.env.step(4)
        self.env.step(4)
        self.assertEqual(self.env.state[3][0], 1)
        self.assertEqual(self.env.state[3][1], -1)

    def test_invalid_action_raises(self):
        with self.assertRaises(ValueError):
            self.env.step(8)

    def test_full_column_removed_from_valid(self):
        for _ in range(3):
            self.env.step(1)
            self.env.step(1)
        self.assertNotIn(1, self.env.validinputs)

    def test_sample_returns_valid_action(self):
        for _ in range(20):
            self.assertIn(self.env.sample(), self.env.validinputs)

    def test_reset_clears_board(self):
        self.env.step(4)
        self.env.reset()
        self.assertTrue(np.all(self.env.state == 0))
        self.assertEqual(self.env.turn, 'red')


class TestWinDetection(unittest.TestCase):
    """Phát hiện thắng theo 4 hướng."""

    def test_win_vertical(self):
        env = play([4, 1, 4, 1, 4, 1, 4])
        self.assertTrue(env.done)
        self.assertEqual(env.reward, 1)
        self.assertEqual(env.info, 'red wins')

    def test_win_horizontal(self):
        env = play([1, 1, 2, 2, 3, 3, 4])
        self.assertTrue(env.done)
        self.assertEqual(env.info, 'red wins')

    def test_win_diagonal_backslash(self):
        env = play(DIAG_BACKSLASH)
        self.assertTrue(env.done)
        self.assertEqual(env.info, 'yellow wins')

    def test_win_diagonal_slash(self):
        env = play(DIAG_SLASH)
        self.assertTrue(env.done)
        self.assertEqual(env.info, 'yellow wins')

    def test_no_win_with_three(self):
        env = play([1, 1, 2, 2, 3])
        self.assertFalse(env.done)


class TestDraw(unittest.TestCase):
    """Hòa thật: bàn đầy 42 quân, không ai có 4 liên tiếp."""

    def test_real_tie(self):
        env = play(TIE_MOVES)
        self.assertTrue(env.done)
        self.assertEqual(env.info, 'tie')
        self.assertEqual(env.reward, 0)
        self.assertEqual(int(np.count_nonzero(env.state)), 42)
        self.assertEqual(env.validinputs, [])

    def test_tie_is_not_win_for_anyone(self):
        env = play(TIE_MOVES)
        self.assertEqual(env.result('red'), 0)
        self.assertEqual(env.result('yellow'), 0)


class TestAfterGameEnd(unittest.TestCase):
    def test_step_after_done_rejected(self):
        env = play([4, 1, 4, 1, 4, 1, 4])
        self.assertTrue(env.done)
        with self.assertRaises(ValueError):
            env.step(2)


class TestEngineAPI(unittest.TestCase):
    """API thêm theo đề: legal_actions, is_terminal, result, clone, undo."""

    def test_legal_actions_initial(self):
        self.assertEqual(play([]).legal_actions(), [1, 2, 3, 4, 5, 6, 7])

    def test_legal_actions_skips_full_column(self):
        env = play([1, 1, 1, 1, 1, 1])   # cột 1 đầy sau 6 quân (không ai thắng)
        self.assertEqual(env.legal_actions(), [2, 3, 4, 5, 6, 7])

    def test_legal_actions_returns_copy(self):
        env = play([])
        env.legal_actions().append(99)
        self.assertNotIn(99, env.validinputs)

    def test_is_terminal(self):
        self.assertFalse(play([4, 1]).is_terminal())
        self.assertTrue(play([4, 1, 4, 1, 4, 1, 4]).is_terminal())

    def test_result_win_loss_playing(self):
        env = play([4, 1, 4, 1, 4, 1, 4])   # đỏ thắng
        self.assertEqual(env.result('red'), 1)
        self.assertEqual(env.result('yellow'), -1)
        self.assertEqual(play([4, 1]).result('red'), 0)   # đang chơi

    def test_result_yellow_wins(self):
        env = play(DIAG_SLASH)
        self.assertEqual(env.result('yellow'), 1)
        self.assertEqual(env.result('red'), -1)

    def test_result_invalid_player(self):
        with self.assertRaises(ValueError):
            play([]).result('green')

    def test_clone_independent(self):
        env = play([4])
        c = env.clone()
        c.step(3)
        self.assertEqual(env.state[2][0], 0)
        self.assertEqual(env.turn, 'yellow')
        self.assertEqual(len(env._history), 1)

    def test_clone_keeps_history_for_undo(self):
        c = play([4, 5]).clone()
        c.undo()
        self.assertEqual(c.state[4][0], 0)
        self.assertEqual(c.turn, 'yellow')

    def test_undo_single_move(self):
        env = play([4])
        env.undo()
        self.assertTrue(np.all(env.state == 0))
        self.assertEqual(env.turn, 'red')

    def test_undo_restores_full_column(self):
        env = play([1, 1, 1, 1, 1, 1])
        self.assertNotIn(1, env.validinputs)
        env.undo()
        self.assertEqual(env.validinputs, [1, 2, 3, 4, 5, 6, 7])

    def test_undo_after_win(self):
        env = play([4, 1, 4, 1, 4, 1, 4])
        env.undo()
        self.assertFalse(env.done)
        self.assertEqual(env.reward, 0)
        self.assertEqual(env.info, '')
        self.assertEqual(env.turn, 'red')
        env.step(2)   # chơi tiếp được sau khi hoàn tác

    def test_undo_after_tie(self):
        env = play(TIE_MOVES)
        env.undo()
        self.assertFalse(env.done)
        self.assertEqual(env.info, '')
        self.assertEqual(env.validinputs, [4])

    def test_undo_empty_raises(self):
        with self.assertRaises(ValueError):
            play([]).undo()

    def test_step_then_undo_many_matches_snapshots(self):
        import random
        rng = random.Random(7)
        for _ in range(100):
            env = ConnectFourEnv()
            env.reset()
            snaps = []
            while not env.done:
                snaps.append((env.state.copy(), env.turn, env.done,
                              env.reward, env.info, list(env.validinputs)))
                env.step(rng.choice(env.validinputs))
            for st, turn, done, reward, info, valid in reversed(snaps):
                env.undo()
                self.assertTrue(np.array_equal(env.state, st))
                self.assertEqual((env.turn, env.done, env.reward, env.info),
                                 (turn, done, reward, info))
                self.assertEqual(env.validinputs, valid)


class TestAgents(unittest.TestCase):
    """Kiểm thử tác tử cơ bản, test chặn assert đúng cột."""

    def test_random_agent_valid(self):
        from agents.random_agent import conn_random
        for _ in range(10):
            env = play([])
            self.assertIn(conn_random(env), env.validinputs)

    def test_think1_takes_win(self):
        from agents.heuristic_agent import conn_think1
        self.assertEqual(conn_think1(play(WIN_VERTICAL)), 4)

    def test_think2_blocks_horizontal(self):
        from agents.heuristic_agent import conn_think2
        self.assertEqual(conn_think2(play(BLOCK_HORIZONTAL)), 4)

    def test_think3_blocks_horizontal(self):
        from agents.heuristic_agent import conn_think3
        self.assertEqual(conn_think3(play(BLOCK_HORIZONTAL)), 4)

    def test_alphabeta_returns_valid(self):
        from agents.alphabeta_agent import alphabeta_agent
        env = play([])
        self.assertIn(alphabeta_agent(env, depth=3), env.validinputs)

    def test_mcts_returns_valid(self):
        from agents.mcts_agent import mcts_agent
        env = play([])
        self.assertIn(mcts_agent(env, n_simulations=50), env.validinputs)

    def test_full_game_random_vs_random(self):
        from game.game_utils import one_conn_game
        from agents.random_agent import conn_random
        for _ in range(5):
            reward, moves = one_conn_game(conn_random, conn_random)
            self.assertIn(reward, [-1, 0, 1])
            self.assertGreater(len(moves), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

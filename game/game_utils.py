import sys
import os


from game.connect4_env import ConnectFourEnv
from typing import Callable, Tuple, List


"""
Utility: Chạy một ván game hoàn chỉnh
=======================================
Theo sách Liu (2024), Ch. 3, one_conn_game().

Cung cấp các hàm để:
- Chạy 1 ván game giữa 2 agent
- Chạy nhiều ván và thống kê kết quả
- Agent thủ công (người chơi)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def one_conn_game(player1: Callable, player2: Callable,
                  render: bool = False,
                  player1_kwargs: dict = None,
                  player2_kwargs: dict = None) -> Tuple[float, List]:
    """
    Chạy một ván game Connect Four hoàn chỉnh.
    
    Theo sách Liu (2024), Ch. 3, one_conn_game().
    
    Parameters
    ----------
    player1 : callable
        Hàm agent cho người chơi 1 (đỏ, đi trước).
        Signature: player1(env) → int (column 1-7)
    player2 : callable
        Hàm agent cho người chơi 2 (vàng).
    render : bool
        True = hiển thị bàn cờ sau mỗi nước (mặc định False).
    player1_kwargs : dict
        Tham số bổ sung cho player1 (vd: {'depth': 5}).
    player2_kwargs : dict
        Tham số bổ sung cho player2.
    
    Returns
    -------
    reward : float
        1 nếu player1 thắng, -1 nếu player2 thắng, 0 nếu hòa.
    moves : list of (turn, action) tuples
        Lịch sử nước đi trong ván.
    """
    if player1_kwargs is None:
        player1_kwargs = {}
    if player2_kwargs is None:
        player2_kwargs = {}

    env = ConnectFourEnv()
    env.reset()
    moves = []

    if render:
        env.render()

    while True:
        # Player 1 (red) đi
        action = player1(env, **player1_kwargs)
        moves.append(('red', action))
        state, reward, done, info = env.step(action)
        if render:
            env.render()
            print(f"  Red played column {action}")
        if done:
            winner_is_p1 = True   # nước cuối do player1 đi
            break

        # Player 2 (yellow) đi
        action = player2(env, **player2_kwargs)
        moves.append(('yellow', action))
        state, reward, done, info = env.step(action)
        if render:
            env.render()
            print(f"  Yellow played column {action}")
        if done:
            winner_is_p1 = False  # nước cuối do player2 đi
            break

    # env.step() trả reward theo góc nhìn NGƯỜI VỪA ĐI (=1 khi người đó thắng).
    # Đổi sang góc nhìn player1: +1 player1 thắng, -1 player2 thắng, 0 hòa.
    if reward == 0:
        return 0, moves
    return (1 if winner_is_p1 else -1), moves


def run_games(player1: Callable, player2: Callable,
              n_games: int = 100,
              player1_kwargs: dict = None,
              player2_kwargs: dict = None,
              verbose: bool = False) -> dict:
    """
    Chạy n_games ván và thống kê kết quả.
    
    Chia đều quyền đi trước: n_games/2 ván player1 đi trước,
    n_games/2 ván player2 đi trước.
    
    Returns
    -------
    dict với keys: wins, losses, draws, score, n_games
    """
    if player1_kwargs is None:
        player1_kwargs = {}
    if player2_kwargs is None:
        player2_kwargs = {}

    wins = 0
    losses = 0
    draws = 0

    for i in range(n_games):
        if i % 2 == 0:
            # player1 đi trước
            reward, _ = one_conn_game(player1, player2,
                                       player1_kwargs=player1_kwargs,
                                       player2_kwargs=player2_kwargs)
            if reward == 1:
                wins += 1
            elif reward == -1:
                losses += 1
            else:
                draws += 1
        else:
            # player2 đi trước (player1 đi sau = yellow)
            reward, _ = one_conn_game(player2, player1,
                                       player1_kwargs=player2_kwargs,
                                       player2_kwargs=player1_kwargs)
            # reward: 1 nếu player2 (đỏ) thắng → player1 thua
            if reward == 1:
                losses += 1
            elif reward == -1:
                wins += 1
            else:
                draws += 1

        if verbose and (i + 1) % 10 == 0:
            print(f"  Game {i+1}/{n_games}: W={wins} L={losses} D={draws}")

    score = (wins + 0.5 * draws) / n_games

    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'score': score,
        'n_games': n_games,
    }


def conn_manual(env: ConnectFourEnv) -> int:
    """
    Agent thủ công: người chơi nhập cột từ bàn phím.
    Theo sách Liu (2024), Ch. 3, conn_manual().
    """
    print(f"\nGame state:\n{env._board_str()}")
    while True:
        try:
            move = int(input(f"{env.turn.upper()} player, enter column (1-7): "))
        except ValueError:
            print("  Vui lòng nhập số từ 1-7.")
            continue
        if move in env.validinputs:
            return move
        print(f"  Cột {move} không hợp lệ. Hợp lệ: {env.validinputs}")

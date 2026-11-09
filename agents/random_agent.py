import random
import sys
import os


from game.connect4_env import ConnectFourEnv


"""
Cấp 1: Random Agent
====================
Theo sách Ch. 3 (conn_random).
Agent chọn ngẫu nhiên một cột hợp lệ.
"""

import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def conn_random(env: ConnectFourEnv) -> int:
    """
    Cấp 1 - Random: Chọn ngẫu nhiên một cột hợp lệ.
    
    Parameters
    ----------
    env : ConnectFourEnv
    
    Returns
    -------
    int : cột được chọn (1-7)
    """
    return env.sample()

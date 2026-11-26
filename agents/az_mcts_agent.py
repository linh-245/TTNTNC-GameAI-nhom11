import os
import sys
import math
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.connect4_env import ConnectFourEnv
from training.model import predict
from agents.nn_agent import load_model
from training.encode import legal_mask

class AZNode:
    """Node cho AlphaZero MCTS (PUCT)."""
    __slots__ = ['env', 'parent', 'action', 'children', 'priors', 'visits', 'value_sum']
    
    def __init__(self, env, parent=None, action=None, prior=0.0):
        self.env = env.copy()
        self.parent = parent
        self.action = action
        self.children = {}  
        self.priors = prior
        self.visits = 0
        self.value_sum = 0.0
        
    def is_expanded(self):
        return len(self.children) > 0
        
    def expand(self, probs):
        """Mở rộng node bằng cách tạo children từ xác suất của mạng."""
        valid_actions = self.env.get_valid_actions()
        for a in valid_actions:
            new_env = self.env.copy()
            new_env.step(a)
            self.children[a] = AZNode(new_env, parent=self, action=a, prior=probs[a-1])
            
    def select_child(self, c_puct=1.0):
        """Chọn node con theo công thức PUCT: Q + U."""
        best_score = -float('inf')
        best_child = None
        for child in self.children.values():
            q_value = -(child.value_sum / child.visits) if child.visits > 0 else 0.0
            u_value = c_puct * child.priors * math.sqrt(self.visits) / (1 + child.visits)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

_MODEL_CACHE = {}

def _get_model(ckpt):
    """Chỉ đọc checkpoint một lần cho mỗi đường dẫn (trước đây đọc lại ở MỖI nước đi)."""
    if ckpt not in _MODEL_CACHE:
        _MODEL_CACHE[ckpt] = load_model(ckpt)
    return _MODEL_CACHE[ckpt]


def az_mcts_agent(env: ConnectFourEnv, n_simulations: int = 200, c_puct: float = 1.0, ckpt: str = 'weights/sl.pt') -> int:
    """Cấp 7: MCTS dùng mạng Policy-Value."""
    start_time = time.perf_counter()
    model = _get_model(ckpt)
    root = AZNode(env)
    
    # Đánh giá node gốc lần đầu
    probs, _ = predict(model, env)
    root.expand(probs)
    
    for _ in range(n_simulations):
        node = root
        path = [node]
        
        # 1. Selection
        while node.is_expanded() and not node.env.done:
            node = node.select_child(c_puct)
            path.append(node)
            
        # 2. Evaluation (Dùng mạng thay vì rollout)
        if not node.env.done:
            probs, value = predict(model, node.env)
            node.expand(probs)
        else:
            # Nếu game kết thúc, lấy kết quả thực tế
            # Quy ước: +1 nếu người vừa đi thắng, -1 nếu thua (góc nhìn người sắp đi)
            value = -1.0 if node.env.reward == 1 else (1.0 if node.env.reward == -1 else 0.0)
            
        # 3. Backpropagation
        # value là góc nhìn của người SẮP ĐI tại lá. Đảo dấu theo độ sâu (không dùng env.turn,
        # vì sau nước thắng env.turn không đổi nên so sánh turn sẽ sai dấu từ độ sâu 2 trở lên).
        for i, n in enumerate(reversed(path)):
            n.value_sum += value if i % 2 == 0 else -value
            n.visits += 1
            
    # Chọn nước đi tốt nhất dựa trên số lần thăm (visits)
    best_action = max(root.children.items(), key=lambda x: x[1].visits)[0]
    latency = (time.perf_counter() - start_time) * 1000.0
    az_mcts_agent.last_stats = {"sims": n_simulations, "latency_ms": round(latency, 2)}
    return best_action

az_mcts_agent.last_stats = {"sims": 0, "latency_ms": 0}
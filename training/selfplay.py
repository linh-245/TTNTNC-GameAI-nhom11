import argparse
import os
import sys
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.connect4_env import ConnectFourEnv
from agents.az_mcts_agent import AZNode
from training.model import predict, PolicyValueNet
from agents.nn_agent import load_model
from training.encode import encode_state, legal_mask

def selfplay_game(model, n_simulations=100, c_puct=1.0, temp_plies=10):
    """Chơi một ván tự chơi và trả về dữ liệu (state, policy, outcome)."""
    env = ConnectFourEnv()
    env.reset()
    states, policies, turns = [], [], []
    
    while not env.done:
        # Chạy MCTS để lấy xác suất nước đi tốt nhất
        root = AZNode(env)
        probs, _ = predict(model, env)
        root.expand(probs)
        
        # Mô phỏng
        for _ in range(n_simulations):
            node = root
            path = [node]
            while node.is_expanded() and not node.env.done:
                node = node.select_child(c_puct)
                path.append(node)
                
            if not node.env.done:
                probs, value = predict(model, node.env)
                node.expand(probs)
            else:
                value = -1.0 if node.env.reward == 1 else (1.0 if node.env.reward == -1 else 0.0)
                
            for i, n in enumerate(reversed(path)):
                n.value_sum += value if i % 2 == 0 else -value
                n.visits += 1
                
        # Tính xác suất từ số lần thăm (visit counts)
        visit_counts = [root.children[a].visits if a in root.children else 0 for a in range(1, 8)]
        total_visits = sum(visit_counts)
        if total_visits == 0:
            # Fallback nếu không có nước nào được thăm
            valid = env.get_valid_actions()
            visit_probs = np.zeros(7)
            for v in valid: visit_probs[v-1] = 1.0 / len(valid)
        else:
            visit_probs = np.array(visit_counts) / total_visits
            
        states.append(encode_state(env))
        policies.append(visit_probs)
        turns.append(env.turn)
        
        # Khám phá: trong `temp_plies` nước đầu lấy MẪU theo tỷ lệ số lần thăm (nhiệt độ 1);
        # sau đó chọn argmax. Nếu chỉ dùng argmax thì mọi ván tự chơi giống hệt nhau
        # (cùng mạng, cùng thế đầu) và dữ liệu bị lặp lại.
        if len(states) <= temp_plies:
            best_action = int(np.random.choice(7, p=visit_probs)) + 1
        else:
            best_action = int(np.argmax(visit_probs)) + 1
        env.step(best_action)
        
    # Xác định kết quả cuối cùng
    winner = env.turn if env.reward == 1 else None
    
    # Gán nhãn value cho từng trạng thái (từ góc nhìn người đi tại trạng thái đó)
    values = []
    for t in turns:
        if winner is None:
            values.append(0.0)
        elif t == winner:
            values.append(1.0)
        else:
            values.append(-1.0)
            
    return states, policies, values

def train_step(model, optimizer, states, policies, values, batch_size=128):
    """Huấn luyện mạng trên dữ liệu tự chơi."""
    model.train()
    indices = np.random.permutation(len(states))
    
    total_loss = 0
    for i in range(0, len(states), batch_size):
        batch_idx = indices[i:i+batch_size]
        x_batch = torch.FloatTensor(np.array([states[j] for j in batch_idx]))
        p_batch = torch.FloatTensor(np.array([policies[j] for j in batch_idx]))
        v_batch = torch.FloatTensor(np.array([values[j] for j in batch_idx]))
        
        logits, v_pred = model(x_batch)
        
        # Loss = Cross Entropy (Policy) + MSE (Value)
        p_loss = F.cross_entropy(logits, p_batch)
        v_loss = F.mse_loss(v_pred, v_batch)
        loss = p_loss + v_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / (len(states) // batch_size + 1)

def main():
    ap = argparse.ArgumentParser(description="Tự chơi kiểu AlphaZero (cấp 8)")
    ap.add_argument("--init", default="weights/sl.pt", help="checkpoint khởi tạo")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--games", type=int, default=40, help="số ván tự chơi mỗi vòng")
    ap.add_argument("--sims", type=int, default=50, help="số mô phỏng mỗi nước")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--temp-plies", type=int, default=10, help="số nước đầu lấy mẫu thay vì argmax")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="weights/az_iter", help="ghi {prefix}_{N}.pt")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    print("Bắt đầu vòng lặp AlphaZero Self-Play...", vars(args))
    model = load_model(args.init)  # khởi tạo từ SL
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for it in range(args.iterations):
        print(f"\n--- Vòng lặp {it+1}/{args.iterations} ---")
        all_states, all_policies, all_values = [], [], []

        # 1. Tự chơi để sinh dữ liệu
        t0 = time.time()
        for g in range(args.games):
            s, p, v = selfplay_game(model, n_simulations=args.sims, temp_plies=args.temp_plies)
            all_states.extend(s)
            all_policies.extend(p)
            all_values.extend(v)
            print(f"  Ván {g+1}/{args.games} xong. Số trạng thái: {len(s)}")

        print(f"Thu thập {len(all_states)} trạng thái trong {time.time()-t0:.1f}s")

        # 2. Huấn luyện mạng
        print("Đang huấn luyện mạng...")
        for epoch in range(args.epochs):
            loss = train_step(model, optimizer, all_states, all_policies, all_values)
            print(f"  Epoch {epoch+1}: Loss = {loss:.4f}")

        # 3. Lưu checkpoint mới
        out = f"{args.out_prefix}_{it+1}.pt"
        torch.save({"model": model.state_dict(), "config": {"channels": 64, "blocks": 3}}, out)
        print(f"Đã lưu {out}")

if __name__ == '__main__':
    main()

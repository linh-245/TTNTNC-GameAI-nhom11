import json
import os
import sys
from collections import defaultdict

# Đảm bảo có thể import từ thư mục gốc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SỬA ĐƯỜNG DẪN NÀY CHO ĐÚNG VỚI FILE LOG BẠN MUỐN PHÂN TÍCH
LOG_FILE = 'logs/tournament_final.jsonl'

def analyze():
    if not os.path.exists(LOG_FILE):
        print(f"LỖI: Không tìm thấy file {LOG_FILE}.")
        print("Hãy chắc chắn bạn đã chạy giải đấu và đường dẫn ở trên là chính xác.")
        return

    # Cấu trúc lưu trữ: stats[agent_name] = { 'first_win': 0, 'first_play': 0, ... }
    stats = defaultdict(lambda: {
        'first_win': 0, 'first_play': 0, 
        'second_win': 0, 'second_play': 0, 
        'win_lengths': [], 'loss_lengths': []
    })
    
    total_games = 0
    draws = 0

    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): 
                continue
            game = json.loads(line)
            total_games += 1
            
            p1, p2 = game['pair']
            winner = game.get('winner')
            num_moves = game.get('plies')
            
            # Xác định ai đi trước
            first_player = p1 if game.get('a_first') else p2
            second_player = p2 if game.get('a_first') else p1
            
            if winner is None:
                draws += 1
                
            # --- Cập nhật thống kê cho người đi trước ---
            stats[first_player]['first_play'] += 1
            if winner == first_player:
                stats[first_player]['first_win'] += 1
                stats[first_player]['win_lengths'].append(num_moves)
            elif winner == second_player:
                stats[first_player]['loss_lengths'].append(num_moves)
                
            # --- Cập nhật thống kê cho người đi sau ---
            stats[second_player]['second_play'] += 1
            if winner == second_player:
                stats[second_player]['second_win'] += 1
                stats[second_player]['win_lengths'].append(num_moves)
            elif winner == first_player:
                stats[second_player]['loss_lengths'].append(num_moves)

    # --- In kết quả ---
    print(f"Đã phân tích {total_games} ván đấu từ {LOG_FILE} (Số ván hòa: {draws})")
    print("-" * 95)
    print(f"{'Tác tử':<15} | {'Tỉ lệ thắng đi trước':<22} | {'Tỉ lệ thắng đi sau':<22} | {'Độ dài ván thắng TB'}")
    print("-" * 95)
    
    # Sắp xếp theo tên tác tử cho dễ nhìn
    for agent in sorted(stats.keys()):
        s = stats[agent]
        
        # Tính tỷ lệ đi trước
        if s['first_play'] > 0:
            first_rate = (s['first_win'] / s['first_play']) * 100
            first_str = f"{first_rate:>5.1f}% ({s['first_win']}/{s['first_play']})"
        else:
            first_str = "N/A"
            
        # Tính tỷ lệ đi sau
        if s['second_play'] > 0:
            second_rate = (s['second_win'] / s['second_play']) * 100
            second_str = f"{second_rate:>5.1f}% ({s['second_win']}/{s['second_play']})"
        else:
            second_str = "N/A"
            
        # Tính độ dài ván thắng trung bình
        if s['win_lengths']:
            avg_win_len = sum(s['win_lengths']) / len(s['win_lengths'])
            len_str = f"{avg_win_len:.1f} nước"
        else:
            len_str = "N/A"
            
        print(f"{agent:<15} | {first_str:<22} | {second_str:<22} | {len_str:>10}")

if __name__ == '__main__':
    analyze()
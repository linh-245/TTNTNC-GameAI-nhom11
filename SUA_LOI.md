# Các chỗ đã chỉnh 
Logic của `game/game_utils.py`, `agents/mcts_agent.py` (lan truyền ngược bằng `mover`) và `conn_think3` trong `agents/heuristic_agent.py` trong bản ban đầu đã đúng, được kiểm chứng bằng `tests/test_regression.py`. Vì vậy các mục dưới đây là chỉnh sửa mô tả và bổ sung tính năng, không phải sửa lỗi logic.

| # | File | Nội dung chỉnh | Ghi chú |
|---|------|----------------|---------|
| 1 | `agents/heuristic_agent.py` | Viết lại docstring đầu file, `conn_think2` và `conn_think3` cho khớp với code (Think3: thắng ngay, chặn, lọc nước an toàn, ưu tiên cột giữa; không có bước tìm "fork"). | Không đổi logic. |
| 2 | `agents/mcts_agent.py` | Dùng đối tượng `random.Random` riêng (`rng`) thay cho `random` toàn cục trong `expand` và `_simulate`; thêm tham số `seed` và `budget_ms`; vòng lặp đổi sang `while sims < n_simulations` có kiểm tra thời gian; ghi `mcts_agent.last_stats` (số mô phỏng, độ trễ ms). Rút gọn docstring của `mcts_agent`. | Bổ sung tính năng để chạy giải đấu lặp lại được và giới hạn thời gian. Phần lan truyền ngược bằng `mover` giữ nguyên. |

Kiểm tra: `python -m pytest tests/test_regression.py -v` (kết quả: 7 passed).

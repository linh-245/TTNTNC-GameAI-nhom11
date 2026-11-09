# Nhật ký đóng góp

Mỗi người tự ghi vào mục của mình **mỗi tối**. Cuối tuần dùng bảng này cùng GitHub Insights để khai tỷ lệ đóng góp (tổng 100%).

## Nguyễn Kiều Linh (@linh-245)

| Ngày | Việc đã làm | Commit / PR / Issue |
|------|-------------|---------------------|
| 09/11 | Xem xét và gộp Pull Request #1 đến #4 vào `main` bằng Merge commit, gồm PR #3 (`fix/bup-think3-block`, sửa docstring `heuristic_agent.py` cho khớp code) và PR #4 (`fix/mai-mcts-backprop`, `mcts_agent.py`) | PR #1, #2, #3, #4 (đã merge); commit merge `955e24d` (PR #3) |
| 09/11 | Đối chiếu bản gốc với bản sửa lỗi, xác định thay đổi thật chỉ là docstring của `heuristic_agent.py` và phần `rng`, `seed`, `budget_ms`, `last_stats` của `mcts_agent.py`; viết lại `SUA_LOI.md` cho khớp | commit `Duan_GameAI/Sualoi` (điền mã commit) |
| 09/11 | Tạo nhánh `dev` và đặt làm default branch; thống nhất quy trình: nhánh `feat/fix/test/docs` tách từ `dev`, Pull Request vào `dev`, khi xong mốc mới đưa `dev` vào `main` | nhánh `dev`; Settings > Default branch |

## Nguyễn Huỳnh Mai (@chloestudy0043-stack)
| Ngày | Việc đã làm | Commit / PR / Issue |
|---|---|---|
|09/11 | Viết `alphabeta_agent.py`, `mcts_agent.py` và tạo file eval gồm 2 notebook `_init_.py` và `tournament.ipynb` |PR #2, nhánh `feat/mai`|
|09/11 | Sửa dấu lan truyền ngược MCTS bằng mover trong `mcts_agent.py` |PR #4, nhánh `fix/mai-mcts-backprop`|

## Nguyễn Lương Hồng Búp (@bupnguyen-19)
| Ngày | Việc đã làm | Commit / PR / Issue |
|---|---|---|
| 09/11 | Đưa `agents/__init__.py`, `random_agent.py`, `heuristic_agent.py` vào repo (khung từ sách Liu) | commit `7b39fae`, `cde3d8c`, `612b8ad`; PR #1 (đã merge) |
| 09/11 | Sửa docstring `heuristic_agent.py` (đầu file, `conn_think2`, `conn_think3`) cho khớp code, bỏ mô tả "fork" không có trong code | commit `ff7fee0`, `b59a31a`, `a349ae7`, `0649aec`; PR #3 (đã merge), nhánh `fix/bup-think3-block` |
| 09/11 | Viết `training/encode.py` (mã hóa trạng thái 3x6x7) và `training/__init__.py` | commit `d43ff13`, nhánh `feat/bup-encode-model` (chưa push) |
| 09/11 | Sửa README (badge, Nhóm 11, GitHub thành viên, ghi chú thư mục chưa có, trạng thái Cấp 4) và thêm pytest vào `requirements.txt` | commit `2a462ea`, `360b716`, `e43ef2e`, `e447b19`, `78e52ce`, `8f7bf26`; PR #6, nhánh `docs/bup-readme` |

## Tỷ lệ đóng góp (điền ở ngày cuối)
| Thành viên | MSSV | Công việc chính | Tỷ lệ (%) |
|---|---|---|---|
| Nguyễn Kiều Linh | 3124720039 | | |
| Nguyễn Huỳnh Mai | 3124720043 | | |
| Nguyễn Lương Hồng Búp | 3124720009 | | |
| **Tổng** | | | **100** |

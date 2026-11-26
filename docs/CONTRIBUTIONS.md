# Nhật ký đóng góp

Mỗi người tự ghi vào mục của mình **mỗi tối**. Cuối tuần dùng bảng này cùng GitHub Insights để khai tỷ lệ đóng góp (tổng 100%).

## Nguyễn Kiều Linh (@linh-245)

| Ngày | Việc đã làm | Commit / PR / Issue |
|------|-------------|---------------------|
| 09/11 | Xem xét và gộp Pull Request #1 đến #4 vào `main` bằng Merge commit, gồm PR #3 (`fix/bup-think3-block`, sửa docstring `heuristic_agent.py` cho khớp code) và PR #4 (`fix/mai-mcts-backprop`, `mcts_agent.py`) | PR #1, #2, #3, #4 (đã merge); commit merge `955e24d` (PR #3) |
| 09/11 | Đối chiếu bản gốc với bản sửa lỗi, xác định thay đổi thật chỉ là docstring của `heuristic_agent.py` và phần `rng`, `seed`, `budget_ms`, `last_stats` của `mcts_agent.py`; viết lại `SUA_LOI.md` cho khớp | commit `Duan_GameAI/Sualoi` (điền mã commit) |
| 09/11 | Tạo nhánh `dev` và đặt làm default branch; thống nhất quy trình: nhánh `feat/fix/test/docs` tách từ `dev`, Pull Request vào `dev`, khi xong mốc mới đưa `dev` vào `main` | nhánh `dev`; Settings > Default branch |
| 11/11 | Merge `dev` vào `main` (Đợt 1 đến Đợt 3), chạy `python -m pytest tests/test_regression.py` ra 7 passed | PR #16; commit merge `d7d04d9` |
| 11/11 | Gắn tag mốc giữa kỳ `v0.1-fixed` trên `main` (xóa tag sai tên `v.01-fixed`, tạo lại đúng tên) | tag `v0.1-fixed` → `d7d04d9` |
| 11/11 | Engine: thêm `legal_actions`, `is_terminal`, `result`, `clone`, `undo` vào `connect4_env.py` | PR #18, nhánh `feat/linh-engine-api` |
| 11/11 | Test: chuyển test engine sang `.py`, thêm test hòa và đường chéo, xóa `test_engine.ipynb` | PR #17, nhánh `test/linh-engine-tests` |
| 11/11 | Solver: sinh 500 thế chuẩn cố định theo seed (`make_positions.py`), đo độ chính xác tối ưu (`optimal_accuracy.py`) | PR #19, nhánh `feat/linh-solver-positions` |
| 11/11 | UI: sửa emoji, hiển thị thống kê nút và mô phỏng, thêm menu MCTS theo ngân sách vào `pygame_ui.py` | PR #20, nhánh `feat/linh-ui-stats` |
| 12/11 | Khôi phục API engine bị ghi đè khi gộp giải đấu (`legal_actions`, `is_terminal`, `result`, `clone`, `undo`) và cập nhật README | PR #28, nhánh `fix/linh-restore-engine-api` |
| 12/11 | Chạy `python -m eval.tournament`, ra bảng xếp hạng 6 agent (AlphaBeta, MCTS, Think3, Think2, Think1, Random) | log CSV, JSONL (lưu trên Drive) |
| 12/11 | Merge `dev` vào `main` cho mốc agents | PR #29; commit merge `804ac26` |
| 12/11 | Gắn tag mốc `v0.2-agents` trên `main` | tag `v0.2-agents` → `804ac26` |
| 14/11 | Merge `dev` vào `main` cho mốc giữa kỳ, gắn tag `v0.9-giuaky` | PR #40; commit merge `e9fb5bf`; tag `v0.9-giuaky` |
| 26/11 | Giao diện Pygame: thêm mục AZ-MCTS vào menu `ui/pygame_ui.py` (chỉ hiện khi có checkpoint trong `weights/`); gộp thêm 3 test (nước đi hợp lệ, checkpoint thật, menu giao diện) vào `tests/test_az_mcts.py`; cập nhật `eval/tournament.ipynb` | commit `9602e6c` |
| 26/11 | Demo web: viết `web/app.py` (Gradio) chọn Human hoặc tác tử AI cho cả hai bên (Human vs AI, AI vs AI, Human vs Human), hiển thị lịch sử nước đi kèm thời gian, số nút, số mô phỏng; gọi trực tiếp `game/`, `agents/`, `weights/` của đồ án, không viết lại logic. Chạy thử trên máy | commit `3a49b01`, `80354bb`; PR #56 (commit merge `4c88071`), PR #57 (commit merge `d223189`), nhánh `feat/web-demo` |
| 26/11 | Đưa demo lên Hugging Face Spaces: tạo Space `Linh345/connect4-nhom11`, chép `app.py`, `game/`, `agents/`, `training/`, `weights/`, viết `README.md` và `requirements.txt` riêng cho Space (torch bản CPU), xử lý lỗi ZeroGPU bằng hàm `@spaces.GPU` giả; Space ở trạng thái Running | https://huggingface.co/spaces/Linh345/connect4-nhom11 (log Space lưu trên Hugging Face) |
| 26/11 | README: thêm mục "Demo web", dòng `web/` trong cấu trúc thư mục, link demo trực tuyến, cập nhật phần kế thừa/tự làm | commit `497870c`; PR #57 (commit merge `d223189`), nhánh `feat/web-demo` |
| 26/11 | Xem xét và gộp PR #55 (`docs/mai-nhat-ky`, nhật ký của Mai 17/11 đến 25/11) vào `dev` | PR #55; commit merge `dafd35b` (commit của Mai `7c3f4b9`) |
| 26/11 | Merge `dev` vào `main` cho mốc cuối kỳ | PR #58; commit merge `2fccb6c` |

## Nguyễn Huỳnh Mai (@chloestudy0043-stack)
| Ngày | Việc đã làm | Commit / PR / Issue |
|---|---|---|
|09/11 | Viết `alphabeta_agent.py`, `mcts_agent.py` và tạo file eval gồm 2 notebook `_init_.py` và `tournament.ipynb` |PR #2(đã merge), nhánh `feat/mai`|
|09/11 | Sửa dấu lan truyền ngược MCTS bằng mover trong `mcts_agent.py` |PR #4(đã merge), nhánh `fix/mai-mcts-backprop`|
|10/11 | Thêm cờ use_pruning, use_ordering và thêm bộ đếm nút trong `alphabeta_agent.py` |PR #13(đã merge), nhánh `feat/mai-alphabeta-pruning`|
|11/11 | Tăng tốc `eval_payoff_conn` bằng cửa sổ tính trước, thêm `budget_ms` (deepening lặp) và tùy chọn bảng chuyển vị trong `alphabeta_agent.py` |PR #21(đã merge), nhánh `feat/mai-alphabeta-budget`|
|12/11 | Thêm `random.Random(seed)` riêng, `budget_ms` và `last_stats` cho MCTS trong `mcts_agent.py` |PR #24(đã merge), nhánh `feat/mai-mcts-budget`|
|12/11 | Thêm `tests/test_search.py`: kiểm thử Alpha-Beta và MCTS |PR #25(đã merge), nhánh `test/mai-search-tests`|
|12/11 | Viết `eval/tournament.py`: giải đấu round-robin đa tiến trình, đổi bên đi trước, khai cuộc ngẫu nhiên; ghi log JSONL, tính Wilson CI và p95 |PR #26(đã merge), nhánh `feat/mai-tournament`|
|13/11 | Thêm `eval/ablation.py`: ablation A1 (bật/tắt cắt tỉa và sắp xếp nước) và A2 (MCTS theo `n_simulations` × `c`) |PR #27(đã merge), nhánh `feat/mai-ablation`|
|13/11 | Chạy giải đấu chế độ 1 và RQ1 (`eval/rq1.py`), thêm bảng tóm tắt `logs/summary_*.csv` và hình `results/figures/fig4_*.png` |PR #30(đã merge), nhánh `feat/mai-tournament-results`|
|13/11 | Viết `notebooks/01_search_results.ipynb`: ma trận Win Score + Wilson CI, thắng/hòa/thua, độ trễ p95, nút/giây, A1, A2 |PR #31(đã merge), nhánh `docs/mai-search-notebook`|
| 13/11 | Chạy thực nghiệm Ablation (A1: bật/tắt cắt tỉa, sắp xếp nước; A2: MCTS theo `n_simulations` x `c`) và cập nhật kết quả vào notebook. | PR #32 (đã merge), nhánh `feat/mai-ablation-results` |
| 13/11 | Di chuyển `01_search_results.ipynb` từ `notebooks/` sang `eval/`, sửa lỗi hàm `spec_budget` và bộ lọc trùng lặp để tổng hợp đúng dữ liệu RQ1 cả 2 mức 200ms và 1000ms. Cập nhật lại bảng tóm tắt `logs/summary_rq1.csv`. | PR #36 (đã merge), nhánh `fix/mai-move-notebook` |
| 13/11 | Viết tài liệu `docs/README_eval.md` hướng dẫn chi tiết cách chạy giải đấu, RQ1 và Ablation (cờ dòng lệnh, cách đọc kết quả). Cập nhật `README.md` và `docs/CONTRIBUTIONS.md` | PR #37 (đã mở chờ review), nhánh `docs/mai-report` |
| 17/11 | Thêm `--out` mặc định vào `logs/` và làm tròn số khi ghi CSV (script đo độ chính xác nước tối ưu) | commit `145a401`; PR #41 (đã merge), nhánh `feature/optimal-accuracy-csv` |
| 24/11 | Hoàn thiện Cấp 7 và Cấp 8 (AlphaZero), cập nhật giải đấu, các script đánh giá và file kết quả đánh giá | commit `2afd4e4`, `ac4866e`; PR #45 (đã merge), nhánh `feature/mai-az-mcts-final` |
| 25/11 | Sửa lỗi đảo dấu Q trong AZ-MCTS và cập nhật kết quả giải đấu | commit `3026771`; PR #46 (đã merge), nhánh `feature/mai-az-mcts-final` |

## Nguyễn Lương Hồng Búp (@bupnguyen-19)
| Ngày | Việc đã làm | Commit / PR / Issue |
|---|---|---|
| 09/11 | Đưa `agents/__init__.py`, `random_agent.py`, `heuristic_agent.py` vào repo (khung từ sách Liu) | commit `7b39fae`, `cde3d8c`, `612b8ad`; PR #1 (đã merge) |
| 09/11 | Sửa docstring `heuristic_agent.py` (đầu file, `conn_think2`, `conn_think3`) cho khớp code, bỏ mô tả "fork" không có trong code | commit `ff7fee0`, `b59a31a`, `a349ae7`; PR #3 (đã merge); commit `0649aec` (bỏ ký tự thừa trong docstring `conn_think3`); PR #9 (đã merge vào `main`); nhánh `fix/bup-think3-block`|
| 09/11 | Viết `training/encode.py` (mã hóa trạng thái 3x6x7) và `training/__init__.py` | commit `d43ff13`; PR #11 (đã merge vào `dev`), nhánh `feat/bup-encode-model` |
| 09/11 | Sửa README (badge, Nhóm 11, GitHub thành viên, ghi chú thư mục chưa có, trạng thái Cấp 4) và thêm pytest vào `requirements.txt` | commit `2a462ea`, `360b716`, `e43ef2e`, `e447b19`, `78e52ce`, `8f7bf26`; PR #6, nhánh `docs/bup-readme` |
| 10/11 | Viết `training/model.py` (mạng CNN policy-value, 3 khối residual, đầu policy 7 cột và đầu value) và `tests/test_learning.py` (6 test mã hóa trạng thái) | commit `c750f6e`, `3f5bcf4`; PR #11 (đã merge vào `dev`), nhánh `feat/bup-encode-model` |
| 10/11 | Viết `training/generate_data.py` (sinh ván tự chơi với giáo viên `conn_think3`, có khai cuộc ngẫu nhiên và exploration) và `training/dataset.py` (loại trùng thế, chia train/val/test theo ván) | commit `1ca0a72`, `a9b861f`; PR #12 (đã merge vào `dev`), nhánh `feat/bup-data-pipeline` |
| 10/11 | Chạy thử: 200 ván (seed 0) ra 5269 thế, sau loại trùng còn 4194; chia train/val/test = 3247/518/429 thế (theo ván 159/20/20); `pytest` 13 passed | chạy trên máy, dữ liệu `.npz` không đẩy lên GitHub |
| 11/11 | Viết `training/train_sl.py` (huấn luyện có giám sát: loss cross-entropy cho policy có che cột đầy cộng MSE cho value, chọn checkpoint theo loss tập val, lưu checkpoint và log .json gồm seed, thiết bị, thời gian) | commit `e1d702b`; PR #14, nhánh `feat/bup-train-sl` (đã mở, Linh đã merge vào dev) |
| 11/11 | Viết `agents/nn_agent.py` (agent cấp 5 chọn cột theo policy của mạng, có `last_stats`) và `tests/test_nn_agent.py` (5 test, tự bỏ qua khi thiếu torch) | commit `639683c`, `e092457`; PR #15, nhánh `feat/bup-nn-agent` (đã mở, Linh đã merge vào dev) |
| 11/11 | Chạy thử chức năng: 1000 ván giáo viên (seed 1) cho 26040 thế, train 5 epoch trên CPU 31,4 giây; nn_agent thắng Random 94/100. Chỉ để kiểm tra, chưa phải số liệu báo cáo | chạy trên máy; dữ liệu và checkpoint không đưa lên GitHub |
| 12/11 | Huấn luyện thật: sinh 5000 ván giáo viên Think3, chạy `train_sl.py`, lưu checkpoint tốt nhất `weights/sl.pt` (khoảng 0,93 MB), `weights/sl.json` (tham số chạy) và log `logs/training/sl_log.csv` | commit `b0aa1d9`; PR #23, nhánh `feat/bup-sl-checkpoint` (Linh đã merge vào `dev`) |
| 12/11 | Viết `training/plot_curves.py` vẽ đường học (loss tổng, policy, value theo epoch) ra `docs/images/sl_curves.png`; viết `docs/README_learning.md` (cách sinh dữ liệu, huấn luyện, đánh giá) | commit `c229e43`, `b0aa1d9`; PR #23 |
| 12/11 | Chạy thử `pytest`: 18 passed | chạy trên máy |
| 13/11 | Sửa lỗi tích hợp: đổi tên tham số `nn_agent` từ `checkpoint` thành `ckpt` cho khớp chỗ gọi trong `ui/pygame_ui.py` và `solver/optimal_accuracy.py` (trước đó gọi sẽ báo `TypeError`); cập nhật 5 chỗ gọi trong `tests/test_nn_agent.py`; `pytest`: 98 passed | PR #33 (đã merge vào `dev`), nhánh `fix/bup-nn-ckpt-name` |
| 13/11 | Viết `training/train_rl.py`: khung huấn luyện tăng cường bằng tự chơi (policy gradient có baseline), khởi tạo từ `weights/sl.pt`; mới chạy thử nhỏ 2 vòng x 4 ván để kiểm tra chạy được, chưa huấn luyện thật, chưa có `weights/rl.pt` | PR #34 (đã merge vào `dev`), nhánh `feat/bup-train-rl` |
| 13/11 | Viết `docs/report/bup.md`: phần báo cáo của Búp về học máy cấp 5 | commit `69e0171`; PR #35 (đã merge vào `dev`), nhánh `docs/bup-report` |
| 21/11 | Huấn luyện RL thật bằng `train_rl.py`: 300 vòng x 32 ván tự chơi, seed 0, khởi tạo từ `weights/sl.pt`, chạy CPU; lưu `weights/rl.pt`, `weights/rl.json`, `logs/training/rl_log.csv`. Thêm tùy chọn `--save-every` để lưu checkpoint trung gian. Trong log, tự chơi chuyển từ bên đi trước thắng nhanh (vòng 1 đến 50, đỏ thắng trung bình 31,3/32) sang hòa đầy bàn (vòng 251 đến 300, hòa trung bình 29,8/32) | commit `58c9445`, `d7574d0`; PR #42 (đã merge vào `dev`), nhánh `feat/bup-rl-eval-anchors` |
| 21/11 | Thêm `training/eval_anchors.py`: đo checkpoint với Random, Think1, Think2, Think3, 100 ván mỗi cặp, đổi bên, 2 nước khai cuộc ngẫu nhiên, khoảng tin cậy Wilson 95%; chạy cho bản SL và 6 mốc RL, ghi `logs/training/rl_curve.csv`. Kết quả (một seed): với Think2 Win Score từ 0,275 (SL) lên 0,48 đến 0,62 (RL); với Think3 và Think1 chưa thấy khác biệt | commit `dca3eb9`, `def96c6`; PR #42 |
| 21/11 | Chạy `python solver/optimal_accuracy.py` với `weights/sl.pt` và `weights/rl.pt` trên 500 thế chuẩn: NN (SL) 50,0% [45,6–54,4], RL 49,4% [45,0–53,8], 0 sai nước, tức RL chưa cải thiện độ chính xác nước tối ưu so với SL; cập nhật `logs/optimal_accuracy.csv` | commit `62938d2`; PR #42 |
| 21/11 | Thêm `training/plot_rl_curves.py` (vẽ đường cong học RL từ log CSV) và `notebooks/02_learning_curves.ipynb`, tạo `results/figures/fig_learning_curve.png` (kết quả tự chơi mỗi vòng, độ dài ván, Win Score theo vòng) | commit `8ca1605`, `ff52bc0`; PR #43 (đã merge vào `dev`), nhánh `docs/bup-learning-curves` |
| 22/11 | Gỡ phần báo cáo cá nhân `docs/report/bup.md` theo thống nhất với nhóm trưởng (báo cáo chung viết lại theo nhóm) | commit `a2fab94`; PR #44 (đã merge vào `dev`), nhánh `docs/bup-go-bao-cao` |
| 25/11 | Sửa lỗi dấu trong AZ-MCTS: lan truyền ngược đổi dấu theo độ sâu thay vì so `env.turn` (sau nước thắng `env.turn` không đổi nên cách cũ sai dấu từ độ sâu 2 trở lên), sửa ở `agents/az_mcts_agent.py` và `training/selfplay.py`; thêm bộ nhớ đệm mô hình `_get_model` (trước đây đọc lại checkpoint ở mỗi nước). Thêm `tests/test_az_mcts.py` (3 test: ăn nước thắng, chặn dọc, chặn ngang), `pytest` 3 passed | commit `33a8982`, `5d6589c`; PR #47 (đã merge vào `dev`), nhánh `fix/bup-az-mcts` |
| 25/11 | Cấp 8: thêm khám phá vào tự chơi (10 nước đầu lấy mẫu theo phân phối lượt thăm) và tham số dòng lệnh cho `training/selfplay.py`; thêm tác tử `AZ8-N` vào `eval/tournament.py` | commit `5d6589c`, `62671eb`; PR #47 |
| 25/11 | Chạy tự chơi cấp 8 thật: 10 vòng x 40 ván x 50 mô phỏng, seed 0, khởi tạo từ `weights/sl.pt`, lưu `weights/az_iter_1..10.pt` và `logs/training/selfplay_log.txt` | commit `cd9266f`; PR #48 (đã merge vào `dev`), nhánh `results/bup-all10` |
| 25/11 | Chạy giải đấu vòng tròn 10 tác tử (seed 42, 45 cặp x 100 ván = 4500 ván, 0 lỗi, mất khoảng 22 phút): AlphaBeta-7 8,44, MCTS-200 6,575, AZ8-10 6,36, AZ-MCTS 6,08, Think3 4,825, RL 4,29, NN (SL) 3,445, Think2 3,115, Think1 1,36, Random 0,51. Ghi `logs/tournament_all10_run2.jsonl`, `_console.txt`, `_matrix.csv` | commit `192bc1f`; PR #49 (đã merge vào `dev`), nhánh `results/bup-all10` |
| 26/11 | Cập nhật 4 README (README.md, docs/README_duan.md, README_eval.md, README_learning.md) cho cấp 6 đến 8, giải đấu 10 tác tử, solver | PR #51, nhánh `docs/bup-readme-cap78` |
| 26/11 | Dọn logs: xóa kết quả AZ-MCTS cũ chạy trước khi sửa lỗi dấu (`tournament_az*`, `tournament_final*`, `tournament_games_matrix.csv`), đổi `eval/analyze_games.py` sang `tournament_all10_run2.jsonl`, sinh lại `summary_all.md` và `summary_all_long.csv` | PR #52, nhánh `chore/logs-cleanup` |
| 26/11 | Solver: thêm AZ-MCTS (cấp 7) và AZ8-10 (cấp 8) vào `solver/optimal_accuracy.py`, đo trên 500 thế chuẩn (200 mô phỏng): AZ-MCTS 69,2% [65,0–73,1], AZ8-10 68,6% [64,4–72,5], 0 sai nước; ghi `logs/optimal_accuracy_az.csv`, sinh lại `summary_all.*`; thêm file chạy thử `*_test.*` vào `.gitignore` | PR #54, nhánh `feat/bup-solver-az` |
| 26/11 | Test: phát hiện 3 test lỗi dấu AZ-MCTS (ăn nước thắng, chặn dọc, chặn ngang) bị ghi đè khi `tests/test_az_mcts.py` được thay bằng bộ test mới; khôi phục thành file riêng `tests/test_az_mcts_search.py`, `pytest` 6 passed cùng `test_az_mcts.py` | PR #60 (đã merge vào `dev`), nhánh `test/bup-az-sign-tests` |

## Tỷ lệ đóng góp (điền ở ngày cuối)
| Thành viên | MSSV | Công việc chính | Tỷ lệ (%) |
|---|---|---|---|
| Nguyễn Kiều Linh | 3124720039 | | |
| Nguyễn Huỳnh Mai | 3124720043 | | |
| Nguyễn Lương Hồng Búp | 3124720009 | | |
| **Tổng** | | | **100** |
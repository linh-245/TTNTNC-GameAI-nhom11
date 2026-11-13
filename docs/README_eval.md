# Hướng dẫn chạy đánh giá (giải đấu, RQ1, ablation)

Tài liệu này mô tả cách chạy lại phần đánh giá do `eval/tournament.py`, `eval/rq1.py`, `eval/ablation.py` và `eval/01_search_results.ipynb` thực hiện. **Chạy mọi lệnh từ thư mục gốc của repo**: các đường dẫn mặc định (`logs/...`) tính theo thư mục hiện tại.

## 1. Yêu cầu

- Python 3.10 trở lên, `numpy` (cho `eval/` và `agents/`). Các số liệu trong báo cáo được chạy bằng Python 3.13.13.
- Notebook cần thêm `pandas`, `matplotlib`, `jupyter`.

```bash
pip install numpy pandas matplotlib jupyter
```

Khi đo độ trễ hoặc chạy theo ngân sách thời gian (`--budget-ms`), đóng các chương trình nặng và không huấn luyện mô hình song song.

### Máy chạy tham chiếu

Số liệu trong báo cáo (`docs/mai.md`) chạy trên: Intel Core i3-1115G4 @ 3,00 GHz (2 lõi, 4 luồng), RAM 3,7 GB, Windows 10 Home Single Language (10.0.19045), Python 3.13.13. Kết quả theo ngân sách thời gian và độ trễ chỉ so sánh được trên cùng một máy; với máy chỉ có 2 lõi, nên dùng `--workers 1` khi cần đo thời gian.

## 2. Giải đấu round-robin: `eval/tournament.py`

Mỗi cặp tác tử đấu `--games` ván; mọi ván của mọi cặp được dồn vào một pool tiến trình chung.

```bash
# Chế độ ngân sách thời gian cho Alpha-Beta và MCTS (ví dụ)
python eval/tournament.py --games 200 --seed 42 --budget-ms 50 --workers 1 \
       --out logs/tournament_games.jsonl

# Cấu hình mặc định (không budget): Alpha-Beta độ sâu 5, MCTS 200 mô phỏng
python eval/tournament.py --games 100 --seed 42 --workers 1 \
       --out logs/tournament_default_games.jsonl

# Chạy thử nhanh
python eval/tournament.py --games 10 --agents Think2 Random --budget-ms 50 --workers 1
```

Các lệnh trên là ví dụ cấu hình. Cấu hình thực tế của từng ván nằm trong trường `spec1`/`spec2` của log (lưu ý: log chỉ ghi `spec` của hai tác tử ở **ván đang đọc**, nên ván đầu tiên của log giải đấu là `Random` vs `Think1` và không cho biết cấu hình Alpha-Beta/MCTS).

### Cấu hình đã dùng cho báo cáo hiện tại

| Mục | Giải đấu | RQ1 | Ablation |
|---|---|---|---|
| Số ván | 100 mỗi cặp (15 cặp, 1500 ván) | 100 mỗi mức ngân sách | 100 mỗi cấu hình (`reduced = false`) |
| Ngân sách | không đặt `--budget-ms` (suy ra: MCTS luôn 200 mô phỏng/nước) | 200 ms và 1000 ms | 50 ms và 200 ms (trần độ sâu 12) cho đo độ sâu |
| Tác tử | Alpha-Beta độ sâu cố định (suy ra mặc định 5), MCTS 200 mô phỏng | MCTS `budget_ms`; Alpha-Beta lặp sâu dần tối đa độ sâu 6 trong `budget_ms` | A1: Alpha-Beta độ sâu 4; A2: MCTS 50 đến 1000 mô phỏng |
| Khai cuộc | 2 nước ngẫu nhiên | không có (bàn cờ trống, chỉ đổi bên đi trước) | 2 |
| Seed | 42  | 42 (seed ván = 42 + `game_id`) | 42 |
| Workers | _chưa ghi lại_ (mặc định `cpu_count // 2` = 2 trên máy này) | 1 (tuần tự, một tiến trình) | 1 |
| Đối thủ cố định | không có | không có | `Think2` |

### Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--games` | 100 | Số ván mỗi cặp (nên chẵn và từ 100 trở lên) |
| `--seed` | 42 | Seed gốc; seed ván và seed khai cuộc suy ra từ đây |
| `--budget-ms` | không đặt | Ngân sách thời gian mỗi nước cho `AlphaBeta` và `MCTS` |
| `--agents` | `Random Think1 Think2 Think3 AlphaBeta MCTS` | Danh sách tác tử |
| `--openers` | 2 | Số nước đầu ngẫu nhiên theo seed (0 đến 4) |
| `--workers` | nửa số lõi CPU | Số tiến trình song song |
| `--depth` | 5 | Độ sâu Alpha-Beta khi **không** có `--budget-ms` |
| `--sims` | 200 | Số mô phỏng MCTS khi **không** có `--budget-ms` |
| `--out` | `logs/tournament_games.jsonl` | Log JSONL từng ván; ma trận CSV ghi cạnh đó thành `tournament_games_matrix.csv` |

### Tên tác tử cho `--agents`

| Tên | Ý nghĩa |
|---|---|
| `Random` | Chọn ngẫu nhiên |
| `Think1`, `Think2`, `Think3` | Tác tử luật: nhìn 1, 2, 3 bước (`agents/heuristic_agent.py`) |
| `AlphaBeta` hoặc `AlphaBeta-<d>` | Alpha-Beta; `<d>` ghi đè độ sâu (chỉ có tác dụng khi không có budget) |
| `MCTS` hoặc `MCTS-<n>` | MCTS; `<n>` ghi đè số mô phỏng (chỉ có tác dụng khi không có budget) |

### Tác dụng của `--budget-ms`

- `AlphaBeta`: iterative deepening từ độ sâu 1 đến độ sâu tối đa (chính là tham số `depth` của `alphabeta_agent`), trả nước của độ sâu hoàn thành gần nhất. Giải đấu dùng trần 12 (tham số `max_depth` của `spec_from_name`, mặc định 12, không có tùy chọn dòng lệnh), ablation dùng `--max-depth` (mặc định 12), RQ1 dùng trần 6. Nước tốt nhất của độ sâu trước luôn được thử đầu tiên ở nút gốc, kể cả khi tắt sắp xếp nước.
- `MCTS`: lặp mô phỏng đến hết giờ, bỏ qua số mô phỏng cố định.
- `Random` và `Think1-3` không bị ảnh hưởng.
- Khi có budget, `--depth` và `--sims` không còn dùng trong `eval/tournament.py`.
- Riêng RQ1 truyền cả `depth = 6` lẫn `budget_ms` cho Alpha-Beta, nên độ sâu 6 là **trần** (xác nhận trong `eval/rq1.py`) và Alpha-Beta không dùng hết ngân sách; xem mục 3.

### Công bằng và tái lập

- Ván chẵn: tác tử 1 đi trước; ván lẻ: tác tử 2 đi trước. Ván `2j` và `2j+1` của một cặp dùng chung khai cuộc, nên mỗi tác tử chơi cả hai phía của cùng một khai cuộc.
- Seed của ván, seed khai cuộc và seed của MCTS ở từng nước đều suy ra từ `--seed`, nên kết quả tái lập được, **trừ** khi dùng `--budget-ms` (số mô phỏng và độ sâu đạt được phụ thuộc tốc độ máy).
- Nước sai luật hoặc tác tử báo lỗi bị xử thua và ghi vào `forfeit` và `error`.

### Về `--workers`

Chạy song song làm độ trễ đo được chịu ảnh hưởng của tải CPU (nhắc trong đầu file `tournament.py`). Nếu **không** đặt `--workers`, mặc định là `cpu_count // 2` (máy tham chiếu: 4 luồng nên 2 workers), vì vậy muốn đo độ trễ sạch phải đặt `--workers 1` rõ ràng. Khi cần độ trễ, nút/giây hoặc kết quả theo budget đáng tin, dùng `--workers 1` và ghi số workers vào báo cáo. Gợi ý kiểm tra: nếu thông lượng MCTS trong giải đấu thấp hơn rõ so với ablation chạy 1 worker (báo cáo hiện tại: ~6.800 so với ~13.000 mô phỏng/giây), hãy xem lại số workers đã dùng.

## 3. RQ1: Alpha-Beta vs MCTS theo ngân sách: `eval/rq1.py`

Mỗi mức ngân sách (200 ms và 1000 ms) chạy 100 ván Alpha-Beta vs MCTS, ghi vào `rq1_200ms_games.jsonl` và `rq1_1000ms_games.jsonl` (cả hai hiện có đủ 100 ván).

```bash
python eval/rq1.py
```

**Cảnh báo:** script **không có tham số dòng lệnh** (không dùng argparse). `python eval/rq1.py --help` không in trợ giúp mà chạy toàn bộ thực nghiệm và **ghi đè** các file `rq1_*ms_games.jsonl` đang có. Muốn đổi ngân sách, số ván hoặc seed, sửa lời gọi `run_rq1_experiment(time_budgets_ms=(200, 1000), num_games=100, seed=42)` ở cuối file. Chạy đủ hai mức khá lâu vì MCTS dùng hết ngân sách ở mỗi nước.

**Thư mục log:** `eval/logs/` khi chạy từ gốc repo, hoặc `logs/` khi chạy từ trong `eval/`. Khác với `logs/` mặc định của `tournament.py` và `ablation.py`; nếu các log nằm rải rác, chuyển về cùng một chỗ hoặc sửa ô cấu hình đầu notebook.

Thiết kế (theo code):
- Alpha-Beta: `depth = 6` kèm `budget_ms` nghĩa là iterative deepening từ độ sâu 1 đến **tối đa 6** (trần), bật cắt tỉa, sắp xếp nước và hàm đánh giá, không dùng bảng chuyển vị. Do có trần, Alpha-Beta thường dừng sớm và không dùng hết ngân sách (ở 1000 ms: độ sâu TB 5,89, p95 ~260 ms).
- MCTS: chạy theo `budget_ms`; seed mỗi nước là `seed + game_id + ply`, nhưng vì dừng theo thời gian nên không tái lập hoàn toàn.
- Bàn cờ bắt đầu trống, không có khai cuộc ngẫu nhiên; ván chẵn (`game_id` chẵn) Alpha-Beta đi trước, ván lẻ MCTS đi trước (mỗi bên 50 ván).
- Seed gốc 42 (`game_id` chạy từ 1 đến 100); chạy tuần tự trong một tiến trình, không có `--workers`.

Định dạng log RQ1 khác log giải đấu, xem mục 5. Điểm cần nhớ: `pair[0]` là **người đi trước** và `score1` tính theo góc nhìn `pair[0]`; điểm theo góc nhìn Alpha-Beta là `score1` nếu `pair[0] == "alphabeta"`, ngược lại là `1 - score1`.

Notebook lấy mức ngân sách từ `spec.kwargs.budget_ms` trong từng ván; nếu log không có thì lấy từ tên file (`..._200ms_...`). Nếu `summary_rq1.csv` chỉ có một hàng (hiện chỉ có hàng 1000 ms), kiểm tra cả hai log đã nằm trong thư mục mà notebook tìm, rồi chạy lại notebook (đừng chạy lại `rq1.py` chỉ vì lý do này).

## 4. Ablation: `eval/ablation.py`

```bash
# Đầy đủ (A1 + A2, 100 ván/cấu hình)
python eval/ablation.py --part all --games 100 --workers 1 --out logs/ablation.jsonl

# Chạy thử cấu hình nhỏ
python eval/ablation.py --part all --games 4 --positions 5 --a2-sims 50,100 \
       --a2-c 1.0,1.414 --out logs/ablation_smoke.jsonl
```

**A1:** Alpha-Beta với 4 tổ hợp (cắt tỉa bật/tắt) × (sắp xếp nước bật/tắt) ở cùng độ sâu `--depth`.
- Số nút và ms/nước trên `--positions` thế cờ cố định theo seed (6 đến 26 nước đã đi, bỏ thế mà người sắp đi có nước thắng ngay).
- Độ sâu đạt được trong các ngân sách `--budgets` (iterative deepening, trần `--max-depth`).
- Tỷ lệ thắng trước đối thủ cố định `--opponent`.

**A2:** MCTS với `n_simulations` ∈ `--a2-sims` và `c` ∈ `--a2-c`, tỷ lệ thắng trước cùng đối thủ. Với mặc định có **5 mức mô phỏng × 4 giá trị c = 20 cấu hình** (kể cả mức 1000 mô phỏng); `summary_a2.csv` có đủ 20 hàng.

| Tham số | Mặc định |
|---|---|
| `--part` | `all` (`A1`, `A2` hoặc `all`) |
| `--games` | 100 (ít hơn sẽ ghi `reduced = true`) |
| `--seed` | 42 |
| `--opponent` | `Think2` |
| `--depth` | 4 (độ sâu cố định cho A1) |
| `--positions` | 20 |
| `--budgets` | `50,200` |
| `--max-depth` | 12 |
| `--a2-sims` | `50,100,200,500,1000` |
| `--a2-c` | `0.5,1.0,1.414,2.0` |
| `--openers` | 2 |
| `--workers` | 1 |
| `--out` | `logs/ablation.jsonl` |

Ghi chú:
- Các phép đo số nút, thời gian và độ sâu theo budget (A1) luôn chạy tuần tự trong một tiến trình; `--workers` chỉ áp dụng cho các trận đấu.
- Nếu `--games` nhỏ hơn 100, dòng `meta` và mọi bản ghi match có `reduced = true` kèm ghi chú số ván thực tế. Không đưa kết quả giảm số ván vào báo cáo như kết quả đầy đủ.
- Ngoài file tổng hợp, script ghi log từng ván vào `<tên>_games.jsonl` (ví dụ `logs/ablation_games.jsonl`).
- Dải seed của A1 (`pair_index` 0 đến 3) tách khỏi A2 (từ 100).
- Lần chạy dùng cho báo cáo (`meta`, 2026-11-13 00:44:45): `part = all`, 100 ván, `reduced = false`, đối thủ `Think2`, 20 thế cờ, `depth` 4, `budgets` 50 và 200 ms, 2 khai cuộc, 1 worker, seed 42.
- Đối thủ `Think2` khá yếu so với Alpha-Beta độ sâu 4 và MCTS từ 200 mô phỏng (cùng đạt 100/100 ở nhiều cấu hình), nên Win Score bị chạm trần. Nếu cần phân biệt các cấu hình mạnh, chạy lại với `--opponent Think3`.

## 5. Định dạng log

### Log từng ván (`*_games.jsonl`, mỗi dòng một ván)

| Trường | Ý nghĩa |
|---|---|
| `game_id` | `<agent1>_vs_<agent2>#<số thứ tự ván>` |
| `pair` | `[agent1, agent2]`; mọi kết quả tính theo góc nhìn **agent 1** |
| `first`, `a_first` | Tác tử đi trước; `a_first` đúng nếu agent 1 đi trước |
| `result`, `score1` | `win`/`draw`/`loss` và điểm 1 / 0,5 / 0 |
| `winner`, `forfeit`, `error` | Người thắng; tác tử bị xử thua và nguyên nhân |
| `seed`, `opener_seed`, `n_random_openers`, `openers` | Thông tin tái lập khai cuộc (`n_random_openers` có thể rỗng ở log RQ1) |
| `plies` | Tổng số nước, gồm khai cuộc |
| `moves` | Mỗi nước của tác tử: `agent`, `col`, `ms`, và `nodes`/`depth` (Alpha-Beta) hoặc `sims` (MCTS) |
| `spec1`, `spec2` | Cấu hình tác tử (có `budget_ms` nếu dùng). Log giải đấu và ablation dùng khóa `kind` (`{"kind": "alphabeta", "kwargs": {...}}`); log RQ1 dùng khóa `name` |
| `tag` | Nhãn thí nghiệm (A1/A2 ghi `exp` và `config`) |

`ms` trong `moves` của log giải đấu đo quanh lệnh gọi tác tử, nên gồm cả thời gian sao chép bàn cờ.

Nước thắng ngay của Alpha-Beta không chạy tìm kiếm: `depth` = 0 và `nodes` rất nhỏ. Khi tính trung bình độ sâu hoặc nút từ log ván, nên lọc các nước này.

### Log RQ1 (`rq1_<ms>ms_games.jsonl`)

Trường ít hơn log giải đấu: `game_id`, `seed`, `pair`, `spec1`, `spec2`, `score1`, `moves`. Khác biệt:
- `pair = [người đi trước, người đi sau]`, đổi theo ván; `score1` là điểm của `pair[0]`, `spec1` là cấu hình của `pair[0]`.
- `moves` gồm `ply`, `agent`, `move` (không phải `col`), `ms`, và `depth`, `nodes` (Alpha-Beta) hoặc `sims` (MCTS). `ms` là thời gian do tác tử tự báo (`time_ms` / `latency_ms`).
- Không có `first`, `a_first`, `result`, `winner`, `forfeit`, `error`, `openers`, `n_random_openers`, `plies`, `tag`.

### Log tổng hợp ablation (`ablation.jsonl`, trường `kind`)

| `kind` | Nội dung |
|---|---|
| `meta` | Thời gian, tham số, `reduced`, số workers, phiên bản Python, máy chạy |
| `A1_cost` | Thống kê (`mean`, `median`, `min`, `max`, `p95`) của số nút và ms/nước, kèm `per_position` |
| `A1_budget` | Độ sâu đạt được trong từng budget và `timed_out_rate` |
| `A1_match` | Thắng/hòa/thua, `score`, `wilson_ci`, độ trễ của cấu hình |
| `A2_match` | Như `A1_match`, cho từng cặp (`n_simulations`, `c`) |

## 6. Các chỉ số

- **Win Score** = (thắng + 0,5 × hòa) / số ván, theo góc nhìn tác tử đang xét.
- **Khoảng tin cậy Wilson 95%** tính trên (thắng + 0,5 × hòa) và số ván. Nếu khoảng chứa 0,5 thì chưa đủ bằng chứng để nói tác tử mạnh hơn hay yếu hơn. Khi so sánh hai ô, hai khoảng chồng nhau thường nghĩa là chênh lệch chưa có ý nghĩa thống kê.
- **Độ trễ trung bình và p95** tính theo từng nước: gom mọi nước đi của tác tử trong các ván, không tính theo ván.
- **Ma trận CSV** (`<tên>_matrix.csv`): ô (hàng, cột) là Win Score của tác tử ở hàng khi gặp tác tử ở cột; cột `total` là tổng các ô của hàng, dùng để xếp hạng. Với 6 tác tử, tổng cột `total` phải bằng 15 (một điểm cho mỗi cặp), có thể dùng để kiểm tra nhanh.
- **Nút/giây, mô phỏng/giây** (notebook): tổng nút (hoặc mô phỏng) chia tổng thời gian của các nước đó. Lưu ý: MCTS đếm mọi vòng lặp là một "mô phỏng", kể cả vòng chạm nút đã kết thúc (không rollout), nên mô phỏng/giây cao hơn nhiều ở cuối ván; khi so sánh giữa các thí nghiệm, dùng trung vị hoặc tách theo `ply` thay vì trung bình số học.

## 7. Tổng hợp bằng notebook

1. Mở `eval/01_search_results.ipynb`.
2. Nếu tên log khác, sửa ở ô cấu hình đầu notebook: `TOURNAMENTS`, `RQ1_LOGS`, `ABLATION_LOG`.
3. Chọn **Run All**.

Notebook xuất hình `results/figures/fig4_*.png` và bảng `logs/summary_*.csv`. Mục nào thiếu log sẽ bị bỏ qua kèm thông báo.

| Hình | Nội dung |
|---|---|
| `fig4_winscore_matrix_tournament.png` | Ma trận Win Score + Wilson CI |
| `fig4_wdl_tournament.png` | Thắng/hòa/thua theo cặp |
| `fig4_latency.png` | Độ trễ trung bình và p95 |
| `fig4_first_second_tournament.png` | Đi trước và đi sau |
| `fig4_rq1_budget.png` | RQ1 |
| `fig4_a1_cost.png`, `fig4_a1_depth_budget.png`, `fig4_a1_match.png` | A1 |
| `fig4_a2_mcts.png` | A2 |

| Bảng | Nội dung |
|---|---|
| `summary_pairs_tournament.csv` | Thắng/hòa/thua, Win Score, Wilson CI theo cặp |
| `summary_latency_tournament.csv` | Độ trễ trung bình, p95, lớn nhất theo tác tử |
| `summary_throughput_tournament.csv` | Nút/giây, mô phỏng/giây |
| `summary_first_second_tournament.csv` | Điểm khi đi trước và đi sau |
| `summary_rq1.csv` | RQ1 theo mức ngân sách |
| `summary_a1_cost.csv`, `summary_a1_budget.csv`, `summary_a1_match.csv`, `summary_a2.csv` | Ablation |

Hậu tố `_tournament` trong tên file là nhãn của giải đấu trong `TOURNAMENTS`; nếu thêm giải đấu khác, đặt nhãn mới và notebook sẽ tạo bộ file riêng.

## 8. Quản lý dung lượng

Log thô (`*_games.jsonl`) lớn nên để trên Drive(đường dẫn:https://drive.google.com/file/d/1oY4uesRlc2iGbDTfjyQQui3D8_8xZgZO/view?usp=sharing); chỉ đưa các `summary_*.csv`, `<tên>_matrix.csv`, `ablation.jsonl` và hình trong `results/figures/` lên GitHub (tổng dưới 50 MB).

## 9. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| `ModuleNotFoundError: game` hoặc `agents` | Chạy lệnh từ thư mục gốc repo |
| Log không nằm ở `logs/` | Chạy từ thư mục khác sẽ tạo `logs/` ở đó; `rq1.py` còn ghi vào `eval/logs/` khi chạy từ gốc repo. Chuyển file về cùng một thư mục (ví dụ `logs/` ở gốc repo) |
| `python eval/rq1.py --help` bắt đầu chạy thực nghiệm | Script không có tham số dòng lệnh; dừng bằng Ctrl+C. Log cũ có thể đã bị ghi đè, nên sao lưu `rq1_*ms_games.jsonl` trước khi chạy |
| Notebook báo "không thấy ... .jsonl" | Sửa đường dẫn ở ô cấu hình đầu notebook |
| `summary_rq1.csv` thiếu một mức ngân sách dù log có đủ | Đặt cả hai log RQ1 vào `logs/` rồi chạy lại notebook (**Run All**) |
| Tiêu đề CSV hiện chữ lỗi (ví dụ `cáº¥u hÃ¬nh`) | File CSV là UTF-8 nhưng bị đọc bằng bảng mã Windows (cp1252). Đọc bằng `pd.read_csv(..., encoding="utf-8")`, `Get-Content -Encoding UTF8` trong PowerShell, hoặc mở trong Excel bằng *Data > From Text/CSV* với UTF-8 |
| Độ trễ dao động mạnh | Dùng `--workers 1`, đóng chương trình nền |
| Hai cấu hình cùng số nút nhưng thời gian khác nhiều (ví dụ A1 khi tắt cắt tỉa) | Là nhiễu đo hoặc tải máy, không do thuật toán; đo lặp lại và đóng chương trình nền |
| Số mô phỏng MCTS khi dùng budget lệch xa so với ablation hoặc giải đấu | Không phải lỗi ghi log: `sims` là số vòng lặp, gồm cả vòng chạm nút kết thúc; trung bình bị nước cuối ván kéo lên. Kiểm tra trung vị hoặc số mô phỏng theo `ply`; cũng xem số workers (mặc định `cpu_count // 2`) |
| Có ván `forfeit` | Xem trường `error` trong log |
| Chạy quá lâu | Thử `--games` nhỏ trước; `--budget-ms` lớn và `--workers 1` làm giải đấu lâu nhất |

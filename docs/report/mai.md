# Báo cáo phần đánh giá: Alpha-Beta, MCTS, giải đấu và ablation (Mai)

> **Về số liệu.** Phần mô tả tác tử, giao thức và chỉ số viết theo đúng mã nguồn (`agents/alphabeta_agent.py`, `agents/mcts_agent.py`, `eval/tournament.py`, `eval/ablation.py`). Số liệu trong các bảng lấy từ `logs/summary_*.csv`, `tournament_games_matrix.csv`, dòng `meta` của `logs/ablation.jsonl` và cấu hình ghi ở ván đầu tiên của từng log. Các phần trăm giảm nút, tỷ lệ gấp, điểm trung bình là phép tính trực tiếp từ các số đó. 

## 1. Phạm vi

| Thực nghiệm | Script | Log | Bảng tóm tắt | Hình |
|---|---|---|---|---|
| Giải đấu round-robin | `eval/tournament.py` | `logs/tournament_games.jsonl` | `summary_pairs_tournament.csv`, `summary_latency_tournament.csv`, `summary_throughput_tournament.csv`, `summary_first_second_tournament.csv`, `tournament_games_matrix.csv` | `fig4_winscore_matrix_tournament.png`, `fig4_wdl_tournament.png`, `fig4_latency.png`, `fig4_first_second_tournament.png` |
| RQ1: Alpha-Beta vs MCTS theo ngân sách | `eval/rq1.py` | `logs/rq1_200ms_games.jsonl`, `logs/rq1_1000ms_games.jsonl` | `summary_rq1.csv` | `fig4_rq1_budget.png` |
| Ablation A1 (Alpha-Beta) | `eval/ablation.py` | `logs/ablation.jsonl`, `logs/ablation_games.jsonl` | `summary_a1_cost.csv`, `summary_a1_budget.csv`, `summary_a1_match.csv` | `fig4_a1_cost.png`, `fig4_a1_depth_budget.png`, `fig4_a1_match.png` |
| Ablation A2 (MCTS) | `eval/ablation.py` | như trên | `summary_a2.csv` | `fig4_a2_mcts.png` |

Hình và bảng tóm tắt đều do `eval/01_search_results.ipynb` dựng lại từ log. Lưu ý: `eval/rq1.py` ghi log vào `eval/logs/` khi chạy từ gốc repo (các script còn lại mặc định ghi `logs/`). Cách chạy chi tiết: [`docs/README_eval.md`](docs/README_eval.md).

## 2. Các tác tử

### 2.1 Alpha-Beta (`agents/alphabeta_agent.py`)

- **Thuật toán:** negamax, có hoặc không có cắt tỉa alpha-beta; độ sâu mặc định 5.
- **Hàm đánh giá** `eval_payoff_conn` (từ góc nhìn người sắp đi): tổng trọng số vị trí (ưu tiên cột giữa và các hàng giữa) cộng điểm theo 69 cửa sổ 4 ô (24 ngang, 21 dọc, 24 chéo). Cửa sổ chỉ có một phe: 3 quân mình +50, 3 quân đối thủ −80, 2 quân mình +10, 2 quân đối thủ −15. Điểm được chia cho 1000.
- **Nút kết thúc:** thua tính `-1000 - depth` (thua sớm tệ hơn), hòa tính 0. Nếu người sắp đi có nước thắng ngay thì đi luôn.
- **Sắp xếp nước** từ cột giữa (cột 4) ra hai bên, tắt được bằng `use_ordering=False`.
- **Các công tắc cho ablation:** `use_pruning`, `use_ordering`, `use_eval`, `budget_ms`, `use_tt`. Bảng chuyển vị Zobrist có sẵn nhưng **tắt mặc định** và không dùng trong các thí nghiệm.
- **Giới hạn thời gian** (`budget_ms`): iterative deepening từ độ sâu 1 đến độ sâu tối đa, kiểm tra thời gian ở mỗi nút; nếu hết giờ giữa chừng thì bỏ độ sâu đang làm dở và trả nước của độ sâu hoàn thành gần nhất. Độ sâu tối đa chính là tham số `depth` của hàm (RQ1 dùng 6; ablation dùng trần `--max-depth` 12). Nước tốt nhất của độ sâu trước được thử đầu tiên **ở nút gốc** của độ sâu sau; việc này vẫn xảy ra khi `use_ordering=False`, vì `use_ordering` chỉ điều khiển thứ tự ở các nút trong. Nếu hết giờ trước khi xong độ sâu 1 thì trả nước dự phòng là nước đầu tiên trong danh sách (cột 4 nếu bật sắp xếp).
- **Thống kê mỗi nước** (`last_stats`): số nút (cộng dồn qua các độ sâu, gồm nút gốc), độ sâu hoàn thành sâu nhất, thời gian, cờ hết giờ. Khi người sắp đi có nước thắng ngay, hàm trả về ngay mà không tìm kiếm: `depth` = 0 và số nút rất nhỏ; trung bình độ sâu hoặc nút tính trên log ván nên lọc các nước này (A1 đã bỏ các thế như vậy).

### 2.2 MCTS/UCT (`agents/mcts_agent.py`)

- UCT = W/N + c·√(ln N_cha / N), mặc định c = √2; bốn pha chọn, mở rộng, mô phỏng ngẫu nhiên đều, lan truyền ngược (dấu kết quả đổi theo người vừa đi).
- Chọn nước có số lượt thăm lớn nhất tại gốc.
- Dùng `random.Random(seed)` riêng: cùng seed và cùng số mô phỏng (không có budget) cho cùng nước đi.
- `budget_ms`: lặp mô phỏng đến hết giờ, bỏ qua `n_simulations`; số mô phỏng đạt được phụ thuộc tốc độ máy.
- `last_stats`: số mô phỏng (`sims`) và độ trễ (`latency_ms`, tính từ sau khi tạo nút gốc). Một "mô phỏng" là một vòng lặp chọn, mở rộng, rollout, lan truyền ngược, **kể cả vòng chạm nút đã kết thúc** (khi đó không có rollout, chỉ tốn chọn và lan truyền). Vì vậy chi phí mỗi mô phỏng rẻ hơn nhiều ở cuối ván, khi cây đã gần được giải hết.
- Cây được dựng lại từ đầu ở mỗi nước (không tái sử dụng cây); không có kiểm tra thắng ngay hay chặn thua ngay riêng ở nút gốc, mọi thứ do UCT tự phát hiện. Nếu hết giờ trước khi chạy được mô phỏng nào thì chọn ngẫu nhiên một nước chưa thử; với `seed=None` cả quá trình không tái lập.
- Mặc định của hàm là 800 mô phỏng; giải đấu dùng `--sims` (mặc định 200) khi không có budget.

### 2.3 Tác tử cơ sở

`Random` và `Think1`, `Think2`, `Think3` (tác tử luật nhìn 1, 2, 3 bước, trong `agents/`). Chúng không bị giới hạn bởi `--budget-ms`.

## 3. Giao thức giải đấu (`eval/tournament.py`)

- **Round-robin** 6 tác tử mặc định (`Random`, `Think1`, `Think2`, `Think3`, `AlphaBeta`, `MCTS`) = 15 cặp; mọi ván của mọi cặp chạy trong một pool tiến trình.
- **Cấu hình thực nghiệm của giải đấu:**
  - Số ván mỗi cặp: **100** (`n` = 100 ở cả 15 cặp; log có 1500 ván = 15 × 100).
  - `--openers`: **2** (đọc từ `n_random_openers` trong log).
  - `--budget-ms`: **không đặt** (suy ra: MCTS luôn đúng 200,0 mô phỏng/nước trong `summary_throughput_tournament.csv`, tức số mô phỏng cố định). Khi đó, theo `eval/tournament.py`, Alpha-Beta chạy độ sâu cố định `--depth` (mặc định 5) và MCTS chạy `n_simulations` = `--sims` (mặc định 200). Độ sâu 5 là suy ra: nút TB/nước 1115,8 phù hợp mức này, nhưng `spec` của Alpha-Beta không đọc được vì ván đầu tiên trong log là `Random` vs `Think1`.
  - `--seed`: mặc định 42 .
  - `--workers`: _chưa ghi lại_. Nếu không đặt, mặc định là `cpu_count // 2`, tức 2 trên máy này (4 luồng). Thông lượng MCTS trong giải đấu thấp hơn khoảng 2 lần so với ablation chạy 1 worker (mục 5) phù hợp với việc chạy 2 tiến trình song song, nhưng đây vẫn là suy luận.
- **Công bằng đi trước/đi sau:** ván chẵn tác tử 1 đi trước, ván lẻ tác tử 2 đi trước. Ván `2j` và `2j+1` dùng chung khai cuộc nên mỗi tác tử chơi cả hai phía của cùng một khai cuộc.
- **Khai cuộc:** `--openers` nước đầu ngẫu nhiên theo seed (mặc định 2).
- **Tái lập:** seed ván, seed khai cuộc và seed của MCTS ở từng nước đều suy ra từ `--seed`; riêng kết quả theo budget phụ thuộc tốc độ máy. Cụ thể (theo `tournament.py`): seed ván = `seed × 1.000.003 + chỉ số cặp × 10.007 + chỉ số ván`; seed MCTS mỗi nước = `seed ván × 131 + ply`; seed khai cuộc dùng chung cho ván `2j` và `2j+1`.
- **Lỗi:** nước sai luật hoặc tác tử báo lỗi bị xử thua, ghi vào `forfeit` và `error`. Ở cả 15 cặp, cột `forfeits` của `summary_pairs_tournament.csv` đều bằng 0.
- **Máy chạy:** Intel Core i3-1115G4 @ 3,00 GHz (2 lõi, 4 luồng), RAM 3,7 GB, Windows 10 Home Single Language (10.0.19045), Python 3.13.13. Vì các tác tử chạy song song khi `--workers` > 1, độ trễ chịu ảnh hưởng của tải CPU; máy chỉ có 2 lõi và 3,7 GB RAM nên càng dễ bị nhiễu.

### Chỉ số

- Win Score = (thắng + 0,5 × hòa) / số ván, theo góc nhìn tác tử đang xét.
- Khoảng tin cậy Wilson 95% trên (thắng + 0,5 × hòa) và số ván.
- Độ trễ trung bình và p95 tính **theo từng nước** (gom mọi nước đi từ log), không tính theo ván. Giá trị `ms` trong log giải đấu đo quanh lệnh gọi tác tử nên gồm cả thời gian sao chép bàn cờ; riêng log RQ1 dùng thời gian do chính tác tử báo (`time_ms` của Alpha-Beta, `latency_ms` của MCTS), chỉ dùng phép đo bên ngoài khi thiếu.
- Thứ hạng = tổng Win Score của tác tử trên các cặp (cột `total` của ma trận CSV).

## 4. Kết quả

### 4.1 Giải đấu

![Ma trận Win Score](../results/figures/fig4_winscore_matrix_tournament.png)

Xếp hạng (từ `tournament_games_matrix.csv`, cột `total`; tối đa 5 điểm mỗi tác tử, tổng cả sáu tác tử = 15,000):

| Hạng | Tác tử | Tổng điểm |
|---|---|---|
| 1 | AlphaBeta | 4,515 |
| 2 | MCTS | 3,840 |
| 3 | Think3 | 3,565 |
| 4 | Think2 | 1,840 |
| 5 | Think1 | 0,860 |
| 6 | Random | 0,380 |

Nhận xét:

- **Khoảng tin cậy:** 14 trong 15 cặp có khoảng Wilson 95% không chứa 0,5. Ngoại lệ duy nhất là `Think3` vs `MCTS` (Think3 được 0,415, CI [0,323; 0,513]) nên chưa đủ bằng chứng để nói MCTS mạnh hơn Think3 dù MCTS xếp trên Think3 theo tổng điểm.
- **Alpha-Beta vs MCTS:** Alpha-Beta thắng 73, hòa 3, thua 24; Win Score 0,745, CI [0,652; 0,820]. Trong cấu hình giải đấu này (độ sâu cố định, MCTS 200 mô phỏng), Alpha-Beta mạnh hơn MCTS có ý nghĩa thống kê.
- **So với `Think3`:** Alpha-Beta thắng Think3 với điểm 0,770 (Think3 chỉ được 0,230, CI [0,158; 0,322]; 12 ván hòa, nhiều nhất trong các cặp). MCTS hơn Think3 ở mức 0,585 nhưng chưa vượt ngưỡng tin cậy như trên. Khoảng cách xếp hạng MCTS–Think3 (0,275 điểm) chủ yếu đến từ cặp trực tiếp (0,170), việc Think3 mất điểm trước Think1 và Think2 (tổng 0,080) và cặp với Alpha-Beta (0,025).
- **Các tác tử luật:** thứ tự Random < Think1 < Think2 < Think3 rõ ràng ở mọi cặp liền kề (Think3 thắng Think2 96/100, 2 hòa; Think2 thắng Think1 92/100). Think1 thắng Random 73/100, tức còn thua Random 27 ván.
- **Hòa:** rất hiếm, 20/1500 ván (Think3–Alpha-Beta 12, Think3–MCTS 3, Alpha-Beta–MCTS 3, Think2–Think3 2).

![Thắng/hòa/thua](../results/figures/fig4_wdl_tournament.png)

### 4.2 Độ trễ và thông lượng

Từ `summary_latency_tournament.csv` và `summary_throughput_tournament.csv`:

| Tác tử | Số nước | Trung bình (ms) | p95 (ms) | Lớn nhất (ms) | Thông lượng |
|---|---|---|---|---|---|
| AlphaBeta | 4174 | 50,11 | 149,82 | 624,97 | 22.265 nút/giây (gộp); trung vị mỗi nước 28.236 |
| MCTS | 4561 | 29,25 | 64,37 | 347,52 | 6.838 mô phỏng/giây (gộp); trung vị mỗi nước 7.526 |

Các tác tử cơ sở rất nhẹ: Random 0,02 ms, Think1 0,14 ms, Think2 0,66 ms, Think3 0,81 ms (trung bình mỗi nước; p95 lần lượt 0,02; 0,32; 1,61; 2,31 ms).

Độ sâu Alpha-Beta đạt được trong giải đấu: `summary_throughput_tournament.csv` không có cột độ sâu. Vì giải không dùng budget nên độ sâu là cố định (suy ra mặc định 5), trung bình 1115,8 nút/nước. MCTS dùng đúng 200 mô phỏng/nước. Alpha-Beta chậm hơn MCTS gần 1,7 lần mỗi nước (50,11 so với 29,25 ms) và biến động mạnh hơn (p95 149,82 ms, lớn nhất 624,97 ms) vì số nút phụ thuộc thế cờ. Với budget, p95 có thể xấp xỉ hoặc vượt nhẹ ngân sách vì thời gian chỉ được kiểm tra ở mỗi nút (Alpha-Beta) hoặc sau mỗi mô phỏng (MCTS).

![Độ trễ](../results/figures/fig4_latency.png)

### 4.3 Đi trước và đi sau

Từ `summary_first_second_tournament.csv` (mỗi ô 250 ván):

| Tác tử | Đi trước | Đi sau |
|---|---|---|
| AlphaBeta | 0,894 [0,850; 0,926] | 0,912 [0,870; 0,941] |
| MCTS | 0,748 [0,691; 0,798] | 0,788 [0,733; 0,834] |
| Think3 | 0,702 [0,643; 0,755] | 0,724 [0,666; 0,776] |
| Think2 | 0,392 [0,334; 0,454] | 0,344 [0,288; 0,405] |
| Think1 | 0,172 [0,130; 0,224] | 0,172 [0,130; 0,224] |
| Random | 0,092 [0,062; 0,134] | 0,060 [0,037; 0,097] |

Điểm trung bình của bên đi trước trên mọi cặp: **0,500** (tổng 3,000 trên 6 tác tử, tức 750/1500 điểm), nên không có lợi thế đi trước ở mức tổng thể. Với từng tác tử, hai khoảng tin cậy đều chồng lên nhau, nên chênh lệch đi trước/đi sau không có ý nghĩa thống kê. Ở đây "đi trước" (`first` trong log) nghĩa là cầm quân đỏ, tức bên đi nước thứ 3 sau hai nước khai cuộc ngẫu nhiên; hai nước ngẫu nhiên đầu có thể làm loãng lợi thế đi trước vốn có của trò chơi.

![Đi trước và đi sau](../results/figures/fig4_first_second_tournament.png)

### 4.4 RQ1: Alpha-Beta vs MCTS theo ngân sách

Từ `summary_rq1.csv`, điểm theo góc nhìn Alpha-Beta. Cấu hình (xác nhận từ `eval/rq1.py` và log): MCTS chạy với `budget_ms` (200 hoặc 1000); Alpha-Beta chạy `depth = 6` kèm cùng `budget_ms`, tức iterative deepening từ độ sâu 1 đến **tối đa 6**, bật cắt tỉa, sắp xếp nước và hàm đánh giá, không dùng bảng chuyển vị. Mỗi mức 100 ván, seed 42, chạy tuần tự trong một tiến trình, bàn cờ bắt đầu trống (không có khai cuộc ngẫu nhiên). Ván chẵn Alpha-Beta đi trước, ván lẻ MCTS đi trước (mỗi bên 50 ván).

| Ngân sách | Ván | Thắng / hòa / thua | Win Score [CI 95%] | Độ sâu TB | Mô phỏng TB của MCTS |
|---|---|---|---|---|---|
| 200 ms | 100 | 48 / 2 / 50 | 0,490 [0,394; 0,587] | 5,72 | 12.119 |
| 1000 ms | 100 | 50 / 9 / 41 | 0,545 [0,448; 0,639] | 5,89 | 74.127 |

Ở 200 ms, p95 thời gian mỗi nước là 200,02 ms với Alpha-Beta và 200,23 ms với MCTS. 
Ở 1000 ms, p95 thời gian mỗi nước là 259,78 ms với Alpha-Beta và 1000,24 ms với MCTS.

![RQ1](../results/figures/fig4_rq1_budget.png)

Kết luận:

Ở cả hai mức ngân sách, khoảng tin cậy (CI 95%) của Alpha-Beta đều chứa 0,5 (200 ms: [0,394; 0,587] và 1000 ms: [0,448; 0,639]). Điều này cho thấy chưa đủ bằng chứng thống kê để khẳng định bên nào mạnh hơn. Tuy nhiên, có hai điểm quan trọng cần đọc kèm khi phân tích kết quả:

Alpha-Beta bị chặn ở độ sâu 6: Dù ngân sách tăng từ 200 ms lên 1000 ms, độ sâu trung bình của Alpha-Beta gần như không đổi (từ 5,72 lên 5,89) và p95 thời gian mỗi nước chỉ rơi vào khoảng ~200 ms - 260 ms, thấp hơn rất nhiều so với ngân sách 1000 ms. Điều này xác nhận Alpha-Beta (với cấu hình depth=6 trong rq1.py) gần như luôn duyệt xong độ sâu 6 rồi dừng lại, chứ không tận dụng hết thời gian. Ngược lại, MCTS liên tục dùng hết ngân sách cho phép (p95 xấp xỉ 200,23 ms ở mức 200 ms và 1000,24 ms ở mức 1000 ms). Do đó, kết quả này thực chất đang đo "Alpha-Beta tối đa độ sâu 6" so với "MCTS tối đa thời gian".

Tính bất đối xứng khi tăng ngân sách: Từ 200 ms lên 1000 ms, MCTS có thêm thời gian để chạy nhiều mô phỏng hơn (trung bình tăng từ ~12.119 lên ~74.127 mô phỏng/nước), trong khi Alpha-Beta đã chạm "trần" độ sâu 6. Chính vì vậy, việc điểm số của Alpha-Beta nhích nhẹ từ 0,490 lên 0,545 chủ yếu phản ánh nỗ lực của phía MCTS chứ không phải do Alpha-Beta tính toán sâu hơn.

Ghi chú kỹ thuật: Độ sâu trung bình của Alpha-Beta có thể bị kéo xuống nhẹ nếu các nước thắng ngay (depth = 0) được tính vào giá trị trung bình.

### 4.5 Ablation A1: cắt tỉa và sắp xếp nước

Cùng độ sâu cố định (mặc định 4), các thí nghiệm A1 đo trên một tập thế cờ cố định theo seed, độ sâu đạt được trong các ngân sách 50 và 200 ms, và tỷ lệ thắng trước đối thủ cố định (mặc định `Think2`). Theo dòng `meta` của `logs/ablation.jsonl` (chạy lúc 2026-11-13 00:44:45): đối thủ `Think2`, 100 ván mỗi cấu hình (`full_games = 100`, `reduced = false`), 20 thế cờ, độ sâu cố định 4, ngân sách 50 và 200 ms với trần độ sâu 12, 2 nước khai cuộc, seed 42, 1 worker.

| Cấu hình | Nút TB | ms/nước TB | Độ sâu @50 ms | Độ sâu @200 ms | Win Score trước đối thủ [CI] |
|---|---|---|---|---|---|
| prune on, order on | 344,3 | 11,376 | 5,50 | 6,75 | 1,000 [0,963; 1,000] |
| prune on, order off | 605,7 | 12,859 | 4,75 | 6,00 | 1,000 [0,963; 1,000] |
| prune off, order on | 2155,1 | 49,402 | 3,30 | 4,15 | 1,000 [0,963; 1,000] |
| prune off, order off | 2155,1 | 69,823 | 3,25 | 4,25 | 1,000 [0,963; 1,000] |

Số nút và ms/nước là trung bình trên 20 thế cờ (độ sâu 4). Ở cả tám ô độ sâu, tỷ lệ hết giờ là 1,0, nghĩa là mọi cấu hình đều dùng hết ngân sách và không bao giờ chạm trần độ sâu 12. p95 ms/nước trong 100 ván đấu: 18,9 / 22,1 / 68,3 / 70,0 ms theo thứ tự các hàng ở bảng trên.

![A1 nút và thời gian](../results/figures/fig4_a1_cost.png)
![A1 độ sâu theo ngân sách](../results/figures/fig4_a1_depth_budget.png)
![A1 tỷ lệ thắng](../results/figures/fig4_a1_match.png)

Đọc kết quả:

- **Số nút:** cắt tỉa kèm sắp xếp giảm 84,0% số nút so với không tối ưu (2155,1 → 344,3). Chỉ cắt tỉa giảm 71,9% (→ 605,7); sắp xếp nước giảm thêm 43,2% so với chỉ cắt tỉa. Khi tắt cắt tỉa, bật hay tắt sắp xếp đều cho đúng 2155,1 nút (trung vị 2436,5), khớp lý thuyết.
- **Thời gian:** cấu hình đầy đủ nhanh hơn không tối ưu 6,1 lần (11,4 so với 69,8 ms), sát với tỷ lệ nút (6,3 lần). Sắp xếp nước cắt nút nhiều hơn là cắt thời gian (nút −43%, thời gian chỉ −11,5%, 12,9 → 11,4 ms), có thể do chi phí sắp xếp.
- **Độ sâu theo ngân sách:** cắt tỉa tăng độ sâu khoảng 1,5 tầng ở 50 ms và 1,75 tầng ở 200 ms (so với không tối ưu); thêm sắp xếp nước cộng thêm 0,75 tầng ở cả hai mức. Tổng cộng cấu hình đầy đủ sâu hơn 2,25 tầng ở 50 ms và 2,5 tầng ở 200 ms.
- **Điểm bất thường cần kiểm tra:** khi tắt cắt tỉa, bật sắp xếp cho **cùng số nút** nhưng nhanh hơn rõ rệt (49,4 so với 69,8 ms; p95 70,3 so với 131,9 ms). Số nút bằng nhau nên chênh lệch này không do thuật toán; có khả năng do nhiễu đo (máy 2 lõi, 3,7 GB RAM, đo tuần tự từng cấu hình). Nên đo lặp lại trước khi kết luận về chi phí sắp xếp.

**Lưu ý về đo độ sâu theo ngân sách.** Khi có `budget_ms`, `alphabeta_agent` luôn thử nước tốt nhất của độ sâu trước đầu tiên ở nút gốc, kể cả khi `use_ordering=False`. Vì vậy chênh lệch độ sâu giữa "order on" và "order off" (+0,75 tầng) chỉ phản ánh sắp xếp ở các nút trong; cấu hình "order off" vẫn giữ lợi thế sắp xếp ở gốc.

**Lưu ý khi đọc cột Win Score.** Cả bốn cấu hình thắng 100/100 trước `Think2`, nên cột này **không phân biệt được** các cấu hình (hiệu ứng trần: `Think2` quá yếu so với Alpha-Beta độ sâu 4). Ở cùng độ sâu, bật hay tắt cắt tỉa về lý thuyết cho cùng giá trị gốc, và mã chọn nước đầu tiên có điểm lớn nhất, nên hai cấu hình này thường chọn cùng nước. Mỗi cấu hình dùng một `pair_index` riêng nên khai cuộc khác nhau; vì vậy nếu có chênh lệch Win Score giữa chúng thì nhiều khả năng là nhiễu lấy mẫu, không phải khác biệt của thuật toán. Tác động thực sự của cắt tỉa và sắp xếp nước thể hiện ở số nút, thời gian và độ sâu đạt được, như đã thấy ở trên. Sắp xếp nước có thể đổi nước chọn khi nhiều nước bằng điểm.

### 4.6 Ablation A2: MCTS

Win Score của MCTS trước đối thủ cố định `Think2` (từ `summary_a2.csv`, 100 ván mỗi ô):

| n_simulations \ c | 0,5 | 1,0 | 1,414 | 2,0 |
|---|---|---|---|---|
| 50 | 0,695 | 0,790 | 0,755 | 0,705 |
| 100 | 0,810 | 0,950 | 0,920 | 0,880 |
| 200 | 0,990 | 1,000 | 0,975 | 0,990 |
| 500 | 0,990 | 1,000 | 1,000 | 1,000 |
| 1000 | 1,000 | 1,000 | 1,000 | 1,000 |

Độ trễ trung bình mỗi nước (ms) với c = 1,414: 4,07 (n=50), 7,23 (100), 14,43 (200), 39,71 (500), 74,45 (1000); p95 lần lượt 7,24 / 11,98 / 25,61 / 73,41 / 124,13 ms.

![A2 MCTS](../results/figures/fig4_a2_mcts.png)

Nhận xét:

- **Theo số mô phỏng:** Win Score tăng đơn điệu và bão hòa. Với c = 1,0: 0,79 (n=50) → 0,95 (100) → 1,00 (từ 200 trở lên). Từ n = 200, mọi giá trị c đạt 0,975 đến 1,000; ở n = 1000 cả bốn giá trị c đều đạt 1,000, nên không còn phân biệt được.
- **Giá trị c:** c = 1,0 cho điểm cao nhất ở n = 50, 100 và 200 (và đồng hạng 1,000 ở n ≥ 500). Tuy nhiên chênh lệch giữa các c thường chưa có ý nghĩa thống kê: ví dụ ở n = 50, các khoảng tin cậy của c = 1,0 [0,700; 0,858], c = 1,414 [0,662; 0,829] và c = 0,5 [0,599; 0,777] chồng lên nhau. Ngoại lệ rõ nhất là n = 100: c = 0,5 (0,810 [0,722; 0,875]) thấp hơn c = 1,0 (0,950 [0,888; 0,979]) và hai khoảng không chồng nhau. Giá trị mặc định c = √2 vẫn nằm trong khoảng tin cậy của c = 1,0 ở mọi n.
- **Độ trễ:** tăng gần tuyến tính theo số mô phỏng, khoảng 0,07 đến 0,08 ms mỗi mô phỏng (tương đương ~13.000 mô phỏng/giây). Tăng từ 200 lên 500 mô phỏng làm độ trễ tăng gấp ~2,7 lần (14,5 → ~39 ms), từ 200 lên 1000 gấp ~5,2 lần, trong khi Win Score ở n = 200 đã gần tối đa.
- **Hạn chế khi đọc:** vì Win Score chạm trần 1,000 [0,963; 1,000] từ n = 200, A2 không cho biết MCTS còn cải thiện thêm bao nhiêu khi tăng mô phỏng; cần đối thủ mạnh hơn (ví dụ `Think3` hoặc Alpha-Beta).

## 5. Hạn chế
Độ trễ chịu ảnh hưởng của tải CPU: Nếu giải chạy với --workers > 1, độ trễ, nút/giây và độ sâu đạt được bị nhiễu do tranh chấp tài nguyên. Số workers đã dùng: ablation 1 worker (theo meta); RQ1 chạy tuần tự trong một tiến trình (theo rq1.py); giải đấu chưa ghi lại (mặc định 2 nếu không đặt --workers, xem mục 3). Máy chạy chỉ có 2 lõi vật lý và 3,7 GB RAM.
Thông lượng MCTS khác nhau giữa các thí nghiệm, và chỉ một phần là do định nghĩa: Khoảng 6.800 mô phỏng/giây trong giải đấu (200 mô phỏng cố định, 29,25 ms/nước), khoảng 13.000 trong A2 (1 worker) và khoảng 74.000 trong RQ1 (74.127 mô phỏng trong 1000 ms). Mức chênh ~2 lần giữa giải đấu và A2 phù hợp với chạy song song (mặc định 2 workers). Với RQ1, rq1.py chỉ chép nguyên last_stats["sims"], và mcts_agent.py đếm mọi vòng lặp là một mô phỏng, kể cả vòng chạm nút kết thúc không có rollout (mục 2.2). Ở cuối ván, khi cây gần được giải hết, số vòng lặp trong 1000 ms có thể rất lớn, và trung bình số học trên mọi nước bị các nước cuối ván kéo lên. Đây là một giả thuyết cần kiểm chứng bằng trung vị hoặc số mô phỏng theo ply từ rq1_1000ms_games.jsonl trước khi dùng con số 74.127 để so sánh với A2 hay giải đấu.
RQ1 đã có đủ hai mức ngân sách trong bảng tóm tắt: File summary_rq1.csv hiện đã ghi nhận đầy đủ cả 2 hàng 200 ms và 1000 ms (tương ứng với 100 ván mỗi mức). Ngoài ra, Alpha-Beta trong RQ1 bị chặn ở độ sâu 6 nên không dùng hết ngân sách (xem mục 4.4). RQ1 cũng không có khai cuộc ngẫu nhiên và Alpha-Beta là tất định, nên mọi ván có cùng một vị thế xuất phát (chỉ đổi bên đi trước); sự đa dạng giữa các ván chỉ đến từ MCTS ngẫu nhiên, nên khoảng tin cậy Wilson (coi các ván độc lập) có thể hơi lạc quan.
Kết quả theo budget phụ thuộc máy: Sức mạnh của Alpha-Beta (độ sâu đạt được) và MCTS (số mô phỏng) thay đổi theo tốc độ máy, nên kết quả chỉ so sánh được trên cùng một máy.
Độ chắc chắn thống kê: Với số ván mỗi cặp ở mức trăm, khoảng tin cậy Wilson vẫn khá rộng; chênh lệch nhỏ có thể chưa đủ bằng chứng (ví dụ Think3 vs MCTS, và phần lớn so sánh giữa các giá trị c ở A2). Hòa tính 0,5 nên Win Score không phân biệt hòa với "thắng một nửa".
Hiệu ứng trần trong ablation: Alpha-Beta (A1, cả bốn cấu hình) và MCTS (A2, từ 200 mô phỏng) đều thắng gần như tuyệt đối trước Think2, nên Win Score không phân biệt được các cấu hình mạnh.
Khai cuộc ngẫu nhiên: Chỉ vài nước đầu ngẫu nhiên nên số khai cuộc khác nhau còn hạn chế; mỗi cặp chơi cả hai phía của cùng một khai cuộc để giảm thiên lệch.
Đối thủ cố định trong ablation: Kết luận A1 và A2 là "trước đối thủ đã chọn" (Think2), không phải sức mạnh tuyệt đối.
Hàm đánh giá của Alpha-Beta: Có trọng số cố định, chưa được tinh chỉnh bằng thực nghiệm; bảng chuyển vị (transposition table) không được dùng.
Chạy giảm số ván: reduced = false và full_games = 100 trong meta của ablation, nên kết quả ablation là đầy đủ.
## 6. Tái lập

```bash
python eval/tournament.py --games <số ván> --seed 42 --budget-ms <ms> --workers <n> --out logs/tournament_games.jsonl
python eval/rq1.py                
python eval/ablation.py --part all --games 100 --workers 1 --out logs/ablation.jsonl
```
Cấu hình thực tế đã dùng: giải đấu 100 ván/cặp, 2 khai cuộc, không `--budget-ms` (tương đương `python eval/tournament.py --games 100 --seed 42 --out logs/tournament_games.jsonl`; `--workers` ); RQ1 100 ván mỗi mức 200 ms và 1000 ms (seed 42, tuần tự, log ghi vào `eval/logs/` nếu chạy từ gốc repo); ablation đúng lệnh thứ ba ở trên (đối thủ `Think2`, 1 worker). Sau đó chạy `eval/01_search_results.ipynb` để dựng lại hình và bảng. Log thô lớn để trên Drive (đường dẫn:https://drive.google.com/file/d/1oY4uesRlc2iGbDTfjyQQui3D8_8xZgZO/view?usp=sharing); chỉ bản tóm tắt dưới 50 MB lên GitHub.

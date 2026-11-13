# Phần báo cáo của Búp: học máy (cấp 5, mạng nơ-ron)

Phần này mô tả cách nhóm xây dựng agent cấp 5 (Neural Network) cho Connect Four 7×6: mã hóa thế cờ, mạng, dữ liệu, huấn luyện có giám sát (SL) và hiện trạng phần học tăng cường (RL). Mọi số liệu lấy từ `weights/sl.json` và `logs/training/sl_log.csv`. Cách chạy lại ở `docs/README_learning.md`.

## 1. Mục tiêu và thiết kế chung

Agent cấp 5 là một mạng CNN nhận thế cờ và cho ra hai thứ: policy (xác suất chọn từng cột trong 7 cột) và value (ước lượng người sắp đi sẽ thắng hay thua, trong khoảng từ -1 đến +1). Agent hiện chỉ dùng policy: chọn cột có xác suất cao nhất trong các cột còn đánh được. Chưa kết hợp tìm kiếm MCTS.

## 2. Mã hóa trạng thái (`training/encode.py`)

Thế cờ được đổi thành tensor 3×6×7 theo góc nhìn của người sắp đi, nên mạng không cần biết đỏ hay vàng:
- kênh 0: quân của người sắp đi;
- kênh 1: quân của đối thủ;
- kênh 2: ô quân sẽ rơi xuống nếu đánh vào mỗi cột còn đánh được.

Kênh 2 cũng cho biết cột nào còn hợp lệ, nên khi tính loss và khi chọn nước đi, cột đầy bị che (xác suất bằng 0).

## 3. Mạng (`training/model.py`)

Mạng gồm một lớp conv 3×3, ba khối residual (mỗi khối hai lớp conv 3×3, BatchNorm, cộng đường tắt), rồi tách hai đầu: đầu policy (conv 1×1, linear ra 7 cột) và đầu value (conv 1×1, linear, tanh). Dùng 64 kênh.

## 4. Dữ liệu (`training/generate_data.py`, `training/dataset.py`)

- **Giáo viên:** agent Rule-based `conn_think3` (đã sửa lỗi không chặn được nước thắng). Với mỗi thế cờ, nhãn policy là nước giáo viên sẽ chọn; nhãn value là kết quả cả ván theo góc nhìn người sắp đi (+1, -1, 0).
- **Tạo ván khác nhau:** `conn_think3` luôn đi giống nhau ở cùng một thế, nên nếu cho tự đấu thì mọi ván giống hệt. Vì vậy 2 nước đầu của mỗi ván đi ngẫu nhiên, và mỗi nước sau có 20% đi ngẫu nhiên. Nhãn policy vẫn luôn là nước giáo viên chọn ở thế đó.
- **Quy mô:** 5000 ván (seed 42) cho 130241 thế; sau khi loại thế trùng còn 76175 thế.
- **Chia tập:** chia theo ván (seed 0) để các thế của cùng một ván không rơi vào hai tập khác nhau: train 60693, val 7863, test 7619 thế. Tập test chưa được dùng.

## 5. Huấn luyện có giám sát (`training/train_sl.py`)

- Loss = cross-entropy cho policy (che cột đầy) + 1,0 × MSE cho value.
- Adam, lr 1e-3, weight decay 1e-4, batch 128, 20 epoch, seed 0, chạy trên CPU, 514,2 giây.
- Chọn checkpoint theo loss tổng (policy + value) thấp nhất trên tập val: epoch 4 (`weights/sl.pt`).

### Kết quả (trên tập val)

| | Epoch 4 (checkpoint) |
|---|---|
| Loss policy (val) | 0,136 |
| Loss value (val) | 0,595 |
| Độ chính xác policy so với giáo viên (val) | 0,954 |
| Độ chính xác policy (train) | 0,965 |

Qua 20 epoch, độ chính xác policy trên val dao động quanh 0,95 đến 0,967. Loss value trên val tăng từ 0,565 (epoch 3) lên 0,798 (epoch 20) trong khi trên train giảm xuống 0,153. Đường cong học ở `docs/images/sl_curves.png`.

## 6. Hạn chế

- Đây là kết quả huấn luyện, **chưa phải kết quả thi đấu**. Sức mạnh của `nn_agent` sẽ đo trong giải đấu của nhóm (khai cuộc ngẫu nhiên, đổi bên). Độ chính xác 0,954 chỉ nói mạng bắt chước giáo viên giống đến đâu, không nói mạng chơi giỏi đến đâu.
- Mạng học bắt chước `conn_think3` nên sức mạnh bị giới hạn bởi giáo viên và có nguy cơ chỉ học lại heuristic.
- Đầu value học thuộc dữ liệu train (loss val tăng dần). Nhãn value là kết quả cả ván và bị nhiễu bởi các nước đi ngẫu nhiên. `nn_agent` chưa dùng value.
- Chọn checkpoint theo loss tổng nghiêng về đầu value; chọn theo loss policy có thể cho độ chính xác policy cao hơn (đỉnh 0,967 ở epoch 18). Chưa thử.
- Chỉ chạy một seed (0) nên chưa biết kết quả dao động bao nhiêu giữa các lần chạy.

## 7. Học tăng cường (RL): hiện trạng

`training/train_rl.py` là bộ khung tự chơi: khởi tạo từ `weights/sl.pt`, cho mạng tự đấu với chính nó, cập nhật bằng policy gradient có baseline là value. **Nhóm chưa huấn luyện RL đầy đủ, chưa có `weights/rl.pt` và chưa có kết quả RL nào.** Mới chạy thử nhỏ (2 vòng, 4 ván) để kiểm tra chương trình chạy được. Khi nào có kết quả thật sẽ cập nhật mục này.

## 8. Các file của phần này

`training/encode.py`, `training/model.py`, `training/generate_data.py`, `training/dataset.py`, `training/train_sl.py`, `training/plot_curves.py`, `training/train_rl.py`, `agents/nn_agent.py`, `tests/test_learning.py`, `tests/test_nn_agent.py`, `weights/sl.pt`, `weights/sl.json`, `logs/training/sl_log.csv`, `docs/images/sl_curves.png`, `docs/README_learning.md`.
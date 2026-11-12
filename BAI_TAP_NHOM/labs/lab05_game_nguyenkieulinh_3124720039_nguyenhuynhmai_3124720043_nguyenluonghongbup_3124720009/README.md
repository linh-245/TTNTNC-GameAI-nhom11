## Cách chạy
```
pip install numpy pandas matplotlib jupyter
jupyter notebook          # mở từng notebook và Run All
```
Thời gian chạy trên 1 CPU: Bài 4 ≈ 0,5 phút, Bài 5 ≈ 2 phút, Bài 9 ≈ 25 phút. Mọi thí nghiệm ngẫu nhiên đặt seed 2026 nên số thắng/hòa/thua tái hiện được; thời gian (ms) phụ thuộc máy.

## Giả định về luật Mean Connect 4
Quân bị lấy mang màu người đi (`recolor=True`); một nước tạo 4 liên tiếp cho cả hai bên là hòa; ván quá 200 nước là hòa. Đổi cách hiểu luật bằng tham số `recolor` của `MeanConnect4`.

## Lưu ý
Ở mục 4.2 của Bài 9, bảng "số lần sai" ghi 3 cho Connect 4 do phép kiểm tra tính nhầm bàn B5 (không có nước thắng ngay trong luật gốc); số lần sai thật là 0 (đã ghi chú trong notebook).

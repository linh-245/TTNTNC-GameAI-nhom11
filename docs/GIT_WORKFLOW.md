# Quy trình Git và GitHub của nhóm 12

Đáp ứng yêu cầu của giảng viên: cung cấp link repo; cập nhật thường xuyên; lịch sử commit thể hiện rõ người làm, nội dung, tiến độ; dùng README, Issues/Projects, Branch/PR, Insights; không dồn code sát hạn.

## 1. Cấu hình danh tính (mỗi người làm một lần)

Commit chỉ được tính cho đúng người khi email trùng với email đã thêm vào tài khoản GitHub (Settings → Emails; có thể dùng địa chỉ `...@users.noreply.github.com`).

```bash
git config --global user.name "Nguyễn Kiều Linh"
git config --global user.email "email-da-gan-voi-github@example.com"
```

Mỗi người commit bằng tài khoản của mình trên máy của mình. Không commit hộ nhau.

## 2. Quy ước commit message

Giảng viên yêu cầu commit message ghi rõ bài nào và làm gì. Với **đề tài cuối kỳ** nhóm không ghi tuần, dùng dạng:

```
CuoiKy/<Khu vực>: <việc đã làm>
```

```
CuoiKy/Engine: thêm undo() và legal_actions()
CuoiKy/MCTS: sửa dấu lan truyền ngược, thêm test hồi quy
CuoiKy/Docs: cập nhật README bảng trạng thái các cấp
```

- Khu vực: Engine, Agents, AlphaBeta, MCTS, UI, Eval, Training, Test, Fix, Docs.
- Phần sau dấu hai chấm viết ngắn, thể hiện được tiến độ. Một commit làm một việc; commit nhỏ, thường xuyên; không dùng message như "update", "fix", "abc".
- Với bài tập nhóm khác trong môn (không phải đề tài cuối kỳ) dùng dạng của giảng viên: `Tuan??/Bai??/Làm gì`, ví dụ `Tuan05/Bai02/Thêm hàm tìm kiếm BFS`.

Tiêu đề Pull Request cũng đặt theo quy ước này.

## 3. Nhánh và Pull Request

| Nhánh | Dùng để |
|---|---|
| `main` | Bản ổn định, mỗi mốc gắn tag (v0.1-fixed, v0.2-agents, v0.3-learning, v0.4-results, v1.0-final) |
| `develop` | Nơi gộp công việc hằng ngày |
| `feat/<tên>-<việc>` | Mỗi người làm việc của mình, ví dụ `feat/mai-alphabeta-pruning` |

Quy trình mỗi việc:

```bash
git checkout develop && git pull --rebase origin develop
git checkout -b feat/mai-alphabeta-pruning
# ... sửa code, chạy test ...
git add agents/alphabeta_agent.py tests/test_search.py
git commit -m "CuoiKy/AlphaBeta: thêm cờ use_pruning và bộ đếm nút"
git push -u origin feat/mai-alphabeta-pruning
# mở Pull Request vào develop trên GitHub, gắn Issue, nhờ 1 người review
```

- Review chéo: Mai review Linh, Búp review Mai, Linh review Búp.
- Gộp bằng **Merge commit** (không Squash) để từng commit của mỗi người còn trong lịch sử.
- Bật Branch protection cho `main` (Settings → Branches): cần PR và CI xanh.
- Sửa conflict bằng `git pull --rebase origin develop` đầu mỗi ngày; chỉ sửa file thuộc phần mình.

## 4. Issues, Labels, Milestones, Projects

1. **Labels** (Issues → Labels): `engine`, `search`, `learning`, `ui`, `eval`, `docs`, `bug`, `task`, và tên từng người (`linh`, `mai`, `bup`).
2. **Milestones** theo các mốc tag: `v0.1-fixed`, `v0.2-agents`, `v0.3-learning`, `v0.4-results`, `v1.0-final`, mỗi mốc có ngày hạn.
3. **Issues:** mỗi dòng trong bảng nhiệm vụ (L1, M1, B1...) là một Issue, tiêu đề `[M1] Alpha-Beta: cờ cắt tỉa và đếm nút`, gán người, milestone, label. Dùng mẫu `task` trong `.github/ISSUE_TEMPLATE`.
4. **Projects** (tab Projects → New project → Board): cột `Backlog`, `Đang làm`, `Chờ review`, `Xong`. Thêm tất cả Issues vào bảng.
5. PR ghi `Closes #12` để Issue tự đóng khi gộp.

## 5. Nhịp làm việc để không dồn sát hạn

- Mỗi người commit và push ít nhất một lần mỗi ngày làm việc.
- Đầu ngày: `git pull --rebase origin develop`. Cuối ngày: push và cập nhật `docs/CONTRIBUTIONS.md`.
- Notebook và tài liệu cũng commit đều theo tiến độ, không chỉ code.
- Không xóa lịch sử, không `push --force` lên `develop` hoặc `main`, không đặt ngày commit giả. Lịch sử trung thực là căn cứ chấm điểm đóng góp.

## 6. Những thứ KHÔNG đưa lên repo

- Thư mục môi trường (`.conda/`, `.venv/`), `__pycache__/`.
- File sách và PDF có bản quyền (ví dụ sách Liu): để link trong README, không commit.
- Dữ liệu lớn và file hơn 100 MB (GitHub chặn): để trên Drive và ghi link; checkpoint nhỏ thì được commit.

## 7. Minh chứng đóng góp (GitHub Insights)

Cuối tuần chụp màn hình để đưa vào phụ lục báo cáo: **Insights → Contributors**, **Commits**, **Code frequency**, **Pulse**, **Network**; cùng ảnh bảng Project và danh sách Pull Request đã gộp.
Lưu ý: Insights chỉ đếm commit có email đã gắn với tài khoản GitHub và nằm trên nhánh mặc định.

## 8. Checklist cuối cùng

- [ ] Link repo ghi trong báo cáo và file nộp; nếu repo private thì đã thêm giảng viên làm collaborator.
- [ ] README đầy đủ: mô tả, thành viên, cài đặt, chạy, kết quả, phần kế thừa
- [ ] Commit message đúng dạng `CuoiKy/<Khu vực>: <việc>`
- [ ] Mỗi thành viên có commit, PR, Issue riêng
- [ ] Các tag mốc đã gắn
- [ ] Clone sạch về máy khác chạy lại được theo README

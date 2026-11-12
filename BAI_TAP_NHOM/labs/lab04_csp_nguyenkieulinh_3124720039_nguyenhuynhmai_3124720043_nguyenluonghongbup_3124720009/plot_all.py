"""plot_all.py - Gộp CSV của cả nhóm, vẽ biểu đồ và xuất bảng 4.2 (n-Queens).

Cách dùng:   python plot_all.py [thu_muc_chua_csv]      (mặc định: thư mục hiện tại)

Đọc:   results_*.csv  (thuat_toan, n, seed, thanh_cong, h_cuoi, so_buoc, so_danh_gia, thoi_gian_s)
       history_*.csv  (thuat_toan, vong_lap, h)
       timing_*.csv   (thuat_toan, n, thoi_gian_trung_binh_s)
Ghi:   figures/plot_a_success_time.png, figures/plot_b_convergence.png, figures/plot_c_loglog.png,
       table_4_2_nqueens.csv, table_4_2_nqueens.md
Thuật toán/file nào chưa có sẽ bị bỏ qua, nên script chạy được khi mới có một phần dữ liệu.
"""
import sys, glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES_COLS = ["thuat_toan", "n", "seed", "thanh_cong", "h_cuoi", "so_buoc", "so_danh_gia", "thoi_gian_s"]
HIS_COLS = ["thuat_toan", "vong_lap", "h"]
TIM_COLS = ["thuat_toan", "n", "thoi_gian_trung_binh_s"]
LABELS = {"steepest_hc": "Steepest HC", "stochastic_hc_inefficient": "Inefficient Stoch. HC",
          "stochastic_hc_efficient": "Efficient Stoch. HC", "random_restart": "Random Restart",
          "sa_geometric": "SA (hình học)", "sa_linear": "SA (tuyến tính)"}
ORDER = list(LABELS)
SIZES = (4, 8, 12, 16, 20)

def load(folder, prefix, cols):
    frames = []
    for f in sorted(glob.glob(os.path.join(folder, f"{prefix}_*.csv"))):
        df = pd.read_csv(f)
        missing = set(cols) - set(df.columns)
        if missing:
            print(f"[CẢNH BÁO] {os.path.basename(f)} thiếu cột {sorted(missing)} -> bỏ qua"); continue
        frames.append(df[cols])
    if not frames:
        print(f"[CẢNH BÁO] không tìm thấy {prefix}_*.csv"); return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    df["thuat_toan"] = df["thuat_toan"].astype(str)
    return df

def lab(a): return LABELS.get(a, a)
def algos(df): return [a for a in ORDER if a in set(df.thuat_toan)] + sorted(set(df.thuat_toan) - set(ORDER))
def color(a, algs): return plt.cm.tab10(algs.index(a) % 10)

def table_42(res):
    g = res.groupby(["thuat_toan", "n"])
    t = g.agg(so_luot=("seed", "count"), thanh_cong_pct=("thanh_cong", lambda x: 100 * x.mean()),
              h_cuoi_tb=("h_cuoi", "mean"), so_danh_gia_tb=("so_danh_gia", "mean"),
              thoi_gian_tb_s=("thoi_gian_s", "mean")).reset_index()
    t["_o"] = t.thuat_toan.map(lambda a: ORDER.index(a) if a in ORDER else 99)
    t = t.sort_values(["n", "_o"]).drop(columns="_o")
    t.insert(0, "Thuật toán", t.thuat_toan.map(lab))
    return t.round({"thanh_cong_pct": 1, "h_cuoi_tb": 3, "so_danh_gia_tb": 1, "thoi_gian_tb_s": 5})

def to_md(t):
    cols = ["Thuật toán", "n", "so_luot", "thanh_cong_pct", "h_cuoi_tb", "so_danh_gia_tb", "thoi_gian_tb_s"]
    head = ["Thuật toán", "n", "Số lượt", "Thành công (%)", "h cuối TB", "Số lần đánh giá TB", "Thời gian TB (s)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in t[cols].iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)

def plot_a(res, out):
    sub = res[res.n.isin([4, 8])]
    if sub.empty: return
    algs = algos(sub); ns = sorted(sub.n.unique()); w = 0.8 / max(len(algs), 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, a in enumerate(algs):
        d = sub[sub.thuat_toan == a].groupby("n")
        succ = [100 * d.get_group(n).thanh_cong.mean() if n in d.groups else np.nan for n in ns]
        tm = [d.get_group(n).thoi_gian_s.mean() if n in d.groups else np.nan for n in ns]
        x = np.arange(len(ns)) + (i - (len(algs) - 1) / 2) * w
        axes[0].bar(x, succ, w, label=lab(a), color=color(a, algs))
        axes[1].bar(x, tm, w, label=lab(a), color=color(a, algs))
    for ax, ttl in zip(axes, ["Tỷ lệ chạy đạt nghiệm tối ưu (h = 0), %", "Thời gian chạy trung bình mỗi lượt (s)"]):
        ax.set_xticks(range(len(ns))); ax.set_xticklabels([f"n = {n}" for n in ns]); ax.set_title(ttl); ax.grid(alpha=.3, axis="y")
    axes[0].set_ylim(0, 128); axes[1].set_yscale("log"); axes[0].legend(fontsize=8, ncol=3, loc="upper center")
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()

def plot_b(his, out):
    if his.empty: return
    algs = algos(his)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for a in algs:
        g = his[his.thuat_toan == a].sort_values("vong_lap")
        ax.step(g.vong_lap + 1, g.h, where="post", label=f"{lab(a)} ({g.vong_lap.max()} vòng)", color=color(a, algs))
    ax.set_xscale("log"); ax.set_xlabel("Vòng lặp + 1 (thang log)"); ax.set_ylabel("h")
    ax.set_title("Hội tụ của h tại n = 8 (một lượt đại diện, seed = 0)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()

def plot_c(tim, out):
    if tim.empty: return
    algs = algos(tim)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    notes = []
    for a in algs:
        g = tim[tim.thuat_toan == a].sort_values("n")
        g = g[g.thoi_gian_trung_binh_s > 0]
        if len(g) >= 2:
            s = np.polyfit(np.log(g.n), np.log(g.thoi_gian_trung_binh_s), 1)[0]; notes.append(f"{lab(a)}: ~ n^{s:.2f}")
            ax.loglog(g.n, g.thoi_gian_trung_binh_s, "o-", label=f"{lab(a)} (độ dốc {s:.2f})", color=color(a, algs))
    ax.set_xticks(SIZES); ax.set_xticklabels(SIZES); ax.minorticks_off()
    ax.set_xlabel("n (log)"); ax.set_ylabel("Thời gian trung bình (s, log)"); ax.set_title("Thời gian chạy theo kích thước bàn cờ")
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
    return notes

def main(folder="."):
    res, his, tim = load(folder, "results", RES_COLS), load(folder, "history", HIS_COLS), load(folder, "timing", TIM_COLS)
    figs = os.path.join(folder, "figures"); os.makedirs(figs, exist_ok=True)
    print("Thuật toán có kết quả:", [lab(a) for a in algos(res)] if len(res) else "(chưa có)")
    if len(res):
        t = table_42(res)
        t.to_csv(os.path.join(folder, "table_4_2_nqueens.csv"), index=False)
        md = to_md(t); open(os.path.join(folder, "table_4_2_nqueens.md"), "w", encoding="utf-8").write(md + "\n")
        print(md); plot_a(res, os.path.join(figs, "plot_a_success_time.png"))
    plot_b(his, os.path.join(figs, "plot_b_convergence.png"))
    notes = plot_c(tim, os.path.join(figs, "plot_c_loglog.png"))
    if notes: print("Độ dốc log-log:", "; ".join(notes))
    print("Đã lưu biểu đồ vào", figs)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")

"""
Tổng hợp kết quả của tất cả thực nghiệm (mọi CSV trong logs/ hoặc thư mục chỉ định).

Chạy:
  python solver/summarize_results.py
  python solver/summarize_results.py --dirs logs eval

Đầu ra (trong thư mục logs/):
  summary/<tên>.csv     bảng sạch của từng thực nghiệm (bỏ cột trống)
  summary_all_long.csv  gộp dọc: source,row,agent,metric,value
  summary_all.md        bảng Markdown cho từng thực nghiệm
"""
import argparse
import csv
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def tidy(v):
    """Làm tròn số thực, nan -> rỗng, giữ nguyên chuỗi."""
    if v is None:
        return ''
    s = str(v).strip()
    try:
        x = float(s)
    except ValueError:
        return s
    if x != x:
        return ''
    return int(x) if x.is_integer() else round(x, 4)


def read_clean(path):
    """Đọc CSV, làm tròn số, bỏ cột trống hoàn toàn. Trả về (cột, dòng)."""
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = [{k: tidy(v) for k, v in r.items() if k} for r in csv.DictReader(f)]
    if not rows:
        return [], []
    cols = [c for c in rows[0] if any(r.get(c, '') != '' for r in rows)]
    return cols, [{c: r.get(c, '') for c in cols} for r in rows]


def md_table(cols, rows):
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        out.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dirs', nargs='+', default=[os.path.join(ROOT, 'logs')])
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'logs'))
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, 'summary'), exist_ok=True)
    long_rows, md = [], ['# Tổng hợp kết quả các thực nghiệm\n']

    for d in args.dirs:
        for path in sorted(glob.glob(os.path.join(d, '*.csv'))):
            tag = os.path.splitext(os.path.basename(path))[0]
            if tag.startswith('summary_all'):
                continue
            cols, rows = read_clean(path)
            if not rows:
                print(f"Bỏ qua {tag} (rỗng)")
                continue

            with open(os.path.join(args.out_dir, 'summary', tag + '.csv'),
                      'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)

            for i, r in enumerate(rows, 1):
                agent = r.get('agent', r.get(cols[0], ''))
                for c in cols:
                    if c in ('agent',) or r[c] == '':
                        continue
                    long_rows.append({'source': tag, 'row': i, 'agent': agent,
                                      'metric': c, 'value': r[c]})

            md += [f'## {tag}\n', md_table(cols, rows), '']
            print(f"Đã xử lý {tag}: {len(rows)} dòng, {len(cols)} cột")

    if not long_rows:
        raise SystemExit("Không tìm thấy CSV hợp lệ trong " + ', '.join(args.dirs))

    p_long = os.path.join(args.out_dir, 'summary_all_long.csv')
    with open(p_long, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['source', 'row', 'agent', 'metric', 'value'])
        w.writeheader()
        w.writerows(long_rows)
    p_md = os.path.join(args.out_dir, 'summary_all.md')
    with open(p_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f"Đã ghi {os.path.abspath(p_long)}\nĐã ghi {os.path.abspath(p_md)}")


if __name__ == '__main__':
    main()
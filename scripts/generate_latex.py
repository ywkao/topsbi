import csv
from collections import defaultdict
from pathlib import Path

fig_root = Path("/eos/user/y/ykao/www/topsbi/figures")
manifest_path = fig_root / "manifest.csv"
tex_out = fig_root / "gallery.tex"

wcs    = ["ctGRe", "ctj1", "cQj31"]
values = ["1.0", "3.0", "5.0"]
tags   = ["all", "compact", "lowlevel", "highlevel"]

# 讀 manifest,依 (wc, fig) 分組,組內用 (value, tag) 當 key 查路徑
rows = defaultdict(dict)  # rows[(wc, fig)][(value, tag)] = path
figs_seen = set()

with open(manifest_path) as f:
    for r in csv.DictReader(f):
        key = (r["wc"], r["fig"])
        rows[key][(r["value"], r["tag"])] = r["path"]
        figs_seen.add(r["fig"])

def esc(s):
    # LaTeX 檔名/文字裡的底線跳脫
    return s.replace("_", r"\_")

lines = []
lines.append(r"\documentclass{article}")
lines.append(r"\usepackage[margin=1cm,landscape]{geometry}")
lines.append(r"\usepackage{graphicx}")
lines.append(r"\usepackage{adjustbox}")
lines.append(r"\graphicspath{{" + str(fig_root) + r"/}}")
lines.append(r"\begin{document}")

for wc in wcs:
    for fig in sorted(figs_seen):
        key = (wc, fig)
        if key not in rows:
            continue  # 這個 wc 底下沒有這種圖,跳過,不會產生空頁

        lines.append(r"\begin{table}[p]")
        lines.append(r"\centering")
        lines.append(r"\renewcommand{\arraystretch}{0}")
        lines.append(r"\setlength{\tabcolsep}{2pt}")
        lines.append(r"\begin{tabular}{" + "c" * len(tags) + "}")

        for vi, value in enumerate(values):
            row_cells = []
            for tag in tags:
                path = rows[key].get((value, tag))
                if path is None:
                    row_cells.append("")  # 缺圖:留白,不報錯
                    continue
                cell = (
                    r"\begin{minipage}{4.2cm}\centering"
                    r"{\tiny\ttfamily " + esc(f"{wc}_{value}_{tag}") + r"}\\[2pt]"
                    r"\adjustbox{valign=t}{\fbox{\includegraphics[width=4.2cm,height=3.2cm,keepaspectratio]{"
                    + path + r"}}}"
                    r"\end{minipage}"
                )
                row_cells.append(cell)
            lines.append(" & ".join(row_cells) + r" \\[4pt]")

        lines.append(r"\end{tabular}")
        lines.append(r"\caption{" + esc(f"{wc} / {fig}") + r"}")
        lines.append(r"\end{table}")
        lines.append(r"\clearpage")

lines.append(r"\end{document}")

tex_out.write_text("\n".join(lines))
print(f"[ok] wrote {tex_out}")

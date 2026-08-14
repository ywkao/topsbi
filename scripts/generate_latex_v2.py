# scripts/generate_latex_v2.py
import csv
from collections import defaultdict
from itertools import islice

from config import FIG_ROOT

manifest_path = FIG_ROOT / "manifest.csv"
tex_out = FIG_ROOT / "gallery.tex"

COLS, ROWS = 6, 4
PER_PAGE = COLS * ROWS

IMG_W, IMG_H = "4.2cm", "2.9cm"

# ---- 讀 manifest,依 wc 分組 ----
groups = defaultdict(list)
with open(manifest_path) as f:
    for r in csv.DictReader(f):
        groups[r["wc"]].append(r)

def sort_key(r):
    return (r["fig"], float(r["value"]), r["tag"])

def esc(s):
    return s.replace("_", r"\_")

def chunk(iterable, n):
    it = iter(iterable)
    while chunk_ := list(islice(it, n)):
        yield chunk_

# ---- 組 LaTeX ----
lines = [
    r"\documentclass{article}",
    r"\usepackage[margin=0.5cm,landscape]{geometry}",
    r"\usepackage{array}",
    r"\usepackage{graphicx}",
    r"\graphicspath{{" + str(FIG_ROOT) + r"/}}",
    r"\setlength{\parindent}{0pt}",
    r"\setlength{\extrarowheight}{0pt}",
    r"\begin{document}",
    r"\pagestyle{empty}",
]

for wc in sorted(groups):
    entries = sorted(groups[wc], key=sort_key)

    for page in chunk(entries, PER_PAGE):
        lines.append(r"\begin{table}[p]")
        lines.append(r"\centering")
        lines.append(r"\setlength{\tabcolsep}{1pt}")
        lines.append(r"\begin{tabular}{" + "c" * COLS + "}")

        for row in chunk(page, COLS):
            cells = []
            for r in row:
                label = f"{r['fig']}_{r['value']}_{r['tag']}"
                cell = (
                    r"\begin{minipage}[t]{4.3cm}\centering"
                    r"\vspace{0pt}"
                    r"{\tiny\ttfamily " + esc(label) + r"}\\[-2pt]"
                    r"\fbox{\includegraphics[width=" + IMG_W + r",height=" + IMG_H
                    + r",keepaspectratio]{" + r["path"] + r"}}"
                    r"\end{minipage}"
                )
                cells.append(cell)
            while len(cells) < COLS:
                cells.append("")
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\end{tabular}")
        lines.append(r"\caption{" + esc(wc) + r"}")
        lines.append(r"\end{table}")
        lines.append(r"\clearpage")

lines.append(r"\end{document}")

tex_out.write_text("\n".join(lines))
total = sum(len(v) for v in groups.values())
print(f"[ok] wrote {tex_out}, {total} images total across {len(groups)} wc group(s)")

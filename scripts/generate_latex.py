import csv
from collections import defaultdict

from config import FIG_ROOT

manifest_path = FIG_ROOT / "manifest.csv"
tex_out = FIG_ROOT / "gallery.tex"

rows = defaultdict(dict)
figs_seen, wcs_seen, values_seen, tags_seen = set(), set(), set(), set()

with open(manifest_path) as f:
    for r in csv.DictReader(f):
        key = (r["wc"], r["fig"])
        rows[key][(r["value"], r["tag"])] = r["path"]
        figs_seen.add(r["fig"])
        wcs_seen.add(r["wc"])
        values_seen.add(r["value"])
        tags_seen.add(r["tag"])

wcs    = sorted(wcs_seen)
values = sorted(values_seen, key=float)
tags   = sorted(tags_seen)

def esc(s):
    return s.replace("_", r"\_")

lines = [
    r"\documentclass{article}",
    r"\usepackage[margin=1cm,landscape]{geometry}",
    r"\usepackage{graphicx}",
    r"\graphicspath{{" + str(FIG_ROOT) + r"/}}",
    r"\begin{document}",
]

for wc in wcs:
    for fig in sorted(figs_seen):
        key = (wc, fig)
        if key not in rows:
            continue

        lines.append(r"\begin{table}[p]")
        lines.append(r"\centering")
        lines.append(r"\renewcommand{\arraystretch}{0}")
        lines.append(r"\setlength{\tabcolsep}{2pt}")
        lines.append(r"\begin{tabular}{" + "c" * len(tags) + "}")

        for value in values:
            row_cells = []
            for tag in tags:
                path = rows[key].get((value, tag))
                if path is None:
                    row_cells.append("")
                    continue
                cell = (
                    r"\begin{minipage}[t]{4.2cm}\centering"
                    r"{\tiny\ttfamily " + esc(f"{wc}_{value}_{tag}") + r"}\\[2pt]"
                    r"\fbox{\includegraphics[width=4.2cm,height=3.2cm,keepaspectratio]{"
                    + path + r"}}"
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

import shutil
from itertools import product
from pathlib import Path

from config import (
    WCS, VALUES, TAGS, NODES, LRSCH,
    SRC_ROOT, INDEX_SRC, FIG_ROOT, SUBDIRS,
)

FIG_ROOT.mkdir(parents=True, exist_ok=True)
shutil.copy2(INDEX_SRC, FIG_ROOT / "index.php")

manifest = []

for tag, node, lr, value, wc in product(TAGS, NODES, LRSCH, VALUES, WCS):
    complete_dir = SRC_ROOT / f"{tag}_{node}_dropout_{lr}" / wc / value / "complete"

    if not complete_dir.is_dir():
        print(f"[skip] source folder not found: {complete_dir}")
        continue

    for subdir, pattern in SUBDIRS:
        src_dir = complete_dir / subdir if subdir else complete_dir
        if not src_dir.is_dir():
            print(f"[skip] source folder not found: {src_dir}")
            continue

        files = sorted(src_dir.glob(pattern))
        if not files:
            print(f"[warn] no files matching {pattern} in {src_dir}")
            continue

        for src in files:
            fig = f"{subdir}/{src.stem}" if subdir else src.stem
            dst = FIG_ROOT / fig / f"{wc}_value_{value}_feature_{tag}_node_{node}_lr_{lr}{src.suffix}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(INDEX_SRC, dst.parent / "index.php")
            shutil.copy2(src, dst)

            if src.suffix.lower() == ".png":
                manifest.append({
                    "fig": fig, "wc": wc, "value": value,
                    "tag": tag, "node": node, "lr": lr,
                    "path": str(dst.relative_to(FIG_ROOT)),
                })

        print(f"[ok] {wc} {subdir or 'top'}: {len(files)} figures copied")

import csv
manifest_path = FIG_ROOT / "manifest.csv"
if manifest:
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    print(f"[ok] manifest written: {manifest_path} ({len(manifest)} entries)")
else:
    print("[warn] manifest empty, nothing written")

import csv
import shutil
from itertools import product

from config import (
    TAGS, NODES, LRSCH,
    SRC_ROOT, INDEX_SRC, FIG_ROOT, SUBDIRS,
)

FIG_ROOT.mkdir(parents=True, exist_ok=True)
shutil.copy2(INDEX_SRC, FIG_ROOT / "index.php")

manifest = []
indexed  = set()  # dirs that already got index.php

for tag, node, lr in product(TAGS, NODES, LRSCH):
    run_dir = SRC_ROOT / f"{tag}_{node}_dropout_{lr}"
    # stitched layout: {term}/seed_{seed}/complete, e.g. ctGIm_ctGRe/seed_2/complete
    complete_dirs = sorted(run_dir.glob("*/seed_*/complete"))
    print(f"[info] {run_dir.name}: {len(complete_dirs)} trainings found (expect 152 terms x n_seeds)")

    for complete_dir in complete_dirs:
        term = complete_dir.parent.parent.name
        seed = complete_dir.parent.name.removeprefix("seed_")

        for subdir, pattern in SUBDIRS:
            src_dir = complete_dir / subdir if subdir else complete_dir
            files = sorted(src_dir.glob(pattern))
            if not files:
                print(f"[warn] no files matching {pattern} in {src_dir}")
                continue

            for src in files:
                fig = f"{subdir}/{src.stem}" if subdir else src.stem
                dst = FIG_ROOT / fig / f"{term}_seed_{seed}_feature_{tag}_node_{node}_lr_{lr}{src.suffix}"
                if dst.parent not in indexed:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(INDEX_SRC, dst.parent / "index.php")
                    indexed.add(dst.parent)
                shutil.copy2(src, dst)

                manifest.append({
                    "fig": fig, "term": term, "seed": seed,
                    "tag": tag, "node": node, "lr": lr,
                    "path": str(dst.relative_to(FIG_ROOT)),
                })

            print(f"[ok] {term} seed_{seed} {subdir or 'top'}: {len(files)} figures copied")

manifest_path = FIG_ROOT / "manifest.csv"
if manifest:
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    print(f"[ok] manifest written: {manifest_path} ({len(manifest)} entries)")
else:
    print("[warn] manifest empty, nothing written")

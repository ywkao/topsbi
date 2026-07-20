import shutil
from itertools import product
from pathlib import Path

wcs = [
    "cQd1", "cQd8", "cQj11", "cQj18", "cQj31", "cQj38",
    "cQu1", "cQu8", "ctd1", "ctd8", "ctGIm", "ctGRe",
    "ctj1", "ctj8", "ctu1", "ctu8",
]

index_src = Path("/eos/user/y/ykao/www/topsbi/index.php")
fig_root  = Path("/eos/user/y/ykao/www/topsbi/figures")

# put index_src under figures/
fig_root.mkdir(parents=True, exist_ok=True)
shutil.copy2(index_src, fig_root / "index.php")

wcs = ["ctGRe", "ctj1", "cQj31"]
values = ["1.0", "3.0", "5.0"]
tags = ["all", "compact", "lowlevel", "highlevel"]
nodes = ["n128"]
lrSch = ["cosine"]

# /eos/cms/store/user/ykao/topsbi/results/fastTrain/gen/all_n128_dropout_cosine/
# /eos/cms/store/user/ykao/topsbi/results/fastTrain/gen/all_n128_dropout_cosine/cQd1/5.0/complete/
#
# each .../complete/ dir now also contains per-feature plots in subfolders:
#   complete/kinematics/<feature>.png
#   complete/animations/<feature>.gif
# these are collected the same way as the top-level pngs, just nested under
# figures/kinematics/<feature>/... and figures/animations/<feature>/...

subdirs = [(None, "*.png"), ("kinematics", "*.png"), ("animations", "*.gif")]

for tag, node, lr, value, wc in product(tags, nodes, lrSch, values, wcs):
    src_dir_tmpl = "/eos/cms/store/user/ykao/topsbi/results/fastTrain/gen/{tag}_{node}_dropout_{lr}/{wc}/{value}/complete"
    dst_tmpl     = "/eos/user/y/ykao/www/topsbi/figures/{fig}/{wc}_value_{value}_feature_{tag}_node_{node}_lr_{lr}{ext}"

    complete_dir = Path(src_dir_tmpl.format(tag=tag, node=node, lr=lr, wc=wc, value=value))

    if not complete_dir.is_dir():
        print(f"[skip] source folder not found: {complete_dir}")
        continue

    for subdir, pattern in subdirs:
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
            dst = Path(dst_tmpl.format(tag=tag, fig=fig, wc=wc, value=value, node=node, lr=lr, ext=src.suffix))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(index_src, dst.parent / "index.php")
            shutil.copy2(src, dst)

        print(f"[ok] {wc} {subdir or 'top'}: {len(files)} figures copied")

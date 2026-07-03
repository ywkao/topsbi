import shutil
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

# /eos/cms/store/user/ykao/topsbi/results/fastTrain/gen/all/ctu8/3.0/complete

values = ["1.0", "3.0", "5.0"]
tags = ["all", "pt_mass"]

#--------------------------------------------------
# tmp
#--------------------------------------------------
wcs = ["cQd8"]
values = ["3.0"]
tags = ["pt_mass"]
feat = "pt_mass"
nodes = ["n128", "n64", "n32"]
lrSch = ["plateau", "cosine"]

# /eos/cms/store/user/ykao/topsbi/results/fastTrain/gen/pt_mass_n64_dropout_plateau/cQd8/3.0/complete
# /eos/cms/store/user/ykao/topsbi/results/fastTrain/gen/pt_mass_n128_dropout_cosine
# /eos/cms/store/user/ykao/topsbi/results/fastTrain/gen/pt_mass_n128_dropout_plateau
# /eos/cms/store/user/ykao/topsbi/results/fastTrain/gen/pt_mass_n32_dropout_cosine
# /eos/cms/store/user/ykao/topsbi/results/fastTrain/gen/pt_mass_n32_dropout_plateau
# /eos/cms/store/user/ykao/topsbi/results/fastTrain/gen/pt_mass_n64_dropout_cosine
# /eos/cms/store/user/ykao/topsbi/results/fastTrain/gen/pt_mass_n64_dropout_plateau

for tag in tags:
    for node in nodes:
        for lr in lrSch:
            for value in values:
                src_dir_tmpl = "/eos/cms/store/user/ykao/topsbi/results/fastTrain/gen/{tag}_{node}_dropout_{lr}/{wc}/{value}/complete"
                dst_tmpl     = "/eos/user/y/ykao/www/topsbi/figures/{fig}/{wc}_value_{value}_feature_{tag}_node_{node}_lr_{lr}.png"

                for wc in wcs:
                    src_dir = Path(src_dir_tmpl.format(tag=tag, node=node, lr=lr, wc=wc, value=value))

                    if not src_dir.is_dir():
                        print(f"[skip] source folder not found: {src_dir}")
                        continue

                    pngs = sorted(src_dir.glob("*.png"))
                    if not pngs:
                        print(f"[warn] no pngs in {src_dir}")
                        continue

                    for src in pngs:
                        fig = src.stem
                        dst = Path(dst_tmpl.format(tag=tag, fig=fig, wc=wc, value=value, node=node, lr=lr))
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(index_src, dst.parent / "index.php")
                        shutil.copy2(src, dst)

                    print(f"[ok] {wc}: {len(pngs)} figures copied")

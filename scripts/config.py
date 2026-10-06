# scripts/config.py
from pathlib import Path


DIR    = "gen"
TAGS   = ["highlevel"] # ["all", "compact", "highlevel", "lowlevel"]
NODES  = ["n256"]
LRSCH  = ["cosine"]

SRC_ROOT   = Path(f"/eos/cms/store/user/ykao/topsbi/results/fastTrain/{DIR}")
INDEX_SRC  = Path("/eos/user/y/ykao/www/topsbi/index.php")
FIG_ROOT   = Path("/eos/user/y/ykao/www/topsbi/figures")

# stitched training runs with plotEvery=0, so complete/animations is empty
SUBDIRS = [(None, "*.png"), ("kinematics", "*.png")]

# scripts/config.py
from pathlib import Path

DIR    = "gen"
WCS    = ["ctGRe", "ctGIm", "ctj8"]
VALUES = ["1.0", "3.0", "5.0"]
TAGS   = ["all", "compact", "highlevel", "lowlevel"]
NODES  = ["n256"]
LRSCH  = ["cosine"]

SRC_ROOT   = Path(f"/eos/cms/store/user/ykao/topsbi/results/fastTrain/{DIR}")
INDEX_SRC  = Path("/eos/user/y/ykao/www/topsbi/index.php")
FIG_ROOT   = Path("/eos/user/y/ykao/www/topsbi/figures")

SUBDIRS = [(None, "*.png"), ("kinematics", "*.png"), ("animations", "*.gif")]

# wcs = [
#     "cQd1", "cQd8", "cQj11", "cQj18", "cQj31", "cQj38",
#     "cQu1", "cQu8", "ctd1", "ctd8", "ctGIm", "ctGRe",
#     "ctj1", "ctj8", "ctu1", "ctu8",
# ]

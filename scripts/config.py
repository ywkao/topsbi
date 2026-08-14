# scripts/config.py
from pathlib import Path

DIR    = "gen"
WCS    = ["ctGIm"]
VALUES = ["5.0"]
TAGS   = ["all"]
NODES  = ["n128"]
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

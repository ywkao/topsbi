# scripts/config.py
from pathlib import Path

DIR    = "gen"
WCS = ['ctGIm', 'ctGRe']
VALUES = ["-0.1", "-0.5", "-1.5"]
# WCS = ['cQj38', 'cQj18', 'cQu8', 'cQd8', 'ctj8', 'ctu8', 'ctd8', 'cQj31', 'cQj11', 'cQu1', 'cQd1', 'ctj1', 'ctu1', 'ctd1']
# VALUES = ["0.5", "1.5", "2.5"]
TAGS   = ["highlevel"] # ["all", "compact", "highlevel", "lowlevel"]
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

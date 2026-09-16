# scripts/config.py
from pathlib import Path


WC_TRAINING_VALUES = {
    # WC          [ pone,  pfive,  pten,  nfive ]  <- constructive x3 + destructive x1 (nfive)
    'ctGIm':      [ 0.40,   0.90,  1.28, -0.91],
    'ctGRe':      [-0.03,  -0.15, -0.30,  4.82],
    'cQj38':      [ 1.50,   3.56,  5.12, -3.94],
    'cQj18':      [ 0.78,   2.60,  4.08, -5.39],
    'cQu8':       [ 1.06,   3.27,  5.02, -5.78],
    'cQd8':       [ 1.40,   4.26,  6.50, -7.27],
    'ctj8':       [ 0.63,   2.17,  3.44, -4.90],
    'ctu8':       [ 1.12,   3.45,  5.30, -6.10],
    'ctd8':       [ 1.44,   4.40,  6.73, -7.58],
    'cQj31':      [ 0.65,   1.48,  2.10, -1.51],
    'cQj11':      [-0.63,  -1.46, -2.08,  1.53],
    'cQu1':       [ 0.79,   1.85,  2.64, -1.99],
    'cQd1':       [ 1.06,   2.41,  3.42, -2.47],
    'ctj1':       [ 0.59,   1.38,  1.97, -1.49],
    'ctu1':       [-0.80,  -1.82, -2.59,  1.87],
    'ctd1':       [-0.99,  -2.28, -3.25,  2.38],
}

DIR    = "gen"
TAGS   = ["highlevel"] # ["all", "compact", "highlevel", "lowlevel"]
NODES  = ["n256"]
LRSCH  = ["cosine"]

SRC_ROOT   = Path(f"/eos/cms/store/user/ykao/topsbi/results/fastTrain/{DIR}")
INDEX_SRC  = Path("/eos/user/y/ykao/www/topsbi/index.php")
FIG_ROOT   = Path("/eos/user/y/ykao/www/topsbi/figures")

SUBDIRS = [(None, "*.png"), ("kinematics", "*.png"), ("animations", "*.gif")]

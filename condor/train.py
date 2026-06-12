#!/usr/bin/env python3
"""Train one topsbi model from a YAML config.  Usage: python train.py <config.yml>"""
import os
# 必須在 import torch 之前設好
os.environ.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

import sys, yaml

ANALYSIS = "/eos/user/y/ykao/SWAN_projects/analysis"
sys.path.insert(0, f"{ANALYSIS}/topsbi")
sys.path.insert(0, ANALYSIS)
from topsbi.train import main

if len(sys.argv) != 2:
    sys.exit(f"usage: {sys.argv[0]} <config.yml>")
cfg_path = sys.argv[1]

with open(cfg_path) as f:
    config = yaml.safe_load(f)
config['device'] = 'cuda'

print(f"[train.py] config = {cfg_path}", flush=True)
for k, v in config.items():
    print(f"  {k} = {v if not isinstance(v, list) else f'list(len={len(v)})'}", flush=True)

main(config)
print(f"[train.py] done: {cfg_path}", flush=True)

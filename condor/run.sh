#!/bin/bash
set -e
CONFIG=$1
echo "[run.sh] host    = $(hostname)"
echo "[run.sh] config  = $CONFIG"
nvidia-smi -L || true

source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh

# pick up user-installed packages (e.g. wandb) not present in the LCG view
export PYTHONPATH=/eos/user/y/ykao/.local/lib/python3.13/site-packages:$PYTHONPATH

# uncomment if worker nodes can't reach api.wandb.ai; sync later with `wandb sync <run-dir>`
# export WANDB_MODE=offline

python /eos/home-y/ykao/SWAN_projects/analysis/topsbi/condor/train.py "$CONFIG"

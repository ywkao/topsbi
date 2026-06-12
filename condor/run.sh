#!/bin/bash
set -e
CONFIG=$1
echo "[run.sh] host    = $(hostname)"
echo "[run.sh] config  = $CONFIG"
nvidia-smi -L || true

source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh

python /eos/home-y/ykao/SWAN_projects/analysis/topsbi/condor/train.py "$CONFIG"

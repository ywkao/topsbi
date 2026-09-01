#!/bin/bash
set -e
cd /eos/home-y/ykao/SWAN_projects/analysis/topsbi
mkdir -p ctj8 ctGRe

run() {
    local cfg=$1 model=$2 eos_out=$3 local_out=$4
    python3 shap.py "$cfg" --model "$model" --outdir "$eos_out"
    cp "$eos_out"/shap_summary.png "$eos_out"/shap_ranking.txt "$local_out/"
}

run condor/configs/ctj8.yml \
    /eos/cms/store/user/ykao/topsbi/results/fastTrain/gen/all_n128_dropout_cosine_cpu_v1.2/ctj8/5.0/model.pt \
    /eos/cms/store/user/ykao/topsbi/results/fastTrain/gen/all_n128_dropout_cosine_cpu_v1.2/ctj8/5.0/complete \
    ./ctj8

run condor/configs/ctGRe.yml \
    /eos/cms/store/user/ykao/topsbi/results/fastTrain/gen/all_n128_dropout_cosine/ctGRe/5.0/model.pt \
    /eos/cms/store/user/ykao/topsbi/results/fastTrain/gen/all_n128_dropout_cosine/ctGRe/5.0/complete \
    ./ctGRe

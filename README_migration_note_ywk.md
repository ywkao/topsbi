# Migration note: dev → eddie/main (ywk)

全部只在 local，沒有 push。

## Branches (base: `eddie/main` @ 6115263)

| branch | commit | 內容 |
|---|---|---|
| `fix/crashes` | 9dc4e4a | `ensemble`：seeds 取平均（原本是總和，r(SM)=N）；`full_likelihood` 沒有 `sm` 時不再 crash |
| | cb871fc | `toy_builder`：計數 accepted events，而不是 batches |
| | 540354f | `train.py`：移除重複的 feature 標準化（merge c36ce8c 帶回來的；main 自己的 test 因此 fail） |
| `port/dev` (on top) | c796612 | stitched：`get_probabilities` 用 float64 |
| | e2e3b47 | evaluation 時關掉 dropout（`eval()`） |
| | 0004d96 | stitched：`sanitize_mask` 丟掉 w≤0 或極端 weights（config key `sanitize`） |
| | 8e4bdc3 | 用 `config['device']` 訓練、`no_grad` evaluation、`BatchSampler` |
| | 82b6d7e | `scripts/scan.py` / `replot_scan.py`：ensemble vs truth 的 1D scan |
| | 7b0126d | `scripts/shap_features.py`：單一網路的 SHAP feature ranking |
| | dabf409 | calibration 和 kinematics 圖改用 p0 加權（原本比的是 mean(p1/p0)，不是網路學的 Σp1/Σp0）；修正 r 和 r̂ 的 legend 標反 |
| | dbcb604 | `scan.py --data-values`：把樣本 reweight 成任意點的 pseudo-data（近 SM closure） |
| | 23da3fd | `full_likelihood`：morph unnormalized 的 w(c1)/w(c0)（normalized ratio 不是二次式；main 也有這個 bug） |
| | 7c5f83b | README notes |
| | 3bcc25e | `scan.py --window`：每個 WC 掃 data ± frac·\|data\| |
| | 209ba10 | `train.py`：config key `bootstrap_seed`，固定 train/test split，和 `seed` 分開 |

- Tests：`pytest tests/` → 21 passed, 1 xfailed（2026-10-14）。
- 沒修（upstream 已在處理）：C `bootstrap: false`（issue #4）、D `validation.py`（#5，on hold）、E（PR #10 已修）。
- 還沒 port 的 dev 功能：wandb、plotEvery、optuna `tune.py`、`prepare_config.py`、latex。shap 已 port（7b0126d）；collect_plots 有 local 版本（`local/collect_plots.py`）。
- 3bcc25e、209ba10 的 commit message 還是 `Co-Authored-By:`，還沒改成 `Assisted-by:`。

`fix/crashes` 沒有改寫，表中前三個 hash 在那個 branch 上仍然正確；`port/dev` 上對應的是改寫後的 045b3dd、bccb0ab、b31f197。

### Commit hash 對照（2026-10-13）

`port/dev` 的 commit message 統一改成 `Assisted-by: claude-code:claude-opus-5-5`，程式碼沒變，但 hash 全部換了。**run 名稱、output 目錄、EOS 路徑、condor configs 和 worktree（`topsbi-wt/port-a6ff4bd`、`port-eae2959`）都沿用舊 hash**，因為那是當時實際用來訓練的 code，舊 commit 仍可從這兩個 worktree 取得。之後的 `port-209ba10` 用的是改寫後的新 hash。

| 舊 | 新 | commit |
|---|---|---|
| 9dc4e4a | 045b3dd | Fix ensemble mean and full_likelihood without an sm network |
| cb871fc | bccb0ab | toy_builder: count accepted events, not batches |
| 540354f | b31f197 | Fix double feature standardisation in train.py |
| 045f8a7 | c796612 | stitched: compute get_probabilities in float64 like get_weights |
| 97cb288 | e2e3b47 | Turn dropout off whenever the network is evaluated |
| 6a898ab | 0004d96 | stitched: drop events with non-positive or extreme weights |
| a6ff4bd | 8e4bdc3 | Train on config['device'] and evaluate without autograd |
| 482bf4d | 82b6d7e | Add 1D likelihood scan of the ensemble against the analytic truth |
| e4a12f4 | 7b0126d | Add SHAP feature ranking for a trained network |
| 44fc479 | dabf409 | Weight calibration and kinematic plots by p0 |
| eae2959 | dbcb604 | scan.py: closure against pseudo-data reweighted to any point |
| e01fb36 | 23da3fd | Morph the unnormalized ratio in full_likelihood |

## Environment

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH=$PWD/src:$PYTHONPATH   # 要 append，覆蓋掉會找不到 torch
```

Local 檔案都放在 `local/`（已加進 `.git/info/exclude`）。

## Data（main 格式）

`/eos/cms/store/user/ykao/topsbi/pretraining_main_highlevel/{train,test,validation}.p`

```bash
python3 local/convert_pretraining.py <dev pretraining dir> <out dir> features_highlevel.yml
```

- 只保留 13 個 highlevel features（float32），順序和 `features_highlevel.yml` 相同；ids 欄位補零。
- **Off-diagonal coefs ×√2**：main 的 `get_lower_tri(off_diag=1)` 才會和 dev 的 weights 一致。可用 w(cg) 的 std/mean 驗證：0.007 是對的，0.46 是錯的。

## Training

```bash
# 從 condor/configs 的 dev configs 產生 port configs → local/condor/configs/<tag>_<point>.yml
python3 local/make_configs.py <tag> <point> [<point> ...]
# 單一 job，在 local 跑（要 GPU，或把 config 裡的 device 改成 cpu）
python3 -m topsbi.train local/condor/configs/<tag>_<point>.yml
```

Output 位置：`/eos/cms/store/user/ykao/topsbi/results/port/<tag>/<point>/seed_2`

## Condor (lxplus)

Jobs 從凍結的 worktree 讀 code，所以之後在 repo 裡切 branch 不會影響正在跑的 jobs。Submit file 不能含 /eos 路徑，所以要從 AFS 送出。

```bash
git worktree add --detach ../topsbi-wt/port-<sha> <sha>
cd ~/topsbi-condor          # run.sh, submit.sub, submit_list.sub
# 用 glob 送出
condor_submit code=/eos/home-y/ykao/SWAN_projects/analysis/topsbi-wt/port-<sha> \
    configs="/eos/home-y/ykao/SWAN_projects/analysis/topsbi/local/condor/configs/<tag>_*.yml" submit.sub
# 或用 list 送出（每行一個 config 路徑）
condor_submit code=<worktree> configlist=<list.txt> submit_list.sub
```

Resources：1 GPU（Capability ≥ 7.5）、16 GB、longlunch；每個 job 約 5 分鐘（A100/H100）。

## Checks

```bash
# 單一網路 vs truth（corr(log r)、E_SM[r̂]），dev 和 port 並排比較
python3 local/compare_networks.py local/condor/configs/a6ff4bd_SM_cQj31.yml \
    dev=/eos/cms/store/user/ykao/topsbi/results/fastTrain/gen/highlevel_n256_dropout_cosine/SM_cQj31/seed_2 \
    port=/eos/cms/store/user/ykao/topsbi/results/port/a6ff4bd/SM_cQj31/seed_2
# SHAP feature ranking（預設用 validation.p，CPU 約 1 分鐘）→ <name>/complete/shap_{summary.png,ranking.txt}
python3 scripts/shap_features.py local/condor/configs/a6ff4bd_SM_cQj31.yml [-o <outdir>]
# 把一個 tag 的訓練圖整理到網頁（每種圖一個資料夾，含 index.php 和 performance.csv；重跑只複製新的檔案，456 個網路約 30 分鐘）
python3 local/collect_plots.py /eos/cms/store/user/ykao/topsbi/results/port/<tag> /eos/home-y/ykao/www/topsbi/results_<date>_<tag>
# Ensemble closure scan（lxplus 沒有 GPU：ensemble 裡的 configs 要用 device: cpu）
python3 scripts/scan.py -p local/ensemble_a6ff4bd.yml --wc <WCs> --truth --threshold 0 \
    --drop-negative --others cg -o output/port_a6ff4bd_closure
```

## Results (a6ff4bd, 152 networks)

- 單一網路 port ≈ dev：SM_cQj31 corr 0.660 vs 0.657；cQd8_cQd8 0.265 vs 0.266。
- **Ensemble closure 失敗，dev 和 port 都一樣**：
  - truth 的最小值全部落在 cg（16/16），但 ensemble 的 best fit 偏差 ~1–2，68% 區間卻只有 ~±0.02（系統性偏差）。
  - port 有 8/16 個 WC 撞到 scan 範圍的邊界。
  - 結果在 `output/port_a6ff4bd_closure/`；dev 的對照在 `output/scan_all_closure_subset/`。
- 推測（未驗證）：lstsq morphing 外推到 cg 時放大了少數網路的偏差；136/152 個 training points 有 w(c1) ≤ 0 的 events。

## Next

1. 152 個網路逐一跑 `compare_networks.py`，找出 E_SM[r̂] 或 corr 異常的網路。
2. 把 truth r 餵進 lstsq：如果 closure 成立，問題在網路；如果不成立，問題在 training points 或 morphing 本身。

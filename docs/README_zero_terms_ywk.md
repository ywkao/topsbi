# Zero terms：用 quantile bins 選出要丟掉的項（2026-10-28）

會議（2026-10-27）建議用 quantile bins 取代全域的 RMS 來選 terms。全域 RMS 會把「只在某個區域很大、其他地方抵消」的項藏起來。

## 方法

和 upstream `topsbi.tools.data.find_zero_terms`（eddie/main `c1e4b70`）相同：
- 每個 feature 切成 20 個等量（equal-count）quantile bins；離散的 feature 會少於 20 個。
- 在每個 bin 裡，算第 q 項的大小 Σ|s_q| / Σ|s_SM|。s_q 是 `coefs` 第 q 欄，也就是 structure constant。取絕對值，所以正負不會抵消。
- 一項的 **size** = 所有 feature、所有 bin 裡最大的那個比值。
- size < tol 的項就丟掉。upstream 預設 tol = 1e-6，即小 6 個數量級。

upstream 的函式只回傳要丟的清單。`local/zero_terms_quantile.py` 用同樣的算法，另外把每一項的 size、最大值出現在哪個 feature 和 bin 都輸出。

## 指令

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
cd /eos/home-y/ykao/SWAN_projects/analysis/topsbi
python3 local/zero_terms_quantile.py \
    /eos/cms/store/user/ykao/topsbi/pretraining_main_highlevel_cos_log_pilot \
    features_highlevel_cos_log.yml \
    local/condor/configs/fullset_s1/fullset_s1_SM_SM_seed2.yml \
    -o local/zero_terms_hlcos --tol 4e-3          # 約 1 分鐘
```
- 參數依序：資料目錄（讀 `train.p`）、要切 bin 的 features、任一個訓練 config（只用來讀 `wcs` 的順序）。
- 選項：`--tol`（預設 1e-6）、`--n-bins`（預設 20）。
- 輸出：
  - `local/zero_terms_hlcos.csv`：153 項，照 size 由大到小。欄位：`term`、`kind`（sm/linear/quadratic/cross）、`size`、`inclusive`（不分 bin、全體加總的比值）、`feature`、`bin`（最大值的位置）、`dropped`。
  - `local/zero_terms_hlcos.yml`：要丟的 pair，格式就是 config 的 `zero_terms`。

## 結果（highlevel_cos pilot 樣本，train.p）

1. **tol = 1e-6 一項都不丟。** 最小的 size 約 2.6e-4：每個 event 的二次 fit 係數帶有雜訊，取絕對值後不會抵消，所以「LO 下應為 0」的項也停在約 3e-4。1e-6 適用於結構上真正為 0 的樣本。
2. **size 分布有一段乾淨的空白：**

   | size | 交叉項數目 |
   |---|---|
   | 2.6e-4 – 1.1e-3 | 64（雜訊底線） |
   | 1.1e-3 – 1.6e-2 | 0 |
   | ≥ 1.6e-2 | 56 |

   門檻落在空白裡（約 2e-3 – 1e-2）結果都一樣：丟 64 項，留 1 + 16 + 16 + 56 = **89 項**，和同事從 153 逐步刪到的 89 相同。所以用 `--tol 4e-3`。
3. **和 rms_ev 分群（`cross_terms_survey.py`，5%）比較：**
   - 丟掉的 64 項全部在 rms_ev 的「小」72 項裡。
   - 留下的 56 個交叉項 = 強 27 + 中 21 + 8 個 rms_ev 判為「小」的 ctGRe 交叉項（ctGRe×ctGIm、ctGRe×cQj11、×cQj31、×cQu1、×cQd1、×ctj1、×ctu1、×ctd1）。rms_ev 乘了 5% 時的 c 值，而 ctGRe 的 c 只有 0.15，所以看起來小；quantile size 只看係數本身。
4. **幾乎所有項的最大值都在 `log_m_ttbar` 的最後一個 bin**（高 m_tt 尾巴），在全體加總裡會被稀釋。最大的項是 cQj11×cQj31（size 3.6）。

## 怎麼用

在訓練或 validation 的 config 加：
```yaml
zero_terms: /eos/home-y/ykao/SWAN_projects/analysis/topsbi/local/zero_terms_hlcos.yml
```
- 訓練：這些欄位從權重裡拿掉。fit：這些 γ 被 mask 掉，設為 0。
- **同一個 ensemble 的所有網路（包括 SM_SM）的 `zero_terms` 必須相同**，否則 `full_likelihood` 會報錯。現有的 357 個網路訓練時沒有 `zero_terms`，所以這個檔案是給下一輪訓練用的。
- zero terms 取決於樣本：換成 reco samples 後要重跑。

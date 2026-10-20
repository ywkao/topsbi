# Batch 2：feature set 比較（cluster 12707721，32 jobs，2026-10-17 送出）

一頁看完這批訓練：設定、為什麼這樣設、可以做哪些 studies、做了哪些、還缺什麼。細節和數字的出處：
- feature sets 的定義、資料製作、Fisher 估計為什麼不可行：`README_featsets_ywk.md`
- 完整的結果表：`README_featsets_results_ywk.md`
- highlevel 基準來自 pilot：`README_batch_pilot_ywk.md`

## 目的

選出 morphing 網路要用的 feature set，結果要能放進 AN：同樣的訓練、同樣的 events，只換 features，比較網路抓到多少資訊。

## 設定

### 程式碼和 job

- code：`/eos/home-y/ykao/SWAN_projects/analysis/topsbi-wt/port-0c49da2`（和 pilot 相同）
- configs：主目錄 `local/condor/configs/pilot_{top30,compact,lowlevel,all}/`
- condor list：`~/topsbi-condor/list_featsets_5p.txt`
- 輸出：`/eos/cms/store/user/ykao/topsbi/results/port/0c49da2_pilot_<set>/<wc>_5p/seed_<n>`

### 程式碼版本和方法（2026-10-20 確認）

- **程式碼**：commit `0c49da2`（worktree `topsbi-wt/port-0c49da2`；port/rebased 在 2026-10-17 rebase 到 eddie/main c810b07 之後的版本，0c49da2 本身只改 READMEs）。
  - 確認方式：condor 的 `run.sh` 會印出 `git log --oneline -1`，三個 clusters 的 `~/topsbi-condor/logs/<cluster>.0.out` 都是 `0c49da2 update READMEs`：12702676（pilot，含 highlevel 基準）、12707721（top30、compact、lowlevel、all）、13369189（highlevel_cos）。
- **方法：`weights_only`**（`pilot/`、`pilot_*/` 的 95 個 configs 都是），`lumi: [1.0]`、`year_loc: null`、`group_cols: []`，holdout `validation.p`。網路學 r = w(c1)/w(SM)，包含 rate。
- 訓練設定：`wo_log_bs10k` 模板（batch 10000、cosine、lr 1e-3、256×3、dropout 0.2、100 epochs、patience 20）。
- 六組的程式碼、方法、訓練設定完全相同，只差 `data`、`features`、`holdout`、`name`。

### Feature sets

| 名稱 | 數量 | 資料（`/eos/cms/store/user/ykao/topsbi/`） | 來源 |
|---|---|---|---|
| highlevel（基準） | 13 | `pretraining_main_highlevel_log_pilot/` | pilot（cluster 12702676） |
| top30 | 30 | `pretraining_main_top30_log_pilot/` | 這批 |
| compact | 26 | `pretraining_main_compact_log_pilot/` | 這批 |
| lowlevel | 31 | `pretraining_main_lowlevel_log_pilot/` | 這批 |
| all | 74 | `pretraining_main_all_log_pilot/` | 這批 |

- 同樣的 log 規則和 w ≤ 0 filter（見 pilot）。
- **五組的 events 完全相同、順序相同**（coefs 逐筆比對過），所以同一個 seed 的 train/test 切分也相同。

### 訓練

- 只訓練 **5% 點**（4 個 WC，WC 值見 pilot 的表）× **seed 2、3** = 4 組 × 8 = 32 jobs。原本規劃全部點 × 3 seeds（204 個），縮減成只看最有訊號的點。
- configs 和 pilot 完全相同，只差 `data`、`features`、`features_to_animate`（lowlevel 沒有 m_ttbar，改畫 `log_lep_top_pt`）、`holdout`、`name`。
- 基準：pilot 的 `<wc>_5p/seed_{2,3}`。

### 設計的取捨

- **配對比較**：五組在同樣的 holdout events 上評估，per-event 的雜訊大部分抵消，所以 2 個 seeds 就夠分辨（Fisher 估計就是因為 per-event 雜訊太大才失敗）。
- **只用 5% 點**：訊號最大，feature 間的差異最明顯；代價是這些網路不能 morph、不能 scan。

## 可以做的 studies

| study | 狀態 | 結果 / 位置 |
|---|---|---|
| holdout BCE gain、frac（相對 truth）、配對 delta | ✅ | `local/compare_featsets.py`；`www/topsbi/featsets_trained_20261018/` |
| auc、pull、rms_bias 比較 | ✅ | 除了 all 在 ctGIm，各組差不多 |
| 找出 ctGIm 的資訊來自哪些 features | ✅ | `cos_{lep,had}_{n,r,k}` → highlevel_cos |
| 確認 highlevel_cos | ✅ | ctGIm gain 是 highlevel 的 4.2 倍（all 的 1.8 倍），其他 WC 沒有損失 → default feature set |
| 用其他 feature set 做 morphing / scan | ❌ 不行 | 只有 5% 點 |

### 主要結論

- **ctGIm：all 的 gain 是 highlevel 的 2.8 倍**（delta +4.2 到 +4.4 × 1e-4，超過誤差 20σ），來自只有 all 才有的 `cos_{lep,had}_{n,r,k}`（CP-odd 自旋關聯的原料）。
- **CP-even 的 WC（ctGRe、cQj11、cQd8）：highlevel 最好**，其他組低 1–8%。
- **lowlevel 最差**：ctGRe 什麼都沒學到，其他 WC 少 25–45%。
- 建議 **highlevel_cos** = highlevel + 6 個 cos（19 個 features）。

## 後續：highlevel_cos（cluster 12948860，2026-10-18）

- 資料：`pretraining_main_highlevel_cos_log_pilot/`（從 all 取出 19 欄，前 13 欄、coefs 和 highlevel 完全相同）
- configs：`local/condor/configs/pilot_highlevel_cos/`（4 WC × seed 2、3 = 8 個），list：`~/topsbi-condor/list_featsets_highlevel_cos.txt`
- **狀態：8 個 jobs 全部在開始幾秒內失敗**（2026-10-18 17:57）：送出時 `code=../topsbi-wt/port-0c49da2` 是相對路徑，worker 上不存在（`cd: No such file or directory` → `No module named 'topsbi'`）。沒有輸出。
- 重送（用絕對路徑，和前兩批一樣）：
  `cd ~/topsbi-condor && condor_submit code=/eos/home-y/ykao/SWAN_projects/analysis/topsbi-wt/port-0c49da2 configlist=list_featsets_highlevel_cos.txt submit_list.sub`
- **已重送：cluster 13369189**（2026-10-19，8 jobs，絕對路徑）。
- **結果（2026-10-20，`www/topsbi/featsets_trained_20261020/`）**：ctGIm 的 delta +7.7 / +7.9（all 是 +4.2 / +4.4），cQj11、cQd8 和 0 一致，ctGRe −0.006 / −0.008（gain 的 2–3%）。細節見 `README_featsets_results_ywk.md`。
- **highlevel_cos 是接下來訓練的 default**：`local/plans/hlcos.yml`（±c 點，見 `README_jupyter.md`）。

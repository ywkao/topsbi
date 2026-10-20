# Batch 1：pilot（cluster 12702676，51 jobs，2026-10-17 送出）

一頁看完這批訓練：設定、為什麼這樣設、可以做哪些 studies、做了哪些、還缺什麼。細節和數字的出處：
- closure、scan、診斷 1–4：`README_closure_pilot_ywk.md`
- 資料處理（w ≤ 0、為什麼不 sanitize）：`README_weights_ywk.md`
- feature set 比較（用到這批的 5% 網路）：`README_batch_featsets_ywk.md`

## 目的

1. 用 rebase 之後的程式碼（`port-0c49da2`）和 upstream 推薦的 `weights_only`，訓練一組**可以 morph 的網路**（SM_SM + 每個 WC 數個點），測試整條 morphing → scan 的流程。
2. 順便提供 feature set 比較的 highlevel 基準（5% 點）。

## 設定

### 程式碼和 job

- code：`/eos/home-y/ykao/SWAN_projects/analysis/topsbi-wt/port-0c49da2`（固定的 worktree，不要動）
- configs：主目錄 `local/condor/configs/pilot/`（CPU 版在 `local/cpu_configs/pilot/`，`device: cpu`）
- condor list：`~/topsbi-condor/list_pilot.txt`
- 輸出：`/eos/cms/store/user/ykao/topsbi/results/port/0c49da2_pilot/<label>/seed_<n>`，label = `<wc>_<pct>p` 或 `SM_SM`

### 方法

- `method: weights_only`：網路學 r = w(c1)/w(c0)，**包含 rate**。
- `lumi: [1.0]`、`year_loc: null`、`group_cols: []`。
- c0 = SM。
- `holdout`：`validation.p`，訓練完全沒用到，最後的 metrics 都在它上面算。

### 資料和 features

- 資料：`/eos/cms/store/user/ykao/topsbi/pretraining_main_highlevel_log_pilot/`
- **只去掉 w ≤ 0 的 events**（train / test / validation 各 116 / 13 / 21 個，數值上都是 0）。不做 cap、lr cut 等其他 sanitize，原因見 `README_weights_ywk.md`。
- features：highlevel_log，13 個（`features_highlevel_log.yml`）。log 規則：m_ttbar 取 log；pt_tt、HT、mT_W、`*_pt`、`*_mass`、`m_*` 取 log1p。

### 訓練超參數（`wo_log_bs10k` 模板）

| | |
|---|---|
| network | 256 × 3 隱藏層，LeakyReLU，dropout 0.2（第 2、3 層和輸出層），Sigmoid 輸出 |
| batch | 10000（降低 seed 間差異） |
| lr | 1e-3，cosine scheduler，warmup 5 epochs |
| epochs | 100，early stopping patience 20 |
| train/test | `bootstrap: true`，由 seed 決定 75/25 切分 |
| seeds | 2、3、4 |

### 訓練點

每個 WC 單獨動（其他 15 個 = 0），取 **constructive root**：yield ratio Y(c)/Y(SM) = 1 + 0.5 / 1 / 2.5 / 5%。WC 值（`local/wc_targets_pilot_train.yml`，用所有 splits 解出）：

| WC | 0.5% | 1% | 2.5% | 5% |
|---|---|---|---|---|
| ctGIm | −0.2859 | −0.4047 | −0.6405 | −0.9062 |
| ctGRe | −0.0158 | −0.0314 | −0.0778 | −0.1532 |
| cQj11 | −0.4535 | −0.6543 | −1.0531 | −1.5027 |
| cQd8 | 0.8099 | 1.402 | 2.7044 | 4.2535 |

另外 **SM_SM**（c1 = c0 = SM，理想輸出 = 1），當作 upstream `full_likelihood` / `ensemble` 需要的 `sm` 網路（lstsq 的 SM 列）。

jobs：(4 WC × 4 點 + SM_SM) × 3 seeds = 51。

### 設計的取捨

- **四個 WC 的代表性**：ctGIm（CP-odd）、ctGRe（CP-even、二體算符）、cQj11（四費米子 color singlet，yield 幾乎只有二次項，5% 時 b·c 佔 4%）、cQd8（color octet，和 SM 的 qq̄ → g → tt̄ 有干涉，5% 時 b·c 佔 41%）。
- **以 yield 定點**：不同 WC 的點在「訊號大小」上可比較。
- **只取一個正負號（constructive）**：destructive 側的 root 有些很遠（ctGRe 約 4.7），shape 變化很大。代價見下面「限制」。
- **四個點**：可以用其中三個 morph（{SM_SM, 1%, 5%}），另外兩個（0.5%、2.5%）當 hold-out closure 的 dedicated 網路。

## 可以做的 studies

| study | 狀態 | 結果 / 位置 |
|---|---|---|
| SM_SM null test | ✅ | rms_bias 0.00125，通過。**看 rms_bias，不看 pull**（c0 = c1 時 truth 的統計誤差是 0，pull 沒意義） |
| 單點網路的校準（performance.yml 的 rms_bias、pull vs \|c\|） | 部分 | cQj11、cQd8 在 2.5% 約 0.01，ctGRe 約 0.001 |
| hold-out closure：{SM_SM, 1%, 5%} morph 到 0.5%、2.5%，和 dedicated 比較 | ✅ 通過 | parametric 的 rms_bias ≈ dedicated；`www/topsbi/closure_pilot_20261018/` |
| 1D scan（weights_only extended likelihood，seed ensemble） | ✅ | `all` 和 `1p5p` 結果一樣；`www/topsbi/scan_pilot_20261018/` |
| 診斷 1：rate / shape 分解 | ✅ | 偏差來自 shape 項，rate 正確 |
| 診斷 2：重新校準 r̂、子樣本漲落 | ✅ | 偏差和 MC 統計漲落同樣大 |
| 診斷 3：bootstrap closure 不確定度 | ✅ | 見下 |
| 診斷 4：gammas vs 真實 α、β | ✅ | 見下 |
| seed 間差異、ensemble 隨 seed 數的改善 | ❌ 還沒做 | 有 3 個 seeds 可以做 |
| pseudo-data 放在 SM（c = 0） | ❌ 還沒做 | |
| feature set 比較的 highlevel 基準 | ✅ | 5% 點的 seed 2、3（見 `README_batch_featsets_ywk.md`） |

### 主要結論

- **closure（bootstrap，pull =（NN − 期望）/ std）**：
  - cQj11、ctGIm：通過（\|pull\| < 1）。
  - ctGRe：固定的小偏移（\|c\| 大 0.0005–0.0009，pull −6 到 −17），但只有 scan ci68 的約 0.1σ，目前可以忽略。
  - cQd8：0.5% 點偏差顯著（pull +4.5），1%、2.5% 在 2σ 以內。
- **MC 樣本太小**：四費米子在 data 統計精度下無法驗證（cQd8 2.5% 的 MC 漲落 0.22，比 scan 的 ci68 半寬 0.05 大 4 倍）。
- **NN 的 scan 區間比 truth 寬**（ctGIm 3.1 倍）：NN 只看 reco features，「吻合」的標準是 best fit 沒偏差。

## 限制（下一批訓練要改的）

1. **一次項和二次項分不開**（診斷 4 的根本原因）：所有點都在同一側，lstsq 解出的 γ1、γ2 互相補償。
   - 沒學到的係數：ctGRe 的 γ2、cQj11 的 γ1、ctGIm 的 γ1。
   - 改善：四費米子、ctGIm 加 **±c 對稱**的點；ctGRe 加**較大 \|c\|** 的點。
2. γ̂ 的尾端太分散（cQd8 最低的 γ1 bin 正負號都錯）→ 更多 seeds。
3. highlevel 看不到 ctGIm 的 CP-odd 資訊 → highlevel_cos（見 `README_batch_featsets_ywk.md`）。
4. 每個 WC 單獨訓練，沒有交叉項，不能做 2D scan。

## 後續：±c 點的 weights 和 LR 分布（2026-10-20）

為了決定下一批的第二個點用 −c（`_neg`）還是 destructive root，直接看 truth weights（不用網路）。

- 圖：`www/topsbi/weights_20261020_pm_points/`，每個 WC 一張 png，列 = 0.5/1/2.5/5%；藍 = +c（constructive），綠 = −c，紅虛線 = destructive。
  - 第 1 欄：|w1| / mean(w0_SM) 的分布。
  - 第 2 欄：每個 event 的真實 log10 LR = log10(w1/w0)（只畫 w1 > 0）。
  - 第 3 欄：**逐 bin** 的 Σw1/Σw0（每個 m_ttbar bin 內的 events 加總後相除 = 該 bin 的 yield ratio；超出邊界的 events 放進最邊的 bin）。
  - `summary.txt`：每個點的 c、1 + b·c + a·c² 的各項、n_eff/N、w1 ≤ 0 的比例。
- 資料：`pretraining_main_highlevel_cos_log_pilot/train.p`（只去掉 w ≤ 0，沒有其他 sanitize）。
- 重現：
  ```
  source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
  export PYTHONPATH=/eos/home-y/ykao/SWAN_projects/analysis/topsbi-wt/port-0c49da2/src:$PYTHONPATH
  cd /eos/home-y/ykao/SWAN_projects/analysis/topsbi
  python3 local/plot_pm_points.py /eos/cms/store/user/ykao/topsbi/pretraining_main_highlevel_cos_log_pilot \
      local/wc_targets_pilot_train.yml /eos/user/y/ykao/www/topsbi/weights_20261020_pm_points [wc ...]
  ```
  不給 wc 時畫 ctGIm、ctGRe、cQj11、cQd8。

ctGRe 5% 的分解（a = 0.0676，b = −0.3159）：

| 點 | c | 1 + b·c + a·c² | Y/Y_SM | n_eff/N |
|---|---|---|---|---|
| +c（constructive） | −0.1532 | 1 + 0.0484 + 0.0016 | 1.050 | 0.77 |
| −c（`_neg`） | +0.1532 | 1 − 0.0484 + 0.0016 | 0.953 | 0.75 |
| destructive（舊 `<wc>_<wc>`） | +4.8252 | 1 − 1.5243 + 1.5743 | 1.050 | **0.05** |

觀察：
- **ctGRe**：±c 幾乎互為鏡像，LR 只跨 1–2 個數量級，m_ttbar 上是平的 1.05 / 0.95；destructive 的 LR 跨 8 個數量級，高 m_ttbar 到 7 倍 → 不適合當訓練點。
- **cQd8**：±c 的差異就是一次項的資訊。−c 多了低側的尾巴（LR 到 10⁻³，干涉相消），+c 只有往上的尾巴；−c 的總 yield 幾乎不變（5% 時 +0.9%），但 shape 有變。高 m_ttbar 的上升 ±c 差不多（二次項），destructive 陡約 3 倍（5% 時 65 vs 23 倍）。n_eff/N 三者都 0.68–0.78。
- **cQj11**：−c ≈ destructive（b 很小）。三者的 LR 分布幾乎一樣，最高到 log10 LR ≈ 5.3（5%），n_eff/N 在 5% 時 0.54–0.62。
- **ctGIm**：−c 和 destructive 完全重合（b ≈ 0）。±c 的 LR 分布相同，但 m_ttbar > 1.3 TeV 時分開（5% 時 +c 1.27、−c 1.20）：per-event 的一次項在 m_ttbar 上不是 0（積分約 0）。CP-odd 算符在 CP-even 變數上不應該有一次項，可能來自 per-event 二次 fit 的 coefs，還沒查。
- 結論：第二個點用 **−c（`_neg`）**，見 `local/plans/hlcos.yml`、`README_jupyter.md`。

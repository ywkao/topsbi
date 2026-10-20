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

- **四個 WC 的代表性**：ctGIm（CP-odd）、ctGRe（CP-even、二體算符）、cQj11、cQd8（四費米子，yield 分別以一次、二次項為主）。
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

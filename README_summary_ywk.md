# 開發與 closure 總結（ywk, 2026-10-13，2026-10-16 更新）

所有變更都只在 local，沒有 push。細節見 `README_migration_note_ywk.md`（branch、環境、condor）和 `README_closure_ywk.md`（closure 的指令和 TODO）。

## 1. 開發：依序解決的問題

Base：`eddie/main` @ 6115263。下表是 2026-10-13 改寫 trailer 之後的 hash；run 名稱和路徑（例如 a6ff4bd、eae2959）沿用舊 hash，對照見 `README_migration_note_ywk.md`。

| # | commit | 問題 → 修正 | 影響 |
|---|---|---|---|
| 1 | 045b3dd | `ensemble` 把各 seed 的 gammas 加總而不是平均（r(SM) = N）；`full_likelihood` 沒有 `sm` 時 crash | 多 seed ensemble 的結果錯誤 |
| 2 | bccb0ab | `toy_builder` 計數的是 batches，不是 accepted events | toy 的 event 數錯誤 |
| 3 | b31f197 | `train.py` 對 features 做了兩次標準化（merge c36ce8c 帶回來的） | main 自己的 test fail |
| 4 | c796612 | stitched 的 `get_probabilities` 改用 float64 | 避免 float32 的精度損失 |
| 5 | e2e3b47 | evaluation 時沒關 dropout | r̂ 有隨機性、不可重現 |
| 6 | 0004d96 | `sanitize_mask`：丟掉 w ≤ 0 或極端 weights 的 events | BCE 不再被少數 events 主導 |
| 7 | 8e4bdc3 | 依 `config['device']` 訓練、evaluation 用 `no_grad`、用 `BatchSampler` | 訓練約 5 分鐘一個網路 |
| 8 | 82b6d7e | 新增 `scripts/scan.py`：ensemble vs truth 的 1D likelihood scan | closure 測試工具 |
| 9 | 7b0126d | 新增 `scripts/shap_features.py` | feature ranking |
| 10 | dabf409 | calibration 和 kinematics 圖用了錯誤的權重（比的是 mean(p1/p0)，不是網路學的 Σp1/Σp0）；r 和 r̂ 的 legend 標反 | 上週看到的「calibration 和 kinematics 不一致」其實是畫圖的 bug |
| 11 | dbcb604 | `scan.py --data-values`：可以把樣本 reweight 成任意點的 pseudo-data | 近 SM closure |
| 12 | 23da3fd | **`full_likelihood` 拿 mean-normalized 的 ratio 做 quadratic morphing**，但 normalized ratio 不是二次式 | 離訓練點越遠偏差越大（見第 2 節） |

- 1–3 在 `fix/crashes`；4–12 在 `port/dev`。
- 10 和 12 也是 main 本身的 bug，開 PR 前應該移到 `fix/crashes`。
- Tests：21 passed、1 xfailed；ruff 全過。

其他完成的工作：
- Data 轉成 main 格式（13 個 highlevel features，off-diagonal coefs ×√2）。
- `local/wc_targets.yml`：每個 WC 在 yield 變化 5/10/20% 時的值（用 train+test+validation 合計算出）。
- 訓練了 152 個網路（a6ff4bd，5%）和 456 個網路（eae2959，5/10/20% × 1 seed），全部成功。
- 文件：`README_migration_note_ywk.md`、`README_jupyter.md`、`README_closure_ywk.md`。

### 還是 open 的問題

| 問題 | 說明 |
|---|---|
| **近 SM closure 失敗** | 主要的 open 問題。**主要來源已定位：網路在 m_ttbar > 1 TeV 的 calibration**（嘗試 19、20）。1.3 TeV 以上 r̂ 高估 8% 到數倍，1.0–1.3 TeV 低估 1–6%。只用 m_ttbar < 1300 GeV 時 cQj11 偏差 ≤ 3.5%；cQd8 翻成 +4 ~ +6% |
| cg closure 失敗 | 先放一邊：近 SM 都還沒 closure，cg 只會更差 |
| 高 r̂ tail | r̂ > 10 的 events，網路預測約 30–40，實際約 20；網路普遍高估 E[r̂] 約 4%。**就是高 m_ttbar 的外推**（嘗試 19）：線性標準化後 8.5 TeV 是 z = 18，LeakyReLU 網路在那裡線性外推 logit，r̂ 指數成長 |
| ctGIm 近 SM 偏小約 5% | 可能是 highlevel features 缺少 CP-odd 的觀測量 |
| 訓練不穩定 | 同一個點不同 seeds 的 log r̂ 逐 event 差約 12%。4 種訓練設定中只有 batch 10000 降低 20%（嘗試 17）。seed 間 best fit 的差異大部分也來自 tail 外推（嘗試 20） |
| `--drop-negative` 依賴掃描範圍 | 丟掉的 events 隨範圍改變，best fit 也跟著變（cQu1：1.665 vs 1.585） |
| 0.1% 的點 | 低於 validation 樣本（18 萬個 events）的靈敏度，closure 沒有意義（嘗試 16） |
| `validation.py` 的 `kinematic_ratio_plot` | 和 dabf409 一樣的權重問題（upstream 正在重寫，issue #5） |
| `bootstrap: false` | 已知問題（upstream issue #4），沒修 |
| 還沒 port 的 dev 功能 | wandb、plotEvery、optuna、`prepare_config.py`、latex（collect_plots 有 local 版本 `local/collect_plots.py`） |

## 2. Closure：依序的嘗試

### 背景

- **Closure 的意思**：用 pseudo-data 做 likelihood scan，−2ΔlnL 的最小值應該落在產生 data 的真實點上（Gibbs inequality 保證期望值在真實點最小）。
- **cg closure**：樣本在 cg 產生（16 個 WC 都不為零：ctGIm、ctGRe = −0.5，其他 = 1.5）。掃描單一 WC，其他 WC 固定在 cg。
- **Truth 曲線**用解析的 p1/p0，每次都 16/16 落在 cg，所以 scan 本身沒問題。
- **Morphing**：r̂(c) = Σᵢ aᵢ(c) r̂ᵢ，由各訓練點的網路用 lstsq 擬合二次式。a(c) 只由訓練點的設計決定；‖a‖₂ 是獨立誤差的放大倍數，Σ|aᵢ| 是共同 bias 的最壞放大倍數。

### 嘗試和結果

| # | 嘗試 | 結果 | 得到的資訊 |
|---|---|---|---|
| 1 | 152 個網路（5%）cg closure，dev 和 port 都跑 | 0/16；best fit 偏 1–2，68% 區間只有 ±0.02 | 系統性偏差，不是統計漲落；port 的程式碼和 dev 行為一致 |
| 2 | 修正 calibration/kinematics 的權重（dabf409） | 網路本身看起來比之前以為的好得多（chi_excl 8235 → 87） | 之前「網路不準」的印象一部分來自畫圖的 bug |
| 3 | 近 SM closure（data 在 5%、2.5%、10% 的點，其他 WC = 0），只跑 ctGIm、ctGRe | ctGRe 三種情況都在 68% 內；ctGIm 偏小約 5% | 訓練點附近的網路基本上是準的 |
| 4 | 計算不同訓練設計在 cg 的 ‖a‖₂ | 5% × 1：22–30；5/10/20%：3.8 | 推測失敗來自外推放大誤差 → 訓練 456 個網路 |
| 5 | 456 個網路（5/10/20%）cg closure | 0/16，沒改善（10/16 撞到邊界） | ‖a‖₂ 降到 1/6 卻沒改善 → **放大獨立雜訊不是主因** |
| 6 | 把 exact truth 餵進同一個 morphing | normalized p1/p0：0/16，失敗模式和網路一樣；unnormalized w(c1)/w(SM)：16/16 | **找到 morphing 的 bug**：normalized ratio 不是二次式 |
| 7 | 修 morphing（23da3fd）後重跑 456 個網路 | 仍然 0/16 | 修正是對的（truth 可以 closure），但網路還有別的誤差 |
| 8 | 每個網路先 self-normalize（E[p0 r̂] = E[p1]）再 morph | 仍然 0/16；縮放係數 0.58–1.06，平均 0.96 | 網路的整體 normalization 有偏差，但**不是 closure 失敗的原因**；剩下的是 event-dependent 的形狀誤差 |
| 9 | 檢查 data sanitization（0004d96）的影響：每個訓練點在 validation 上被丟掉多少 events 和 Σp1 | 最多丟 0.17% 的 events、0.1% 的 Σp1；預測的縮放偏移是 +0.6%，方向和觀察到的（−4%）相反，相關係數 −0.16 | **sanitization 不是原因**。但縮放偏差隨訓練點變遠而變大：5p 0.989、10p 0.973、20p 0.921 → 20p 的網路最不準 |
| 10 | 混合測試（`local/truth_morph_scan.py --mode hybrid --truth-for <regex>`）：先全部 456 個換成 truth 當 sanity check，再只把 152 個 20p 換成 truth | 全部換：16/16；只換 20p：仍然 0/16，而且 r ≤ 0 被丟掉的 events 從約 500 增加到約 15000 | 修正後的 morphing 在 truth 下完全正確；~~只有 20p 不準不是主因~~（**這個結論無效**，見嘗試 11） |
| 11 | 混合測試，依點的類型換：只換 360 個交叉點、只換 96 個單一 WC 的點 | 兩種都 0/16，r ≤ 0 被丟掉的 events 多達 19k–50k | **混合測試的設計本身有問題**：truth 是每個 event 的 r(x,z)，網路學的是 r(x) = E[r(x,z) \| x]，兩者是不同的函數（corr(log r) 約 0.66）。混在一起做 lstsq，無論網路好壞都會不一致。嘗試 10、11 都無法用來定位問題 |
| 12 | 近 SM closure 補齊 16 個 WC，兩個 ensemble × 3 個 data 點，用修正後的 morphing（`output/nearSM_fixed_{a6ff4bd,eae2959}_{on_5p,half_5p,10p}/`，表格：`python3 local/nearsm_table.py <dirs>`） | data 落在 ensemble 68% 內：a6ff4bd 3/16、3/16、3/16；eae2959 1/16、0/16、3/16。best fit 相對 data 偏 5–10%（rms）。a6ff4bd 有 12–13/16 偏向 SM（平均 −4~−5%）；兩個 ensemble 的偏差相關係數只有 −0.03~+0.22 | **之前「近 SM 可信」是只看 ctGRe/ctGIm 得出的，推廣錯了**。a6ff4bd 在 on_5p 時，r̂ 只由 SM_wc、wc_wc 兩個網路決定，所以單一網路本身就有偏差。兩次獨立訓練的偏差幾乎不相關 → 大部分是訓練的隨機誤差，多個 seeds 平均應該有幫助 |
| 13 | 近 SM closure 的 pull 用 pseudo-data 的有效 event 數修正（`local/neff_pulls.py <data.yml> <scan dirs>`） | N_eff/N 多數約 0.7–0.8（cQj31、cQj11 在 10p 低到 0.17），區間只放寬 1.1–1.2 倍（最多 2.4 倍）；修正後在 68% 內的仍只有 1–4/16 | **統計量不足不是原因**，偏差是真的 |

結果都在 `output/`：

| 嘗試 | 資料夾 |
|---|---|
| 1 | `port_a6ff4bd_closure/`（dev：`scan_all_closure_subset/`） |
| 3 | `port_a6ff4bd_nearSM_{on_5p,half_5p,10p}/` |
| 5 | `port_eae2959_closure/` |
| 6 | `truthmorph_eae2959_{norm,unnorm}/` |
| 7 | `port_eae2959_morphfix_closure/` |
| 8 | `port_eae2959_selfnorm_closure/`（縮放係數：`port_eae2959_selfnorm_closure_factors.txt`） |

### 目前的理解

| 可能原因 | 狀態 |
|---|---|
| scan 本身 | 排除（truth 每次都在 cg） |
| morphing 的 normalization | 已修（嘗試 6、7） |
| 外推放大獨立雜訊 | 不是主因（嘗試 5） |
| 網路的整體 normalization | 不是主因（嘗試 8） |
| data sanitization | 不是主因（嘗試 9） |
| **網路的形狀誤差，經過 morphing 累加到 cg** | 最可能 |
| 樣本按 event 數混合（stitched 除以 pg） | 不是近 SM 偏差的原因（嘗試 19）：metrics 只差 < 10%，closure 的 pseudo-data 也同樣除以 pg |
| **網路在 m_ttbar > 1 TeV 的 calibration** | **近 SM 偏差的主要來源**（嘗試 19、20） |

兩個加重問題的因素：
- **cg 離所有訓練點都很遠**：cg 的 16 個 WC 都不為零，而每個訓練點最多只有 2 個 WC 不為零。r̂(cg) 必須由全部 153 個網路組合而成，每個網路的誤差都會進來。
- **統計精度非常高**：truth 的 68% 區間只有 ±0.01–0.05（18 萬個 events），r̂ 只要有約 1% 的形狀偏差，就會讓 best fit 偏移好幾個 σ。

### 下一步（2026-10-16 更新）

1. **m_ttbar、pT 類 features 改用 log 後重訓**：4 個點 × 3 seeds（和 bs10k 同樣規模），看 1 TeV 以上的 calibration 和近 SM closure 是否改善。要改 feature 處理、送 condor。
2. 如果 log 不夠：增加 tail 的訓練統計（tail 加權或過採樣），或在 tail 截斷。
3. **修正 `--drop-negative` 依賴掃描範圍的問題**（tail 修好後，被丟掉的 events 應該也會變少）。
4. 決定之後用 `weights_only` 還是 `stitched`（upstream 推薦 `weights_only`，需要 era 資訊，但 highlevel 資料沒有 era）；這也決定 23da3fd 要不要 port 到 `port/rebased`。
5. 之後再回到 cg closure：leave-one-out 一致性、誤差容忍度、直接在 cg 訓練一個網路。

## 2026-10-13 晚上：overnight（`local/overnight_20261013.sh`，log：`output/overnight_20261013.log`）

- Condor：cluster 12543449（0.1%/1% 建設性設計，152 個網路，`results/port/constructive/`）、12543452（a6ff4bd 的 10 個 WC × 2 個點，seeds 3、4，40 個網路，`results/port/a6ff4bd/<point>/seed_{3,4}`）。
- lxplus，依序：
  1. 細格點（801 點、原本的範圍）近 SM closure：`output/nearSM_fine_{a6ff4bd,eae2959}_{on_5p,half_5p,10p}/`
  2. eae2959 在 1%/0.1% 建設性點的 closure（對照組）：`output/nearSM_fine_eae2959_{1p,0.1p}/`
  3. seeds：`output/seed_compare_a6ff4bd.txt`（逐 event 比較 r̂），`output/nearSM_fine_a6ff4bd_seeds{3,4,234}_*/`
  4. 建設性設計的 closure：`output/nearSM_fine_constructive_{1p,0.1p}/`；圖：`www/topsbi/results_20261013_constructive/`
- **新的 open issue：`--drop-negative` 的結果依賴掃描範圍。** 被丟掉的是「範圍內任何一點 r ≤ 0」的 events，範圍不同，丟掉的就不同。a6ff4bd on_5p 的 cQu1：原本範圍丟 996 個，best fit 1.665；只掃 data ±30% 丟 44 個，best fit 1.585（data 1.85；truth 兩次都是 1.85）。`output/test_window_cQu1_on_5p/`。所以今晚的 scan 都用原本的範圍，結果才能和之前比。

### Overnight 結果（2026-10-14）

| # | 嘗試 | 結果 | 得到的資訊 |
|---|---|---|---|
| 14 | 細格點（801 點）重跑近 SM closure | 相對偏差和 101 點最多差 1%；68% 內仍只有 0–3/16 | 格點不是原因；偏差是真的 |
| 15 | a6ff4bd 10 個 WC 的 `SM_wc`、`wc_wc` 多訓練 seeds 3、4（`output/seed_compare_a6ff4bd.txt`、`output/nearSM_fine_a6ff4bd_seeds{3,4,234}_*/`） | 每個 event 的 log r̂ 在 seeds 之間差約 11–13%（相對於 log r̂ 本身的大小）；best fit 的 seed 間 std 約 3.5–7%；seed 2 剛好是最差的（rms 6–12%，seeds 3、4 是 3–8%）；三個 seeds 平均後 rms 3.6–5.9%，比 seed 2 改善約 40–50% | **訓練的隨機誤差是最大的來源**；但平均後仍有約 3–5% 的系統性偏差，有些 WC 三個 seeds 都偏同一個方向（cQd8、cQu1、cQu8 偏小，cQj31 偏大） |
| 16 | 0.1%/1% 建設性設計（152 個網路）在 1% 和 0.1% data 點的 closure，對照 eae2959 | 1%：68% 內 5/16（eae2959 是 2/16），rms 15%（eae2959 是 20%）；0.1%：兩者都失敗（1–2/16），很多撞到 scan 邊界 | 1% 有改善但還不夠；**0.1% 低於這個樣本的靈敏度**：truth 的 68% 區間本身就有 c 的 ±7–27%，網路被雜訊主導 |

## 3. 降低訓練的隨機誤差（2026-10-14，cluster 12547015，48 個 jobs）

**動機**（嘗試 15 和訓練 log）：
- 同一個點不同 seeds 的 log r̂ 逐 event 差約 11–13%，是近 SM 偏差的最大來源。
- Early stopping 的 best epoch 分散在 10–78（多數 25–40），這時 cosine 的 learning rate 還很高（第 30 個 epoch 約 0.84×10⁻³）。
- Test loss 只比 ln2 = 0.6931 低 0.001–0.009，訊號很弱，等於在雜訊中挑 epoch。

**設計**：每個變體只改一件事。

| 變體 | 改什麼 | 要回答的問題 |
|---|---|---|
| `fixboot` | 三個 seeds 用同一個 bootstrap split（`bootstrap_seed: 2`，209ba10 新增） | seed 間差異有多少來自資料取樣（訓練設定消除不了）、多少來自初始化和 batch 順序 |
| `cos40` | 40 個 epochs 跑完整個 cosine，不 early stop | learning rate 降到 10⁻⁶ 再挑 best epoch，是否比較穩定 |
| `bs10k` | batch 10000 | 梯度雜訊少 10 倍，是否比較穩定 |
| `nodrop` | 不用 dropout | dropout 是否增加 seed 間差異 |

- 點：`SM_cQj11`、`cQj11_cQj11`（主要是隨機誤差）、`SM_cQd8`、`cQd8_cQd8`（有系統性偏差），a6ff4bd 的 5% 值；seeds 2、3、4。
- 評估：
  1. 主要指標是 `seed_compare.py` 的逐 event seed 差異比例（baseline 約 0.11–0.12），越低越好。
  2. 每個 seed 的 normalization。
  3. 有改善的變體再跑近 SM scan：把 `local/ensemble_a6ff4bd_cpu.yml` 裡這 4 個點換成 `local/cpu_configs/var/<variant>_<point>.yml`，再用 `local/truth_morph_scan.py --mode seeds --seeds 2,3,4`。
- 指令和檔案位置見 `README_closure_ywk.md` 最後一節。

### 結果（2026-10-15/16）

| # | 嘗試 | 結果 | 得到的資訊 |
|---|---|---|---|
| 17 | 4 種訓練設定（`fixboot`、`cos40`、`bs10k`、`nodrop`）的 `seed_compare.py` | seed 差異比例：baseline 0.116；fixboot −5%；cos40 −8% 但不一致；**bs10k −20%**（四個點都降）；nodrop +6%（best epoch 2–16，過擬合） | seed 間差異主要來自初始化和 batch 順序，不是 train/test split；dropout 要保留 |
| 18 | bs10k 的近 SM scan（4 個網路換成 bs10k，seeds 2、3、4 和平均） | cQd8：seed 間 std 從 0.7–6.7% 降到 0.1–1.6%，但仍一致偏小（−1.8 ~ −6.7%）；cQj11：seed 3 壞掉（`half_5p` 跳到 +0.365），平均 3 seeds 偏 +10 ~ +17.5% | bs10k 降低隨機誤差，但系統性偏差還在；cQj11 的異常當時無法解釋 |
| 19 | 用 upstream metrics（`local/eval_metrics.py`）評估 24 個網路，按 m_ttbar 分段看 calibration | < 840 GeV（80% 的 events）±1%；1.0–1.3 TeV 低估 1–6%；1.3–2 TeV 高估 8–25%；> 2 TeV 高估 1–6.5 倍，一個 7 TeV 的 event r̂ = inf。除以 pg 與否只差 < 10% | **網路在高 m_ttbar 外推失準**；pg 不是原因 |
| 20 | 近 SM scan 只用 m_ttbar < 1300 / < 2000 GeV 的 events（`local/scan_mttcap.sh`） | cQj11：全部 events 最多偏 21%，< 2000 ≤ 9%，< 1300 ≤ 3.5%（bs10k 的異常也消失）。cQd8：< 2000 小幅改善；< 1300 時 seed 2 在 ±2% 內，但平均 3 seeds 翻成 +4 ~ +6% | **確認 tail 是近 SM 偏差的主要來源**。cQd8 有兩個方向相反的來源（> 1.3 TeV 高估、1.0–1.3 TeV 低估），原本部分抵消。用 cut 修不可行 → log features 重訓 |

結果資料夾：嘗試 17 `README_closure_ywk.md`「降低訓練的隨機誤差」；18 `output/nearSM_fine_bs10k_seeds{2,3,4,234}_*/`；19 `output/eval_metrics_a6ff4bd_bs10k.txt`；20 `output/nearSM_mtt{1300,2000}_*/`。

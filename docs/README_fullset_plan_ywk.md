# Full set 的訓練計畫：三階段（2026-10-22）

目標：一組涵蓋全部 153 項（SM、16 個一次項、16 個二次項、120 個交叉項）的網路，交給 upstream 的 `validation.py`（bias test、歸因、drop test、point search）檢查，最後做多維 scan。

背景：
- 2026-10-20 組會的討論：交叉項的點、±c、ESS、validation 流程。
- 單一 WC 的經驗：`README_batch_pilot_ywk.md`、`README_jupyter.md`（hlcos）。
- 同事從 153 項逐步刪到約 89 項。我們先做 153 項，之後由 validation 決定刪哪些。

## 設計原則

1. **用大的點，不用 0.5–2.5%**
   - 網路的雜訊是絕對量：SM_SM 的 rms_bias 約 0.0006，r 約 ±0.001。γ 的誤差 ∝ 雜訊 / c，所以 c 大，γ 才準。
   - hlcos ctGRe 的結果：0.5% 的 pull 最高到 12，rms_bias 和訊號一樣大；5–20% 的 pull 在 0.3–1。
   - morphing 是精確的二次式，訓練點不必放在要評估的地方（SM 附近）。即使 Run 2 的 EFT 效應不明顯，這個結論也不變。
   - 小點只用來做 closure：從大點 morph 下來，和小點的 dedicated 網路比較。hlcos 的四個 WC 已經有了。
2. **±c**：c_p 是 constructive root（b·c > 0），−c_p 是它的反號（`_neg`）。r(+c) − r(−c) ∝ γ1，r(+c) + r(−c) ∝ γ2，一次項和二次項分得開。不用 destructive root（ctGRe 在 4.8，ESS 0.05）。
3. **交叉項**：每個 WC 取自己的 ±c_p，一個 pair 有 (+,+)、(+,−)、(−,+)、(−,−) 四種組合。
   - 優先選 ESS 高的組合。（原本寫「a_ij·c_i·c_j < 0、交叉貢獻抵消的方向 ESS 較高」，第 2 階段檢查發現不成立：交叉 yield 明顯不為 0（|yield| > 0.001）的 19 個中，有 12 個選中的是交叉貢獻為正的方向。例如 cQj11×cQj31 的 (+,+) 是抵消方向，ESS 只有 0.10；(+,−) 0.82。）
   - 對角線的 ±c 已經固定了單一 WC 的部分，所以一個交叉點在代數上就能解出 γ_ij。兩個點（一組加上它的鏡像）提供冗餘，讓 drop test 可以用。
   - 會議提到的「找 2D yield 的極小值」暫時不做：b 小的 pair 極小值在原點附近，含 ctGRe 的 pair 極小值很遠、ESS 低。它真正有用的部分，就是上面的正負號規則。
4. **ESS（= n_eff/N = (Σw)²/Σw² / N）**：每個點都要算。SM 本身是 0.76；低於約 0.3 的點不用。
5. **先做會有結果的部分，學不到的項不浪費 trainings。**

## 交叉項的 survey（truth，不用網路）

`local/cross_terms_survey.py`，輸出在 `www/topsbi/cross_terms_survey_20261022/survey_{0.05,0.1}.txt`：
- **rms_ev**：每個 event 的交叉貢獻 c_i·c_j·w_ij / w_SM，以 w0 加權的 RMS，也就是網路要學的訊號大小。
- **yield**：這個交叉項對總 yield 的貢獻。
- **ess++ / ess+−**：兩種正負號組合的 ESS。
- 對角線部分：16 個 WC 在 ±c_p 的 ESS。

5% 時 120 個交叉項分成三群：

| 群 | 數量 | rms_ev | 內容 |
|---|---|---|---|
| 強 | 27 | > 0.25（最大 7.5） | 同一種輕夸克的 singlet/octet 對，例如 cQj11×cQj31、cQj18×cQj38、ctd1×ctd8、ctu1×ctu8 |
| 中 | 21 | 0.01–0.25 | 主要是 ctGIm、ctGRe 和四費米子 |
| 小 | 72 | ≤ 0.01 | 多半是不同初態夸克（例如 u 類 × d 類），LO 物理上是 0；這裡不是 0，大概是 per-event 二次 fit 的雜訊 |

- 「小」那群和網路的雜訊同量級，網路學不到。同事刪到 89 項，大概就是刪掉這一群。
- **注意**：我們的資料裡這些 coefs 不是精確的 0，upstream `full_likelihood` 的 mask 只會去掉全為 0 的欄，所以不會自動去掉它們。
- 要去掉的話，在資料裡把那些欄設為 0（truth 會變動約 0.001，和雜訊同量級），或者向同事要那 89 項的清單或規則。

## 三階段

| 階段 | 內容 | 每個 seed | × 3 seeds | 產出 |
|---|---|---|---|---|
| **1** | SM_SM + 16 個 WC × {±5%, ±10%} | 65 | **195** | 16 個 1D scan；validation 在單一 WC 上的測試 |
| 2 | 強的 27 個交叉項，5%，2 組正負號（ESS 高的那組和它的鏡像） | 54 | 162 | 主要的 2D 相關 |
| 3 | 中等的 21 個交叉項，視第 2 階段的 validation 結果而定 | 約 42 | 約 126 | 補完 |
| — | 小的 72 個交叉項 | 0 | 0 | γ 設為 0，不訓練 |

- 合計約 480 jobs。對照組：全部 153 項 × 5 個百分比的規劃約 4100 jobs；全部 120 個交叉項 × 2 組約 915 jobs。
- seeds 先用 3 個。會議提到「約 10 seeds、800 個網路」，等 validation 的 point search 看過之後再決定要不要擴充。
- 交叉項先用 5%（會議經驗：5% 好、10% 開始變差）。

## 第 1 階段（2026-10-22 送出：cluster 13415949，195 jobs）

- **程式碼：`topsbi-wt/port-3507174`**（固定的 detached worktree）。
  - 由 branch `port/rebased-fa6f982`（worktree `topsbi-wt/rebased-fa6f982`）而來：upstream eddie/main fa6f982（含 PR #17、validation 重寫、`ensembleCache.py`）加上我們 `port/rebased` 的 patches（cherry-pick，全部沒有衝突；5de0af6 已經在 upstream，所以跳過）。
  - 為什麼不用 fa6f982 本身：upstream 不支援 `year_loc: null`，我們的資料只有 19 欄，沒有 `year_int`。1 epoch 的測試在 `get_weights` 出現 `IndexError: index 6 is out of bounds`。
  - 檢查：pytest 28 passed、ruff 通過、1 epoch CPU 測試跑完（含 holdout metrics）。
  - **`port/rebased` 沒有動**（ce8d502）。
- plan：`local/plans/fullset_s1.yml`。configs：`local/condor/configs/fullset_s1/`（195 個）。list：`~/topsbi-condor/list_fullset_s1.txt`。
- 輸出：`/eos/cms/store/user/ykao/topsbi/results/port/3507174_fullset/<label>/seed_<n>`。
- 資料、features、訓練設定和 hlcos 相同：highlevel_cos、`weights_only`、`wo_log_bs10k` 模板。
- WC 值：`local/wc_targets_hlcos.yml`（0.5–5% 和 pilot 相同，另外加了 10%、20%）。
- 10% 時的 ESS：大多 ≥ 0.6；cQj38 的 −c 0.45；cQj11、cQj31 0.30–0.37（低但可以接受，要注意它們的結果）。
- 送出前檢查：65 個點從 configs 的 c1 重新算 yield，+c 都是 1.0500 或 1.1000，−c 是 −c_p，動的 WC 和 label 一致，每個 seed 都有 65 個。
- 送出的指令：
  `cd ~/topsbi-condor && condor_submit code=/eos/home-y/ykao/SWAN_projects/analysis/topsbi-wt/port-3507174 configlist=list_fullset_s1.txt submit_list.sub`

### 第 1 階段的 jobs 設定（複習）

**點**：每個 WC 單獨動（其他 15 個為 0，c0 = SM）。16 個 WC × {+5%, −5%, +10%, −10%}，加 SM_SM，每個 seed 65 個點；seeds 2、3、4，共 195 個。

**WC 值**：yield 比 SM 多 5% 或 10% 的 constructive root；`_neg` 是反號。「符號」是 constructive root 本身的正負號。

| WC | 5% | 10% | 符號 |
|---|---|---|---|
| ctGIm | 0.906 | 1.282 | − |
| ctGRe | 0.153 | 0.298 | − |
| cQj38 | 3.61 | 5.19 | + |
| cQj18 | 2.63 | 4.13 | + |
| cQu8 | 3.26 | 5.00 | + |
| cQd8 | 4.25 | 6.50 | + |
| ctj8 | 2.18 | 3.45 | + |
| ctu8 | 3.43 | 5.27 | + |
| ctd8 | 4.39 | 6.71 | + |
| cQj31 | 1.52 | 2.16 | + |
| cQj11 | 1.50 | 2.14 | − |
| cQu1 | 1.84 | 2.63 | + |
| cQd1 | 2.41 | 3.42 | + |
| ctj1 | 1.38 | 1.98 | + |
| ctu1 | 1.82 | 2.58 | − |
| ctd1 | 2.27 | 3.23 | − |

**訓練設定**（模板 `pilot_cQd8_5p_seed2.yml`，和 hlcos、pilot 一樣）：
- `method: weights_only`，`bootstrap: true`（train.p 依 seed 切成 75/25）。
- features `features_highlevel_cos_log.yml`，`year_loc: null`。
- 網路 256-256-256-1，LeakyReLU，dropout 0.2。
- lr 0.001，cosine scheduler，warmup 5 epochs；最多 100 epochs，batch 10k，early stopping patience 20。
- holdout `validation.p`；`device: cuda`。

**位置**：
- 資料：`/eos/cms/store/user/ykao/topsbi/pretraining_main_highlevel_cos_log_pilot`
- configs：`local/condor/configs/fullset_s1/`；CPU 版（給 validation、scan）：`local/cpu_configs/fullset_s1/<label>/`
- 輸出：`/eos/cms/store/user/ykao/topsbi/results/port/3507174_fullset/<WC>_<5p|10p>[_neg]/seed_<n>`

**沒有放進來的**：0.5–2.5%（雜訊蓋過，hlcos 的 4 個 WC 留著做 closure）、20%、交叉項（第 2 階段）。

## 第 1 階段之後（結果和手動操作：`README_fullset_s1_validation_ywk.md`）

1. 各網路的 `performance.yml`：pull、ci_diff 應該和 0 一致。
2. 16 個 1D scan：`scan.py`（weights_only、seed ensemble）。
3. `validation.py`：建立 ensemble cache（`--refit`），看 bias test 和 Z heatmap。
   - **注意**：validation.py 第 725 行 `features[:, config.get('year_loc', 21)]` 也還不支援 `year_loc: null`，跑之前要再 patch。
4. 第 2 階段：擴充 `make_training_configs.py`，支援兩個 WC 的點（自動選正負號、印出 ESS）。

## 第 2 階段（2026-10-23 送出：cluster 13422990，162 jobs；細節和結果：`README_fullset_s2_ywk.md`）

- 只做強的 27 個交叉項（5% 時 rms_ev > 0.25）。中的 21 個等這批的 validation 結果再決定。
- 每個 pair 兩個點，都用 5%：
  - ESS 高的那組正負號（直接比 ESS；不一定是交叉貢獻抵消的方向），例如 `cQj11_5p_x_cQj31_5p_neg`。
  - 它的鏡像（兩個 WC 都反號，c_i·c_j 不變、一次項反號），例如 `cQj11_5p_neg_x_cQj31_5p`。
- 54 個點 × seeds 2、3、4 = **162 個 jobs**。程式碼、模板、資料、輸出目錄都和第 1 階段相同（`3507174_fullset/<label>/seed_<n>`）。
- n_eff/N：54 個點都在 0.51–0.84（最低的是 cQj31×cQj38、cQj11×cQj38 的鏡像）。每個 pair 的值記在 plan 的註解裡。
- plan：`local/plans/fullset_s2.yml`；configs：`local/condor/configs/fullset_s2/`；list：`~/topsbi-condor/list_fullset_s2.txt`。
- validation config：`local/validation/fullset_s2.yml` = 第 1 階段的 64 個網路 + 這 54 個（每個 seed 118 個，SM_SM 沿用第 1 階段），輸出到 `validation_fullset_s2`。
- `make_training_configs.py` 新增兩個 plan key：`pairs`（兩個 WC 的點）和 `validation_base`（把新點加到既有的 validation config）。
- 送出前檢查：162 個 configs 和第 1 階段 config 只差 c1、seed、name；每個都正好動兩個 WC，和 label 一致；輸出目錄都還沒有 model.pt。
- 送出的指令：
  `cd ~/topsbi-condor && condor_submit code=/eos/home-y/ykao/SWAN_projects/analysis/topsbi-wt/port-3507174 configlist=list_fullset_s2.txt submit_list.sub`
- 跑完之後：`python -m topsbi.validation -c $R/local/validation/fullset_s2.yml --refit`（手動步驟同 `README_fullset_s1_validation_ywk.md`）。

## 還沒決定的

- 89 項的清單（同事），或者用 survey 的門檻自己定。
- 交叉項要不要也試 10%。
- 最終的 seeds 數量。
- **（2026-10-24 更新）r ≤ 0 的根源找到了**：全域 fit 的 lstsq 沒有加權，γ_sm 被雜訊大的四費米子網路拉走；SM_SM 列加權 ×100 後 r ≤ 0 幾乎消失。細節見 `README_fullset_s2_ywk.md` 第 5 節。正式的修正方式（加權方式）待決定。原本列的選項：r ≤ 0 的處理方式（scan 時去掉、對 γ 加限制、找出造成的網路並 drop）。這些都在訓練之後做，**不影響第 2 階段 jobs 怎麼送**。唯一可能的影響：如果診斷出 r ≤ 0 來自低 ESS 的點，交叉項就要選 ESS 高的符號組合、用 5%，而這兩點本來就在計畫裡。
- 第 2 階段只訓練 27 個交叉項。其餘 93 個不用另外處理：沒有任何網路動到的交叉項，在訓練矩陣裡那一欄全為 0，`full_likelihood` 會自動 mask（`buildLikelihood.py:83`），γ_ij = 0。代價只是這些項的物理效應被當成 0（小的 72 個本來就在雜訊以下；中的 21 個看第 2 階段結果再決定）。

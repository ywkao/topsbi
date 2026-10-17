# Feature set 比較（ywk，2026-10-17）

目的：比較 5 組 feature sets，結果要能放進 AN。分兩部分：不用訓練的 Fisher information 估計，以及實際訓練的網路。

## Feature sets

都從 `/eos/cms/store/user/ykao/topsbi/pretraining/`（74 個 features，dev 格式）用 `local/convert_pretraining.py` 轉成 main 格式，再用 `local/make_log_features.py` 取 log，最後用 `local/filter_nonpositive.py` 去掉 w ≤ 0 的 events（見 `README_weights_ywk.md`）。

| 名稱 | 數量 | 資料（`/eos/cms/store/user/ykao/topsbi/`） | features 檔（主目錄） |
|---|---|---|---|
| highlevel（基準） | 13 | `pretraining_main_highlevel_log_pilot/` | `features_highlevel_log.yml` |
| top30 | 30 | `pretraining_main_top30_log_pilot/` | `features_top30_log.yml` |
| compact | 26 | `pretraining_main_compact_log_pilot/` | `features_compact_log.yml` |
| lowlevel | 31 | `pretraining_main_lowlevel_log_pilot/` | `features_lowlevel_log.yml` |
| all | 74 | `pretraining_main_all_log_pilot/` | `features_all_log.yml` |

- top30、compact、lowlevel 的清單在主目錄 `local/feature_sets.yml`；all 就是 `features.yml` 的 74 個。
- **log 規則**（`make_log_features.py` 的 `log_fn`）：m_ttbar 取 log；pt_tt、HT、mT_W、`*_pt`、`*_mass`、`m_*` 取 log1p。對 highlevel 選到的還是原本的 4 個欄位。mT_W 是事後加的（沒取 log 時 z 最大 42.5，取了之後 −5.1 到 4.3）。
- 取 log 之後 z 仍然偏大的欄位：`lep_mass`（lowlevel、all，30.7，基本上只有 e、μ 兩個值，照原樣保留）、`njets`（11.3，離散）、`dr_tt`（9.5，highlevel 本來就有）。
- **五組的 events 完全相同**：coefs 逐筆比對相同、順序相同（train、validation 都確認過）。所以同一個 seed 的 train/test 切分也相同（`sample_boostrap` 用 `manual_seed(seed)`）。

## 訓練（AN 用的比較）

| | 內容 | jobs | cluster |
|---|---|---|---|
| highlevel | pilot（`0c49da2_pilot`）裡 4 個 WC 的 5% 點，**只用 seed 2、3** | （pilot 的一部分） | 12702676 |
| 其他四組 | 4 個 WC 的 5% 點 × seed 2、3（`0c49da2_pilot_<set>`） | 32 | 12707721 |

- configs 除了 `data`、`features`、`features_to_animate`、`holdout`、`name` 之外完全相同。lowlevel 沒有 m_ttbar，animation 改畫 `log_lep_top_pt`。
- 比較指標：holdout BCE loss（proper scoring rule，同一組 (c0, c1) 下越低表示抓到越多資訊）、`rms_bias`、`pull`。兩個 seeds 的差距當誤差尺度；差距接近 seed 差異的組再補 seed 4。
- **這是配對比較**：五組在同樣的 events 上評估，per-event 的雜訊大部分會互相抵消。下面 Fisher 估計失敗的原因（per-event 雜訊太大）在這裡影響小得多。

## 訓練結果（2026-10-18）

見 `README_featsets_results_ywk.md`。摘要：ctGIm 用 all 的 gain 是 highlevel 的 2.8 倍（來自 `cos_{lep,had}_{n,r,k}`）；其他 WC 是 highlevel 最好，lowlevel 最差。建議 highlevel + 這 6 個 cos（`highlevel_cos`，configs 已準備）。

## 不用訓練的 Fisher information 估計：這個方法不可行

Script：主目錄 `local/fisher_featsets.py`。結果：`/eos/user/y/ykao/www/topsbi/fisher_featsets_20261017/`（`fisher_featsets.txt`、`.yml`、`.png`）。

**做法**：每個 event 的真實 ratio 是 r(c) = 1 + α·c + β·c²。網路最多能學到 E[r|x]。用 gradient boosting（weights w0）從 features x 回歸兩個目標，算 held-out 的 weighted R²，當作「保留了多少比例的資訊」：
- α：SM 處的 score，shape Fisher information 是 Var_w(α)，只用 features 時是 Var_w(E[α|x])；
- r5：5% 點的真實 ratio。

另外加了一組 `mtt_only`（只用 log m_ttbar）當參考。

**結果**（held-out R²，± 是 validation 兩半的差距）：

| α | mtt_only | highlevel | top30 | compact | lowlevel | all |
|---|---|---|---|---|---|---|
| ctGIm | −0.000 | −0.002 | −0.001 | −0.002 | −0.001 | −0.001 |
| ctGRe | 0.008 | **0.096** | 0.095 | 0.095 | 0.004 | 0.096 |
| cQj11 | 0.000 | −0.005 | −0.008 | −0.009 | −0.007 | −0.007 |
| cQd8 | 0.004 | 0.011 | 0.003 | 0.007 | −0.004 | −0.007 |

r5 的結果類似，cQj11、cQd8 還出現很負的值（例如 cQd8 all −0.86 ± 0.78）。

**為什麼不可行**（validation 上的診斷）：
- **每個 event 的 α 主要由 parton-level 的資訊決定，reco features 看不到**。四費米子 operator 只影響 qq̄ 初態，gg 初態的 events α ≈ 0：cQj11 的 |α| 中位數是 1e-6，但 0.01% 的 events（18 個）就佔了 Var_w(α) 的 50%，1% 的 events 佔 81%。
- **features 能看到的部分不到總變異的 1%**：只用 m_ttbar 分 11 個 bins，shape 資訊佔 Var_w(α) 的比例是 ctGIm 0.02%、ctGRe 0.85%、cQj11 0.20%、cQd8 0.40%。
- 所以「保留比例」的分母沒有意義，而且回歸的目標被少數 events 主導：gradient boosting 大多在 10–20 次迭代就因為 validation 沒有改善而停止，R² 在 0 附近、被雜訊主導。

**能讀出的有限結論**（只有 ctGRe 的訊號高於雜訊）：
- ctGRe：highlevel ≈ top30 ≈ compact ≈ all（0.095–0.096），遠高於只用 m_ttbar（0.008）。也就是說，ctGRe 的資訊主要不在 m_ttbar，而在 highlevel 已經包含的變數（自旋關聯等），其他三組沒有額外的資訊。
- lowlevel 在 ctGRe 只有 0.004，但這是回歸方法的限制（boosted trees 組不出 m_ttbar、自旋角這類組合變數），不能當成 lowlevel 本身不好；NN 可能學得到。
- ctGIm：所有組都是 0。CP-odd 的資訊需要 triple product 或帶正負號的角度，74 個 features 裡只有 φ 角這類原料，boosted trees 組不出來。所以這個方法無法回答「highlevel 是否缺少 CP-odd 觀測量」。
- cQj11、cQd8：雜訊太大，無法比較。

**要讓不用訓練的估計可行**，需要和訓練一樣的配對比較（同樣的 events 上比較兩組的 loss 差），以及比 boosted trees 強的回歸（也就是 NN），那就等於訓練本身。所以 **AN 的 feature set 比較以訓練結果為主**；Fisher 估計最多在 AN 裡當一段說明：per-event 的資訊大多是 reco-level 看不到的。

## 待辦

- [x] 比較 script 和圖（上面「訓練結果」）。
- [ ] ctGIm 的 CP-odd 問題：送出 `highlevel_cos`（8 jobs，見 `README_featsets_results_ywk.md`），確認能拿到 all 的 gain。之後也可以試明確的 CP-odd 組合（c_rk − c_kr 等）。

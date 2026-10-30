# 會議 2026-10-27：follow-ups 和進度

## 會議重點（原文摘要）

1. **改用 reco samples**，不再調 gen-level 的結果。框架作者會開始做 2016 central sample 的 reco skims。
2. **用 quantile bins 選 terms**，取代全域 RMS。輸出要丟的 terms 的 yaml，訓練和 validation 共用。
3. **Feature sensitivity**：每個 structure constant，在每個 input feature 的 quantile bins 裡畫相對 SM 的 yield。平的或很雜，代表網路只在學那個 feature 的雜訊。
4. **分母改用 generation point**，不用 SM：從 generation point 到 SM 的有效統計量低，SM 當分母的權重 variance 很大。
5. **SM–SM 權重 100**：擔心蓋過 EFT 效應。(4) 之後若還需要，scan 50 / 100 / 200 / 1000 看 validation。γ 矩陣 full rank 之前出現負值是預期的、會被 mask；只有 full γ 矩陣仍有負值才要擔心。
6. **closure / misclosure 圖少了 SM anchor**：SM 網路有列在 yaml 裡，要查有沒有被讀到。（SM 當分母時可能根本不需要 SM 網路。）
7. **已知有問題的項**：cQj11、cQj31、cQj38、cQj18、ctGRe，可能還有 cQd8。可能是 sampling 問題：generation 時所有係數用同一個 WC 值，有些 pair 變成完全平方，分不開。reweighting 解決不了，丟掉樣本裡為 0 的項。

validation script 的補充：每個 feature 的 Z = (χ² − ndf)/√(2 ndf)，用 quantile bins 算，對離散的 feature（例如 njets）比較不可靠。drop test 只在 Z 改善時保留 drop，輸出 dropped / to-retrain 的 yaml。

## 進度

### (2) quantile 選 terms：完成（2026-10-28）

見 `README_zero_terms_ywk.md`。結論：tol 用 4e-3（size 分布的空白中間），丟 64 個交叉項、留 89 項，和同事的數字相同。upstream 預設的 1e-6 在我們的樣本一項都不丟。

### (6) SM anchor：SM 網路有被讀到，只是圖上沒標出來（2026-10-28）

- SM 網路有讀進來、也有用在 fit：`ensemble_diagnostics.yml` 的 `network_misclosure` 裡有 `fullset_s1/SM_SM: 0.00035`，是所有網路裡最小的。
- 但 `plot_network_misclosure`（`tools/plots.py`）用**網路名稱**判斷種類：名稱等於 `sm` 才畫成 SM anchor（星號），名稱以 `cross_terms/` 開頭才畫成 interference point（三角形）。
- 網路名稱是 config 所在目錄相對於共同路徑的名字。我們的是 `fullset_s1/SM_SM`、`fullset_s2/<pair>`，所以：
  - SM_SM 被畫成灰色圓點（x = 1.00、y = 3.5e-4 那一點），不是星號。
  - 54 個交叉項網路也都畫成圓點，沒有三角形。
- 修法：改用 Θ 判斷種類（Θ 那一列只有 sm 欄非 0 → SM；動到兩個以上的 WC → interference），不依賴目錄名稱。只影響畫圖，不影響任何數字。
- **已修好（2026-10-28）**：`validation.point_kinds` 從 Θ 算出每個點的種類，傳給 `plot_network_misclosure(point_kinds=...)`；沒給時照舊用名稱猜。測試：`tests/test_ensemble.py::test_point_kinds_from_training_points`。
- 新圖：`/eos/cms/store/user/ykao/topsbi/results/port/3507174_fullset/validation_fullset_s2_smw100/plots/network_misclosure.png`。SM_SM 是星號（m_n = 3.5e-4，比所有其他網路小 1–3 個數量級），54 個交叉項網路是三角形。validation 的數字不變（27/60 flagged、drop 14、10 組新點）。
- 重畫的指令（用現有 cache，不加 `--refit`，約 5 分鐘）：
  `python3 -m topsbi.validation -c $R/local/validation/fullset_s2_smw100.yml`

### (4) generation point 當分母（2026-10-28 先算數字）

- 我們樣本的 generation point cg：ctGIm、ctGRe = −0.5，其他 14 個 = 1.5；yield 是 SM 的 1.61 倍。
- 有效統計量（n_eff/N）：在 cg 是 1.000（events 在 cg 是等權重的）；reweight 到 SM 是 0.757。
- 所以在這個 gen-level pilot 樣本，cg → SM 的統計量損失約 24%，不算低。會議的擔心可能對 reco / central samples 比較重要，換樣本後要重算。
- closure test 要不要放在 cg：目前判斷不需要。
  - 會議第 4 點講的是**訓練的分母**（c0），不是 closure 的 pseudo-data 點；兩者可以分開決定。
  - cg 的 16 個 WC 同時不為 0、yield 多 61%，遠在訓練點（單一 WC 5–10%、交叉項 5%）之外。在 cg 做 closure，測的是 16 維的外插，不是分析會用到的 SM 附近。
  - 分母換成 cg 之後，closure 仍應放在 SM 附近（例如 2.5%）。換分母的好處要看「在 SM 附近的 scan 有沒有變好」。
  - cg 點的 pseudo-data 不用 reweight（原始樣本），`scan.py --others cg` 已支援；可以當作程式的 sanity check，但不期待它 close。

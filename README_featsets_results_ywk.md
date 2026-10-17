# Feature set 比較：訓練結果（ywk，2026-10-18）

資料、log 規則、訓練設計、為什麼不用 Fisher 估計，見 `README_featsets_ywk.md`。這份只放訓練的結果和結論。

## 設定

- 5 組 feature sets：highlevel（13）、top30（30）、compact（26）、lowlevel（31）、all（74）。**events 完全相同、順序相同**，同一個 seed 的 train/test 切分也相同。
- 4 個 WC（ctGIm、ctGRe、cQj11、cQd8）的 constructive 5% 點，c0 = SM，weights_only，seed 2、3。除了 features 之外，configs 完全相同。
- 網路：
  - highlevel：`results/port/0c49da2_pilot/<wc>_5p/seed_<n>`（cluster 12702676）
  - 其他四組：`results/port/0c49da2_pilot_<set>/<wc>_5p/seed_<n>`（cluster 12707721）
- Script：主目錄 `local/compare_featsets.py`。
- 輸出：`/eos/user/y/ykao/www/topsbi/featsets_trained_20261018/`（`compare_featsets.txt`、`.yml`、`.png`）。

## 指標

都在共同的 holdout `validation.p`（177,883 events）上算，w0、w1 除以 holdout 的平均 w0。每個 event 的 loss 是 l = −w0·log(1−s) − w1·log(s)。

- **gain** = L_const − L_net：網路的 weighted BCE 比最好的常數分類器（s = Σw1 / Σ(w0+w1)）低多少，單位是 nats / SM event。BCE 是 proper scoring rule，所以 gain 越大表示網路抓到越多 c1 vs c0 的資訊。
- **frac** = gain / (L_const − L_true)，L_true 用每個 event 的真實 ratio s = w1/(w0+w1)。真實 ratio 包含 reco 看不到的 parton-level 資訊，所以就算網路完美，frac 也遠小於 1。
- **delta** = gain(set) − gain(highlevel)，同 WC、同 seed。這是配對比較：誤差來自逐 event 的 loss 差，per-event 的雜訊大部分會抵消。
- **auc、pull、rms_bias**：從各網路的 `complete/performance.yml`。

## 結果

gain，單位 1e-4 nats / SM event（seed 2 / seed 3）：

| | highlevel | top30 | compact | lowlevel | all |
|---|---|---|---|---|---|
| ctGIm | 2.50 / 2.39 | 2.43 / 2.44 | 2.49 / 2.40 | 1.31 / 1.40 | **6.91 / 6.59** |
| ctGRe | 0.316 / 0.318 | 0.309 / 0.305 | 0.309 / 0.288 | −0.000 / −0.003 | 0.294 / 0.291 |
| cQj11 | 96.5 / 95.8 | 89.6 / 90.7 | 94.5 / 93.9 | 69.8 / 73.2 | 92.3 / 93.2 |
| cQd8 | 51.3 / 51.5 | 48.4 / 49.0 | 50.0 / 50.4 | 36.5 / 37.8 | 49.4 / 50.5 |

delta（相對 highlevel，配對誤差），單位 1e-4（seed 2 / seed 3）：

| | top30 | compact | lowlevel | all |
|---|---|---|---|---|
| ctGIm | −0.07 ± 0.06 / +0.05 ± 0.06 | −0.01 ± 0.06 / 0.00 ± 0.05 | −1.19 ± 0.14 / −0.99 ± 0.12 | **+4.41 ± 0.19 / +4.20 ± 0.21** |
| ctGRe | −0.008 ± 0.002 / −0.013 ± 0.001 | −0.007 ± 0.002 / −0.030 ± 0.002 | −0.316 ± 0.006 / −0.321 ± 0.005 | −0.022 ± 0.002 / −0.027 ± 0.002 |
| cQj11 | −6.9 ± 2.5 / −5.1 ± 1.9 | −2.0 ± 1.1 / −1.8 ± 1.2 | −26.7 ± 4.4 / −22.6 ± 3.3 | −4.2 ± 2.5 / −2.6 ± 1.4 |
| cQd8 | −2.9 ± 0.6 / −2.5 ± 0.5 | −1.4 ± 0.4 / −1.1 ± 0.4 | −14.8 ± 1.0 / −13.7 ± 1.0 | −1.9 ± 0.6 / −1.0 ± 0.6 |

frac（highlevel）：ctGIm 0.02、ctGRe 0.09、cQj11 0.41、cQd8 0.25。

calibration（pull / rms_bias）：
- 各組在 ctGRe、cQj11、cQd8 上差不多，差異在 seed 之間的差異以內。
- 唯一的例外是 all 在 ctGIm 上：pull 0.9–1.2、rms_bias 0.008–0.012，highlevel 是 ±0.1、0。

## 結論

1. **ctGIm：all 的 gain 是 highlevel 的 2.8 倍**，兩個 seeds 一致，超過誤差 20σ。
   - 只有 all 有 `cos_lep_{n,r,k}`、`cos_had_{n,r,k}`（lepton 和 hadronic top 分支在 helicity basis 的自旋投影）。它們是 CP-odd 自旋關聯（例如 c_rk − c_kr）的原料。
   - highlevel、top30、compact 只有 c_hel、c_han、cos_theta_l、cos_theta_had 這類 CP-even 或單一軸的量，所以三組在 ctGIm 上一樣。
   - 不用訓練的 Fisher 估計（boosted trees）在 ctGIm 全部是 0，沒有看到這一點：NN 能從這六個 cos 組出相關性，boosted trees 不行。
2. **CP-even 的 WC（ctGRe、cQj11、cQd8）：highlevel 最好**。
   - compact 和 all 低 1–8%，top30 低 3–6%。
   - 多出的 features 沒有新的資訊，反而在同樣的 epochs 和網路大小下讓訓練稍微變難。
3. **lowlevel 明顯最差**。在 ctGRe 上什麼都沒學到（gain ≈ 0，auc 0.500），在其他 WC 上少了 25–45%。同樣的網路組不出 m_ttbar、自旋角這類組合變數。
4. **建議**：highlevel + 6 個 `cos_{lep,had}_{n,r,k}`（`highlevel_cos`，19 個 features）。預期 ctGIm 能拿到 all 的 gain，其他 WC 和 highlevel 一樣。

## 確認：highlevel_cos（cluster 12948860，2026-10-18 送出）

- 資料：`/eos/cms/store/user/ykao/topsbi/pretraining_main_highlevel_cos_log_pilot/`
  - 從 `pretraining_main_all_log_pilot` 用 `local/select_features.py` 取出 19 個欄位。
  - 檢查過：前 13 欄和 `pretraining_main_highlevel_log_pilot` 完全相同（max diff 0），coefs 也完全相同。
- features 檔：主目錄 `features_highlevel_cos_log.yml`。
- configs：`local/condor/configs/pilot_highlevel_cos/`（4 WC × seed 2、3 = 8 個）。和 top30 的 configs 只差 `data`、`features`、`holdout`、`name`。
- condor list：`~/topsbi-condor/list_featsets_highlevel_cos.txt`
- 輸出：`results/port/0c49da2_pilot_highlevel_cos/<wc>_5p/seed_<n>`
- 送出：`cd ~/topsbi-condor && condor_submit code=../topsbi-wt/port-0c49da2 configlist=list_featsets_highlevel_cos.txt submit_list.sub`
- 跑完後：`local/compare_featsets.py` 的 `SETS` 已經加了 `highlevel_cos`，直接重跑即可。
- 判斷標準：
  - ctGIm 的 delta 接近 all（+4.2 到 +4.4）。
  - 其他 WC 的 delta 和 0 一致（在配對誤差和 seed 差異以內）。

# Pilot 的 1D hold-out closure（ywk，2026-10-18）

問題：只用 {SM, 1%, 5%} 三個網路 morph 出來的 LR，在**沒有訓練過的點**（0.5%、2.5%）是否和那兩個點的 dedicated 網路一樣好？

## 做法

- 工具：eddie/main 的 `src/topsbi/validation.py`（parametric vs dedicated，沒有修改），用主目錄 `local/run_validation.py` 執行。
  - validation.py 的 truth 用 `get_probabilities`，p0、p1 各自除以自己的平均，得到的是 r / (Y1/Y0)，也就是 stitched 學的東西。
  - weights_only 學的是 r = w1/w0，所以 wrapper 把 `get_probabilities` 換成 weights_only 的 weights（都除以 mean(w0)）。
  - wrapper 另外寫 `summary.yml`。
- parametric：`full_likelihood`，`sm` = SM_SM，networks = `<wc>_1p`、`<wc>_5p`（同一個 seed）。lstsq 是 3 列 × 3 欄（1、c、c²），剛好解出。configs 在 `local/closure/pilot_<wc>_seed<n>.yml`。
- dedicated：`<wc>_0.5p`、`<wc>_2.5p` 的網路。
- CPU 版的 pilot configs（`device: cpu`）在 `local/cpu_configs/pilot/`。
- 資料：`pretraining_main_highlevel_log_pilot/validation.p`（177,883 events，訓練時的 holdout）。
- 規模：4 WC × 2 點 × seed 2、3、4 = 24 次，每次約 25 秒。
- 輸出：`/eos/user/y/ykao/www/topsbi/closure_pilot_20261018/<wc>_<pct>p/seed_<n>/`
  - validation.py 的 `lr*.png`、`kinematics/`、`performace.yml`
  - `summary.yml`
  - 總表 `summary.txt`

## 指標

- `rms_bias_{dedicated,parametric}`：`lr_calibration_curve` 對 truth（50 個 r̂ bins，扣掉統計雜訊），和訓練時 `performance.yml` 的 rms_bias 一樣。**主要看這個**。
- `par_vs_ded` = rms_w(plr − dlr) / rms_w(dlr − 1)：parametric 和 dedicated 的差，相對於學到的形狀。
- E[r]：E_w0[r̂] 和真實的 yield ratio Y1/Y0。
- validation.py 自己的 `pChiIncl`、`dChiIncl` 是 1000 個 bins 的 Σ(x − mean)²，沒有除誤差，被尾端的 bins 主導（cQj11 到 1e5）。**不適合用來判斷**，只列在 `summary.txt` 裡。

## 結果（`summary.txt`）

rms_bias（seed 2、3、4 的範圍）：

| 點 | dedicated | parametric |
|---|---|---|
| ctGIm 0.5% | 0.0014–0.0026 | 0–0.0015 |
| ctGIm 2.5% | 0–0.0013 | 0–0.0015 |
| ctGRe 0.5% | 0.0003–0.0016 | 0.0003–0.0004 |
| ctGRe 2.5% | 0.0007–0.0010 | 0.0004–0.0017 |
| cQj11 0.5% | 0.0001–0.0039 | 0–0.0027 |
| cQj11 2.5% | 0.0081–0.0116 | 0.0060–0.0109 |
| cQd8 0.5% | 0.0009–0.0025 | 0.0009–0.0018 |
| cQd8 2.5% | 0.0091–0.0100 | 0.0077–0.0085 |

E[r]（yield ratio）：
- parametric、dedicated 和 truth 的差大多在 0.1% 以內。
- 例外是 cQj11：兩者都偏低。0.5% 是 1.0036–1.0048 vs 1.0053；2.5% 是 1.0231–1.0261 vs 1.0266。

par_vs_ded：
- 2.5%：0.09–0.47。
- 0.5%：0.19–1.08。0.5% 的 dlr − 1 很小，分母小，所以比例大。

## 結論

1. **closure 通過**：在沒有訓練過的 0.5%、2.5%，parametric 的 rms_bias 和 dedicated 一樣，甚至略好。用 {SM_SM, 1%, 5%} morph 不會增加 calibration 誤差。
2. 剩下的誤差來自網路本身，不是 morphing：cQj11、cQd8 在 2.5% 時 rms_bias 約 1%，dedicated 也一樣。
3. cQj11 的 E[r̂] 比 validation 上的真實 yield ratio 低 0.05–0.3%（parametric、dedicated、三個 seeds 都是）。**這不是網路的錯**（2026-10-18 更正）：validation.p 的 cQj11 yield ratio 本身因為 MC 統計漲落偏高（2.5% 點：train 1.02484，validation 1.02655），網路學到的是 train 的值（1.02485）。cQj11 的 yield 由少數大 weight 的 qq̄ events 主導，所以漲落大。
4. par_vs_ded 在 0.5% 很大，但兩者的 rms_bias 都小。也就是說，它們在 per-event 上不同，但平均起來都校準。這是網路的 per-event 雜訊，ensemble（seed 平均）會降低它。

## 1D scan（step 2，2026-10-18）

### scan.py 的修改（rebased worktree，`scripts/scan.py`，還沒 commit）

- `--method {weights_only, stitched}`，預設讀第一個網路 config 的 `method`。
  - **weights_only**：網路學的是 w(c)/w(SM)，包含 rate。
    - −2 ln L 用 extended likelihood：−2 Σ w_data log r(c) + 2 Σ w_SM r(c)。
    - 第二項是預期的 yield，N·Y(c)/Y(SM)，用的是 r 自己的平均（NN 或 truth）。
    - truth 是 w(c)/w(SM)。
    - pseudo-data 的 weights 是 w(data)/mean(w(SM))，總和是 N·Y(data)/Y(SM)。
  - **stitched**：原本的 shape-only 寫法，沒有改。
- config 裡有 `seeds`（`{seed: {sm, networks}}`）時，用 upstream 的 `ensemble` 做 seed 平均；否則用 `full_likelihood`。兩種都需要 `sm`，用 SM_SM。
- 每次都會印出、並寫進 summary：r ≤ 0 的 events 數（`n_negative`），以及它們佔 SM weight 的比例（`negative_sm_weight_frac`）。
- 檢查：truth 的最小值剛好在 data 點（ctGRe −0.0778），這驗證了 extended likelihood 的寫法。

### 設定

- ensembles：主目錄 `local/scan/pilot_<wc>_{all,1p5p}.yml`。
  - `all` = SM_SM + 0.5/1/2.5/5%；`1p5p` = SM_SM + 1/5%。
  - 都是 seed 2、3、4 平均。
- **pseudo-data 放在 hold-out 的 2.5% 點**（`local/scan/data_pilot_2.5p.yml`）：
  - 被 scan 的 WC 設在下表 2.5% 的值，**其他 15 個 WC 都是 0**（`--others sm`）。
  - 樣本的 generation point cg（ctG = −0.5，其他 = 1.5）在這裡**沒有用到**：每個 event 的 weight 是完整的 quadratic w(c)，reweight 到任何點都是精確的，cg 只影響 MC events 分布在哪裡，也就是統計效率。
  - 所以**期望的 best fit 就是下表 2.5% 那一欄**，而不是 cg。truth 曲線的最小值應該剛好在這裡。
  - 要測試 cg，要用 `--others cg` 而且不給 `--data-values`（舊的 cg closure，還沒用 weights_only 跑過）。

pilot 各點的 WC 值（constructive root，yield ratio Y(c)/Y(SM) = 1 + 比例；`local/wc_targets_pilot_train.yml`）：

| WC | 0.5% | 1% | **2.5%（pseudo-data、期望 best fit）** | 5% |
|---|---|---|---|---|
| ctGIm | −0.2859 | −0.4047 | **−0.6405** | −0.9062 |
| ctGRe | −0.0158 | −0.0314 | **−0.0778** | −0.1532 |
| cQj11 | −0.4535 | −0.6543 | **−1.0531** | −1.5027 |
| cQd8 | 0.8099 | 1.402 | **2.7044** | 4.2535 |

- 選項：`--threshold 0 --truth --npoints 201 --window 0.5`，也就是 data ±50%，沒有 drop。
- 輸出：`/eos/user/y/ykao/www/topsbi/scan_pilot_20261018/<wc>_<ens>_data2.5p/`。

### 結果

| | 期望 best fit（2.5% 點的 WC 值） | NN best fit（ci68） | truth best fit（ci68） | 寬度比 NN/truth | r ≤ 0 |
|---|---|---|---|---|---|
| ctGIm | −0.6405 | −0.650（−0.674, −0.622） | −0.6405（−0.649, −0.632） | 3.1 | 0 |
| ctGRe | −0.0778 | −0.0786（−0.0859, −0.0714） | −0.0778（−0.0837, −0.0719） | 1.2 | 0 |
| cQj11 | −1.0531 | **−0.979**（−0.993, −0.968） | −1.0531（−1.062, −1.044） | 1.4 | 5（2e-8） |
| cQd8 | 2.7044 | **2.353**（2.299, 2.395） | 2.7044（2.674, 2.735） | 1.6 | 14（4e-8） |

（`all` ensemble；`1p5p` 的 best fit 完全一樣，區間差 < 0.003。）

### 結論

1. **r ≤ 0 不是問題**：最多 14 個 events，佔 SM weight 4e-8，所以不 drop（item 2）。
2. **ensemble 選哪些點不重要**：`all` 和 `1p5p` 的 best fit 相同。
3. **ctGIm、ctGRe 的 best fit 沒有偏差**，和 truth 的差在 68% 區間以內。
4. **cQj11、cQd8 往 SM 偏**：偏了 7%、13%，大約是 6、7 個 68% 半寬。這是下一步要診斷的。
   - 可能的原因：2.5% 點的 r − 1 只有約 0.025，但 cQj11、cQd8 網路的 rms_bias 約 0.01（見上面 closure），相對於訊號是 40% 的誤差。
   - ctGRe 的 rms_bias 是 0.001，相對訊號約 4%，所以沒有偏差。
5. **NN 的區間一定比 truth 寬**：truth 用每個 event 的 parton-level 資訊，NN 只看 reco features。
   - 寬度比和 feature-set 比較的 frac 一致：ctGIm 用 highlevel 只抓到 2% 的 truth gain，寬度比 3.1。
   - highlevel_cos（cluster 12948860）應該能縮小 ctGIm 的區間。
   - 所以「吻合」的標準是 best fit 沒有偏差，不是區間一樣寬。

### 診斷 1：rate 項和 shape 項（2026-10-18）

Script：主目錄 `local/scan_decompose.py`。輸出：`/eos/user/y/ykao/www/topsbi/scan_pilot_20261018/decompose_data2.5p/`（`<wc>_decompose.{png,yml}`）。

拆法：−2 ln L 拆成兩項，加起來和 scan.py 完全相同。
- rate = 2(ν − n log ν)，ν = N·E_SM[r(c)]，n 是 pseudo-data 的 yield。
- shape = −2 Σ w log(r / E_SM[r])。

另外算了兩個混合：NN shape + truth rate，以及 truth shape + NN rate。

各欄都是 best fit。期望值是 2.5% 點的 WC 值（其他 WC = 0）；truth 的四種組合（truth 總和、truth rate、truth shape、truth shape + NN rate）都剛好落在期望值，表中只列 truth shape + NN rate。

| | 期望 best fit（2.5% 點） | NN 總和 | NN rate | NN shape | NN shape + truth rate | truth shape + NN rate | 曲率 rate / shape（truth） |
|---|---|---|---|---|---|---|---|
| ctGIm | −0.6405 | −0.650 | −0.644 | −0.663 | −0.647 | −0.6405 | 2074 / 24803 |
| ctGRe | −0.0778 | −0.0786 | −0.0778 | **−0.0953** | −0.0786 | −0.0778 | 37090 / 20511 |
| cQj11 | −1.0531 | −0.979 | −1.090 | **−0.974** | −0.979 | −1.0531 | 825 / 22636 |
| cQd8 | 2.7044 | 2.353 | 2.691 | **2.326** | 2.353 | 2.7044 | 64 / 2008 |

yield ratio Y(c)/Y(SM) 在 2.5% 點（NN vs validation 上的 truth）。WC 值是用 train + test + validation 全部（`pretraining_main_highlevel_log`，去掉 w ≤ 0 之前）解出 yield = 1.025 的 root；validation.p 上的 truth 不是剛好 1.025，是 MC 統計漲落（train.p：ctGIm 1.02507、ctGRe 1.02498、cQj11 1.02484、cQd8 1.02497）：
- ctGIm 1.02447 vs 1.02459
- ctGRe 1.02500 vs 1.02502
- cQj11 1.02485 vs 1.02655
- cQd8 1.02509 vs 1.02496

**結論**：
1. **偏差來自 NN 的 shape 項**。換成 truth 的 rate 不會修正偏差（cQj11 −0.979、cQd8 2.353）；只要 shape 換成 truth，偏差就消失。
2. NN 的 rate 項幾乎沒有偏差。cQj11 的 NN yield 比 validation 的 truth 低 0.17%，所以 rate-only 的 best fit 是 −1.09。但這是 validation.p 的 MC 漲落：NN 的 1.02485 和 train.p 的 1.02484 一致。而且 cQj11 的 rate 曲率只有 shape 的 1/27，影響很小。
3. **ctGRe 的 NN shape 也偏了 22%**（−0.095），只是 ctGRe 的資訊主要在 rate（rate 曲率比 shape 大），所以總和看起來沒有偏差。ctGIm 的 shape 偏 3.5%。
4. cQj11、cQd8 的資訊幾乎都在 shape（truth 的 shape 曲率是 rate 的 27、31 倍），所以 shape 的偏差直接變成 best fit 的偏差。
5. cQd8 ≈ 1.8 的 kink 出現在 NN 的 shape 項，就是那 14 個 r ≤ 0 的 events。

所以四個 WC 的 NN shape 都有偏差，但方向不同：
- cQj11、cQd8 往 SM 偏。
- ctGIm、ctGRe 往遠離 SM 的方向偏（|c| 變大）。

接下來的診斷 2（重新校準 r̂）要確認偏差是不是 calibration 造成的。

### 診斷 2：重新校準 r̂，以及 MC 統計漲落（2026-10-18）

Script：主目錄 `local/scan_recalib.py`。輸出：`/eos/user/y/ykao/www/topsbi/scan_pilot_20261018/recalib_data2.5p/`（`<wc>_recalib.{png,yml}`）。pseudo-data 和期望 best fit 同上（2.5% 點，其他 WC = 0）。

做法：
- **重新校準**：validation.p 分成兩半。
  - 在偶數 events 上，依 data 點的 r̂ 分 quantile bins（20、50、200 個），每個 bin 的 r̂(c) 換成 E_SM[t(c) | bin]。
  - 套到奇數 events 上，只用奇數 events scan。
  - bins 在所有 c 固定，所以曲線是平滑的。第一版每個 c 重新分 bins，大 weight 的 events 會換 bin，曲線鋸齒狀、best fit 不可靠，已經改掉。
  - 另外對 truth 本身做同樣的分 bin（「truth recalibrated」），當作這個方法本身的雜訊。
- **MC 漲落**：同一個 ensemble、同樣的 likelihood，只用 4 個互不重疊的子樣本（event index mod 4）各自 scan，看 NN best fit 的分散。

結果（best fit；期望值 = 2.5% 點的 WC 值）：

| | 期望 | NN 全部 | NN 奇數半 | NN 偶數半 | NN 4 個子樣本 | 重新校準 20 / 50 / 200 bins（奇數半） | truth 分 50 bins |
|---|---|---|---|---|---|---|---|
| ctGIm | −0.6405 | −0.650 | −0.663 | −0.634 | −0.609, −0.663, −0.660, −0.663 | −0.660 / −0.663 / −0.650 | −0.647 |
| ctGRe | −0.0778 | −0.0786 | −0.0790 | −0.0786 | −0.0782, −0.0790, −0.0786, −0.0790 | −0.0782 / −0.0782 / −0.0782 | −0.0778 |
| cQj11 | −1.0531 | −0.979 | −0.900 | −1.079 | −0.969, −0.869, −1.201, −0.953 | −1.032 / −0.995 / −0.879 | −1.037 |
| cQd8 | 2.7044 | 2.353 | 2.204 | 2.556 | 2.515, 1.880, 2.596, 2.840 | 2.772 / 2.799 / 2.610 | 2.664 |

**結論**：
1. **cQj11、cQd8 的「偏差」和 MC 統計漲落同樣大**。
   - 4 個子樣本的分散：cQj11 約 ±0.14，cQd8 約 ±0.4。換算到全部樣本大約是 ±0.07、±0.2。
   - 全部樣本的偏差：cQj11 0.074、cQd8 0.35，大約 1σ、1.7σ。
   - 奇數、偶數兩半就差了 0.18、0.35。
2. **scan 的 ci68 嚴重低估了這個 closure test 的不確定度**。
   - pseudo-data 是同一批 MC events reweight 出來的，所以 truth 在任何子樣本上都剛好落在期望值。
   - NN 的 best fit 則由少數大 weight、t 和 r̂ 差很多的 qq̄ events 決定（見 `README_featsets_ywk.md`：cQj11 0.01% 的 events 佔 Var(α) 的 50%）。
   - ci68 只反映 Σw 個「data events」的 Poisson 統計，不包含這些 MC events 本身的漲落。
3. **重新校準**：20、50 bins 讓 cQj11、cQd8 靠近期望值（−1.03 / −1.00、2.77 / 2.80），200 bins 又離開（每個 bin 的 events 太少，校準本身有雜訊）。truth 自己分 50 bins 也偏了 0.016、0.04。這些變化都在上面 MC 漲落的範圍內，**所以不能說是 calibration 的問題，也不能排除**。
4. ctGRe：子樣本分散只有 ±0.0004，偏差 0.0008，沒有問題。ctGIm：分散約 ±0.025，偏差 0.01，沒有問題。
5. **診斷 1 的「NN shape 偏差」也要用同樣的方式解讀**：cQj11、cQd8 的資訊幾乎都在 shape，而 shape 正是被大 weight events 主導的部分。

### 診斷的待辦

- [x] 把 −2ΔlnL 拆成 rate 和 shape：偏差來自 NN 的 shape 項（上面）。
- [x] 重新校準 r̂：結果在 MC 漲落範圍內，無法判定（上面）。
- [ ] **closure test 的不確定度**：用 bootstrap（對 events 重抽，算 NN best fit 的分布）估計 MC 漲落，再看偏差有幾個 σ。這比 scan 的 ci68 更適合判斷 closure。
- [ ] 換 pseudo-data 點（0.5%、1%、SM），看偏差是不是和 |c| 成比例；也要附上 bootstrap 的不確定度。
- [ ] 如果要降低漲落：用更多 MC（train.p 也拿來當 pseudo-data，網路沒有看過的部分只有 validation.p，要小心），或者看大 weight events 的處理。

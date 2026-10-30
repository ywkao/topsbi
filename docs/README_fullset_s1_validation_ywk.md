# Full set 第 1 階段：validation 和 1D scan（2026-10-23）

第 1 階段的網路（cluster 13415949，SM_SM + 16 個 WC × {±5%, ±10%} × seeds 2、3、4）的 validation 和 1D scan。計畫見 `README_fullset_plan_ywk.md`。

## 1. 環境（每次開新的 terminal 都要做）

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
W=/eos/home-y/ykao/SWAN_projects/analysis/topsbi-wt/rebased-fa6f982
export PYTHONPATH=$W/src:$W/scripts:$PYTHONPATH
R=/eos/home-y/ykao/SWAN_projects/analysis/topsbi        # configs 在這裡的 local/
cd $W
```

- 程式碼用 `rebased-fa6f982` worktree（branch `port/rebased-fa6f982`），**不是**訓練用的 `port-3507174`。原因是 `validation.py` 也需要支援 `year_loc: null`：這個 patch 只在這個 worktree，**還沒 commit**。
- 不需要 GPU。validation 和 scan 讀的是 `local/cpu_configs/fullset_s1/<label>/*.yml`（`device: cpu`）。
- 在 SWAN notebook 裡：每個 cell 用 `!` 開頭跑同樣的指令，或者用 `%%bash` cell。環境變數要寫在同一個 cell 裡（`%%bash` 的每個 cell 都是新的 shell）。

## 2. Configs（由 generator 產生，不用手改）

`python3 local/make_training_configs.py local/plans/fullset_s1.yml`（在 `$R` 下執行）會產生：

| 檔案 | 用途 |
|---|---|
| `local/condor/configs/fullset_s1/*.yml` | 訓練（cuda） |
| `local/cpu_configs/fullset_s1/<label>/*.yml` | 同樣的 configs 改成 `device: cpu`，給 validation 和 scan 讀 |
| `local/validation/fullset_s1.yml` | validation：全部 65 個點 × 3 seeds，加上 `name`、`lumi`、`year_loc: null`、`n_threads: 8` |
| `local/scan/fullset_s1_<wc>.yml` | 1D scan：SM_SM + 這個 WC 的 4 個點，3 seeds 平均 |
| `local/scan/data_2.5p.yml` | pseudo-data：每個 WC 在 2.5% 的 constructive root（不是訓練點，是 hold-out） |

**每個點要有自己的目錄**（`<label>/`）：validation 的 ensemble cache 用 config 所在的目錄當網路的名字。第一次跑時 configs 都放在同一個目錄，65 個網路全都叫 `'.'`，attribution、drop test、point search 都把它們當成同一個網路（point search 找到 0 個候選點）。

## 3. Validation

```bash
python3 -m topsbi.validation -c $R/local/validation/fullset_s1.yml --refit   # 約 15–20 分鐘
```

- `--refit`：重新 fit ensemble 並寫入 cache。之後只改畫圖或門檻時可以省略，會直接讀 cache；網路或 configs 變了就要再加。
- 輸出在 config 的 `name`：`/eos/cms/store/user/ykao/topsbi/results/port/3507174_fullset/validation_fullset_s1/`
  - `ensemble/`：cache（每個 seed 的 γ、每個網路的 per-event ratio）。
  - `ensemble_diagnostics.yml`：每個 component 的 Z、worst feature、rms_bias；flagged 清單；suspects 和 drop test；point search。
  - `proposed_points.yml`：建議新增或重訓的點。
  - `plots/`：`z_heatmap_{compact,full}.png`、`closure/{flagged,passed}/`、`attribution/`、`drop_tests_*.png`、`network_misclosure.png`、`point_search_sigma_*.png`、`fit_residuals*.png`。
- 怎麼讀：先看 `z_heatmap_compact.png`（Z ≥ 5 會加紅框），再到 `closure/flagged/` 看 flagged component 在它最差的 feature 上的 closure 圖。
- 列出各 component 的摘要：
  ```bash
  python3 -c "
  import yaml; d = yaml.safe_load(open('/eos/cms/store/user/ykao/topsbi/results/port/3507174_fullset/validation_fullset_s1/ensemble_diagnostics.yml'))
  for k, c in sorted(d['components'].items(), key=lambda x: -x[1]['max_Z']):
      print(f\"{k:10s} flagged={c['flagged']!s:5s} maxZ={c['max_Z']:6.2f} worst={c['worst_feature']:15s} rms_bias={c['rms_bias']:.4f}\")
  print(d['suspects']['drop'], d['suspects']['retrain'])"
  ```

## 4. 1D scan

一個 WC：
```bash
O=/eos/user/y/ykao/www/topsbi/scan_fullset_s1_20261023
python3 scripts/scan.py -p $R/local/scan/fullset_s1_cQu1.yml --wc cQu1 \
    --data-values $R/local/scan/data_2.5p.yml --others sm \
    --threshold 0 --truth --npoints 201 --window 0.5 -o $O/cQu1_data2.5p        # 每個約 30 秒
```

- 選項和 pilot 一樣：pseudo-data 放在 2.5% 點，其他 WC = 0；scan 範圍是 data ±50%；`--threshold 0` 表示不 winsorize；`--truth` 畫出解析的 truth 曲線。
- 加 `--drop-negative`：去掉在 grid 上任何一點 r ≤ 0 的 events（NN 和 truth 用同一組 events）。輸出放到 `<wc>_data2.5p_dropneg`。
- 輸出：`<wc>_scan.{png,yml,npz}`。yml 裡有 `best_fit`、`ci68`、`ci95`、`truth`，以及 `n_negative`（r ≤ 0 的 events 數）。

16 個 WC 一起跑：
```bash
for w in ctGIm ctGRe cQj38 cQj18 cQu8 cQd8 ctj8 ctu8 ctd8 cQj31 cQj11 cQu1 cQd1 ctj1 ctu1 ctd1; do
  python3 scripts/scan.py -p $R/local/scan/fullset_s1_$w.yml --wc $w --data-values $R/local/scan/data_2.5p.yml \
      --others sm --threshold 0 --truth --npoints 201 --window 0.5 -o $O/${w}_data2.5p 2>&1 | grep "^{'wc'"
done
```

## 5. Bootstrap：closure 的 MC 統計不確定度

scan 的 ci68 只包含 data（Poisson）統計，不包含 MC events 本身的漲落。四費米子由少數大 weight 的 events 主導，MC 漲落比 ci68 大好幾倍（pilot 的診斷 3）。所以 **NN 的 best fit 偏差要和 bootstrap std 比較，不能和 ci68 比**。

```bash
cd $R
python3 local/scan_bootstrap.py -p local/scan/fullset_s1_cQu1.yml --wc cQu1 \
    --data-values local/scan/data_2.5p.yml -o $O/bootstrap        # 500 replicas
```

- 輸出：`$O/bootstrap/<wc>_bootstrap.{png,yml}`。pull =（NN best fit − 期望值）/ bootstrap std。

## 6. 結果（2026-10-23）

### 各網路（`performance.yml`）

| 群 | pull | rms_bias |
|---|---|---|
| ctGRe、ctGIm | 0.4–1.4 | 0.002–0.012 |
| 四費米子 octet | 大多 0.4–1.6 | 0.005–0.03 |
| 四費米子 singlet | 大多 1.5–4，最高 6.9（cQu1 5p_neg） | 0.008–0.04 |

### Validation

- rank 33（= SM + 16 個一次項 + 16 個二次項；交叉項還沒有點），沒有 residual outliers。
- **33 個 components 中 15 個被 flag**（Z > 5）：
  - ctGRe²（Z 28）、ctGRe（12）：Z 大是因為 ctGRe 的 MC 誤差很小。scan 沒有偏差（下面），和 pilot 的結論一樣，目前可以忽略。
  - octet 的二次項 cQj18²（21）、cQj38²（11）、cQd8²（10）：都在 log m_ttbar、β\*、log pT(t_ℓ)，也就是高能量的尾巴。
  - sm（16，在 Δφ、y_tt、c_han）。
  - singlet 和 octet 的一次項：cQj31、ctu1、ctd1、cQj18、ctj1、ctj8；以及 ctGIm²、cQj11²、ctj1²。
- drop test 接受去掉 4 個網路：cQd8_5p、cQd8_10p、ctd8_5p、ctd1_10p（去掉之後 closure 變好）。
- `retrain`：cQu8_10p_neg（seed 4）。
- point search：4 個候選，都是重訓上面那些點，沒有新的點。

### 1D scan（pseudo-data 在 2.5% 點）

> **2026-10-24 補充**：這裡的 scan 每個 WC 只用自己的 4 個網路 + SM_SM 來 fit。全部網路一起 fit（validation 用的方式）時，沒加權的 lstsq 讓 γ_sm 壞掉，ctG 也會變差、r ≤ 0 大增；SM_SM 列加權可以修正。見 `README_fullset_s2_ywk.md` 第 5 節。

偏差以 truth 的 ci68 半寬 σ_t 為單位；寬度比 = NN ci68 / truth ci68。

| WC | data | NN best fit | 偏差 / σ_t | 寬度比 | r ≤ 0 events | `--drop-negative` 之後的偏差 / σ_t |
|---|---|---|---|---|---|---|
| ctGIm | −0.6405 | −0.6661 | −3.0 | 2.56 | 0 | −3.0 |
| **ctGRe** | −0.0778 | −0.0786 | **−0.1** | 1.22 | 0 | −0.1 |
| cQj38 | 2.4966 | 2.3218 | −8.2 | 1.52 | 55 | −8.8 |
| cQj18 | 1.6129 | 1.5887 | −1.1 | 0.29 | 185 | −5.2 |
| cQu8 | 2.0583 | 1.8731 | −7.7 | 1.00 | 528 | **0.0** |
| cQd8 | 2.7044 | 2.7450 | 1.3 | 0.23 | 71 | −14.6 |
| ctj8 | 1.3210 | 1.3540 | 1.8 | 0.26 | 151 | −3.2 |
| ctu8 | 2.1676 | 2.3085 | 5.5 | 0.12 | 153 | −6.3 |
| ctd8 | 2.7905 | 2.8742 | 2.7 | 0.42 | 100 | −8.2 |
| cQj31 | 1.0738 | 1.0255 | −5.5 | 1.28 | 298 | −6.7 |
| cQj11 | −1.0531 | −1.0847 | −3.5 | 1.41 | 300 | −2.3 |
| cQu1 | 1.2815 | 1.1021 | −15.9 | 1.28 | 68 | −16.5 |
| cQd1 | 1.6948 | 1.5423 | −10.9 | 1.27 | 540 | −3.6 |
| ctj1 | 0.9615 | 0.9327 | −3.2 | 1.42 | 43 | −3.2 |
| ctu1 | −1.2789 | −1.2213 | 5.3 | 1.45 | 90 | 5.9 |
| ctd1 | −1.5876 | −1.3653 | 16.5 | 1.51 | 58 | 17.1 |

- **ctGRe 通過**（偏差 0.1σ_t）。ctGIm 偏差 −3σ_t，寬度是 truth 的 2.6 倍（和 pilot 一樣：NN 只看 reco features）。
- **四費米子都有 r ≤ 0 的 events**（43–540 個）：每個 event 的二次式 γ0 + γ1·c + γ2·c² 在 grid 的某處變成負值。log r 在那裡不連續，NN 曲線出現鋸齒，octet 的 ci68 也因此比 truth 窄（寬度比 0.12–0.42，不合理）。
  - cQu8 的偏差完全來自這些 events（去掉後 0.0σ_t）；其他 WC 去掉之後偏差還在。
- 以 σ_t 為單位，四費米子偏差 3–17σ_t；換成 c 是 0.03–0.22。這和 pilot 的 MC 漲落（cQd8 2.5% bootstrap std 0.22）同一個量級，**要看 bootstrap 才能判斷是否顯著**（下一節）。

### Bootstrap（500 replicas，`$O/bootstrap/`）

| WC | 期望 | NN best fit | bootstrap std | pull | σ_t（參考） |
|---|---|---|---|---|---|
| cQu1 | 1.2815 | 1.1030 | 0.105 | −1.7 | 0.011 |
| ctd1 | −1.5876 | −1.3634 | 0.088 | **+2.5** | 0.013 |
| cQj11 | −1.0531 | −1.0857 | 0.063 | −0.5 | 0.009 |
| cQd8 | 2.7044 | 2.7418 | 0.215 | +0.2 | 0.031 |

- bootstrap std 是 σ_t 的 7–10 倍。換成 bootstrap 的單位之後，偏差最大的 cQu1、ctd1 只有 −1.7 和 +2.5。
- 和 pilot 一樣：**四費米子的 closure 被 MC 樣本的大小限制**，偏差大多可以用 MC 漲落解釋。ctd1 的 2.5σ 要留意（drop test 也挑出 ctd1_10p）。

## 7. 結論和下一步

1. **流程可以用了**：validation（cache、Z heatmap、drop test、point search）和 16 個 1D scan 都能跑。補了 `validation.py` 的 `year_loc: null`（未 commit），並把 CPU configs 改成一個點一個目錄。
2. **ctGRe 通過**，ctGIm 偏差約 −3σ_t。四費米子的偏差以 σ_t 看很大，但以 bootstrap 看大多 < 2σ，所以目前**不能說網路（包括 singlet）有問題**，也不能說沒問題：MC 統計不夠，驗證不了。
3. **r ≤ 0 是真正的問題**：每個四費米子 WC 有 43–540 個 events 的 morph 結果在 grid 上變成負值。它讓 scan 曲線不平滑（octet 的 ci68 比 truth 還窄），cQu8 的偏差也完全來自這些 events。要決定怎麼處理：scan 時去掉、在 γ 上加限制，或者看是哪些網路造成的。
4. **第 2 階段**：上面的結果沒有指出需要調整訓練設定，validation 也只建議重訓 4 個現有的點，所以可以送。但 r ≤ 0 在交叉項很可能更多（權重更極端），建議先決定第 3 點的處理方式。
5. 待辦：
   - 重訓 point search 建議的點：cQd8_5p、cQd8_10p、ctd8_5p、ctd1_10p，以及 cQu8_10p_neg 的 seed 4。換 seed 或者先看它們的 performance。
   - 其他 12 個四費米子 WC 的 bootstrap（每個約 3–5 分鐘）。
   - commit `validation.py` 的 patch。

# Pilot 點的 weight 和真實 LR 分布，以及要不要 sanitize（ywk，2026-10-17）

目的：在 rebase 後（`port-0c49da2`，weights_only）訓練 1D pilot 之前，先看訓練資料的 weight 和真實 likelihood ratio 分布，決定要不要把 sanitize 加回訓練。

**結論：pilot 不做任何 sanitize；ctGRe 的 destructive 點要換掉。**

## 檔案

| | 位置 |
|---|---|
| Script | 主目錄 `local/diag_weights.py` |
| Pilot 的 WC 點 | 主目錄 `local/wc_targets_pilot.yml` |
| 圖、表 | `/eos/user/y/ykao/www/topsbi/weights_20261017_pilot_sanitize/` |

`wc_targets_pilot.yml` 由下面的指令產生（用 train、test、validation 合計）：

```sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
cd /eos/home-y/ykao/SWAN_projects/analysis/topsbi
export PYTHONPATH=/eos/home-y/ykao/SWAN_projects/analysis/topsbi-wt/port-0c49da2/src:$PYTHONPATH
D=/eos/cms/store/user/ykao/topsbi/pretraining_main_highlevel_log
python local/wc_targets.py $D local/wc_targets_pilot.yml 0.001 0.005 0.01 0.05
python local/diag_weights.py $D local/wc_targets_pilot.yml /eos/user/y/ykao/www/topsbi/weights_20261017_pilot_sanitize
```

`diag_weights.py` 會把 train.p（1.4M events）整份讀進 memory，在 lxplus 上可以跑。

## 設定

- **資料**：`pretraining_main_highlevel_log`，圖用 train，`summary.yml` 另外有 validation（結果幾乎一樣）。
- **Pilot 點**：ctGIm、ctGRe、cQj11、cQd8，每個 WC 單獨變動，總 yield 增加 0.1%、0.5%、1%、5%。每個目標有兩個解：
  - **p（constructive）**：b·c > 0 的解，即現有的 `SM_<wc>` 命名。
  - **n（destructive）**：b·c ≤ 0 的解，即現有的 `<wc>_<wc>` 命名。
  - 共 4 × 4 × 2 = 32 個點，c0 都是 SM。
- **Weights**：和 weights_only 一樣，w(c) = coefs @ T(c)。lumi = 1、乘 1000 都省略，因為所有 weight 都除以 train.p 的 SM 平均 weight（`train.py` 也是這樣做），這些常數會消掉。
- **Sanitize**：用 0004d96（port/dev，當時只用在 stitched）的三個條件：
  - `negative`：w0 ≤ 0 或 w1 ≤ 0；
  - `lr_cap`：w1/w0 在 [1e-4, 1e4] 之外；
  - `weight_cap`：w0 或 w1 ≥ 100（以 SM 平均 weight 為單位）。
- **Union mask**：任何一個 pilot 點丟掉的 event 就全部丟掉，讓所有網路用同一組 events。每個點自己的 mask 會讓不同網路在不同的 events 上訓練，和 `--drop-negative` 讓 scan 不能互相比較是同一個問題。

rebase 之後的程式碼（`port-0c49da2`）**沒有** sanitize：0004d96 只在 port/dev，upstream 的 weights_only 也不做。所以這裡回答的是「要不要把它加回來」。

## 圖怎麼看

每個 WC 一張圖（`<wc>.png`），每一列是一個 yield 目標，三欄是：

1. **左：每個 event 的 w1**，單位是 SM 平均 weight，log 軸。w1 < 0 的 events 畫成同顏色的淡色細線，橫軸是 |w1|。
2. **中：每個 event 的真實 LR，log10(w1/w0)**，只畫 w1/w0 > 0 的 events。
3. **右：每個 m_ttbar bin 的 Σw1/Σw0**（帶點的實線），以及 sanitize 後除以 sanitize 前的比值（帶方塊的虛線）。

顏色：藍色是 constructive，紅色是 destructive。實線是全部 events，虛線是 union sanitize 之後。

另外兩張：`SM.png`（SM weight 分布和 m_ttbar 的 SM yield）、`dropped.png`（每個條件丟掉的 events 比例，以及每個點自己的 mask 丟掉的 Σw1 比例）。數字在 `summary.txt`（train）和 `summary.yml`（train、validation）。

## 結果

### SM 的 weight 很乾淨

- 沒有 w0 ≤ 0 的 event。
- 最大的 w0 是平均的 31.5 倍，低於 weight_cap 的 100。
- n_eff/N = 0.76，n_eff = (Σw)²/Σw²。

### 大部分的點：sanitize 的影響可以忽略

ctGIm、cQj11、cQd8 的全部點，以及 ctGRe 的 constructive 點：

- 每個點自己的 mask 最多丟 0.057% 的 events、0.15% 的 Σw1（cQj11_5p）。
- 網路要學的 yield ratio Σw1/Σw0，sanitize 後最多變 0.2%（ctGIm_5p_n）。
- n_eff/N 在 0.54–0.77。
- union mask 丟 0.141% 的 events（validation 0.144%）。

這和以前在 stitched 上的結論一樣（`README_summary_ywk.md` 嘗試 9：最多丟 0.17% 的 events，不是偏差的原因）。

### 三個條件各切掉什麼

| 條件 | 切掉的 events | 判斷 |
|---|---|---|
| `lr_cap` 下限（w1/w0 ≤ 1e-4） | 只有 ctGRe 的 n 點有，每點 12–16 個；其他 28 個點一個都沒有。放寬到 1e-6 也幾乎沒差 | 不重要 |
| `lr_cap` 上限（w1/w0 ≥ 1e4） | union 丟 0.06% 的 events，主要來自 cQj11 和 cQd8（cQj11_5p 各 542、601 個）。這些 events 的 w0 幾乎是 0，但 w1 不小，cQj11_5p 因此丟掉 0.14–0.15% 的 Σw1 | **會切掉真實的 EFT 訊號**：SM 幾乎沒有、EFT 才有的 phase space，正是網路要學的地方 |
| `weight_cap`（w ≥ 100） | 只在 ctGRe 的 n 點有作用 | 在 ctGRe 的 n 點把最高 m_ttbar 的 bin 壓低約 30%，**造成 bias** |
| `negative`（w ≤ 0） | 1%、5% 的點最多 575 個（cQd8_5p_n，c = −7.25）；ctGRe 的 n 點 330–341 個 | 比例很小；upstream 讓負 weight 直接進 BCE |

所以在 weights_only 之下，這三個條件沒有一個在處理真正的問題。

### ctGRe 的 destructive 點不是近 SM 的點

| | constructive | destructive |
|---|---|---|
| c | −0.003（0.1%）到 −0.15（5%） | 4.68 到 4.83 |
| 真實 LR 分布（中欄） | 在 r = 1 的細針 | 從 1e-4 到 1e4 |
| Σw1/Σw0 隨 m_ttbar（右欄） | 平的，約 1 | 從 0.5（低 m_ttbar）升到 7（> 3 TeV） |
| n_eff/N | 0.76 | 0.05 |
| union sanitize 丟掉的 Σw1 | 0.08% | 4.2%，yield ratio 偏 −4.1% |

**為什麼差這麼多**：每個 event 的真實 LR 是 c 的二次式，

  r_i(c) = 1 + α_i·c + β_i·c²

α_i 是干涉項（一次項），β_i 是 EFT 平方項（二次項），每個 event 都不一樣。總 yield 只是它們的加權平均：

  Y(c)/Y_SM = 1 + b·c + a·c²，b = ⟨α⟩，a = ⟨β⟩

- ctGRe：b = −0.316、a = 0.068，一次項很大（chromomagnetic dipole 和 SM 的 gg → tt̄ 干涉很強）。
- **constructive**：c ≈ t/b 很小，r_i ≈ 1 + α_i·c，每個 event 都只偏離 1 一點點。
- **destructive**：c ≈ −b/a ≈ 4.7，是 b·c + a·c² ≈ 0 的位置，總 yield 幾乎不變。但這只是**平均**抵消：β 隨能量增加得比 α 快很多（dimension-6 平方約 E⁴，干涉約 E²），所以低 m_ttbar 由負的一次項主導（r < 1），高 m_ttbar 由二次項主導（r ≫ 1），兩邊加總剛好抵消。

其他 WC 的 destructive 點的位置：

| WC | b | a | −b/a | destructive 的 c |
|---|---|---|---|---|
| ctGIm | −0.0001 | 0.061 | 0.002 | ±0.13 到 ±0.9，和 constructive 對稱（CP-odd，總 yield 沒有一次項） |
| cQj11 | −0.0014 | 0.021 | 0.067 | 0.25 到 1.57，和 constructive 差不多遠 |
| cQd8 | 0.0049 | 0.0016 | −3.0 | −3.2 到 −7.3，也很遠；w1 ≤ 0 的 events 在這些點最多 |
| ctGRe | −0.316 | 0.068 | 4.67 | 4.68 到 4.83 |

「constructive 和 destructive 都取」只對 |b| 相對 a 很小的 WC 合理。

## 決定和下一步

- **pilot 不做 sanitize**：大部分點不需要，ctGRe 的 n 點反而會被 weight_cap 弄出 bias。之後某個網路的 calibration 有問題，再回頭檢查負 weight。
- **ctGRe 的 n 點要換掉**，候選：
  - 和 constructive 相同 |c| 的正值；
  - 或用更大的 yield 目標（10%、20%）。
- cQd8 的 n 點（c = −3 到 −7）也離 SM 很遠，要決定是否保留。
- 0.1% 的點：weight 分布正常，但真實 LR 幾乎全在 1 附近，網路能學的訊號非常小（以前 0.1% 的 closure 也失敗，`README_closure_ywk.md`「0.1% / 1% 建設性干涉點」）。要決定保留還是換成 2.5% 或 10%。

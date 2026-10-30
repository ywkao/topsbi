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

## 程式碼版本和方法（2026-10-20 確認）

- **程式碼**：commit `0c49da2`（worktree `topsbi-wt/port-0c49da2`；port/rebased 在 2026-10-17 rebase 到 eddie/main c810b07 之後的版本，0c49da2 本身只改 READMEs）。
  - 確認方式：condor 的 `run.sh` 會印出 `git log --oneline -1`，三個 clusters 的 `~/topsbi-condor/logs/<cluster>.0.out` 都是 `0c49da2 update READMEs`：12702676（pilot，含 highlevel 基準）、12707721（top30、compact、lowlevel、all）、13369189（highlevel_cos）。
- **方法：`weights_only`**（`pilot/`、`pilot_*/` 的 95 個 configs 都是），`lumi: [1.0]`、`year_loc: null`、`group_cols: []`，holdout `validation.p`。網路學 r = w(c1)/w(SM)，包含 rate。
- 訓練設定：`wo_log_bs10k` 模板（batch 10000、cosine、lr 1e-3、256×3、dropout 0.2、100 epochs、patience 20）。
- 六組的程式碼、方法、訓練設定完全相同，只差 `data`、`features`、`holdout`、`name`。

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

## 確認：highlevel_cos（cluster 12948860 失敗，重送 13369189）

- 資料：`/eos/cms/store/user/ykao/topsbi/pretraining_main_highlevel_cos_log_pilot/`
  - 從 `pretraining_main_all_log_pilot` 用 `local/select_features.py` 取出 19 個欄位。
  - 檢查過：前 13 欄和 `pretraining_main_highlevel_log_pilot` 完全相同（max diff 0），coefs 也完全相同。
- features 檔：主目錄 `features_highlevel_cos_log.yml`。
- configs：`local/condor/configs/pilot_highlevel_cos/`（4 WC × seed 2、3 = 8 個）。和 top30 的 configs 只差 `data`、`features`、`holdout`、`name`。
- condor list：`~/topsbi-condor/list_featsets_highlevel_cos.txt`
- 輸出：`results/port/0c49da2_pilot_highlevel_cos/<wc>_5p/seed_<n>`
- 送出（2026-10-18 這次用了相對路徑 `../topsbi-wt/...`，8 個 jobs 全部失敗，要用絕對路徑重送）：`cd ~/topsbi-condor && condor_submit code=/eos/home-y/ykao/SWAN_projects/analysis/topsbi-wt/port-0c49da2 configlist=list_featsets_highlevel_cos.txt submit_list.sub`
- 跑完後：`local/compare_featsets.py` 的 `SETS` 已經加了 `highlevel_cos`，直接重跑即可。
- 判斷標準：
  - ctGIm 的 delta 接近 all（+4.2 到 +4.4）。
  - 其他 WC 的 delta 和 0 一致（在配對誤差和 seed 差異以內）。

### 結果（cluster 13369189，2026-10-19 跑完；比較 2026-10-20）

- 重送的 cluster 13369189：8/8 正常結束（return value 0）。
- 輸出：`/eos/user/y/ykao/www/topsbi/featsets_trained_20261020/`（六組一起；舊的 `featsets_trained_20261018/` 保留）。

delta（相對 highlevel，配對誤差），單位 1e-4（seed 2 / seed 3）：

| | highlevel_cos | all（對照） |
|---|---|---|
| ctGIm | **+7.72 ± 0.19 / +7.94 ± 0.20** | +4.41 ± 0.19 / +4.20 ± 0.21 |
| ctGRe | −0.008 ± 0.002 / −0.006 ± 0.001 | −0.022 ± 0.002 / −0.027 ± 0.002 |
| cQj11 | +2.4 ± 2.7 / +0.7 ± 1.1 | −4.2 ± 2.5 / −2.6 ± 1.4 |
| cQd8 | +0.14 ± 0.38 / −0.08 ± 0.35 | −1.9 ± 0.6 / −1.0 ± 0.6 |

- ctGIm：gain 10.2 / 10.3（highlevel 2.5 / 2.4，all 6.9 / 6.6），frac 0.085（highlevel 0.02）。calibration 比 all 好：pull 0.59 / 0.52、rms_bias 0.005 / 0.004（all 0.88 / 1.15、0.008 / 0.012）。

**結論**：
1. **ctGIm：highlevel_cos 的 gain 是 highlevel 的 4.2 倍，也比 all 多 1.8 倍**，兩個 seeds 一致。超過預期（原本預期接近 all）：all 多出的 55 個 features 沒有 ctGIm 的資訊，反而讓同樣的網路和 epochs 學得比較差，和 CP-even 的 WC 上 all 比 highlevel 低 1–8% 是同一個現象。
2. **CP-even 的 WC 沒有損失**：cQj11、cQd8 和 0 一致；ctGRe 低 0.006–0.008（統計上 3–4σ，但只是 gain 的 2–3%，和 top30、compact 同等級）。
3. **highlevel_cos 當成接下來訓練的 default feature set**（`local/plans/hlcos.yml`、`README_jupyter.md`）。

## 怎麼重現（指令）和指標的意義（2026-10-20）

### 指令

```bash
cd /eos/home-y/ykao/SWAN_projects/analysis/topsbi
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH=/eos/home-y/ykao/SWAN_projects/analysis/topsbi-wt/port-0c49da2/src:$PYTHONPATH
python3 local/compare_featsets.py /eos/user/y/ykao/www/topsbi/featsets_trained_<YYYYMMDD>   # 約幾分鐘，CPU
```

- 要比較哪些組：`local/compare_featsets.py` 開頭的 `SETS`、`PILOT`（WC）、`SEEDS`。config 路徑由 `config_path()` 決定：highlevel 讀 `local/condor/configs/pilot/`，其他讀 `local/condor/configs/pilot_<set>/`。
- 輸出：`compare_featsets.txt`（表）、`.yml`（數字）、`.png`（4 列 × 4 個 WC）：
  - 第 1 列 gain，第 2 列 delta（配對，誤差棒是配對誤差），第 3 列 rms_bias，第 4 列 pull。
- gain、delta 是 script 在 holdout 上重新算的；auc、pull、rms_bias 直接讀各網路訓練完寫的 `complete/performance.yml`。
- 檢查某個網路的 rms_bias 是被哪些 bins 決定的：主目錄 `python3 local/rmsbias_check.py <config.yml> ...`（同樣的 PYTHONPATH）（每個 bin 的偏差和誤差，見下面 rms_bias = 0 的說明）。

### gain、nats、paired

- **常數分類器**：每個 event 都輸出同一個 s = Σw1 / Σ(w0 + w1)。它只知道 c1 和 c0 的總 yield 比，不看 x。
- **gain over constant** = L_const − L_net：網路的 weighted BCE 比常數分類器低多少，也就是網路從 x 的 shape 多拿到多少資訊。0 = 什麼都沒學到（例如 lowlevel 的 ctGRe）。BCE 是 proper scoring rule，所以 gain 越大，網路抓到越多 c1 vs c0 的資訊；同一個 WC 點之間可以直接比大小（ctGIm：highlevel_cos 10.3 vs highlevel 2.4 → 4 倍的資訊）。
- **nats**：資訊（log loss）的單位，用自然對數 ln 算時叫 nat（用 log2 叫 bit，1 nat = 1.44 bits）。「1e-4 nats per SM event」：除以 SM 的 event 數（Σw0，w 已經除以平均 w0），所以是每個 SM event 平均多少資訊；不同 WC 之間的數值大小反映 5% 點本身多好分（cQj11 的 shape 變化大，gain 大）。
- **frac** = gain / gain(真實 ratio)：真實 ratio 用了 parton-level 資訊，網路只看 reco features，所以即使網路完美也遠小於 1。
- **paired（配對）**：delta = gain(set) − gain(highlevel) 在**同一組 holdout events、同一個 seed** 上算，誤差來自每個 event 的 loss 差 l_highlevel − l_set 的分布：√N·std(差)/N。兩個網路共有的漲落（例如少數大 weight 的 events 讓兩者的 loss 一起變大）在差裡抵消，所以 delta 的誤差（ctGIm ±0.2）比單獨 gain 的統計誤差小很多，小的差異也看得出來。

### rms_bias 和 pull（upstream `src/topsbi/tools/plots.py` 的 `lr_calibration_curve`、`lr_calibration_difference`）

兩者都用 holdout，把 events 依 r̂ 分成 50 個 quantile bins（每 bin 的 event 數相同，`threshold=0` 包含兩端），events 用 w0（SM）加權。每個 bin 算 ⟨r̂⟩、真實 ⟨r⟩ = ⟨w1/w0⟩ 和它的統計誤差 σ。

- **rms_bias**（偏差有多大，相對值，不隨資料量變）：
  - 每個 bin 的相對偏差 rel = (⟨r̂⟩ − ⟨r⟩)/⟨r⟩，相對誤差 rel_err = σ/⟨r⟩，權重 wk = bin 的 Σw0 比例。
  - rms_bias = √max(Σ wk·rel² − Σ wk·rel_err², 0)：觀察到的偏差平方平均，**扣掉完美網路也會有的統計漲落**。
  - 0.01 = 平均有 1% 的偏差。
- **pull**（偏差是否顯著，會隨資料量變大）：
  - 每個 bin 的 pull = |⟨r⟩ − ⟨r̂⟩| / σ(⟨r − r̂⟩)。
  - pull = 平均 pull − √(2/π)。完美網路的 pull 是標準常態，|Z| 的平均是 √(2/π) ≈ 0.80，所以完美時 pull ≈ 0；誤差 `pull_err` = √((1 − 2/π)/50) ≈ 0.085。
  - 讀法：pull = 1 表示 bins 平均偏了約 1.8σ；pull = 2.5 是平均 3.3σ。holdout 越大，同樣大小的偏差 pull 越大，所以要和 rms_bias 一起看：**rms_bias 小、pull 大 = 偏差很小但資料多到看得出來**。
  - 這次：ctGIm 的 highlevel_cos 0.5–0.6，比 all（0.9–1.15）好；cQj11 在所有 feature sets 都是 1.5–2.8，和 features 無關，是網路本身在這個點的限制（pilot closure 也看到 cQj11 rms_bias 約 0.01）。

**cQj11 highlevel_cos seed 2 的 rms_bias = 0.000**（用 `rmsbias_check.py` 重算，seed 3 的 0.0063 也重現）：
- 觀察到的 Σ wk·rel² = 2.7e-4，比預期的統計漲落 Σ wk·rel_err² = 3.1e-4 還小，max(·, 0) 就截成 0。seed 3 是 3.5e-4 vs 3.1e-4，剛好大一點，得到 0.0063。
- 統計漲落幾乎都來自高 r̂ 尾端的一兩個 bin：bin 45（⟨r̂⟩ ≈ 1.8）的 rel_err 19%，一個 bin 就佔了 55%。cQj11 的 yield 由少數大 weight 的 qq̄ events 主導，所以尾端的 ⟨r⟩ 誤差很大。
- 所以 0 的意思是「偏差在這個雜訊下量不出來」，不是完美；可以量到的下限大約是 √3.1e-4 ≈ 0.018。0.000 和 0.0063 在這個精度下沒有差別。
- 同樣的原因，cQj11 的 pull 卻有 2.5：pull 是每個 bin 各自除以自己的誤差再平均，不會被一個雜訊大的 bin 蓋掉；rms_bias 的雜訊扣除是全部 bins 加在一起，尾端一個 bin 的大誤差會把 bulk 裡小但顯著的偏差一起扣掉。這是推論，沒有逐 bin 驗證 pull。

# Closure study (ywk, 2026-10-12，2026-10-17 更新)

Branch、環境和 condor 的基本設定見 `README_migration_note_ywk.md`。所有變更都只在 local，沒有 push。commit hash 新舊對照見同一份文件。

## 結論

1. **Calibration 和 kinematics 不一致是畫圖的 bug，不是網路的問題**（已修，commit dabf409）。
   - 樣本是在 cg 產生的，網路學的是 Σp1/Σp0；舊的圖比的是 mean(p1/p0)。
   - 正確的比法是 Σp0·r̂ 和 Σp1（SM 用 r̂ reweight 之後，和 c1 真實分布比較）。樣本在 SM 產生時（p0 ≡ 1），新比法會退化回舊的形式。
   - SM_cQj31：m_tt 最後一個 bin 的 r̂/r 從 0.18 變成約 1；chi_excl 從 8235 降到 87。
   - 仍存在的真正問題：r̂ > 10 的 tail，網路預測約 30，實際約 20。
2. **cg 的 closure 失敗來自 morphing 的輸入是 normalized ratio**（2026-10-13 確認，見下面的 TODO）。
   - 之前推測是 lstsq 外推放大誤差：456 個網路把 ‖a‖₂ 從 22 降到 3.8，closure 卻沒改善，所以不是主因。
   - 把 exact truth 餵進 morphing 也失敗；改餵 unnormalized 的 w(c1)/w(SM) 就 16/16 closure。
3. ~~近 SM 的 closure 看起來可信~~：只看了 ctGIm、ctGRe，推廣錯了。16 個 WC 都跑之後（2026-10-13）也失敗，見 `README_summary_ywk.md` 嘗試 12。
   - Truth 精準落在 data 點；ctGRe 在 5%、2.5%、10% 三種情境都在 68% 內。
   - ctGIm 偏小約 5%（1.6–2.5σ），可能是 highlevel features 缺少 CP-odd 的觀測量。
4. **近 SM closure 的系統性偏差主要來自網路在高 m_ttbar 的 calibration**（2026-10-16，見最後兩節）。
   - 1.3–2 TeV r̂ 高估 8–25%，2 TeV 以上高估 1–6.5 倍；1.0–1.3 TeV 反而低估 1–6%。840 GeV 以下（80% 的 events）在 ±1% 內。
   - validation 只保留 m_ttbar < 1300 GeV 時，cQj11 的偏差從最多 21% 降到 ≤ 3.5%；cQd8 從 −4 ~ −12% 翻成 +4 ~ +6%（只剩 1.0–1.3 TeV 的低估）。
   - **log features 重訓**（最後兩節）：cQj11 在 > 2 TeV 的高估從最多 +144% 降到 ≤ +30%；cQd8 只減半（仍約 2 倍），1.0–2 TeV 的偏差沒變。
   - **scan 不能直接比較不同網路**：`--drop-negative` 丟掉的 events 隨網路改變（tail 越差丟越多）。在共同 events 上 log 和 linear 差不多，平均 3 seeds 偏 −8 ~ −1%（「近 SM scan」一節）。
   - **weights_only** 對 cQj11、cQd8 只有小幅改善，cQd8 在 1–2 TeV 的形狀不變；ctGRe_ctGRe 明顯比較好（最後一節）。
5. **不同訓練設計在 cg 的放大倍數 ‖a‖₂**（越小越好）：

   | 設計 | ‖a‖₂ |
   |---|---|
   | 5% × 1 seed | 30 |
   | 5% × 3 seeds | 17 |
   | 1/5/10% × 3 seeds | 4.6 |
   | 5/10/20% × 1 seed（已選這個） | 3.8 |
   | 10/20/50% | 1.5 |

   Seeds 只能平均掉獨立的雜訊；共同 bias 的放大倍數 Σ|a| 不會變。
5. **Feature sets**：all、top30、highlevel 的 AUC 幾乎一樣，只有 ctGIm 用 highlevel 時低約 0.005。現階段先繼續用 highlevel。

## 5/10/20% × 1 seed，共 456 個 NNs（已完成，2026-10-13）

- Code：worktree `../topsbi-wt/port-eae2959`（包含 p0 加權修正後的畫圖程式）。
- Configs：`local/condor/configs/eae2959/{5p,10p,20p}_<point>.yml`。
  - WC 值來自 `local/wc_targets.yml`，用三個 splits 合計算出。
  - 和舊的 5% configs 會差到約 3%，因為舊的只用了 validation.p。
- Output：`/eos/cms/store/user/ykao/topsbi/results/port/eae2959/<pct>/<point>/seed_2/`
- Condor：測試 job 12490854（`20p_cQd8_ctd8`，最極端的點，已成功）；其餘 455 個在 cluster **12493439**（2026-10-12 23:15 送出）。
  - 重送失敗的 jobs：把失敗的 config 路徑寫進一個 list，然後
    `cd ~/topsbi-condor && condor_submit code=/eos/home-y/ykao/SWAN_projects/analysis/topsbi-wt/port-eae2959 configlist=<list> submit_list.sub`

```bash
# 檢查進度
condor_q
L=~/topsbi-condor/logs/12493439.log
grep -c "Job terminated" $L
grep "return value" $L | grep -vc "return value 0"   # 失敗的數量
ls /eos/cms/store/user/ykao/topsbi/results/port/eae2959/*/*/seed_2/model.pt | wc -l   # 應該是 456
```

## 跑完之後要做的事（manual）

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH=$PWD/src:$PYTHONPATH
WCS="ctGIm ctGRe cQj38 cQj18 cQu8 cQd8 ctj8 ctu8 ctd8 cQj31 cQj11 cQu1 cQd1 ctj1 ctu1 ctd1"

# 1. 產生 ensemble yml，所有網路的 device 都設成 cpu（lxplus 沒有 GPU）
python3 - <<'EOF'
import glob, os, yaml
out = 'local/cpu_configs/eae2959'; os.makedirs(out, exist_ok=True); nets = []
for f in sorted(glob.glob('local/condor/configs/eae2959/*.yml')):
    c = yaml.safe_load(open(f)); c['device'] = 'cpu'
    p = f'{out}/{os.path.basename(f)}'; yaml.safe_dump(c, open(p, 'w')); nets.append(p)
e = yaml.safe_load(open('local/ensemble_a6ff4bd.yml')); e['networks'] = nets
yaml.safe_dump(e, open('local/ensemble_eae2959.yml', 'w'))
EOF

# 2. cg closure：和 output/port_a6ff4bd_closure 比較，預期偏差會明顯變小
python3 scripts/scan.py -p local/ensemble_eae2959.yml --wc $WCS --truth --threshold 0 --drop-negative \
    --others cg -o output/port_eae2959_closure

# 3. 近 SM closure（記憶體：一次只跑一組，三組同時跑會超過 57 GB）
for d in on_5p half_5p 10p; do
  python3 scripts/scan.py -p local/ensemble_eae2959.yml --wc $WCS --truth --threshold 0 --drop-negative \
      --others sm --data-values local/data_$d.yml -o output/port_eae2959_nearSM_$d
done
```

- `local/data_on_5p.yml`、`data_half_5p.yml`、`data_10p.yml` 裡的 WC 值是依照舊的 a6ff4bd 5% 訓練點產生的。拿去測 eae2959 時：
  - `data_10p` 現在正好落在訓練點上；
  - 要測內插，可以改用 7.5% 或 15%：用 `python3 local/wc_targets.py <data dir> <out.yml> 0.075 0.15` 算出 WC 值，再寫成 `{wc: value}`。
- 比較 cg closure 結果的腳本：`python3 local/cmp_scans.py <dir A> <dir B>`。

## 還沒做的

- [x] 近 SM closure 補齊 16 個 WC（a6ff4bd、eae2959，2026-10-13）：68% 內只有 0–3/16，見 `README_summary_ywk.md` 嘗試 12–14
- [x] 456 個網路的 cg closure（2026-10-13）：**沒有改善**，10/16 撞到 scan 邊界（a6ff4bd 是 8/16），結果在 `output/port_eae2959_closure/`。‖a‖₂ 從 22 降到 3.8 卻沒改善，表示問題不只是獨立雜訊的放大。
- [x] 456 個網路的近 SM closure（同上）
- [x] 把 truth r 餵進同一個 lstsq morphing（`local/truth_morph_scan.py --mode norm|unnorm <scan.py args>`，結果在 `output/truthmorph_eae2959_{norm,unnorm}/`）：
  - norm（mean-normalized p1/p0，也就是網路學的東西）：0/16 closure，和網路的失敗模式一樣。
  - unnorm（w(c1)/w(SM)）：16/16 剛好落在 cg，68% 區間和 truth 完全一致。
  - **cg closure 失敗的 root cause：`full_likelihood` 拿 normalized ratio 做 quadratic morphing。** normalized r(c) = [w(c)/w_SM]·Y_SM/Y(c) 是兩個二次式相除，不是二次式。修法：lstsq 前每個網路乘上 Y(cᵢ)/Y_SM。
- [x] 修 `full_likelihood`（23da3fd；upstream 也有這個 bug，已放進 PR branch `fix/likelihood-toy-builder`（readonly clone），並加上只對 stitched 做 yield 修正）。重跑 456 個網路的 cg closure（`output/port_eae2959_morphfix_closure/`）：仍然 0/16。
- [x] 每個網路先 self-normalize（E_cg[p0 r̂] = E_cg[p1]）再 morph（`local/truth_morph_scan.py --mode selfnorm`，`output/port_eae2959_selfnorm_closure/`）：仍然 0/16。
  - 縮放係數平均 0.961、std 0.054，範圍 0.58–1.06（`output/port_eae2959_selfnorm_closure_factors.txt`）：網路普遍高估 E[r̂] 約 4%，最差的是 20p 的 cQd8 相關點（低到 0.58）。
  - 所以剩下的偏差是 event-dependent 的形狀誤差，不是整體 normalization。
- [x] ~~混合測試~~（部分網路換成 truth）：設計無效。truth 是 r(x,z)，網路學的是 r(x)，混用一定不一致（`README_summary_ywk.md` 嘗試 10、11）
- [x] seeds study（a6ff4bd 10 個 WC × 2 個點，seeds 3、4，2026-10-14）：訓練的隨機誤差是最大來源，平均 3 個 seeds 改善約 40–50%，但仍有 3–5% 系統性偏差（嘗試 15）
- [x] 降低訓練的隨機誤差（4 個變體，2026-10-14/15）：只有 `bs10k` 有用（seed 差異 −20%），但 cQj11 的 bs10k seed 3 壞掉；後來發現這主要也是 m_ttbar tail 的問題
- [x] 用 upstream（eddie/main a733f77）的 metrics 評估現有網路（`local/eval_metrics.py`，2026-10-15）：除以 pg（count 混合）和 xsec 混合的 metrics 只差 < 10%，**不是近 SM 偏差的原因**；找到 m_ttbar tail 的問題
- [x] m_ttbar cut 的近 SM closure（2026-10-16）：確認 tail 是主要來源（見最後一節）
- [ ] **m_ttbar、pT 類 features 改用 log 後重訓**（4 個點 × 3 seeds，和 bs10k 同樣規模），看 1 TeV 以上的 calibration 和近 SM closure 是否改善。要改 feature 處理、送 condor
- [ ] `--drop-negative` 的結果依賴掃描範圍：被丟掉的 events 要和範圍無關，否則不同設定不能比（`output/test_window_cQu1_on_5p/`）
- [ ] ~~高 r̂ tail：統計 r̂ > 10 的 events~~ → 已定位到 m_ttbar > 1.3 TeV 的外推（見最後兩節），改由 log features 處理
- [ ] 152 個 a6ff4bd 網路用新的 calibration 指標重算（`python3 local/replot.py <config> <outdir>`，每個點約 1 分鐘）
- [ ] ctGIm：加入 CP-odd 的 features 後重訓，看 5% 的偏差是否消失
- [ ] `validation.py` 的 `kinematic_ratio_plot` 有同樣的權重問題（upstream 正在重寫，issue #5）
- [ ] dabf409（plots 權重）是 main 的 bug，開 PR 前應該移到 `fix/crashes`

## 0.1% / 1% 建設性干涉點（2026-10-13 會議決定，已完成）

- WC 值：`local/wc_targets_small.yml`（`python3 local/wc_targets.py <data dir> local/wc_targets_small.yml 0.001 0.01`）
- Configs：`python3 local/make_constructive_configs.py local/condor/configs/a6ff4bd_SM_cQj31.yml constructive local/wc_targets_small.yml`
  → `local/condor/configs/constructive/{0.1p,1p}_<point>.yml`，共 152 個：SM_wc 在 0.1% 和 1%，交叉點在 1%。
- ‖a‖₂：在 1% 的點剛好是 1；外推到 5% 是 20–110，到 cg 是約 3800。只適合驗證 1% 以內的點。
- 送出（訓練程式碼和 eae2959 相同，所以沿用同一個 worktree）：
  `cd ~/topsbi-condor && condor_submit code=/eos/home-y/ykao/SWAN_projects/analysis/topsbi-wt/port-eae2959 configlist=list_constructive.txt submit_list.sub`
- 2026-10-13 22:16 送出，cluster 12543449，152/152 成功。Output：`/eos/cms/store/user/ykao/topsbi/results/port/constructive/{0.1p,1p}/<point>/seed_2/`；圖：`/eos/home-y/ykao/www/topsbi/results_20261013_constructive/`
- 結果（`output/nearSM_fine_constructive_{1p,0.1p}/`，對照 `output/nearSM_fine_eae2959_{1p,0.1p}/`）：
  - 1%：68% 內 5/16（eae2959 是 2/16），rms 偏差 15%（eae2959 是 20%）。有改善但不夠。
  - 0.1%：兩者都失敗（1–2/16），很多撞到 scan 邊界。truth 的 68% 區間本身就有 c 的 ±7–27%，這個樣本（validation 18 萬個 events）分辨不了 0.1% 的效應。

## Overnight 2026-10-13 → 10-14（`local/overnight_20261013.sh`）

全部依序執行（`nice -n 19`、4 threads），用 `setsid nohup` 跑，log 在 `output/overnight_20261013.log`，22:19 開始、02:12 結束。

```bash
setsid nohup bash local/overnight_20261013.sh > output/overnight_20261013.log 2>&1 < /dev/null &
```

| 步驟 | 內容 | Output |
|---|---|---|
| 1 | 細格點近 SM closure（801 點、原本的範圍） | `output/nearSM_fine_{a6ff4bd,eae2959}_{on_5p,half_5p,10p}/` |
| 2 | eae2959 在 1% / 0.1% 建設性點 | `output/nearSM_fine_eae2959_{1p,0.1p}/` |
| 3 | seeds（cluster 12543452，40/40 成功） | `output/seed_compare_a6ff4bd.txt`、`output/nearSM_fine_a6ff4bd_seeds{3,4,234}_*/` |
| 4 | 建設性設計（cluster 12543449） | `output/nearSM_fine_constructive_{1p,0.1p}/` |

新增的工具：
- `scripts/scan.py --window <frac>`（3bcc25e）：每個 WC 掃 data ± frac·|data|。**注意 `--drop-negative` 會因範圍不同而丟掉不同的 events**，所以 overnight 沒用它，改用原本的範圍加密格點（`--npoints 801`）。
- `local/truth_morph_scan.py --mode seeds --seeds 3` 或 `2,3,4`：把網路換成別的 seed，或對多個 seeds 的 r̂ 取平均，再 morph。ensemble configs 要指向 seed_2。
- `local/seed_compare.py <seeds> <configs>`：同一個訓練點不同 seeds 的 r̂ 逐 event 比較，不需要跑 scan。
- `local/nearsm_table.py <dirs>`：近 SM closure 的表格；`local/neff_pulls.py <data.yml> <dirs>`：用 N_eff 修正 pull。

結果摘要（細節見 `README_summary_ywk.md` 嘗試 14–16）：
- 細格點和 101 點最多差 1%：格點不是問題。
- 同一個點不同 seeds 的 log r̂ 逐 event 差約 11–13%；best fit 的 seed 間 std 約 3.5–7%。seed 2（之前所有 a6ff4bd 的結果都用它）剛好是最差的。
- 平均 3 個 seeds：rms 偏差 3.6–5.9%（seed 2 是 6.3–12.4%），但仍有系統性偏差（cQd8、cQu1、cQu8 三個 seeds 都偏小，cQj31 都偏大）。
- Early stopping 的 best epoch 分散在 10–78（多數 25–40），這時 cosine 的 learning rate 還很高；test loss 只比 ln2 低 0.001–0.009，等於在雜訊中挑 epoch。

## 降低訓練的隨機誤差（2026-10-14 送出，cluster 12547015，48/48 成功）

每個變體只改一件事，其他和 baseline（dropout 0.2、batch 1000、lr 1e-3、cosine 100 epochs、patience 20）相同：

| 變體 | 改什麼 | 要回答的問題 |
|---|---|---|
| `fixboot` | `bootstrap_seed: 2`（209ba10 新增的 config key），三個 seeds 用同一個 train/test split | seed 間差異有多少來自資料取樣（訓練設定消除不了）、多少來自初始化和 batch 順序 |
| `cos40` | `epochs: 40`、`patience: 1000`：跑完整個 cosine，不 early stop | learning rate 降到 1e-6 再挑 best epoch，是否比較穩定 |
| `bs10k` | `batchSize: 10000` | 梯度雜訊少 10 倍，是否比較穩定 |
| `nodrop` | 拿掉所有 dropout | dropout 是否增加 seed 間差異 |

- 點：`SM_cQj11`、`cQj11_cQj11`（主要是隨機誤差）、`SM_cQd8`、`cQd8_cQd8`（有系統性偏差），用 a6ff4bd 的 5% 值；seeds 2、3、4。共 4 × 4 × 3 = 48 個 jobs。
- Code：worktree `../topsbi-wt/port-209ba10`。
- Configs：`local/condor/configs/var/<variant>_<point>_seed<s>.yml`；CPU 版本（指向 seed_2）：`local/cpu_configs/var/<variant>_<point>.yml`。
- Output：`/eos/cms/store/user/ykao/topsbi/results/port/a6ff4bd_<variant>/<point>/seed_<s>/`

```bash
# 送出（已執行）
cd ~/topsbi-condor && condor_submit code=/eos/home-y/ykao/SWAN_projects/analysis/topsbi-wt/port-209ba10 \
    configlist=list_variants.txt submit_list.sub
# 檢查進度
L=~/topsbi-condor/logs/12547015.log; grep -c "Job terminated" $L; grep "return value" $L | grep -vc "return value 0"
# 評估 1（主要指標）：逐 event seed 差異比例，baseline 在這 4 個點約 0.11–0.12
for v in fixboot cos40 bs10k nodrop; do echo "== $v"; python3 local/seed_compare.py 2,3,4 local/cpu_configs/var/${v}_*.yml; done
python3 local/seed_compare.py 2,3,4 local/cpu_configs/a6ff4bd/a6ff4bd_{SM_cQj11,cQj11_cQj11,SM_cQd8,cQd8_cQd8}.yml   # baseline
# best epoch 的分布
for f in ~/topsbi-condor/logs/12547015.*.out; do echo "$(grep -o 'configs/var/[a-z0-9]*' $f | head -1 | cut -d/ -f3) \
  $(grep -o 'best \(epoch was\|checkpoint from epoch\) [0-9]*' $f | tail -1 | grep -o '[0-9]*$')"; done | sort | uniq -c
```

注意：`seed_compare.py` 需要 torch，先 `source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh` 並設 `PYTHONPATH=<port/dev src>`。

### 結果（2026-10-15）

逐 event seed 差異比例（越低越好；`seed_compare.py`）：

| 變體 | SM_cQj11 | cQj11_cQj11 | SM_cQd8 | cQd8_cQd8 | 平均 | vs baseline |
|---|---|---|---|---|---|---|
| baseline | 0.114 | 0.115 | 0.120 | 0.114 | 0.116 | — |
| `fixboot` | 0.109 | 0.112 | 0.109 | 0.109 | 0.110 | −5% |
| `cos40` | 0.122 | 0.101 | 0.107 | 0.096 | 0.107 | −8%，不一致 |
| `bs10k` | 0.092 | 0.093 | 0.097 | 0.090 | 0.093 | **−20%** |
| `nodrop` | 0.121 | 0.116 | 0.135 | 0.119 | 0.123 | +6% |

- `fixboot` 只降 5%：seed 間差異主要來自初始化和 batch 順序，不是 train/test split。
- `nodrop` 變差，best epoch 在 2–16，過擬合；dropout 要保留。
- `cos40` 的 best epoch 多在 cosine 中段（9–36），不在 learning rate 最低的尾端。
- `bs10k` 四個點都比 baseline 低，normalization 也比較好（`cQd8_cQd8` 最差的 seed 偏 2.8%，baseline 是 4.8%）。

### bs10k 的近 SM scan（2026-10-14）

把 ensemble 裡這 4 個網路換成 bs10k（`local/ensemble_bs10k_cpu.yml`），設定和 overnight 的 seeds scan 相同（`local/scan_bs10k.sh`，log `output/scan_bs10k.log`，結果 `output/nearSM_fine_bs10k_seeds{2,3,4,234}_{on_5p,half_5p,10p}/`）。

| WC | 版本 | seed 間 std（相對 data） | 平均 3 seeds 的偏差 |
|---|---|---|---|
| cQd8 | baseline | 0.7–6.7% | −4.0 ~ −7.2% |
| cQd8 | bs10k | **0.1–1.6%** | −1.8 ~ −6.7% |
| cQj11 | baseline | 8.0–13.1% | −0.8 ~ −3.8% |
| cQj11 | bs10k | 6.9–85%（seed 3 壞掉） | **+10 ~ +17.5%** |

- cQd8：三個 seeds 幾乎重合，但仍然一致地偏小 → bs10k 降的是隨機誤差，系統性偏差還在。
- cQj11：bs10k 的 seed 3 在 `half_5p` 跳到 +0.365（符號相反、68% 上界是 None），`--drop-negative` 丟 447 個 events（seed 2 只丟 136）。平均 3 seeds 反而被拉向 seed 3。後來確認這主要是 m_ttbar tail 外推（下一節）。

## 用 upstream metrics 評估現有網路（2026-10-15，`local/eval_metrics.py`）

- 用 `port/rebased` 的 src（有 upstream 的 `rms_bias`、`pull`、`group_closure`），在 `validation.p` 上評估 baseline 和 bs10k 的 24 個網路。
- highlevel 資料沒有 era、njets（`ids` 全是 0），所以 `group_closure` 改用 m_ttbar 分位數分組（它是網路的 input，符合 upstream 的要求）。
- 每個網路用兩種權重評估：`train` = w(c)/w(cg)（port/dev 的 `stitched`，樣本按 event 數混合），`xsec` = w(c)（upstream 現在的 `stitched`，按截面混合）。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH=/eos/home-y/ykao/SWAN_projects/analysis/topsbi-wt/rebased/src:$PYTHONPATH OMP_NUM_THREADS=4
P="SM_cQj11 cQj11_cQj11 SM_cQd8 cQd8_cQd8"
python3 local/eval_metrics.py 2,3,4 $(for p in $P; do echo local/cpu_configs/a6ff4bd/a6ff4bd_$p.yml; done) \
    $(for p in $P; do echo local/cpu_configs/var/bs10k_$p.yml; done) > output/eval_metrics_a6ff4bd_bs10k.txt
```

結果：
- `train` 和 `xsec` 只差 < 10%：**除以 pg 不是近 SM 偏差的原因**。closure 的 pseudo-data 也除以 pg，兩邊一致。不過分析真的 data 時應該用 xsec 混合。
- `cQd8_cQd8` 的 `group_rms_bias` 8–31%，cQj11 的 `pull` 3–9：都有顯著的區域性偏差。bs10k seed 3 的 `SM_cQj11` 是 bs10k 裡最差的（pull 4.0），和 scan 的異常一致。
- 按 m_ttbar 分段的 Σw0·r̂ / Σw0·r − 1（xsec 權重）：

| 網路 | < 840 GeV（80%） | 840–1000 | 1000–1300 | 1300–2000 | > 2000（4128 個 events） |
|---|---|---|---|---|---|
| baseline `cQd8_cQd8` | ±1% | −1 ~ −2% | −0.1 ~ −4% | **+18 ~ +25%** | **+118 ~ +655%** |
| bs10k `cQd8_cQd8` | ±1% | −1% | −3 ~ −6% | **+8 ~ +19%** | **+170 ~ +356%** |
| baseline `SM_cQd8` | ±1% | −1% | −1 ~ −3% | **+10 ~ +20%** | **+104 ~ +132%** |

- 一個 7 TeV 的 event 讓 sigmoid 飽和，r̂ = inf。
- 原因：`Net` 內部只做線性標準化（m_ttbar 平均 687、std 432 GeV），8.5 TeV 對應 z = 18；LeakyReLU 網路在資料範圍外是線性外推，r̂ = e^logit 隨 z 指數成長，而真值 r 隨 m_ttbar 大致是多項式。

## m_ttbar cut 的近 SM closure（2026-10-16，`local/scan_mttcap.sh`）

不改程式碼：把 `validation.p` 只保留 m_ttbar < 1300 和 < 2000 GeV（`output/data_mttcap/validation_mtt{1300,2000}.p`，164024 和 173776 個 events，原本 177904），ensemble config 指過去（`local/ensemble_{a6ff4bd,bs10k}_cpu_mtt{1300,2000}.yml`）。truth、pseudo-data、網路都用同一組 events，closure 仍然自洽。

```bash
setsid nohup bash local/scan_mttcap.sh > output/scan_mttcap.log 2>&1 < /dev/null &
```

- **主目錄現在在 `new`（eddie/main），沒有 `scripts/scan.py`**，所以 script 從 `../topsbi-wt/port-209ba10` 執行（stitched 網路需要 port/dev 的程式碼）。
- 18 個 scan，每個約 4 分鐘；結果在 `output/nearSM_mtt{1300,2000}_{a6ff4bd,a6ff4bd_seeds234,bs10k_seeds234}_{on_5p,half_5p,10p}/`，對照 `output/nearSM_fine_*`。

best fit 相對 data 的偏差（cQj11 的 data 是負值，「+」表示絕對值偏小；truth 本身 ≤ 0.3%）：

| WC | 網路 | 全部 events | < 2000 | < 1300 |
|---|---|---|---|---|
| cQj11 | baseline s2 | +10.0 ~ +21.0% | +2.5 ~ +9.3% | **+1.2 ~ +3.0%** |
| cQj11 | baseline 平均 3 seeds | −3.8 ~ −0.8% | −0.3 ~ +5.5% | **0.0 ~ +2.8%** |
| cQj11 | bs10k 平均 3 seeds | +10.0 ~ +17.5% | +0.7 ~ +3.5% | +2.8 ~ +3.5% |
| cQd8 | baseline s2 | −6.2 ~ −11.7% | −4.5 ~ −11.5% | **−0.8 ~ +1.8%** |
| cQd8 | baseline 平均 3 seeds | −4.0 ~ −7.2% | −2.3 ~ −6.5% | +4.2 ~ +5.8% |
| cQd8 | bs10k 平均 3 seeds | −1.8 ~ −6.7% | −1.2 ~ −6.7% | +5.0 ~ +6.0% |

（範圍是三個 data 點 on_5p、half_5p、10p。）

- **cQj11**：幾乎全部來自 2 TeV 以上。之前看起來像「訓練的隨機誤差」或「bs10k seed 3 運氣不好」，主要是各 seed 在 tail 外推的方式不同。
- **cQd8**：有兩個方向相反的來源。> 1.3 TeV 高估 → cQd8 偏小；1.0–1.3 TeV 低估 → cQd8 偏大。原本部分抵消；切掉 tail 後只剩後者，所以翻成 +4 ~ +6%（bs10k 在 1.0–1.3 TeV 低估最多，翻得也最多）。網路在 1 TeV 以上整段都沒學好 r 隨 m_ttbar 成長的形狀。
- 大部分仍在 68% 區間外：16–17 萬個 events 的區間很窄，1–3% 就會超出。
- 用 cut 修不可行（cQd8 會變成偏大，也丟掉最敏感的區域）→ 下一步是 log features 重訓。

## Log features 重訓（2026-10-16，cluster 12555416，24/24 成功）

- 資料：`local/make_log_features.py` 把 m_ttbar 取 log，pt_tt、lep_top_pt、had_top_pt 取 log(1+x)，其他欄位、coefs、event 順序都不變 → `/eos/cms/store/user/ykao/topsbi/pretraining_main_highlevel_log/`，features 檔 `features_highlevel_log.yml`（名稱加 `log_`）。m_ttbar 的最大 z 從 18 降到 5.9，pT 類從 16–19 降到約 4。

**自己準備 log features（manual）**：在 lxplus 的主目錄執行（train.p 有 1.8 GB，會整個讀進記憶體）。

```bash
cd /eos/home-y/ykao/SWAN_projects/analysis/topsbi
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
D=/eos/cms/store/user/ykao/topsbi
python3 local/make_log_features.py $D/pretraining_main_highlevel $D/pretraining_main_highlevel_log \
    features_highlevel.yml features_highlevel_log.yml
```

- 輸入：原本資料夾裡的 `train.p`、`test.p`、`validation.p`，以及原本的 features 檔。輸出：新資料夾裡同名的三個檔，以及新的 features 檔。
- 要取 log 的欄位寫在 script 開頭的 `LOG`（m_ttbar 用 `log`，其他用 `log1p`）；要加減欄位就改這裡。欄位位置從 features 檔的 `loc` 讀，所以換資料集時不用改程式碼。
- 新的 features 檔只改這幾個欄位：名稱加上 `log_`，`label` 加上 log，`min`/`max` 改成 log 尺度（只影響畫圖的範圍）。
- 檢查：script 會印出 train 裡這幾個欄位的 z 範圍，m_ttbar 的最大值應該約 5.9；三個輸出檔的大小應該和原本的一樣。
- 訓練 config 改兩個地方：`data:` 指到新的資料夾，`features:` 指到新的 features 檔。weights_only 的話，`holdout:` 也要一起改。
- 推論時，網路要吃轉換過的 input。用原本的 linear 資料評估時，要先做同樣的轉換（`local/truth_morph_scan.py`、`local/common_events_fit.py` 會依 features 檔裡的 `log_*` 自動轉換）。`local/eval_metrics.py` 不會轉換，它直接讀第一個 config 的資料，所以 log 和 linear 的網路要分開跑。
- 訓練：port/dev 的程式碼（`port-209ba10`，stitched），只換 features，和 baseline、bs10k 可以直接比較。`log`（batch 1000）和 `log_bs10k` 各 4 個點 × seeds 2、3、4。configs 在 `local/condor/configs/log/`，CPU 版本 `local/cpu_configs/log/`。

```bash
cd ~/topsbi-condor && condor_submit code=/eos/home-y/ykao/SWAN_projects/analysis/topsbi-wt/port-209ba10 \
    configlist=list_log.txt submit_list.sub   # 已執行
# 評估（eval_metrics.py 加上了依 m_ttbar 分段的表；linear 和 log 的網路要分開跑，因為只讀第一個 config 的資料）
setsid nohup bash local/eval_log.sh > output/eval_log.log 2>&1 < /dev/null &
# → output/eval_metrics_bs10k_mtt.txt、output/eval_metrics_log.txt
```

依 m_ttbar 分段的 Σw0·r̂ / Σw0·r − 1（xsec 權重，3 個 seeds 的範圍）：

| 點 | m_ttbar | bs10k（linear） | log_bs10k | log（batch 1000） |
|---|---|---|---|---|
| SM→cQj11 | 1.3–2 TeV | −5 ~ −8% | −3 ~ −11% | −14 ~ −16% |
| | > 2 TeV | **+66 ~ +110%** | **+3 ~ +29%** | −4 ~ −11% |
| cQj11→cQj11 | 1.3–2 TeV | −12 ~ +4% | −6 ~ −8% | −4 ~ −7% |
| | > 2 TeV | **+40 ~ +144%** | **+8 ~ +12%** | +4 ~ +20% |
| SM→cQd8 | 1.0–1.3 TeV | −2 ~ −3% | −2 ~ −4% | −3% |
| | 1.3–2 TeV | +10 ~ +11% | +8 ~ +11% | +4 ~ +13% |
| | > 2 TeV | **+46 ~ +158%** | **+79 ~ +112%** | +37 ~ +69% |
| cQd8→cQd8 | 1.0–1.3 TeV | −3 ~ −6% | −3 ~ −7% | −3 ~ −4% |
| | 1.3–2 TeV | +8 ~ +19% | +9 ~ +16% | +14 ~ +17% |
| | > 2 TeV | **+170 ~ +356%** | **+70 ~ +145%** | +66 ~ +196% |

- < 1 TeV（約 85% 的 events）：兩種 features 都在 ±2% 內。r̂ = inf 的 event 從 1 個變 0 個。
- **cQj11**：> 2 TeV 的高估從最多 +144% 降到 ≤ +30%。`rms_bias` 也下降（log_bs10k 0.005–0.011，bs10k 0.008–0.026），但 `pull` 仍 2–5。
- **cQd8**：> 2 TeV 約減半，仍高估約 2 倍；1.0–1.3 TeV 低估、1.3–2 TeV 高估的模式和 linear 完全一樣 → 除了外插，cQd8 還有別的原因（推測：r 在這段隨 m_ttbar 上升很快，網路沒學到；未驗證）。
- log 和 log_bs10k 差不多；log 的 SM→cQj11 seed 3 較差（pull 7.6），bs10k 仍然比較穩。

### 近 SM scan（`local/scan_log.sh`）

- ensemble 是 153 個網路一起解 lstsq，只有這 4 個點有 log 版本。`local/truth_morph_scan.py` 的 `Network` 會把 linear 的 input 轉成 log 再餵給 features 檔有 `log_*` 的網路（驗證：和直接用 log 資料的輸出完全相同），其他 149 個網路不變。
- ensemble：`local/ensemble_log_bs10k_cpu.yml`（`ensemble_bs10k_cpu.yml` 把 4 個點換成 `local/cpu_configs/log/log_bs10k_*`）。
- 設定和 `local/scan_bs10k.sh` 相同；結果在 `output/nearSM_fine_log_bs10k_seeds{2,3,4,234}_{on_5p,half_5p,10p}/`，對照 `output/nearSM_fine_bs10k_seeds*`。

```bash
setsid nohup bash local/scan_log.sh > output/scan_log.log 2>&1 < /dev/null &
```

12 個 scan 都成功（`output/scan_log.log`）。單看 scan，log 讓 cQj11 變好、cQd8 變差（cQd8 平均 3 seeds 從 −2 ~ −7% 變成 −3 ~ −22%）。**但這個比較不公平：**

- `--drop-negative` 丟掉「grid 上任何一點 morph 後 r ≤ 0」的 events，幾乎都是 m_tt > 1.3 TeV。tail 越差，丟得越多，丟掉的又正好是最會拉偏 fit 的 events → **越差的網路被「美化」得越多**，不同網路的 scan 用的是不同的 events。
- 例：cQd8 seed 2 on_5p，bs10k 丟 552 個（544 個 m_tt > 1.3 TeV），log_bs10k 只丟 140 個。

**共同 events 的比較（`local/common_events_fit.py`）**：只有其他 WC 在 SM 時，r(c) = 1 + γx·c + γxx·c² 只由 `SM_<wc>` 和 `<wc>_<wc>` 兩個網路決定，所以可以不載入 ensemble，直接重現 scan.py 的 best fit（yield ratio、drops、每個 grid 點 E_p0[r] = 1、pseudo-data weights；`own` 欄和 scan 的結果一致）。再把所有變體、seeds 的 drops 取聯集，在同一組 events 上 fit（`common`，約 700–1000 個 events，95% 以上 m_tt > 1.3 TeV），以及再加上 m_tt < 2 TeV（`common<2TeV`）。

```bash
setsid nohup bash local/common_fit_log.sh > output/common_fit_bs10k_log.txt 2>&1 < /dev/null &
```

平均 3 seeds，best fit 相對 data 的偏差（own → common → common<2TeV）：

| WC | data | bs10k | log_bs10k |
|---|---|---|---|
| cQj11 | on_5p | +12.0 → −2.8 → +1.7% | +2.5 → −4.2 → −0.2% |
| | half_5p | +17.3 → −3.5 → +3.5% | +7.0 → −3.2 → +3.2% |
| | 10p | +10.0 → −3.0 → +0.7% | +1.0 → −5.0 → −1.8% |
| cQd8 | on_5p | −5.5 → −5.0 → −2.5% | −15.5 → −5.3 → −2.0% |
| | half_5p | −6.7 → −6.2 → −4.0% | −22.2 → −8.0 → −4.2% |
| | 10p | −1.8 → −1.2 → +0.3% | −3.2 → −1.0 → +1.0% |

- **在同一組 events 上，log 和 linear 差不多**。scan 看到的「log 讓 cQd8 變差」完全是 drops 不同造成的；bs10k 原本的 cQj11 +10 ~ +17.5%（含 seed 3 的 +150%）在共同 events 上也消失（seed 3 half_5p：−5.5%）。
- 再切掉 > 2 TeV 後，大多在 ±4% 內；剩下的偏差來自 2 TeV 以上，和 `eval_metrics.py` 的分段表一致。
- 單一 seed 的 common 偏差：bs10k −8 ~ 0%，log_bs10k −8.5 ~ +2.3%，seed 間差異仍然存在。
- **結論：scan 的數字不能直接拿來比較不同網路**，除非 event selection 和網路無關（見 `README_summary_ywk.md` 的下一步）。

## weights_only 和 ctGRe（2026-10-16 送出，cluster 12559217 / 12559218）

- `wo_log_bs10k`：log features + `method: weights_only`，port/rebased 的程式碼（`port-d11cea0`），4 個點 × seeds 2、3、4，加上 ctGRe 的 2 個點 × seed 2，共 14 個。和 `log_bs10k` 比較；兩邊程式碼不同（port/dev vs port/rebased），差很多時不一定是 method 造成的。
- 新的 config keys：`year_loc: null`、`lumi: [1.0]`（單一 era，lumi 的值不影響訓練）、`group_cols: []`、`holdout: <data>/validation.p`。
- `log_bs10k` 的 ctGRe 2 個點 × seed 2（`port-209ba10`）。ctGRe 不是四費米子算符，效應隨能量上升較溫和：用來檢查 log features 和 weights_only 是否對不同類型的算符都有效。
- configs：`local/condor/configs/wo/`、`local/condor/configs/log/log_bs10k_*ctGRe*`；清單 `~/topsbi-condor/list_wo_log.txt`、`list_ctgre_log.txt`。`wo_bs10k`（linear）的 configs 也在 `wo/`，決定不送。

```bash
cd ~/topsbi-condor
condor_submit code=/eos/home-y/ykao/SWAN_projects/analysis/topsbi-wt/port-d11cea0 configlist=list_wo_log.txt submit_list.sub    # 已執行
condor_submit code=/eos/home-y/ykao/SWAN_projects/analysis/topsbi-wt/port-209ba10 configlist=list_ctgre_log.txt submit_list.sub  # 已執行
```

### 結果：calibration（2026-10-17，`local/eval_wo.sh` → `output/eval_metrics_wo_log.txt`）

16 個 jobs 都成功。`eval_metrics.py` 對 weights_only 網路改和沒有 normalize 的 w1/w0 比較（它學的就是這個；E_p0[r̂] = 1.053，yield ratio 1.050），stitched 網路照舊和 p1/p0 比較。

| 點 | rms_bias stitched | rms_bias wo | pull stitched | pull wo | > 2 TeV stitched | > 2 TeV wo |
|---|---|---|---|---|---|---|
| SM_cQj11 | 0.005–0.011 | **0.000** | 2.5–4.9 | 1.9–3.4 | +2.5 ~ +29% | −2 ~ +17% |
| cQj11_cQj11 | 0.005–0.008 | 0.000–0.010 | 2.2–3.4 | 1.7–3.9 | +8 ~ +12% | −5 ~ +14% |
| SM_cQd8 | 0.015–0.023 | 0.013–0.025 | 0.5–0.8 | 0.5–0.8 | +79 ~ +112% | +67 ~ +117% |
| cQd8_cQd8 | 0.011–0.027 | 0.014–0.023 | 0.4–1.8 | 0.4–0.95 | +70 ~ +145% | +68 ~ +127% |
| SM_ctGRe（s2） | 0.0008 | 0.0014 | 0.21 | 1.12 | −0.1% | +0.2% |
| ctGRe_ctGRe（s2） | 0.029 | **0.012** | 0.11 | 0.14 | +12% | +36% |

（stitched = log_bs10k，wo = wo_log_bs10k；都用 log features，seeds 2、3、4 的範圍；`rms_bias` 已扣掉雜訊，0.000 表示小於雜訊。）

- **cQj11、cQd8**：wo 只有小幅改善，在 seed 間差異之內。cQd8 在 1.0–1.3 TeV 低估、1.3–2 TeV 高估、> 2 TeV 約 2 倍的形狀兩種 method 都一樣 → **不是 stitched 造成的**。
- **ctGRe_ctGRe**（AUC 0.69，最好區分的點）：差別最明顯。stitched 每一段都偏（< 2 TeV 是 −2 ~ −24%），wo 在 1 TeV 以下 < 2%、1–2 TeV ±7%。只有一個 seed。
- SM_ctGRe 兩邊都 ≤ 0.4%；wo 的 pull 較大但 `rms_bias` 只有 0.0014，可以忽略。
- 兩邊程式碼版本不同（port/dev vs port/rebased），差異不一定全部來自 method。

### 結果：共同 events 的近 SM fit（2026-10-17，`output/common_fit_wo.txt`）

`common_events_fit.py` 加了 weights_only 的處理（不乘 yield ratio），`local/common_fit_log.sh` 加入 wo_log_bs10k。共同 events 是三個變體 drops 的聯集（多了約 30–90 個），所以 bs10k、log_bs10k 的數字和上一節差 ≤ 0.5%。平均 3 seeds（own → common → common<2TeV）：

| WC | data | log_bs10k | wo_log_bs10k |
|---|---|---|---|
| cQj11 | on_5p | +2.5 → −4.2 → −0.2% | +1.0 → −4.5 → −0.5% |
| | half_5p | +7.0 → −3.2 → +3.2% | +2.8 → −5.5 → +1.5% |
| | 10p | +1.0 → −5.0 → −1.8% | 0.0 → −4.7 → −1.5% |
| cQd8 | on_5p | −15.5 → −5.5 → −2.2% | −14.8 → −5.0 → −2.0% |
| | half_5p | −22.2 → −8.5 → −4.5% | −21.2 → −8.0 → −4.5% |
| | 10p | −3.2 → −1.2 → +0.8% | −2.7 → −0.8 → +0.8% |

- **weights_only 和 stitched 在近 SM fit 上幾乎一樣**，和 calibration 的結論一致。
- wo 的 seed 3 特別好（cQd8 −1.5 ~ +1.2%），seeds 2、4 和 log_bs10k 差不多：seed 間差異仍大於 method 的差異。
- wo 自己丟掉的 events 更少（平均 3 seeds：cQj11 23 個，log_bs10k 86 個），所以和 log_bs10k 一樣，cQd8 的 own 被留下來的 tail 拉偏到 −15 ~ −21%。

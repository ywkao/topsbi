# Closure study (ywk, 2026-10-12)

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
4. **不同訓練設計在 cg 的放大倍數 ‖a‖₂**（越小越好）：

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
- [ ] 降低訓練的隨機誤差：正在跑（見最後一節）
- [ ] `--drop-negative` 的結果依賴掃描範圍：被丟掉的 events 要和範圍無關，否則不同設定不能比（`output/test_window_cQu1_on_5p/`）
- [ ] 高 r̂ tail：統計 r̂ > 10 的 events 和它們在 scan 中的貢獻；試 `--threshold` winsorize
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

## 正在跑：降低訓練的隨機誤差（2026-10-14 送出，cluster 12547015）

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
for f in ~/topsbi-condor/logs/12547015.*.out; do grep -o "best epoch was [0-9]*" $f | tail -1; done | sort | uniq -c
```

評估 2：有明顯改善的變體，再跑近 SM scan，確認 best fit 的 seed 間差異和偏差是否變小。

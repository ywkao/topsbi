# 在 CERN SWAN 的 Jupyter notebook 上跑 training

## 開 session

- Software stack 要和 condor 一樣，選 **LCG 109 cuda**，torch、numpy、matplotlib 都已經內建。如果沒有 GPU，選一般的 LCG 109 也可以。
- **不要選 LCG 110**：它的 mplhep 1.1.0 + matplotlib 3.11 + numpy 2.4 組合下，`mh.cms.label` 在 `savefig` 時會報 `TypeError: only 0-dimensional arrays can be converted to Python scalars`，train 會在第一次畫圖（epoch 0）就停掉。
- **預設用 GPU**：開 session 時在 SWAN 設定裡選 GPU。cell 1 會檢查 `torch.cuda.is_available()`，沒有 GPU 就直接停下來。真的要用 CPU 時，把 cell 2 的 `DEVICE` 改成 `'cpu'`（很慢，見下面「注意」）。
- 在 SWAN 裡，repo 的路徑是 `/eos/user/y/ykao/SWAN_projects/analysis/topsbi`（和 `/eos/home-y/...` 是同一個地方）。

## Notebook

**程式碼和 configs 分開放**：
- 程式碼用固定版本的 worktree **`topsbi-wt/port-0c49da2`**（port/rebased，已 rebase 到 eddie/main c810b07，2026-10-17）。
- 主目錄 `topsbi/` checkout 的是 eddie/main：沒有 `group_cols`、`year_loc: null`，不要拿來跑我們的資料。
- 舊的 worktree 只用來重現已經訓練好的網路：`port-d11cea0`（rebase 前的 weights_only）、`port-209ba10`（port/dev，stitched）。
- configs 在主目錄的 `local/`（untracked，只有主目錄有）。

**現在的 configs：`local/condor/configs/hlcos/`**（2026-10-20）
- 由 `python3 local/make_training_configs.py local/plans/hlcos.yml` 產生（plan 檔記錄所有設定）。同時產生 CPU 版 `local/cpu_configs/hlcos/`、scan 的 ensemble `local/scan/hlcos_<wc>.yml`、condor list `~/topsbi-condor/list_hlcos.txt`。
- feature set：highlevel_cos（highlevel 13 個 + `cos_{lep,had}_{n,r,k}`，見 `README_batch_featsets_ywk.md`）。method：`weights_only`。訓練設定和 pilot 相同（`README_batch_pilot_ywk.md`）。
- 檔名 `hlcos_<label>_seed<n>.yml`，seeds 2、3、4。結果寫到 `config['name']` = `/eos/cms/store/user/ykao/topsbi/results/port/0c49da2_hlcos/<label>/seed_<n>`。
- labels（c0 = SM，只動一個 WC）：

| label | c1 | 舊的名字 |
|---|---|---|
| `SM_SM` | SM（`sm` 網路，morphing 和 scan 需要，每個 seed 一次） | |
| `<wc>_<pct>p` | constructive root c_p：yield Y/Y_SM = 1 + pct | `SM_<wc>` |
| `<wc>_<pct>p_neg` | **−c_p**（和 `<wc>_<pct>p` 成 ±c 對，分開一次項和二次項） | 沒有。**不是**舊的 `<wc>_<wc>`（destructive root） |

pct ∈ {0.5, 1, 2.5, 5}，WC ∈ {ctGRe, ctGIm, cQj11, cQd8}。WC 值：

| WC | 0.5p | 1p | 2.5p | 5p | `_neg` 的 yield 變化（0.5 / 1 / 2.5 / 5） |
|---|---|---|---|---|---|
| ctGRe | −0.0158 | −0.0314 | −0.0778 | −0.1532 | −0.50 / −0.99 / −2.42 / −4.68% |
| ctGIm | −0.2859 | −0.4047 | −0.6405 | −0.9062 | +0.49 / +0.99 / +2.48 / +4.98% |
| cQj11 | −0.4535 | −0.6543 | −1.0531 | −1.5027 | +0.37 / +0.82 / +2.20 / +4.58% |
| cQd8 | 0.8099 | 1.402 | 2.7044 | 4.2535 | −0.29 / −0.36 / −0.13 / +0.87% |

`_neg` 的 WC 值就是上表變號；yield 是 Y/Y_SM − 1 = b c + a c²（a、b 在 `local/wc_targets_pilot_train.yml`）。為什麼不用 destructive root：ctGRe 的在 c ≈ 4.7–4.8（n_eff/N 只有 0.05），octet 四費米子在 |c| = 4.9–7.5（`/eos/user/y/ykao/www/topsbi/weights_20261017_pilot_sanitize/`）。

**先單獨跑幾個（cell 1–4），有感覺之後再用 for loop（cell 5–6）。**

```python
# --- cell 1: path，只加程式碼的 src/ ---
import os, sys
TOP = '/eos/user/y/ykao/SWAN_projects/analysis'
CODE = f'{TOP}/topsbi-wt/port-0c49da2'
REPO = f'{TOP}/topsbi'                     # configs 從這裡讀
CONFIGS = f'{REPO}/local/condor/configs/hlcos'
sys.path.insert(0, f'{CODE}/src')   # 不要加 REPO 或 CODE 本身：根目錄的 metrics.py / schema.py 等會蓋掉同名的 package
%load_ext autoreload
# 改了 src/ 的程式不用重開 kernel（% magic 那一行後面不能加註解，會被當成參數）
%autoreload 2

import torch, yaml
from topsbi import train
print(train.__file__, torch.__version__)   # 確認 __file__ 在 CODE 底下
assert torch.cuda.is_available(), '沒有 GPU：SWAN 設定裡選 GPU，或在 cell 2 設 DEVICE = "cpu"'
print(torch.cuda.get_device_name(0))


def describe(config):
    """c1 裡不為 0 的 WC。"""
    moved = {wc: v for wc, v in zip(config['wcs'], config['c1'][1:]) if v}
    return moved or 'SM'


def is_done(config):
    return all(os.path.exists(f"{config['name']}/{f}") for f in ('model.pt', 'complete/performance.yml'))
```

```python
# --- cell 2: 單獨一個網路：載入 config，覆蓋要改的參數 ---
DEVICE = 'cuda'
POINT = 'ctGRe_5p_neg'    # 例：'SM_SM'、'ctGRe_5p'、'ctGRe_5p_neg'、'cQd8_0.5p'、'cQd8_0.5p_neg'
SEED = 2                  # 2、3、4
TEST = False              # True：5 epochs、不畫 animation、寫到自己的目錄（不碰正式結果）

config = yaml.safe_load(open(f'{CONFIGS}/hlcos_{POINT}_seed{SEED}.yml'))
config['device'] = DEVICE
if TEST:
    config['epochs'] = 5
    config['features_to_animate'] = []   # animate_per_epoch 不能設 0：train.py 用它取 %，會除以零
    config['name'] = f'/eos/user/y/ykao/topsbi_notebook/hlcos_{POINT}_seed{SEED}'
print(POINT, 'seed', SEED, describe(config), '->', config['name'])
if is_done(config):
    print('注意：這個網路已經訓練過，cell 3 會覆蓋它')
```

```python
# --- cell 3: train ---
config = train.main(config)           # model.pt 和圖會寫到 config['name']；會先刪掉那裡的 complete/、incomplete/
```

```python
# --- cell 4: 看結果 ---
from IPython.display import Image
print(open(f"{config['name']}/complete/performance.yml").read())
Image(f"{config['name']}/complete/loss.png")
```

### 一次一個 WC 的 for loop（cell 5–6）

先跑 cell 1。cell 5 對一個 WC 跑過 0.5%、1%、2.5%、5% 的 `<wc>_<pct>p`（constructive）和 `<wc>_<pct>p_neg`（−c）。

```python
# --- cell 5: 一個 WC 的 c_p 和 -c_p，4 個 yield 點 × seeds ---
WC = 'ctGRe'                        # 一次一個：ctGRe、ctGIm、cQj11、cQd8
SEEDS = [2]                         # [2, 3, 4]
PCTS = ['0.5p', '1p', '2.5p', '5p']
KINDS = {'constructive': '', 'neg': '_neg'}   # 只跑一種就刪掉另一個
RUN_SM_SM = False                   # 每個 seed 跑一次就好
SKIP_DONE = True                    # 已經有 model.pt 和 performance.yml 的跳過（kernel 斷掉後可以接著跑）
DEVICE = 'cuda'

labels = (['SM_SM'] if RUN_SM_SM else []) + [f'{WC}_{p}{k}' for k in KINDS.values() for p in PCTS]
done = []
for seed in SEEDS:
    for label in labels:
        config = yaml.safe_load(open(f'{CONFIGS}/hlcos_{label}_seed{seed}.yml'))
        config['device'] = DEVICE
        if SKIP_DONE and is_done(config):
            print(f'skip {label} seed {seed}: already trained')
            continue
        print(f'=== {label} seed {seed}: {describe(config)} -> {config["name"]}')
        train.main(config)
        done.append(config['name'])
print('trained:', *done, sep='\n')
```

```python
# --- cell 6: 這個 WC 的結果一覽 ---
for seed in SEEDS:
    for label in labels:
        config = yaml.safe_load(open(f'{CONFIGS}/hlcos_{label}_seed{seed}.yml'))
        if is_done(config):
            perf = yaml.safe_load(open(f"{config['name']}/complete/performance.yml"))
            print(f"{label:16s} seed {seed}  auc {perf['auc']:.4f}  rms_bias {perf['rms_bias']:.4f}  pull {perf['pull']:.2f}")
        else:
            print(f'{label:16s} seed {seed}  (not trained)')
```

- 結果直接寫進正式的 `results/port/0c49da2_hlcos/`（和 condor 一樣），之後 `local/scan/hlcos_<wc>.yml` 的 scan、closure 可以直接用。`SKIP_DONE = False` 會重訓、覆蓋已經有的結果。
- 一個點（100 epochs，GPU）和 condor 的一個 job 一樣久；一個 WC 一個 seed 是 8 個點。SWAN session 有時間限制，跑不完時 `SKIP_DONE = True` 再執行一次就會接著跑。
- 舊的 configs（`wo/`、`log/`，highlevel_log，只有 5% 點，`SM_<wc>`、`<wc>_<wc>` 的命名）還在，要重現舊的結果時把 cell 2 的 path 改成 `{REPO}/local/condor/configs/wo/wo_log_bs10k_{POINT}_seed{SEED}.yml`，並加上 `config['group_cols'] = []`（stitched 的 `log/` configs 沒設）。

## performance.yml 的意義

### port/rebased（`port-0c49da2`、`port-d11cea0`，upstream 的 metrics）

完整說明在 worktree 的 `examples/training/README.md`「Performance metrics」。有設 `holdout` 時（`hlcos`、`wo_*` configs 都有），用的是沒參與訓練的 `validation.p`。除了 `auc`，校準完美時都是 0；`_err` 是完美網路在同樣大小的資料上會有的漲落，數值在約 2 倍 `_err` 內就和 0 一致。

| key | 意義 |
|---|---|
| `auc` | 分類能力，看不出 calibration；只能比較同一組 c0、c1 的不同次訓練 |
| `pull`、`pull_err` | 50 個 r̂ quantile bins 的平均 \|⟨r⟩ − ⟨r̂⟩\|/σ，減掉 √(2/π)：**顯著性**，資料越多同樣的偏差越顯著 |
| `ci_diff`、`ci_diff_err` | 期望減掉實際在 1、2、3σ 內的 bin 比例 |
| `rms_bias` | 同樣 bins 的相對偏差 (⟨r̂⟩ − ⟨r⟩)/⟨r⟩ 的 RMS，扣掉統計漲落：**偏差的大小**，不隨資料量變 |
| `group_*` | 依 input 分組（`group_cols`）× 5 個 r̂ bins 的 closure。我們的 configs 設 `group_cols: []`，只剩 r̂ bins |

看的順序：偏差多大看 `rms_bias`，是否顯著看 `pull`。依 m_ttbar 分段的 calibration 不在 `performance.yml` 裡，用 `local/eval_metrics.py`。

### port/dev（`port-209ba10`，舊的 metrics，只在看舊的 stitched 結果時用）

由 `networkPlots`（`src/topsbi/tools/plots.py`）寫出。`_excl` 是去掉 $\hat r$ 兩端各 1% 的 event（`threshold=0.01`），`_incl` 是全部 event（`threshold=0`，含大 $r$ 的尾巴）。除了 `auc`，都用 1000 個 quantile bin（每個 bin 的 event 數大致相同）。

- **`auc`**（`metrics.py:netEval`）：把網路輸出當分類器，p0 加權為背景、p1 加權為訊號，算 ROC 下面積。0.5 = 分不開。θ₁、θ₀ 很接近時物理上限本來就低，所以它反映的主要是這組 θ 有多難分，只適合比較同一組 θ 的不同次訓練。
- **`chi_excl` / `chi_incl`**（`lr_calibration_curve`）：$\sum_\text{bins} (\langle\hat r\rangle - \langle r\rangle)^2 / \langle r\rangle$，即 calibration curve 離對角線多遠，越小越好。分母是 $\langle r\rangle$ 不是統計誤差，所以**不是**真正的 χ²，不能除以 ndf 判斷好壞，只能比較同樣設定的不同次訓練。`incl` 通常大很多，因為尾巴 $r$ 大、偏差的絕對值也大。
- **`delta_sig` / `ci_ratio`**（`lr_calibration_difference`）：每個 bin 算偏差 $d = \langle r\rangle - \langle\hat r\rangle$ 和它的統計誤差 $e$，pull $= |d|/e$。三個值對應 $k = 1, 2, 3$：
  - `delta_sig` 第 $k$ 個值 = 平均 pull − $k$。三個值永遠剛好各差 1，真正的資訊只有平均 pull = `delta_sig[0] + 1`。
  - `ci_ratio` 第 $k$ 個值 = pull < $k$ 的 bin 比例，即在 $k\sigma$ 內與真值一致的比例。

  校準完美（偏差只剩統計漲落）時，pull 近似標準常態：

  | | 理想值 |
  |---|---|
  | `ci_ratio` | 0.68 / 0.95 / 0.997 |
  | 平均 pull | ≈ 0.80（`delta_sig` ≈ −0.2 / −1.2 / −2.2） |

  例：5 epochs 的 SM_cQj31 得到 `ci_ratio_excl` = 0.37 / 0.57 / 0.72、平均 pull ≈ 2.2，偏差明顯大於統計誤差，還沒校準好。

**看的優先順序**：校準品質看 `ci_ratio`；比較不同次訓練用 `chi`；`auc` 主要反映這組 θ 本身多難分。

## 注意

- **Path**：只把 `src/` 加進 `sys.path`。scripts 用 `PYTHONPATH=src:$PYTHONPATH` 是同樣的道理：要用 append 的方式加，不能整個覆蓋。
- **讀 data**：data 在 `/eos/cms/store/user/ykao/topsbi/`：`pretraining_main_highlevel/`（linear）和 `pretraining_main_highlevel_log/`（m_ttbar 取 log、pT 類取 log1p，`local/make_log_features.py` 產生）。如果 SWAN 讀不到 `/eos/cms`，就把資料夾複製到 `/eos/user/...`，再改 `config['data']` 和 `config['holdout']`。
- **Memory**：train.p 有 1.4M events。session memory 至少選 16 GB。
- **CPU 很慢**：在 CPU 上，1 epoch 要好幾分鐘。完整訓練（100 epochs）還是送 condor 跑，指令見 `README_migration_note_ywk.md`。
- **Branch**：notebook 讀的是 `CODE` 那個 worktree 的程式碼。worktree 是 detached 的固定版本，不會因為 branch 更新而改變；要用新版本時另外開一個 worktree（`git worktree add --detach ../topsbi-wt/port-<hash> <hash>`）。

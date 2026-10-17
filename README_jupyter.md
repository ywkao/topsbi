# 在 CERN SWAN 的 Jupyter notebook 上跑 training

## 開 session

- Software stack 要和 condor 一樣，選 **LCG 109 cuda**，torch、numpy、matplotlib 都已經內建。如果沒有 GPU，選一般的 LCG 109 也可以。
- **不要選 LCG 110**：它的 mplhep 1.1.0 + matplotlib 3.11 + numpy 2.4 組合下，`mh.cms.label` 在 `savefig` 時會報 `TypeError: only 0-dimensional arrays can be converted to Python scalars`，train 會在第一次畫圖（epoch 0）就停掉。
- **預設用 GPU**：開 session 時在 SWAN 設定裡選 GPU。cell 1 會檢查 `torch.cuda.is_available()`，沒有 GPU 就直接停下來。真的要用 CPU 時，把 cell 2 的 `DEVICE` 改成 `'cpu'`（很慢，見下面「注意」）。
- 在 SWAN 裡，repo 的路徑是 `/eos/user/y/ykao/SWAN_projects/analysis/topsbi`（和 `/eos/home-y/...` 是同一個地方）。

## Notebook

**程式碼和 configs 分開放**：
- 程式碼用固定版本的 worktree **`topsbi-wt/port-0c49da2`**（port/rebased，已 rebase 到 eddie/main c810b07，2026-10-17）。`weights_only` 和 `stitched` 都用它。
- 主目錄 `topsbi/` checkout 的是 eddie/main：沒有 `group_cols`、`year_loc: null`，不要拿來跑我們的資料。
- 舊的 worktree 只用來重現已經訓練好的網路：`port-d11cea0`（rebase 前的 weights_only）、`port-209ba10`（port/dev，stitched）。
- configs 在主目錄的 `local/`（untracked，只有主目錄有）。notebook 預設讀 condor 版本 `local/condor/configs/<組>/`（`device: cuda`）；CPU 版本（`device: cpu`，只有 seed 2）在 `local/cpu_configs/<組>/`，cell 2 反正會用 `DEVICE` 覆蓋 `device`。`wo/` 是 weights_only，`log/` 是 stitched。

**用哪個 method**：預設用 **`weights_only`**。
- upstream 推薦它：weights 保留每個樣本的 xsec × lumi normalization，學到的 ratio 包含 yield 的比例，可以直接拿來做 likelihood scan。
- `stitched` 把 p0、p1 各自除以平均值，只學到形狀的比例；scan 時要另外乘 yield ratio。
- 我們的結果（`README_summary_ywk.md` 嘗試 17–23）：cQj11、cQd8 上兩者差不多；ctGRe_ctGRe 上 weights_only 明顯比較好，但只有一個 seed。
- `stitched` 留著當 cross-check。注意 upstream 改過 stitched 的語意，新程式碼的 stitched 結果不能直接和 `port-209ba10` 的舊結果比。

```python
# --- cell 1: path，只加程式碼的 src/ ---
import sys
TOP = '/eos/user/y/ykao/SWAN_projects/analysis'
CODE = f'{TOP}/topsbi-wt/port-0c49da2'   # weights_only 和 stitched 都用這個
REPO = f'{TOP}/topsbi'                     # configs 從這裡讀
sys.path.insert(0, f'{CODE}/src')   # 不要加 REPO 或 CODE 本身：根目錄的 metrics.py / schema.py 等會蓋掉同名的 package
%load_ext autoreload
# 改了 src/ 的程式不用重開 kernel（% magic 那一行後面不能加註解，會被當成參數）
%autoreload 2

import torch, yaml
from topsbi import train
print(train.__file__, torch.__version__)   # 確認 __file__ 在 CODE 底下
assert torch.cuda.is_available(), '沒有 GPU：SWAN 設定裡選 GPU，或在 cell 2 設 DEVICE = "cpu"'
print(torch.cuda.get_device_name(0))
```

```python
# --- cell 2: 載入 config，覆蓋要改的參數 ---
DEVICE = 'cuda'
METHOD = 'weights_only'   # 或 'stitched'
POINT = 'SM_cQj11'        # weights_only 也可以用 'SM_SM'（`sm` 網路，c0 = c1 = SM）
CFG = {
    'weights_only': f'{REPO}/local/condor/configs/wo/wo_log_bs10k_{POINT}_seed2.yml',
    'stitched':     f'{REPO}/local/condor/configs/log/log_bs10k_{POINT}_seed2.yml',
}[METHOD]
config = yaml.safe_load(open(CFG))
assert config['method'] == METHOD
config['device'] = DEVICE
config['name'] = f'/eos/user/y/ykao/topsbi_notebook/{METHOD}_log_bs10k_{POINT}'   # 改到自己的目錄，避免蓋掉 condor 的結果
config['group_cols'] = []             # highlevel 資料沒有 era/njets 欄位；stitched 的舊 configs 沒設，不加會在畫圖時 IndexError
config.setdefault('holdout', '/eos/cms/store/user/ykao/topsbi/pretraining_main_highlevel_log/validation.p')   # 兩個 method 用同一份 validation 比較
config['epochs'] = 5                  # 快速測試用
config['features_to_animate'] = []    # 不產生 animation，比較快（animate_per_epoch 不能設 0：train.py 用它取 %，會除以零）
```

`weights_only` 的 config 還需要 `year_loc: null`、`lumi: [1.0]`（`wo/` 的 configs 都有）；`stitched` 用不到這兩個。

```python
# --- cell 3: train ---
config = train.main(config)           # model.pt 和圖會寫到 config['name']
```

```python
# --- cell 4: 看結果 ---
from IPython.display import Image
print(open(f"{config['name']}/complete/performance.yml").read())
Image(f"{config['name']}/complete/loss.png")
```

## performance.yml 的意義

### port/rebased（`port-0c49da2`、`port-d11cea0`，upstream 的 metrics）

完整說明在 worktree 的 `examples/training/README.md`「Performance metrics」。有設 `holdout` 時（`wo_*` configs 都有，cell 2 也幫 stitched 補上），用的是沒參與訓練的 `validation.p`。除了 `auc`，校準完美時都是 0；`_err` 是完美網路在同樣大小的資料上會有的漲落，數值在約 2 倍 `_err` 內就和 0 一致。

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

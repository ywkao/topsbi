# 在 CERN SWAN 的 Jupyter notebook 上跑 training

## 開 session

- Software stack 要和 condor 一樣，選 **LCG 109 cuda**，torch、numpy、matplotlib 都已經內建。如果沒有 GPU，選一般的 LCG 109 也可以。
- **不要選 LCG 110**：它的 mplhep 1.1.0 + matplotlib 3.11 + numpy 2.4 組合下，`mh.cms.label` 在 `savefig` 時會報 `TypeError: only 0-dimensional arrays can be converted to Python scalars`，train 會在第一次畫圖（epoch 0）就停掉。
- 要用 GPU 的話，在 SWAN 設定裡選 GPU。沒有 GPU 時，`train.main` 會自動退回 CPU（會印出 Warning）。
- 在 SWAN 裡，repo 的路徑是 `/eos/user/y/ykao/SWAN_projects/analysis/topsbi`（和 `/eos/home-y/...` 是同一個地方）。

## Notebook

```python
# --- cell 1: path，只加 src/ ---
import sys
REPO = '/eos/user/y/ykao/SWAN_projects/analysis/topsbi'
sys.path.insert(0, f'{REPO}/src')   # 不要加 REPO 本身：根目錄的 metrics.py / schema.py 等會蓋掉同名的 package
%load_ext autoreload
# 改了 src/ 的程式不用重開 kernel（% magic 那一行後面不能加註解，會被當成參數）
%autoreload 2

import torch, yaml
from topsbi import train
print(train.__file__, torch.__version__, torch.cuda.is_available())
```

```python
# --- cell 2: 載入 config，覆蓋要改的參數 ---
config = yaml.safe_load(open(f'{REPO}/local/condor/configs/a6ff4bd_SM_cQj31.yml'))
config['name'] = '/eos/user/y/ykao/topsbi_notebook/SM_cQj31'   # 改到自己的目錄，避免蓋掉 condor 的結果
config['epochs'] = 5                  # 快速測試用
config['features_to_animate'] = []    # 不產生 animation，比較快（animate_per_epoch 不能設 0：train.py 用它取 %，會除以零）
```

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
- **讀 data**：data 在 `/eos/cms/store/user/ykao/...`。如果 SWAN 讀不到 `/eos/cms`，就把 `pretraining_main_highlevel/` 複製到 `/eos/user/...`，再改 `config['data']`。
- **Memory**：train.p 有 1.4M events。session memory 至少選 16 GB。
- **CPU 很慢**：在 CPU 上，1 epoch 要好幾分鐘。完整訓練（100 epochs）還是送 condor 跑，指令見 `README_migration_note_ywk.md`。
- **Branch**：notebook 讀的是 repo 目前 checkout 的 branch（現在是 `port/dev`）。如果要固定版本，把 `REPO` 指到 worktree，例如 `.../topsbi-wt/port-a6ff4bd`。

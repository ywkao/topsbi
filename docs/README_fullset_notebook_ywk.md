# 在 SWAN notebook 自己跑 full set 的 validation 和 1D scan（2026-10-27）

照順序把每一段貼進一個 cell 執行。完整的背景和結果見：
- `README_fullset_plan_ywk.md`：三階段計畫、設計
- `README_fullset_s1_validation_ywk.md`：第 1 階段結果、bootstrap
- `README_fullset_s2_ywk.md`：第 2 階段（交叉項）結果、r ≤ 0 的根源

## 0. 先知道的事

- 訓練已經全部跑完，notebook 只做「讀網路 → fit → validation / scan」，CPU 就夠，不用 GPU。
- 要用 `sm_weight` 版本（SM_SM 那一列在 lstsq 裡加權 ×100），程式碼在 worktree `topsbi-wt/rebased-fa6f982`（還沒 commit）。
- 每個 `%%bash` cell 都是新的 shell：環境設定要寫在同一個 cell 的開頭。
- 時間：validation（118 個網路）約 10 分鐘；16 個 WC 的 1D scan 約 10 分鐘。

## 1. 共用的環境設定（每個 `%%bash` cell 開頭都貼這段）

```bash
%%bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
W=/eos/home-y/ykao/SWAN_projects/analysis/topsbi-wt/rebased-fa6f982
R=/eos/home-y/ykao/SWAN_projects/analysis/topsbi
export PYTHONPATH=$W/src:$W/scripts:$PYTHONPATH
cd $W
python3 -c "import topsbi, torch; print(topsbi.__file__, torch.__version__)"   # 確認用的是 rebased-fa6f982
```

## 2. Validation

可以選的 config（都在 `$R/local/validation/`）：

| config | 網路 | 加權 |
|---|---|---|
| `fullset_s1.yml` | 第 1 階段 64 個／seed | 無 |
| `fullset_s2.yml` | 第 1 + 2 階段 118 個／seed | 無（舊結果，γ_sm 壞掉） |
| `fullset_s2_smw100.yml` | 118 個／seed | `sm_weight: 100`（**用這個**） |

```bash
%%bash
# （先貼第 1 節的環境設定）
python3 -m topsbi.validation -c $R/local/validation/fullset_s2_smw100.yml --refit 2>&1 | tail -3
```

- `--refit`：重建 ensemble cache。cache 已經存在、網路沒變時可以拿掉，會快很多。
- 想試別的權重：複製 config，改 `sm_weight`（例如 1000）和 `name`（輸出目錄），再跑。

讀結果（Python cell，不用 LCG 環境）：

```python
import yaml
D = '/eos/cms/store/user/ykao/topsbi/results/port/3507174_fullset/validation_fullset_s2_smw100'
d = yaml.safe_load(open(f'{D}/ensemble_diagnostics.yml'))
comp = d['components']
print('flagged:', len(d['flagged']), '/', len(comp))
for k in sorted(comp, key=lambda k: -comp[k]['max_Z'])[:20]:
    print(f"{k:14s} Z={comp[k]['max_Z']:6.1f}  flagged={comp[k]['flagged']}")
print('drop:', d['suspects']['drop'])
print('retrain:', d['suspects']['retrain'])
for p in d['point_search']['proposed']:
    print(p['kind'], p['points'][0])
```

看圖：

```python
from IPython.display import Image, display
for f in ['z_heatmap_compact.png', 'drop_tests_compact.png', 'network_misclosure.png', 'point_search_objective.png']:
    display(Image(f'{D}/plots/{f}'))
```

## 3. 1D scan

`-p` 給 validation config，scan 就用同一組網路和同一個 `sm_weight`。`--wc` 可以一次給多個 WC，網路只載入一次。

```bash
%%bash
# （先貼第 1 節的環境設定）
O=/eos/user/y/ykao/www/topsbi/scan_fullset_s2_20261024
python3 scripts/scan.py -p $R/local/validation/fullset_s2_smw100.yml \
    --wc ctGIm ctGRe cQj38 cQj18 cQu8 cQd8 ctj8 ctu8 ctd8 cQj31 cQj11 cQu1 cQd1 ctj1 ctu1 ctd1 \
    --data-values $R/local/scan/data_2.5p.yml --others sm --threshold 0 --truth --npoints 201 --window 0.5 \
    -o $O/my_smweight100 2>&1 | grep "^{'wc'"
```

- 只跑一個 WC：`--wc ctGRe`。
- pseudo-data 在 2.5% 點（`data_2.5p.yml`），其他 WC = 0；`--truth` 畫出解析的 truth 曲線。
- 加 `--drop-negative`：去掉在 grid 上任何一點 r ≤ 0 的 events。

整理成表（Python cell；表格全英文，可以一次比較多個輸出目錄，例如 CPU 和 cuda）：

```python
import yaml, numpy as np, pandas as pd

S = '/eos/user/y/ykao/www/topsbi/scan_fullset_s2_20261024'
S1 = '/eos/user/y/ykao/www/topsbi/scan_fullset_s1_20261023'
RUNS = {                      # label -> scan output directory; add or remove runs here
    'cpu': f'{S}/my_smweight100',
    'cuda': f'{S}/my_smweight100_cuda',
}
# a directory per WC: put {wc} in the path (double braces inside an f-string), e.g.
# 's1 per-WC fit': f'{S1}/{{wc}}_data2.5p',
WCS = ['ctGIm','ctGRe','cQj38','cQj18','cQu8','cQd8','ctj8','ctu8','ctd8','cQj31','cQj11','cQu1','cQd1','ctj1','ctu1','ctd1']

def summary(path, wc):
    """bias and ci68 width in units of sigma_t (half width of the truth ci68), and the r <= 0 event count"""
    y = yaml.safe_load(open(f'{path.format(wc=wc)}/{wc}_scan.yml')); t = y['truth']
    sigma_t = (t['ci68'][1] - t['ci68'][0]) / 2
    lo, hi = y['ci68']
    return {
        'best fit': y['best_fit'],
        'truth': t['best_fit'],
        'bias / sigma_t': (y['best_fit'] - t['best_fit']) / sigma_t,
        'width / truth': (hi - lo) / 2 / sigma_t if None not in (lo, hi) else np.nan,
        'n(r<=0)': y['n_negative'],
    }

table = pd.concat(
    {run: pd.DataFrame({wc: summary(path, wc) for wc in WCS}).T for run, path in RUNS.items()}, axis=1
)
table.style.format('{:.3g}')
```

- `bias / sigma_t`：best fit 和 truth 的差，以 σ_t（truth ci68 的半寬，只含 data 統計）為單位。
- `width / truth`：ci68 寬度和 truth 的比；`nan` 表示 ci68 在 scan 範圍內沒有閉合。
- 四費米子的 bias 要和 bootstrap std（σ_t 的 7–10 倍）比，見第 1 階段 md。

畫圖（Python cell）：每個 WC 一格，實線是各 run 的 NN，黑色虛線是 truth 曲線，紅色直線是 pseudo-data 的值（`local/scan/data_2.5p.yml`，也就是 truth 的最低點），水平線是 68%、95% 的 Δ(−2 ln L) = 1、3.84。NN 曲線的最低點離紅線多遠，就是 bias。

```python
import matplotlib.pyplot as plt

DATA = yaml.safe_load(open('/eos/user/y/ykao/SWAN_projects/analysis/topsbi/local/scan/data_2.5p.yml'))  # pseudo-data WC values

def plot_scans(runs, wcs, ncols=4, zoom=None, out=None):
    """
    runs: {label: scan output directory}; wcs: WCs to draw
    zoom: optional y-axis maximum of Delta(-2 ln L), e.g. 10
    out: optional file to save the figure to (png/pdf)
    """
    nrows = -(-len(wcs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
    for ax, wc in zip(axes.flat, wcs):
        truth_drawn, notes = False, []
        for label, path in runs.items():
            d = np.load(f'{path.format(wc=wc)}/{wc}_scan.npz')
            line, = ax.plot(d['grid'], d['nll'], label=label)
            # bias of the NN minimum, in WC units and in sigma_t (half width of the truth ci68); ci68 width ratio
            y = yaml.safe_load(open(f'{path.format(wc=wc)}/{wc}_scan.yml')); t = y['truth']
            sigma_t = (t['ci68'][1] - t['ci68'][0]) / 2
            bias = y['best_fit'] - t['best_fit']
            lo, hi = y['ci68']
            width = f'{(hi - lo) / 2 / sigma_t:.2f}' if None not in (lo, hi) else 'open'
            notes.append((f'{label}: bias {bias:+.3g} ({bias / sigma_t:+.1f}$\\sigma_t$), width {width}', line.get_color()))
            if not truth_drawn and 'nll_true' in d.files:
                ax.plot(d['grid'], d['nll_true'], 'k--', label='truth curve (analytic r)')
                truth_drawn = True
        ax.axvline(DATA[wc], color='red', lw=1, label=f'truth value (data_2.5p.yml) = {DATA[wc]:g}')
        for level in (1, 3.84):
            ax.axhline(level, color='gray', ls=':', lw=0.8)
        ax.set_ylim(0, zoom)
        for k, (text, color) in enumerate(reversed(notes)):  # bottom-right corner, one line per run
            ax.text(0.98, 0.03 + 0.07 * k, text, transform=ax.transAxes, ha='right', va='bottom', fontsize=7, color=color,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        ax.set_xlabel(wc)
        ax.set_ylabel(r'$\Delta(-2\ln L)$')
    for ax in axes.flat[len(wcs):]:
        ax.axis('off')
    for ax in axes.flat[: len(wcs)]:
        ax.legend(fontsize=7)  # per panel: the red line's value differs by WC
    fig.tight_layout()
    if out:
        fig.savefig(out)
    plt.show()

plot_scans(RUNS, WCS, zoom=10)                          # all 16 WCs
plot_scans(RUNS, ['ctGRe', 'ctGIm'], ncols=2, zoom=10)  # only some WCs
```

- 比較所有版本（加權與否 × 用哪些網路），`RUNS` 可以這樣寫：
  ```python
  RUNS = {
      's1 per-WC fit':      f'{S1}/{{wc}}_data2.5p',          # 每個 WC 只用自己的 4 個網路 + SM_SM
      's1, unweighted':     f'{S}/s1global_data2.5p',         # 第 1 階段 64 個一起 fit
      's1, smw100':         f'{S}/my_s1_smweight100',         # 第 1 階段 64 個，SM 列 x100（自己跑，見下）
      's1+s2, unweighted':  f'{S}/data2.5p',                  # 118 個一起 fit
      's1+s2, smw100':      f'{S}/my_smweight100_cuda',       # 118 個，SM 列 x100
  }
  ```
  `s1, smw100` 用 `-p $R/local/validation/fullset_s1_smw100_cuda.yml -o $O/my_s1_smweight100` 跑 scan（CPU 版是 `fullset_s1_smw100.yml`）。
  曲線太多會擠，可以一次只比兩三個。
- 右下角：每個 run 一行，`bias` 是 NN 最低點減 truth value（WC 的單位），括號是以 σ_t（truth ci68 的半寬）為單位；`width` 是 NN ci68 寬度 / truth ci68 寬度（> 1 保守，≪ 1 過度自信；`open` 表示 ci68 在 scan 範圍內沒有閉合）。顏色和曲線相同。
- `zoom=None` 顯示整條曲線；四費米子的曲線很窄，`zoom=10` 比較看得清楚 1σ 附近。
- 存成投影片用的圖：`plot_scans(RUNS, WCS, zoom=10, out='scan_16wcs.pdf')`。
- scan.py 自己畫的圖也在：`display(Image(f"{RUNS['cpu']}/ctGRe_scan.png"))`。

怎麼讀：
- ctGRe、ctGIm：偏差應該在幾個 σ_t 內，寬度比約 1.2（ctGIm 約 2.6，NN 只看 reco features）。
- 四費米子：σ_t 只含 data 統計。MC 統計的 bootstrap std 是 σ_t 的 7–10 倍，所以 10σ_t 的偏差大約只是 1–1.5 個 bootstrap σ。
- r ≤ 0：加權後大多是 0 到幾十；還剩較多的是 cQj18、ctj8、ctu8。

## 4. 已經有的輸出（不想重跑可以直接看）

| 內容 | 位置 |
|---|---|
| validation（加權） | `/eos/cms/store/user/ykao/topsbi/results/port/3507174_fullset/validation_fullset_s2_smw100/` |
| validation（沒加權，s1 / s2） | 同目錄的 `validation_fullset_s1/`、`validation_fullset_s2/` |
| scan（正式 sm_weight 100） | `www/topsbi/scan_fullset_s2_20261024/smweight100_data2.5p/` |
| scan（沒加權、118 個） | 同目錄 `data2.5p/`、`data2.5p_dropneg/` |
| scan（沒加權、只有 s1 的 64 個） | 同目錄 `s1global_data2.5p/` |
| scan（第 1 階段，每個 WC 各自 fit） | `www/topsbi/scan_fullset_s1_20261023/<wc>_data2.5p/` |

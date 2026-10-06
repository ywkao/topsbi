import sys
ANALYSIS = "/eos/user/y/ykao/SWAN_projects/analysis"
sys.path.insert(0, f"{ANALYSIS}/topsbi")
sys.path.insert(0, ANALYSIS)

from topsbi.tools.buildLikelihood import full_likelihood
from topsbi.tools.data import get_probabilities

import matplotlib.pyplot as plt
import mplhep as mh
import numpy as np
import torch

import argparse, os, yaml


def find_cl_crossings(grid, nll, level):
    """
    Linear-interp crossings of nll with `level`, the outermost one on each side of
    the minimum. None if the curve is still below `level` at that grid edge (interval
    not closed within the scan range), even if it pokes above level further in.
    """
    imin = np.argmin(nll)
    lo = hi = None
    if nll[imin] >= level:
        return lo, hi
    left = nll[:imin + 1]
    if left[0] > level:
        i = np.where(left <= level)[0][0]
        lo = np.interp(level, [left[i], left[i - 1]], [grid[i], grid[i - 1]])  # np.interp needs increasing xp
    right = nll[imin:]
    if right[-1] > level:
        i = np.where(right <= level)[0][-1]
        j = imin + i
        hi = np.interp(level, [right[i], right[i + 1]], [grid[j], grid[j + 1]])
    return lo, hi


def robust_nll(r, threshold):
    """
    -2*sum(log r), winsorizing r at the given quantile threshold first.

    The lstsq morphing fit in full_likelihood extrapolates badly for a handful
    of events (huge or negative r); same instability lrMeanPlot/sMeanPlot guard
    against with a quantile `threshold`. Clip instead of drop so N stays fixed
    across the grid.
    """
    lo, hi = np.quantile(r, [threshold, 1 - threshold])
    r = np.clip(r, max(lo, 1e-12), hi)
    return -2 * np.log(r).sum()


def truth_ratio(coefficients, wcs, point):
    """Per-event analytic p1/p0 from the quadratic fit coefficients, c0 = SM."""
    c0 = [1.0] + [0.0] * len(wcs)
    p0, p1 = get_probabilities(coefficients, {'wcs': wcs, 'c0': c0, 'c1': point})
    return (p1 / p0).cpu().numpy()


def summarize(grid, nll):
    """Best fit and 68%/95% CL intervals (None where the curve never crosses)."""
    as_float = lambda x: float(x) if x is not None else None
    return {
        'best_fit': float(grid[np.argmin(nll)]),
        'ci68': [as_float(x) for x in find_cl_crossings(grid, nll, 1.0)],
        'ci95': [as_float(x) for x in find_cl_crossings(grid, nll, 3.84)],
    }


def default_range(cg):
    """Span SM (0) and the sample generation point cg, padded by |cg|/2 on each side."""
    pad = 0.5 * abs(cg) or 1.0
    return min(0.0, cg) - pad, max(0.0, cg) + pad


def plot_scan(grid, nll, nll_true, wc, path, xlim=None, ylim=None):
    """nll_true=None skips the truth curve; ylim defaults to the curves' max inside xlim."""
    xlim = xlim or (grid[0], grid[-1])
    if ylim is None:
        inside = (grid >= xlim[0]) & (grid <= xlim[1])
        curves = [nll] if nll_true is None else [nll, nll_true]
        ylim = (0, max([5] + [c[inside].max() * 1.05 for c in curves if inside.any()]))

    mh.style.use('CMS')
    fig, ax = plt.subplots()
    ax.plot(grid, nll, linewidth=3, label='Ensemble')
    if nll_true is not None:
        ax.plot(grid, nll_true, color='red', linestyle='--', linewidth=3, label=r'Truth ($p_1/p_0$)')
    ax.axhline(1.0, color='grey', linestyle='--', label='68% CL')
    ax.axhline(3.84, color='grey', linestyle=':', label='95% CL')
    ax.set_xlabel(wc)
    ax.set_ylabel(r'$-2\Delta\ln L$')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend()
    mh.cms.label('Preliminary', data=False, lumi=137.64, com=13, ax=ax)
    fig.savefig(path)
    plt.close(fig)


def scan_wc(plr, coefficients, wc, wc_min, wc_max, npoints, output, threshold, truth, base):
    """`base`: SM-inclusive point the other WCs are held at (SM, or cg for a closure test)."""
    wcs = plr.wcs
    idx = wcs.index(wc)

    grid = np.linspace(wc_min, wc_max, npoints)
    nll = np.empty(npoints)
    nll_true = np.zeros(npoints)
    for i, v in enumerate(grid):
        point = list(base)
        point[idx + 1] = v
        r = plr(point).detach().cpu().numpy()
        nll[i] = robust_nll(r, threshold)
        if truth:
            # same winsorization as the ensemble so the two curves differ only by the NN
            nll_true[i] = robust_nll(truth_ratio(coefficients, wcs, point), threshold)
    nll -= nll.min()
    nll_true -= nll_true.min()

    os.makedirs(output, mode=0o755, exist_ok=True)
    plot_scan(grid, nll, nll_true if truth else None, wc, f'{output}/{wc}_scan.png')
    # raw curves, so replot_scan.py can re-zoom without reloading the ensemble
    np.savez(f'{output}/{wc}_scan.npz', grid=grid, nll=nll, **({'nll_true': nll_true} if truth else {}))

    summary = {'wc': wc, **summarize(grid, nll)}
    if truth:
        summary['truth'] = summarize(grid, nll_true)
    with open(f'{output}/{wc}_scan.yml', 'w') as f:
        f.write(yaml.dump(summary))
    print(summary)


def main(parametric, wc_list, wc_min, wc_max, npoints, output, threshold, truth, others):
    with open(parametric) as f:
        config = yaml.safe_load(f)
    with open(config['features']) as f:
        config['features'] = yaml.safe_load(f)

    features, coefficients = torch.load(config['data'], weights_only=False)[:]
    features = features.float()

    # loading the ensemble dominates the runtime, so do it once for all WCs
    plr = full_likelihood(config, features)
    coefficients = coefficients[plr.infFilter]
    with open(config['networks'][0]) as f:
        cg = yaml.safe_load(f)['cg']
    base = cg if others == 'cg' else [1.0] + [0.0] * len(plr.wcs)

    for wc in wc_list:
        lo, hi = default_range(cg[plr.wcs.index(wc) + 1])
        lo = lo if wc_min is None else wc_min
        hi = hi if wc_max is None else wc_max
        scan_wc(plr, coefficients, wc, lo, hi, npoints, output, threshold, truth, base)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parametric', '-p', required=True, help='parametric likelihood config yml (same as used for validation.py)')
    parser.add_argument('--wc', nargs='+', required=True, help='Wilson coefficient name(s) to scan, e.g. ctGRe ctGIm')
    parser.add_argument('--min', dest='wc_min', type=float, default=None, help='default: span SM and cg, see default_range')
    parser.add_argument('--max', dest='wc_max', type=float, default=None, help='default: span SM and cg, see default_range')
    parser.add_argument('--npoints', type=int, default=101)
    parser.add_argument('--output', '-o', required=True, help='directory to save scan plot/summary')
    parser.add_argument('--threshold', type=float, default=0.01, help='quantile to winsorize per-event r at (same convention as lrMeanPlot/sMeanPlot)')
    parser.add_argument('--truth', action='store_true', help='overlay analytic p1/p0 truth curve (winsorized with the same --threshold)')
    parser.add_argument('--others', choices=['sm', 'cg'], default='sm',
                        help='hold the non-scanned WCs at SM (0) or at the sample generation point cg; cg = closure test, truth minimum should sit at cg')

    args = parser.parse_args()
    main(args.parametric, args.wc, args.wc_min, args.wc_max, args.npoints, args.output, args.threshold, args.truth, args.others)

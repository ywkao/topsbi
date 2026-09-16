from topsbi.tools.buildLikelihood import full_likelihood

import matplotlib.pyplot as plt
import mplhep as mh
import numpy as np
import torch

import argparse, os, yaml


def find_cl_crossings(grid, nll, level):
    """Linear-interp crossings of nll with `level`, one on each side of the minimum."""
    imin = np.argmin(nll)
    lo = hi = None
    left = nll[:imin + 1]
    if left.min() < level < left.max():
        i = np.where(left <= level)[0][0]
        lo = np.interp(level, [left[i - 1], left[i]], [grid[i - 1], grid[i]])
    right = nll[imin:]
    if right.min() < level < right.max():
        i = np.where(right <= level)[0][-1]
        j = imin + i
        hi = np.interp(level, [right[i + 1], right[i]], [grid[j + 1], grid[j]])
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


def main(parametric, wc, wc_min, wc_max, npoints, output, threshold):
    with open(parametric) as f:
        config = yaml.safe_load(f)
    with open(config['features']) as f:
        config['features'] = yaml.safe_load(f)

    features, _ = torch.load(config['data'], weights_only=False)[:]
    features = features.float()

    plr = full_likelihood(config, features)
    wcs = plr.wcs
    idx = wcs.index(wc)

    grid = np.linspace(wc_min, wc_max, npoints)
    nll = np.empty(npoints)
    for i, v in enumerate(grid):
        point = [1.0] + [0.0] * len(wcs)
        point[idx + 1] = v
        r = plr(point).detach().cpu().numpy()
        nll[i] = robust_nll(r, threshold)
    nll -= nll.min()

    best_fit = grid[np.argmin(nll)]
    lo68, hi68 = find_cl_crossings(grid, nll, 1.0)
    lo95, hi95 = find_cl_crossings(grid, nll, 3.84)

    mh.style.use('CMS')
    fig, ax = plt.subplots()
    ax.plot(grid, nll, linewidth=3)
    ax.axhline(1.0, color='grey', linestyle='--', label='68% CL')
    ax.axhline(3.84, color='grey', linestyle=':', label='95% CL')
    ax.set_xlabel(wc)
    ax.set_ylabel(r'$-2\Delta\ln L$')
    ax.set_ylim(0, max(5, nll.max() * 1.05))
    ax.legend()
    mh.cms.label('Preliminary', data=False, lumi=137.64, com=13, ax=ax)

    os.makedirs(output, mode=0o755, exist_ok=True)
    fig.savefig(f'{output}/{wc}_scan.png')
    plt.close(fig)

    summary = {
        'wc': wc,
        'best_fit': float(best_fit),
        'ci68': [float(lo68) if lo68 is not None else None, float(hi68) if hi68 is not None else None],
        'ci95': [float(lo95) if lo95 is not None else None, float(hi95) if hi95 is not None else None],
    }
    with open(f'{output}/{wc}_scan.yml', 'w') as f:
        f.write(yaml.dump(summary))
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parametric', '-p', required=True, help='parametric likelihood config yml (same as used for validation.py)')
    parser.add_argument('--wc', required=True, help='Wilson coefficient name to scan, e.g. ctGRe')
    parser.add_argument('--min', dest='wc_min', type=float, default=-5.0)
    parser.add_argument('--max', dest='wc_max', type=float, default=5.0)
    parser.add_argument('--npoints', type=int, default=101)
    parser.add_argument('--output', '-o', required=True, help='directory to save scan plot/summary')
    parser.add_argument('--threshold', type=float, default=0.01, help='quantile to winsorize per-event r at (same convention as lrMeanPlot/sMeanPlot)')

    args = parser.parse_args()
    main(args.parametric, args.wc, args.wc_min, args.wc_max, args.npoints, args.output, args.threshold)

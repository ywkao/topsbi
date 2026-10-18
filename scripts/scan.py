from topsbi.tools.buildLikelihood import ensemble, full_likelihood
from topsbi.tools.data import get_lower_tri, get_probabilities

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
    left = nll[: imin + 1]
    if left[0] > level:
        i = np.where(left <= level)[0][0]
        lo = np.interp(level, [left[i], left[i - 1]], [grid[i], grid[i - 1]])  # np.interp needs increasing xp
    right = nll[imin:]
    if right[-1] > level:
        i = np.where(right <= level)[0][-1]
        j = imin + i
        hi = np.interp(level, [right[i], right[i + 1]], [grid[j], grid[j + 1]])
    return lo, hi


def robust_nll(r, threshold, weights=None):
    """
    -2*sum(weights * log r), winsorizing r at the given quantile threshold first.

    The lstsq morphing fit in full_likelihood extrapolates badly for a handful
    of events (huge or negative r); same instability lrMeanPlot/sMeanPlot guard
    against with a quantile `threshold`. Clip instead of drop so N stays fixed
    across the grid.
    """
    lo, hi = np.quantile(r, [threshold, 1 - threshold])
    r = np.clip(r, max(lo, 1e-12), hi)
    return -2 * (np.log(r) if weights is None else weights * np.log(r)).sum()


def weight(coefficients, point):
    """Per-event w(point) = coefs @ T(point), float64."""
    return (coefficients.to(torch.float64) @ get_lower_tri(point, dtype=torch.float64)).cpu().numpy()


def sm_point(n_wcs):
    return [1.0] + [0.0] * n_wcs


def truth_ratio(coefficients, point, method):
    """
    Per-event analytic ratio to SM, what the networks learn:
    weights_only: w(point)/w(SM), rate included; stitched: p1/p0, each normalized by its mean (shape only).
    """
    if method == 'weights_only':
        return weight(coefficients, point) / weight(coefficients, sm_point(len(point) - 1))
    p0, p1 = get_probabilities(coefficients, {'c0': sm_point(len(point) - 1), 'c1': point})
    return (p1 / p0).cpu().numpy()


def data_weights(coefficients, point, method):
    """
    Per-event pseudo-data weights at `point`, reweighting the sample:
    weights_only: w(point)/mean(w(SM)), so the total is N * Y(point)/Y(SM) (the rate enters the extended likelihood);
    stitched: w(point)/mean(w(point)), mean 1.
    """
    w = weight(coefficients, point)
    return w / (weight(coefficients, sm_point(len(point) - 1)).mean() if method == 'weights_only' else w.mean())


def summarize(grid, nll):
    """Best fit and 68%/95% CL intervals (None where the curve never crosses)."""

    def as_float(x):
        return float(x) if x is not None else None

    return {
        'best_fit': float(grid[np.argmin(nll)]),
        'ci68': [as_float(x) for x in find_cl_crossings(grid, nll, 1.0)],
        'ci95': [as_float(x) for x in find_cl_crossings(grid, nll, 3.84)],
    }


def default_range(c):
    """Span SM (0) and c (the data point along the scanned WC), padded by |c|/2 on each side."""
    pad = 0.5 * abs(c) or 1.0
    return min(0.0, c) - pad, max(0.0, c) + pad


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


def scan_wc(
    plr,
    coefficients,
    wcs,
    cg,
    wc,
    wc_min,
    wc_max,
    npoints,
    output,
    threshold,
    truth,
    base,
    drop_negative,
    data=None,
    method='weights_only',
):
    """
    `method`: weights_only networks learn w(c)/w(SM), so -2 ln L is extended,
    -2 sum(w_data log r(c)) + 2 sum(w_SM r(c)) (the second term is the expected yield, N Y(c)/Y(SM));
    stitched networks learn the shape ratio p1/p0, so -2 ln L = -2 sum(w_data log r(c)).
    `base`: SM-inclusive point the other WCs are held at (SM, or cg for a closure test).
    `data`: SM-inclusive point to reweight the sample to as pseudo-data (None: the raw sample, i.e. data at cg);
    for a closure test the minimum should then sit at data.
    `drop_negative`: drop every event whose ensemble (or truth) r <= 0 anywhere on the grid, so
    both curves use one fixed event subset instead of per-point clamping/winsorizing.
    """
    idx = wcs.index(wc)

    grid = np.linspace(wc_min, wc_max, npoints)
    points = []
    for v in grid:
        point = list(base)
        point[idx + 1] = v
        points.append(point)
    # [npoints, n_events]; ~140 MB each for 200 points x 180k events
    r = np.stack([plr(p).detach().cpu().numpy() for p in points])
    t = np.stack([truth_ratio(coefficients, p, method) for p in points]) if truth else None

    if data is None and method == 'weights_only':
        data = cg  # the raw sample is distributed as at cg; weights_only needs it as explicit weights for the rate
    w = None if data is None else data_weights(coefficients, data, method)
    w_sm = weight(coefficients, sm_point(len(wcs)))
    negative = (r <= 0).any(0)
    n_negative, negative_frac = int(negative.sum()), float(w_sm[negative].sum() / w_sm.sum())
    print(
        f'[{wc}] {n_negative}/{negative.size} events with r <= 0 somewhere on the grid, {negative_frac:.2e} of the SM weight'
    )
    n_dropped = 0
    if drop_negative:
        keep = (r > 0).all(0) & ((t > 0).all(0) if truth else True) & (True if w is None else w > 0)
        n_dropped = int((~keep).sum())
        print(f'[{wc}] dropping {n_dropped}/{keep.size} events with r <= 0 somewhere on the grid')
        sub = coefficients[torch.from_numpy(keep)]
        # restore E_p0[r] = 1 on the subset; get_probabilities does the same for truth below
        p0 = (sub @ get_lower_tri([1.0] + [0.0] * len(wcs), dtype=sub.dtype)).numpy()
        r = r[:, keep]
        if method != 'weights_only':
            r *= (p0.mean() / (r * p0).mean(1))[:, None]
        if truth:
            t = np.stack([truth_ratio(sub, p, method) for p in points])
        if w is not None:
            w = data_weights(sub, data, method)
        w_sm = w_sm[keep]

    def nll_of(ratios):
        # same winsorization for the ensemble and the truth so the two curves differ only by the NN
        nll = np.array([robust_nll(x, threshold, w) for x in ratios])
        if method == 'weights_only':
            nll += 2 * (ratios * (w_sm / w_sm.mean())).sum(1)
        return nll

    nll = nll_of(r)
    nll_true = nll_of(t) if truth else np.zeros(npoints)
    nll -= nll.min()
    nll_true -= nll_true.min()

    os.makedirs(output, mode=0o755, exist_ok=True)
    plot_scan(grid, nll, nll_true if truth else None, wc, f'{output}/{wc}_scan.png')
    # raw curves, so replot_scan.py can re-zoom without reloading the ensemble
    np.savez(f'{output}/{wc}_scan.npz', grid=grid, nll=nll, **({'nll_true': nll_true} if truth else {}))

    summary = {
        'wc': wc,
        'method': method,
        'n_negative': n_negative,
        'negative_sm_weight_frac': negative_frac,
        'n_dropped': n_dropped,
        **summarize(grid, nll),
    }
    if data is not None:
        summary['data'] = float(data[idx + 1])
    if truth:
        summary['truth'] = summarize(grid, nll_true)
    with open(f'{output}/{wc}_scan.yml', 'w') as f:
        f.write(yaml.dump(summary))
    print(summary)


def main(
    parametric,
    wc_list,
    wc_min,
    wc_max,
    npoints,
    output,
    threshold,
    truth,
    others,
    drop_negative,
    data_values,
    window=None,
    method=None,
):
    with open(parametric) as f:
        config = yaml.safe_load(f)
    with open(config['features']) as f:
        config['features'] = yaml.safe_load(f)

    features, coefficients, _ = torch.load(config['data'], weights_only=False)[:]
    features = features.float()

    # loading the ensemble dominates the runtime, so do it once for all WCs
    # `seeds`: {seed: {sm, networks}}, averaged by ensemble; otherwise one set of networks (sm + networks)
    if 'seeds' in config:
        plr = ensemble(config, features)
        first = config['seeds'][next(iter(config['seeds']))]['networks'][0]
    else:
        plr = full_likelihood(config, features)
        first = config['networks'][0]
    coefficients = coefficients[plr.infFilter]
    with open(first) as f:
        network = yaml.safe_load(f)
    cg, wcs = network['cg'], network['wcs']
    method = method or network.get('method', 'stitched')
    print(f'[INFO] {method} networks')
    base = cg if others == 'cg' else [1.0] + [0.0] * len(wcs)
    if data_values:
        with open(data_values) as f:
            data_values = yaml.safe_load(f)

    for wc in wc_list:
        idx = wcs.index(wc) + 1
        data = None
        if data_values:
            data = list(base)
            data[idx] = data_values[wc]
        center = cg[idx] if data is None else data[idx]
        lo, hi = default_range(center)
        if window:
            # per-WC range around the expected minimum, so the grid resolves the 68% interval
            half = window * (abs(center) or 1.0)
            lo, hi = center - half, center + half
        lo = lo if wc_min is None else wc_min
        hi = hi if wc_max is None else wc_max
        scan_wc(
            plr, coefficients, wcs, cg, wc, lo, hi, npoints, output, threshold, truth, base, drop_negative, data, method
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--parametric', '-p', required=True, help='parametric likelihood config yml (same as used for validation.py)'
    )
    parser.add_argument('--wc', nargs='+', required=True, help='Wilson coefficient name(s) to scan, e.g. ctGRe ctGIm')
    parser.add_argument(
        '--min', dest='wc_min', type=float, default=None, help='default: span SM and cg, see default_range'
    )
    parser.add_argument(
        '--max', dest='wc_max', type=float, default=None, help='default: span SM and cg, see default_range'
    )
    parser.add_argument('--npoints', type=int, default=101)
    parser.add_argument('--output', '-o', required=True, help='directory to save scan plot/summary')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='quantile to winsorize per-event r at (same convention as lrMeanPlot/sMeanPlot)',
    )
    parser.add_argument(
        '--truth', action='store_true', help='overlay analytic p1/p0 truth curve (winsorized with the same --threshold)'
    )
    parser.add_argument(
        '--others',
        choices=['sm', 'cg'],
        default='sm',
        help='hold the non-scanned WCs at SM (0) or at the sample generation point cg; cg = closure test, truth minimum should sit at cg',
    )
    parser.add_argument(
        '--drop-negative',
        action='store_true',
        help='drop events with r <= 0 anywhere on the grid (fixed subset for both curves); use with --threshold 0',
    )

    parser.add_argument(
        '--data-values',
        default=None,
        help='yml {wc: value}: pseudo-data at --others with the scanned WC set to value, by reweighting the sample; '
        'closure test near SM with --others sm',
    )

    parser.add_argument(
        '--window',
        type=float,
        default=None,
        help='scan data (or cg) +- window * |value| per WC instead of default_range; --min/--max still override',
    )

    parser.add_argument(
        '--method',
        choices=['weights_only', 'stitched'],
        default=None,
        help='what the networks learned, which sets the truth, pseudo-data weights and likelihood; '
        'default: method in the first network config',
    )

    args = parser.parse_args()
    main(
        args.parametric,
        args.wc,
        args.wc_min,
        args.wc_max,
        args.npoints,
        args.output,
        args.threshold,
        args.truth,
        args.others,
        args.drop_negative,
        args.data_values,
        args.window,
        args.method,
    )

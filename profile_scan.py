from topsbi.tools.buildLikelihood import full_likelihood

import matplotlib.pyplot as plt
import mplhep as mh
import numpy as np
import torch
import yaml
from scipy.optimize import minimize

import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.scan import robust_nll, find_cl_crossings  # noqa: E402  (reuse, not modify)

SANITY_TOL = 1e-3


def infer_wc_ranges(config):
    """
    Per-WC (min, max) inferred from the training points of the single-WC
    networks listed in config['networks']. Each network's c1 differs from
    its c0=SM point in exactly one WC; the SM point (0.0) is always a safe
    endpoint since every network was trained against it.
    """
    values = {}
    for net_path in config['networks']:
        with open(net_path) as f:
            net_cfg = yaml.safe_load(f)
        wcs, c0, c1 = net_cfg['wcs'], net_cfg['c0'], net_cfg['c1']
        for i in range(1, len(c1)):
            if c1[i] != c0[i]:
                values.setdefault(wcs[i - 1], []).append(c1[i])
    return {wc: (min(v + [0.0]), max(v + [0.0])) for wc, v in values.items()}


def build_point(n_wcs, poi_idx, poi_value, free_idx, nuisance):
    values = [0.0] * n_wcs
    values[poi_idx] = poi_value
    for k, i in enumerate(free_idx):
        values[i] = nuisance[k]
    return [1.0] + values


def nuisance_scales(bounds):
    """Per-nuisance normalization for the SM-penalty: the training-range edge
    farthest from 0, so a nuisance exactly at its bound contributes 1.0 (pre
    reg_strength) to the penalty sum, regardless of that WC's own range width."""
    return np.array([max(abs(lo), abs(hi)) for lo, hi in bounds])


def profile_objective(nuisance, plr, n_wcs, poi_idx, poi_value, free_idx, threshold, scales, reg_strength):
    point = build_point(n_wcs, poi_idx, poi_value, free_idx, nuisance)
    r = plr(point).detach().cpu().numpy()
    penalty = reg_strength * np.sum((np.asarray(nuisance) / scales) ** 2)
    return robust_nll(r, threshold) + penalty


def inner_minimize(plr, n_wcs, poi_idx, poi_value, free_idx, bounds, threshold,
                    reg_strength, scales, x0=None, minimize_kwargs=None):
    """
    Minimize `profile_objective` (robust_nll + SM-pull penalty) over the WCs
    indexed by `free_idx`, holding WC `poi_idx` fixed at `poi_value`.

    `plr` (the likelihood ensemble) and `free_idx` (the nuisance subset) are
    both parameters rather than assumptions, so this can be reused for a
    different likelihood object, a smaller/grouped nuisance set, or a
    non-POI reference point without touching the scan loop.

    Gradient-free (Powell) is required: robust_nll winsorizes per grid point,
    which is not differentiable, so L-BFGS-style gradients are not valid here.
    """
    if x0 is None:
        x0 = np.zeros(len(free_idx))
    kwargs = dict(xtol=1e-8, ftol=1e-10, maxiter=2000 * max(len(free_idx), 1))
    if minimize_kwargs:
        kwargs.update(minimize_kwargs)
    return minimize(
        profile_objective, x0,
        args=(plr, n_wcs, poi_idx, poi_value, free_idx, threshold, scales, reg_strength),
        method='Powell', bounds=bounds, options=kwargs,
    )


def run_profile_scan(plr, wcs, wc, grid, bounds, free_idx, threshold, reg_strength, bound_tol=0.01):
    """
    Returns per grid point: total (penalized) NLL, bare (unpenalized) NLL,
    nuisance best-fit values, optimizer convergence flag, and the count of
    nuisances sitting within `bound_tol` (relative) of their training bound
    -- a high count is a sign the optimizer wants to leave the region the
    morphing was actually trained on, even with the SM-penalty active.
    """
    poi_idx = wcs.index(wc)
    n_wcs = len(wcs)
    n_grid = len(grid)
    zero_idx = int(np.argmin(np.abs(grid)))
    scales = nuisance_scales(bounds)

    nll_total = np.full(n_grid, np.nan)
    nll_bare = np.full(n_grid, np.nan)
    nuisance_fit = np.full((n_grid, len(free_idx)), np.nan)
    converged = np.zeros(n_grid, dtype=bool)
    n_at_bound = np.zeros(n_grid, dtype=int)

    def run_direction(indices):
        x0 = np.zeros(len(free_idx))
        for i in indices:
            res = inner_minimize(plr, n_wcs, poi_idx, grid[i], free_idx, bounds,
                                  threshold, reg_strength, scales, x0=x0)
            nll_total[i] = res.fun
            nuisance_fit[i] = res.x
            converged[i] = res.success
            x0 = res.x

            point = build_point(n_wcs, poi_idx, grid[i], free_idx, res.x)
            r = plr(point).detach().cpu().numpy()
            nll_bare[i] = robust_nll(r, threshold)

            n_at_bound[i] = sum(
                1 for v, (lo, hi) in zip(res.x, bounds)
                if abs(v - lo) <= bound_tol * max(abs(lo), 1e-9) or abs(v - hi) <= bound_tol * max(abs(hi), 1e-9)
            )

    run_direction(range(zero_idx, n_grid))
    run_direction(range(zero_idx, -1, -1))
    return nll_total, nll_bare, nuisance_fit, converged, n_at_bound


def fixed_nll_curve(plr, n_wcs, poi_idx, grid, threshold):
    """scan.py-equivalent curve: all non-POI WCs held at SM (0)."""
    nll = np.empty(len(grid))
    for i, v in enumerate(grid):
        point = build_point(n_wcs, poi_idx, v, [], [])
        r = plr(point).detach().cpu().numpy()
        nll[i] = robust_nll(r, threshold)
    return nll


def sanity_check_vs_fixed(grid, profile_rel, fixed_rel, tol=SANITY_TOL):
    abs_diff = np.abs(profile_rel - fixed_rel)
    i_max = int(np.argmax(abs_diff))
    diff = float(abs_diff[i_max])
    verdict = "DEGENERATE TO FIXED-SM (expected)" if diff < tol else "NON-TRIVIAL (investigate)"
    print(f"\n[sanity check] max|profile - fixed-SM| Delta(-2lnL) = {diff:.3e}  ->  {verdict}")
    print(f"[sanity check] driven by grid point v={grid[i_max]:.4f}: "
          f"profile={profile_rel[i_max]:.3e}, fixed={fixed_rel[i_max]:.3e}")
    return diff, verdict


def main(config_path, wc, n_grid, threshold, reg_strength, output):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(config['features']) as f:
        config['features'] = yaml.safe_load(f)

    features, _ = torch.load(config['data'], weights_only=False)[:]
    features = features.float()

    plr = full_likelihood(config, features)
    wcs = plr.wcs
    n_wcs = len(wcs)
    poi_idx = wcs.index(wc)
    free_idx = [i for i in range(n_wcs) if i != poi_idx]

    wc_ranges = infer_wc_ranges(config)
    wc_min, wc_max = wc_ranges[wc]
    bounds = [wc_ranges[wcs[i]] for i in free_idx]

    grid = np.linspace(wc_min, wc_max, n_grid)
    nll_total, nll_bare, nuisance_fit, converged, n_at_bound = run_profile_scan(
        plr, wcs, wc, grid, bounds, free_idx, threshold, reg_strength)
    # the physically meaningful curve is the *bare* NLL at the regularized
    # best-fit nuisances; the penalty is only a device to steer the search.
    profile_rel = nll_bare - np.nanmin(nll_bare)

    print(f"WC = {wc}   range inferred from training points = [{wc_min}, {wc_max}]")
    print(f"nuisance WCs ({len(free_idx)}): {[wcs[i] for i in free_idx]}")
    print(f"reg_strength = {reg_strength}")
    print()
    print(f"{'v':>8} {'-2dlnL':>10} {'conv':>6} {'n@bnd':>6}  nuisance best-fit")
    for i in range(n_grid):
        row = ", ".join(f"{wcs[j]}={nuisance_fit[i, k]:+.3f}" for k, j in enumerate(free_idx))
        print(f"{grid[i]:8.4f} {profile_rel[i]:10.4f} {str(converged[i]):>6} {n_at_bound[i]:6d}  {row}")

    fixed_nll = fixed_nll_curve(plr, n_wcs, poi_idx, grid, threshold)
    fixed_rel = fixed_nll - fixed_nll.min()
    sanity_diff, sanity_verdict = sanity_check_vs_fixed(grid, profile_rel, fixed_rel)

    best_fit = float(grid[np.argmin(profile_rel)])
    lo68, hi68 = find_cl_crossings(grid, profile_rel, 1.0)
    lo95, hi95 = find_cl_crossings(grid, profile_rel, 3.84)
    print(f"best_fit = {best_fit:.4f}")
    print(f"68% CL   = [{lo68}, {hi68}]")
    print(f"95% CL   = [{lo95}, {hi95}]")

    if output:
        os.makedirs(output, mode=0o755, exist_ok=True)

        mh.style.use('CMS')
        fig, ax = plt.subplots()
        ax.plot(grid, profile_rel, linewidth=3, label='Profiled (15 nuisance WCs)')
        ax.plot(grid, fixed_rel, linewidth=2, linestyle='-.', color='C1', label='Fixed-SM (scan.py)')
        ax.axhline(1.0, color='grey', linestyle='--', label='68% CL')
        ax.axhline(3.84, color='grey', linestyle=':', label='95% CL')
        ax.set_xlabel(wc)
        ax.set_ylabel(r'$-2\Delta\ln L$')
        ax.set_ylim(0, max(5, profile_rel.max() * 1.05))
        ax.legend()
        mh.cms.label('Preliminary', data=False, lumi=137.64, com=13, ax=ax)
        fig.subplots_adjust(bottom=0.22)
        fig.text(
            0.5, 0.02,
            f'Profile over {len(free_idx)} nuisance WCs (currently equivalent to fixed-SM due to single-axis training)',
            ha='center', va='bottom', fontsize=9,
        )
        fig.savefig(f'{output}/{wc}_profile_scan.png')
        plt.close(fig)

        np.savez(
            f'{output}/{wc}_profile_scan.npz',
            grid=grid, profile_nll=profile_rel, fixed_nll=fixed_rel,
            nuisance_names=np.array([wcs[i] for i in free_idx]),
            nuisance_fit=nuisance_fit, converged=converged, n_at_bound=n_at_bound,
            best_fit=best_fit, ci68=np.array([lo68, hi68], dtype=float),
            ci95=np.array([lo95, hi95], dtype=float),
            sanity_diff=sanity_diff, sanity_verdict=sanity_verdict,
        )
        print(f"\nSaved: {output}/{wc}_profile_scan.png")
        print(f"Saved: {output}/{wc}_profile_scan.npz")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wc', required=True, help='Wilson coefficient name to scan, e.g. ctGRe')
    parser.add_argument('--config', required=True, help='parametric likelihood config yml (same as scan.py --parametric)')
    parser.add_argument('--n-grid', type=int, default=50)
    parser.add_argument('--threshold', type=float, default=0.01, help='same winsorization convention as scan.py')
    parser.add_argument('--nuisance-reg', type=float, default=1.0, dest='reg_strength',
                         help='L2 penalty strength pulling nuisance WCs toward SM, in units of -2dlnL '
                              'per nuisance fully at its training-range edge (0 disables)')
    parser.add_argument('--output', '-o', default=None, help='directory to save scan plot/npz (omit to only print)')
    args = parser.parse_args()
    main(args.config, args.wc, args.n_grid, args.threshold, args.reg_strength, args.output)

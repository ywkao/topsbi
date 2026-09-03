"""
1D scan of -2 Delta ln L over ctGRe.
Compares ensemble prediction (blue) with truth from analytic ratio (red).
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml

from topsbi.tools.buildLikelihood import full_likelihood
from topsbi.tools.data import get_probabilities


# =========================================================================
# Config
# =========================================================================
PARAMETRIC_CONFIG = './examples/validation/config_ywk.yml'
TARGET_WC         = 'ctGRe'
SCAN_MIN          = -1.0
SCAN_MAX          = 4.0
N_POINTS          = 201
Y_MAX             = 10.0
OUTPUT            = 'scan_ctGRe.png'


# =========================================================================
# Load
# =========================================================================
with open(PARAMETRIC_CONFIG) as f:
    config = yaml.safe_load(f)
with open(config['features']) as f:
    config['features'] = yaml.safe_load(f)

features, coefficients = torch.load(config['data'], weights_only=False)[:]
features = features.float()

# Build parametric likelihood ensemble
plr = full_likelihood(config, features)

# Apply the same inf-filter used by the ensemble
features     = features[plr.infFilter]
coefficients = coefficients[plr.infFilter]

# Resolve WC index
# coefs layout: [SM_constant, wc_0, wc_1, ...]
# plr.wcs      : [wc_0, wc_1, ...]  (does NOT include the SM constant)
idx_in_wcs   = plr.wcs.index(TARGET_WC)
idx_in_coefs = idx_in_wcs + 1

print(f"plr.wcs             = {plr.wcs}")
print(f"target WC           = {TARGET_WC}")
print(f"idx in plr.wcs      = {idx_in_wcs}")
print(f"idx in coefs vector = {idx_in_coefs}")
print(f"n_events after inf filter = {features.shape[0]}")


# =========================================================================
# Prepare truth-ratio config
# =========================================================================
# get_probabilities needs a config-like dict with wcs, c0, c1.
# c0 is the reference point; use SM (all WC = 0, SM constant = 1).
# c1 will be updated inside the scan loop.
truth_config = {
    'wcs': plr.wcs,
    'c0':  [1.0] + [0.0] * len(plr.wcs),
    'c1':  [1.0] + [0.0] * len(plr.wcs),   # placeholder, overwritten per point
}


#----------------------------------------------------------------------------------------------------
# Debug
#----------------------------------------------------------------------------------------------------
# 在訓練點驗證: c1 = [1, 0, +1, 0, ..., 0] (ctGRe=1, 這是訓練點之一)
coefs_train = [1.0] + [0.0] * len(plr.wcs)
coefs_train[idx_in_coefs] = 1.0
truth_config['c1'] = coefs_train

p0, p1 = get_probabilities(coefficients, truth_config)
print(f"p0: shape={p0.shape}, min={p0.min():.3e}, max={p0.max():.3e}, mean={p0.mean():.3e}")
print(f"p1: shape={p1.shape}, min={p1.min():.3e}, max={p1.max():.3e}, mean={p1.mean():.3e}")
print(f"p1/p0: min={(p1/p0).min():.3e}, max={(p1/p0).max():.3e}, mean={(p1/p0).mean():.3e}")

# 對照 dedicated 網路 (ctGRe=1 對應 config['networks'][0])
from topsbi.tools.buildLikelihood import likelihood
dlr = likelihood(config['networks'][0], features.shape[1])
r_dlr = dlr(features).detach()
print(f"dedicated NN ratio: min={r_dlr.min():.3e}, max={r_dlr.max():.3e}, mean={r_dlr.mean():.3e}")

# =========================================================================
# Scan
# =========================================================================
scan_values = np.linspace(SCAN_MIN, SCAN_MAX, N_POINTS)

nll_pred = np.empty(N_POINTS)
nll_true = np.empty(N_POINTS)

for i, v in enumerate(scan_values):
    # ---- Build the WC vector for this scan point ---------------------
    coefs_list = [1.0] + [0.0] * len(plr.wcs)
    coefs_list[idx_in_coefs] = float(v)

    # ---- Ensemble (predicted) ratio ---------------------------------
    coefs_tensor = torch.tensor(coefs_list, dtype=torch.float32)
    r_pred = plr(coefs_tensor)
    r_pred = torch.clamp(r_pred, min=1e-12)
    nll_pred[i] = (-2.0 * torch.log(r_pred).sum()).item()

    # ---- Analytic (true) ratio --------------------------------------
    truth_config['c1'] = coefs_list
    p0, p1 = get_probabilities(coefficients, truth_config)

    # p0, p1 may be numpy arrays or torch tensors depending on the impl;
    # normalise to torch for a consistent code path.
    if not torch.is_tensor(p0):
        p0 = torch.as_tensor(p0)
        p1 = torch.as_tensor(p1)

    r_true = torch.clamp(p1 / p0, min=1e-12)
    nll_true[i] = (-2.0 * torch.log(r_true).sum()).item()

    if i % 20 == 0:
        print(f"  v={v:+.4f}   nll_pred={nll_pred[i]:.3e}   nll_true={nll_true[i]:.3e}")


# =========================================================================
# Normalise to Delta(-2 ln L) so both curves start from 0 at their minima
# =========================================================================
delta_pred = nll_pred - nll_pred.min()
delta_true = nll_true - nll_true.min()

argmin_pred = scan_values[np.argmin(nll_pred)]
argmin_true = scan_values[np.argmin(nll_true)]
print(f"\nBest-fit ctGRe (ensemble) = {argmin_pred:.4f}")
print(f"Best-fit ctGRe (truth)    = {argmin_true:.4f}")


# =========================================================================
# Plot
# =========================================================================
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(scan_values, delta_pred, color='C0', lw=2, label='Ensemble prediction')
ax.plot(scan_values, delta_true, color='red',  lw=2, ls='--', label='Truth  ($p_1/p_0$)')

ax.axhline(1.00, ls='--', color='gray', label=r'1$\sigma$ ($\Delta=1$)')
ax.axhline(3.84, ls=':',  color='gray', label=r'95% CL ($\Delta=3.84$)')

ax.set_xlim(SCAN_MIN, SCAN_MAX)
ax.set_ylim(0, Y_MAX)
ax.set_xlabel(r'$c_{tG}^{\mathrm{Re}}$')
ax.set_ylabel(r'$-2\Delta \ln L$')
ax.legend(loc='upper right', fontsize=9)
ax.set_title(f'1D scan of {TARGET_WC}')

fig.tight_layout()
fig.savefig(OUTPUT, dpi=150)
print(f"\nSaved: {OUTPUT}")

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

from topsbi.tools.buildLikelihood import full_likelihood

# --- load data and trained networks ---
with open('./examples/validation/config_ywk.yml') as f:
    config = yaml.safe_load(f)
with open(config['features']) as f:
    config['features'] = yaml.safe_load(f)

features, _ = torch.load(config['data'], weights_only=False)[:]
features = features.float()
plr = full_likelihood(config, features)
features = features[plr.infFilter] # filter inf events

# --- scan points ---
wcs = plr.wcs
target = 'ctGRe'
idx = wcs.index(target) + 1 # the zeroth will be SM

scan_values = np.linspace(2.9, 3.1, 201)

# --- compute -2 ln L ---
nll = []
for v in scan_values:
    coefs = [1.0] + [0.0] * (len(wcs))  # SM point: [1, 0, 0, ...]
    coefs[idx] = v                      # set ctGRe scan value
    coefs = torch.tensor(coefs, dtype=torch.float32)

    r = plr(coefs)                      # ratio, shape (n_events,)
    r = torch.clamp(r, min=1e-12)
    two_nll = -2.0 * torch.log(r).sum().item()
    nll.append(two_nll)

# evaluate -2 Delta NLL
nll = np.array(nll)
delta = nll - nll.min()

fig, ax = plt.subplots()
ax.plot(scan_values, delta)
ax.set_ylim(0, 10)
ax.axhline(1.0, ls='--', color='gray', label='1σ (Δ=1)')
ax.axhline(3.84, ls=':',  color='gray', label='95% CL (Δ=3.84)')
ax.set_xlabel(r'$c_{tG}^{\mathrm{Re}}$')
ax.set_ylabel(r'$-2\Delta \ln L$')
ax.legend()
fig.savefig('scan_ctGRe.png')

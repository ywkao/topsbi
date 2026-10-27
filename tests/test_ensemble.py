import os

import numpy as np
import pytest
import torch
import yaml

import topsbi.tools.buildLikelihood as build
from topsbi.tools.data import get_lower_tri

N_EVENTS = 6
# two WCs: get_lower_tri([1, a, b]) has 6 terms, 3 of which are kept
ZEROS_MASK = torch.tensor([True, True, False, True, False, False])
SEEDS = {
    # seed: (events kept by full_likelihood, gammas for the kept events)
    's1': (torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool), torch.arange(18.0).reshape(3, 6)),
    's2': (torch.tensor([1, 1, 0, 1, 1, 1], dtype=torch.bool), 10 + torch.arange(15.0).reshape(3, 5)),
}


class FakeFullLikelihood:
    """Stands in for full_likelihood with known per-event coefficients."""

    zeros_masks = {}

    def __init__(self, config, features):
        self.infFilter, self.gammas = SEEDS[config['sm']]
        self.zerosMask = self.zeros_masks.get(config['sm'], ZEROS_MASK)
        self.residuals = torch.rand(int(self.infFilter.sum()))


@pytest.fixture
def ensemble_config(monkeypatch):
    monkeypatch.setattr(build, 'full_likelihood', FakeFullLikelihood)
    FakeFullLikelihood.zeros_masks = {}
    return {
        'device': 'cpu',
        'c0': [1.0, 0.0, 0.0],
        'features': {},
        'seeds': {seed: {'sm': seed, 'networks': []} for seed in SEEDS},
    }


def full_size(seed):
    keep, gammas = SEEDS[seed]
    out = torch.zeros(3, N_EVENTS)
    out[:, keep] = gammas
    return out


def test_ensemble_averages_aligned_events(ensemble_config):
    ens = build.ensemble(ensemble_config, torch.zeros(N_EVENTS, 2))

    kept_by_all = SEEDS['s1'][0] & SEEDS['s2'][0]
    expected = ((full_size('s1') + full_size('s2')) / 2)[:, kept_by_all]
    assert torch.equal(ens.infFilter, kept_by_all)
    assert torch.allclose(ens.gammas, expected)
    expected2 = ((full_size('s1') ** 2 + full_size('s2') ** 2) / 2)[:, kept_by_all]
    assert torch.allclose(ens.gammas2, expected2)

    wcs = [1.0, 2.0, 3.0]
    assert torch.allclose(ens(wcs), get_lower_tri(wcs)[ZEROS_MASK] @ expected)


def test_ensemble_passes_each_seed_fit_to_per_seed(ensemble_config):
    residuals = {}
    build.ensemble(
        ensemble_config, torch.zeros(N_EVENTS, 2), per_seed=lambda seed, fit: residuals.update({seed: fit.residuals})
    )
    assert [len(residuals[s]) for s in SEEDS] == [int(SEEDS[s][0].sum()) for s in SEEDS]


def test_ensemble_rejects_seeds_at_different_points(ensemble_config):
    other = ZEROS_MASK.clone()
    other[2] = True
    FakeFullLikelihood.zeros_masks = {'s2': other}
    with pytest.raises(ValueError):
        build.ensemble(ensemble_config, torch.zeros(N_EVENTS, 2))


class FakeCache:
    """Stands in for EnsembleCache with one seed: SM row last, noisy rows before it."""

    def __init__(self, sm_weight):
        gen = torch.Generator().manual_seed(0)
        self.seeds = ['a']
        self.mask = ZEROS_MASK.numpy()
        self.Theta = torch.tensor([[1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 2.0, 4.0], [1.0, 0.0, 0.0]]).double()
        gammas = torch.rand(3, N_EVENTS, generator=gen, dtype=torch.float64) + 1
        noise = torch.randn(4, N_EVENTS, generator=gen, dtype=torch.float64)
        noise[-1] = 0  # the SM network closes
        self.ratios = (self.Theta @ gammas + noise).numpy()
        self.seed_filter = {'a': np.ones(N_EVENTS, dtype=bool)}
        self.sm_weight = sm_weight

    def network(self, seed):
        return {'networks': ['p1', 'p2', 'p3', 'sm'], 'Theta': self.Theta.numpy()}

    def seed_ratios(self, seed):
        return self.ratios

    def row_weights(self, seed):
        w = np.ones(4)
        w[-1] = self.sm_weight
        return w


def test_sm_weight_pins_gamma_sm_to_sm_ratio():
    from topsbi.validation import refit_without

    unweighted = refit_without(FakeCache(1), [])
    weighted = refit_without(FakeCache(1e4), [])
    sm_ratio = FakeCache(1).ratios[-1]
    assert np.abs(unweighted.gammas[0] - sm_ratio).max() > 1e-2
    assert np.allclose(weighted.gammas[0], sm_ratio, atol=1e-6)


def test_likelihood_device_overrides_training_config(train_config, tmp_path):
    """A network trained with device: cuda loads and evaluates on the device given at inference, returning CPU ratios."""
    from conftest import N_FEATURES
    from topsbi.model.net import Net
    from topsbi.tools.buildLikelihood import likelihood

    config = dict(train_config, device='cuda')
    os.makedirs(f'{config["name"]}/complete')
    torch.save(Net(N_FEATURES, 'cpu', config['network']).state_dict(), f'{config["name"]}/model.pt')
    with open(f'{config["name"]}/complete/performance.yml', 'w') as f:
        yaml.safe_dump({}, f)
    path = tmp_path / 'network.yml'
    path.write_text(yaml.safe_dump(config))

    network = likelihood(str(path), N_FEATURES, 'cpu')
    lr = network(torch.rand(10, N_FEATURES))
    assert network.config['device'] == 'cpu'
    assert lr.device.type == 'cpu' and lr.shape == (10,) and (lr > 0).all()

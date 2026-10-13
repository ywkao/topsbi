import torch

from topsbi.tools import buildLikelihood
from topsbi.tools.data import get_lower_tri

N_EVENTS = 50
# SM + 2 WCs -> 6 quadratic terms; 5 networks + the built-in SM row determine them exactly
POINTS = [[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [1.0, 1.0, 1.0]]


def test_full_likelihood_without_sm_reproduces_normalized_ratio(monkeypatch):
    # w(c)/w(SM) = l(c) @ gammas per event; Y(c) ~ l(c) @ yields. Networks learn the normalized
    # ratio w(c)/w(SM) * Y(SM)/Y(c), which is not quadratic in c
    gammas = torch.rand(6, N_EVENTS)
    gammas[0] = 1.0  # ratio is 1 at the SM
    yields = torch.tensor([1.0, 0.3, 0.5, -0.2, 0.1, 0.4], dtype=torch.float64)
    sm = [1.0, 0.0, 0.0]

    def normalized(c):
        y = get_lower_tri(c, dtype=torch.float64) @ yields / (get_lower_tri(sm, dtype=torch.float64) @ yields)
        return get_lower_tri(c) @ gammas / float(y)

    class FakeLikelihood:
        def __init__(self, config, n_features):
            self.config = config

        def __call__(self, features):
            return normalized(self.config['c1'])

    monkeypatch.setattr(buildLikelihood, 'likelihood', FakeLikelihood)
    monkeypatch.setattr(buildLikelihood, 'mean_coefs', lambda *args: yields)
    config = {
        'features': ['x'],
        'networks': [{'c0': sm, 'c1': p, 'data': None, 'bootstrap': False, 'seed': 0} for p in POINTS],
    }
    plr = buildLikelihood.full_likelihood(config, torch.zeros(N_EVENTS, 1))

    c = [1.0, 1.5, -1.5]  # far from the training points
    torch.testing.assert_close(plr(c), normalized(c), rtol=1e-4, atol=1e-4)


def test_ensemble_averages_seeds_and_combines_inf_filters(monkeypatch):
    seeds = {'a': 1.0, 'b': 3.0}
    dropped = {'a': 0, 'b': 7}

    class FakeFull:
        def __init__(self, config, features):
            self.zerosMask = torch.ones(6, dtype=torch.bool)
            self.infFilter = torch.ones(N_EVENTS, dtype=torch.bool)
            self.infFilter[dropped[config['sm']]] = False
            self.gammas = torch.full((6, N_EVENTS - 1), seeds[config['sm']])
            self.network_config = None

    monkeypatch.setattr(buildLikelihood, 'full_likelihood', FakeFull)
    config = {
        'device': 'cpu',
        'c0': [1.0, 0.0, 0.0],
        'cg': [1.0, 0.0, 0.0],
        'features': ['x'],
        'seeds': {s: {'sm': s, 'networks': []} for s in seeds},
    }
    ens = buildLikelihood.ensemble(config, torch.zeros(N_EVENTS, 1))

    assert ens.infFilter.sum() == N_EVENTS - 2
    torch.testing.assert_close(ens.gammas, torch.full((6, N_EVENTS - 2), 2.0))

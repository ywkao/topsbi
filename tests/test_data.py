import torch

from conftest import CG, N_WCS, make_dataset
from topsbi.tools.data import get_probabilities


def test_get_probabilities_accepts_float64_coefs():
    _, coefs, _ = make_dataset(100)[:]
    assert coefs.dtype == torch.float64
    c1 = [1.0] + [0.0] * N_WCS
    c1[4] = 1.0
    p0, p1, pg = get_probabilities(coefs, {'c0': [1.0] + [0.0] * N_WCS, 'c1': c1, 'cg': CG})
    assert p0.dtype == p1.dtype == pg.dtype == torch.float32
    assert torch.isfinite(p1 / p0).all()

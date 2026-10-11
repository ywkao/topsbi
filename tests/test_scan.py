import os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from scan import find_cl_crossings  # noqa: E402


def test_find_cl_crossings_parabola():
    grid = np.linspace(-3, 3, 601)
    nll = (grid - 0.5) ** 2  # crossings of level 1 at 0.5 +- 1
    lo, hi = find_cl_crossings(grid, nll, 1.0)
    assert np.isclose(lo, -0.5, atol=1e-3) and np.isclose(hi, 1.5, atol=1e-3)


def test_find_cl_crossings_open_interval():
    grid = np.linspace(0, 1, 11)
    lo, hi = find_cl_crossings(grid, grid**2 * 2, 1.0)  # minimum at the left edge, so lo is open
    assert lo is None and np.isclose(hi, 2**-0.5, atol=0.02)

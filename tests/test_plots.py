import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from topsbi.tools.plots import lr_calibration_curve  # noqa: E402


def test_perfect_ratio_is_calibrated_only_with_p0_weights():
    """
    r depends on an unobserved z, so the best lr_hat(x) is sum(p1)/sum(p0) over z, not mean(p1/p0).
    Weighting by p0 puts it on the diagonal; unit weights (the old lr_norm_factor) do not.
    """
    rng = np.random.default_rng(0)
    x = rng.integers(0, 20, 200_000)
    z = rng.random(x.size)
    p0 = 0.5 + z
    p1 = p0 * (1 + 0.05 * x * (z - 0.3))
    lrhat = np.zeros(x.size)
    for v in range(20):
        k = x == v
        lrhat[k] = p1[k].sum() / p0[k].sum()

    _, ax = plt.subplots()
    chi_p0 = lr_calibration_curve(ax, p1 / p0, lrhat, p0, 20, threshold=0.0)
    chi_ones = lr_calibration_curve(ax, p1 / p0, lrhat, np.ones(x.size), 20, threshold=0.0)
    plt.close('all')
    assert chi_p0 < 1e-8
    assert chi_ones > 1e-3

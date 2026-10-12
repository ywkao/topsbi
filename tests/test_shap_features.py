import os, sys

import numpy as np
import pytest
import torch

pytest.importorskip('shap')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from shap_features import run_shap  # noqa: E402

from topsbi.model.net import Net  # noqa: E402


def test_shap_is_additive_with_internal_normalization(tmp_path, monkeypatch):
    """SHAP values sum to s(x) - E_bg[s] even though Net standardizes raw features in forward."""
    torch.manual_seed(0)
    network = [
        {'type': 'Linear', 'out': 8, 'activation': 'LeakyReLU'},
        {'type': 'Linear', 'in': 8, 'out': 1, 'activation': 'Sigmoid', 'dropout': 0.2},
    ]
    net = Net(3, 'cpu', network)
    features = torch.randn(300, 3) * torch.tensor([1.0, 50.0, 1000.0]) + 500
    net.set_feature_normalization(features.mean(0), features.std(0))
    net.eval()

    captured = {}
    import shap

    original = shap.DeepExplainer.shap_values

    def spy(self, x, **kwargs):
        captured['x'], captured['expected'] = x, self.expected_value
        captured['values'] = original(self, x, **kwargs)
        return captured['values']

    monkeypatch.setattr(shap.DeepExplainer, 'shap_values', spy)
    ranking = run_shap(net, features, ['a', 'b', 'c'], str(tmp_path), n_background=100, n_samples=50, seed=0)

    values = np.asarray(captured['values']).reshape(50, 3)
    with torch.no_grad():
        s = net(captured['x']).numpy().flatten()
    assert np.allclose(values.sum(1), s - np.asarray(captured['expected']).item(), atol=1e-4)
    assert sorted(name for name, _ in ranking) == ['a', 'b', 'c']
    assert (tmp_path / 'shap_ranking.txt').exists()

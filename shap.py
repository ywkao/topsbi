import argparse, yaml, torch
import numpy as np
import matplotlib.pyplot as plt
import shap

from topsbi.model.net import Model
from topsbi.tools.data import prepare_features, get_probabilities, parameterize_weights

from .schema import FEATURE_NAMES

def get_feature_indices(config):
    if config.get('features', 'all') == 'all':
        return list(range(len(FEATURE_NAMES)))
    selected = config['features']
    return [FEATURE_NAMES.index(f) for f in selected]

def load_model_and_data(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    feat_idx = get_feature_indices(config)
    used_names = [FEATURE_NAMES[i] for i in feat_idx]

    test_feats, test_coefs = torch.load(f'{config["data"]}/test.p', weights_only=False)[:]
    test_feats, test_coefs = test_feats.float(), test_coefs.float()
    test_feats = test_feats[:, feat_idx]
    test_feats = prepare_features(test_feats)

    model = Model(nFeatures=test_feats.shape[1], method=config['method'],
                  device=config['device'], config=config['network'], seed=config['seed'])
    model.net.load_state_dict(torch.load(f'{config["name"]}/model.pt', map_location=config['device']))
    model.net.eval()

    return model, test_feats, used_names, config

def run_shap(model, test_feats, feature_names, outdir, n_background=200, n_samples=500):
    net = model.net
    device = next(net.parameters()).device
    test_feats = test_feats.to(device)

    bg_idx     = torch.randperm(test_feats.shape[0])[:n_background]
    sample_idx = torch.randperm(test_feats.shape[0])[:n_samples]

    background = test_feats[bg_idx]
    samples    = test_feats[sample_idx]

    explainer   = shap.DeepExplainer(net, background)
    shap_values = explainer.shap_values(samples)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap.summary_plot(shap_values, samples.cpu().numpy(),
                       feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/shap_summary.png", dpi=150)
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking  = sorted(zip(feature_names, mean_abs), key=lambda x: -x[1])

    with open(f"{outdir}/shap_ranking.txt", "w") as f:
        for name, val in ranking:
            f.write(f"{name:20s} {val:.6e}\n")

    return ranking

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='trained config yml (must have model.pt in cfg["name"])')
    parser.add_argument('--n_background', type=int, default=200)
    parser.add_argument('--n_samples', type=int, default=500)
    args = parser.parse_args()

    model, test_feats, feature_names, config = load_model_and_data(args.config)
    ranking = run_shap(model, test_feats, feature_names, config['name'],
                        n_background=args.n_background, n_samples=args.n_samples)

    print("[INFO] Top 10 features by mean |SHAP|:")
    for name, val in ranking[:10]:
        print(f"  {name:20s} {val:.6e}")

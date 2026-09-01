import argparse, os, sys, yaml, torch
import numpy as np
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topsbi.model.net import Model
from topsbi.schema import FEATURE_NAMES

def get_feature_indices(config):
    if config.get('features', 'all') == 'all':
        return list(range(len(FEATURE_NAMES)))
    selected = config['features']
    return [FEATURE_NAMES.index(f) for f in selected]

def load_model_and_data(config_path, model_path=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    feat_idx = get_feature_indices(config)
    used_names = [FEATURE_NAMES[i] for i in feat_idx]

    # match train.py: standardize with TRAIN mean/std, applied to test
    train_feats = torch.load(f'{config["data"]}/train.p', weights_only=False)[:][0].float()[:, feat_idx]
    test_feats  = torch.load(f'{config["data"]}/test.p',  weights_only=False)[:][0].float()[:, feat_idx]
    test_feats  = (test_feats - train_feats.mean(0)) / train_feats.std(0)

    model = Model(nFeatures=test_feats.shape[1], method=config['method'],
                  device=config['device'], config=config['network'], seed=config['seed'])
    model.net.load_state_dict(torch.load(model_path or f'{config["name"]}/model.pt', map_location=config['device']))
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

    mean_abs = np.abs(shap_values).mean(axis=0).squeeze()
    ranking  = sorted(zip(feature_names, mean_abs), key=lambda x: -x[1])

    with open(f"{outdir}/shap_ranking.txt", "w") as f:
        for name, val in ranking:
            f.write(f"{name:20s} {val:.6e}\n")

    return ranking

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='trained config yml (must have model.pt in cfg["name"] unless --model given)')
    parser.add_argument('--model',  default=None, help='override model.pt path')
    parser.add_argument('--outdir', default=None, help='override output dir (default: cfg["name"])')
    parser.add_argument('--n_background', type=int, default=200)
    parser.add_argument('--n_samples', type=int, default=500)
    args = parser.parse_args()

    model, test_feats, feature_names, config = load_model_and_data(args.config, model_path=args.model)
    outdir = args.outdir or config['name']
    ranking = run_shap(model, test_feats, feature_names, outdir,
                        n_background=args.n_background, n_samples=args.n_samples)

    print("[INFO] Top 10 features by mean |SHAP|:")
    for name, val in ranking[:10]:
        print(f"  {name:20s} {val:.6e}")

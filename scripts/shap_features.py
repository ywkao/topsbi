from topsbi.model.net import Net

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

import argparse, os, yaml


def feature_names(features_yml):
    """Names ordered by their `loc`, i.e. by data column."""
    with open(features_yml) as f:
        features = yaml.safe_load(f)
    return sorted(features, key=lambda name: features[name]['loc'])


def load_net(config, n_features):
    net = Net(n_features, 'cpu', config['network'])
    net.load_state_dict(torch.load(f'{config["name"]}/model.pt', map_location='cpu'))
    return net.eval()


def run_shap(net, features, names, outdir, n_background, n_samples, seed):
    """
    Mean |SHAP| of the network output s per feature, on raw (unnormalized) features:
    Net.forward standardizes them with the training-set statistics stored in the model.
    """
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(features.shape[0], generator=generator)[: n_background + n_samples]
    background, samples = features[idx[:n_background]], features[idx[n_background:]]

    values = shap.DeepExplainer(net, background).shap_values(samples, check_additivity=False)
    values = np.asarray(values).reshape(n_samples, len(names))

    os.makedirs(outdir, mode=0o755, exist_ok=True)
    shap.summary_plot(values, samples.numpy(), feature_names=names, show=False)
    plt.tight_layout()
    plt.savefig(f'{outdir}/shap_summary.png', dpi=150)
    plt.close()

    ranking = sorted(zip(names, np.abs(values).mean(0)), key=lambda x: -x[1])
    with open(f'{outdir}/shap_ranking.txt', 'w') as f:
        for name, value in ranking:
            f.write(f'{name:20s} {value:.6e}\n')
    return ranking


def main(config_path, outdir, split, n_background, n_samples, seed):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    names = feature_names(config['features'])
    # validation.p is used by neither training nor the bootstrap split, so it is independent of every seed
    features = torch.load(f'{config["data"]}/{split}.p', weights_only=False)[:][0].float()
    net = load_net(config, features.shape[1])

    ranking = run_shap(net, features, names, outdir or f'{config["name"]}/complete', n_background, n_samples, seed)
    print('[INFO] features by mean |SHAP|:')
    for name, value in ranking:
        print(f'  {name:20s} {value:.6e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='training config yml; model.pt is read from its `name`')
    parser.add_argument('--outdir', '-o', default=None, help='default: <name>/complete')
    parser.add_argument('--split', default='validation', choices=['train', 'test', 'validation'])
    parser.add_argument('--n-background', type=int, default=200)
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    main(args.config, args.outdir, args.split, args.n_background, args.n_samples, args.seed)

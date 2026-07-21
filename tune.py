"""Optuna hyper-parameter sweep for topsbi train.py.

Usage:
    python tune.py <base_config.yml> [--trials N] [--study NAME] [--storage sqlite:///optuna.db]

The base yml sets everything you want held fixed (feature set, scheduler, epochs,
data path, etc.). Optuna overrides the tunable knobs per trial.
"""
import argparse, copy, os, sys, yaml, optuna

_here   = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
sys.path.insert(0, _parent)
sys.path.insert(0, _here)
from topsbi.train import main


def build_network(n_layers, n_nodes, dropout):
    """n_layers = number of hidden Linear(width=n_nodes) layers before the output layer.
    Matches the reference yml style: first Linear has no `in`, hidden layers carry dropout,
    output is Linear(n_nodes, 1) + Sigmoid."""
    layers = [{'type': 'Linear', 'out': n_nodes, 'activation': 'LeakyReLU'}]
    for _ in range(n_layers - 1):
        layers.append({'type': 'Linear', 'in': n_nodes, 'out': n_nodes,
                       'activation': 'LeakyReLU', 'dropout': dropout})
    layers.append({'type': 'Linear', 'in': n_nodes, 'out': 1,
                   'activation': 'Sigmoid', 'dropout': dropout})
    return layers


def objective(trial, base_config):
    cfg = copy.deepcopy(base_config)

    # architecture (asked for)
    n_layers = trial.suggest_categorical('n_layers', [3, 4, 5])
    n_nodes  = trial.suggest_categorical('n_nodes',  [64, 128, 256])
    # extras worth sweeping — comment out any you want to freeze
    dropout             = trial.suggest_float('dropout', 0.0, 0.4)
    cfg['optimizer']    = trial.suggest_categorical('optimizer', ['Adam', 'NAdam'])
    cfg['learningRate'] = trial.suggest_float('learningRate', 1e-4, 1e-2, log=True)
    cfg['batchSize']    = trial.suggest_categorical('batchSize', [512, 1000, 2048])
    cfg['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    cfg['network']   = build_network(n_layers, n_nodes, dropout)
    cfg['name']      = os.path.join(base_config['name'], 'optuna', f'trial_{trial.number:04d}')
    cfg['wandb']     = False        # per-trial wandb runs are noise; flip to True if you disagree
    cfg['skipPlots'] = True         # kill per-epoch PNGs / animations for tuning speed

    _, best_test_loss = main(cfg)
    return best_test_loss


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('config',           help='base yml config (fixed knobs live here)')
    ap.add_argument('--trials',  type=int, default=30)
    ap.add_argument('--study',   default='topsbi')
    ap.add_argument('--storage', default=None,
                    help='e.g. sqlite:///optuna.db — enables resume + parallel workers')
    args = ap.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)
    base_config.setdefault('device', 'cuda')

    study = optuna.create_study(
        direction='minimize',
        study_name=args.study,
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=base_config.get('seed', 42)),
    )
    study.optimize(lambda t: objective(t, base_config), n_trials=args.trials)

    os.makedirs(os.path.join(base_config['name'], 'optuna'), exist_ok=True)
    with open(os.path.join(base_config['name'], 'optuna', 'best.yml'), 'w') as f:
        yaml.safe_dump({'params': study.best_trial.params,
                        'value':  study.best_trial.value,
                        'number': study.best_trial.number}, f)
    print(f'\nbest trial #{study.best_trial.number}')
    print(f'  params: {study.best_trial.params}')
    print(f'  loss  : {study.best_trial.value:.6f}')

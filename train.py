from topsbi.model.net import Model
from topsbi.tools.plots import networkPlots, kinematic_histogram, animate_plots
from topsbi.tools.data import parameterize_weights, prepare_features, get_probabilities

import argparse, glob, math, os, tqdm, torch, wandb, yaml

from .schema import FEATURE_NAMES

def check_loss(name, value, epoch):
    """Warn if a loss value is non-finite or non-positive (signals training instability)."""
    if not math.isfinite(value) or value <= 0:
        print(f"[WARNING] {name} at epoch {epoch} is non-finite or non-positive: {value}")

def get_feature_indices(config):
    """
    回傳要使用的 feature indices。
    config['features'] == 'all'  -> 全部
    config['features'] == [...]  -> 指定名稱的 subset
    """
    if config.get('features', 'all') == 'all':
        print(f"[INFO] using all {len(FEATURE_NAMES)} features")
        return list(range(len(FEATURE_NAMES)))

    selected = config['features']
    unknown  = [f for f in selected if f not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"Unknown feature name(s) in config: {unknown}")

    indices = [FEATURE_NAMES.index(f) for f in selected]
    print(f"[INFO] using {len(indices)} / {len(FEATURE_NAMES)} features: {selected}")
    return indices

def main(config):
    with open(f'{config["data"]}/features.yml', 'r') as f:
        features_config = yaml.safe_load(f)
    if config['device'] != 'cpu' and not torch.cuda.is_available():
        print("Warning, you tried to use cuda, but its not available. Will use the CPU")
        config['device'] = 'cpu'
    torch.manual_seed(config['seed'])

    print("[INFO] loading training samples...")
    test_feats,  test_coefs  = torch.load(f'{config["data"]}/test.p',  weights_only=False)[:]
    train_feats, train_coefs = torch.load(f'{config["data"]}/train.p', weights_only=False)[:]
    test_feats,  test_coefs  = test_feats.float(),  test_coefs.float()
    train_feats, train_coefs = train_feats.float(), train_coefs.float()

    # ── feature selection ──────────────────────────────────────────────
    feat_idx    = get_feature_indices(config)
    test_feats  = test_feats[:,  feat_idx]
    train_feats = train_feats[:, feat_idx]

    if config.get('features', 'all') != 'all':
        selected_set = set(config['features'])
        loc_remap = {orig: new for new, orig in enumerate(feat_idx)}
        features_config = {
            name: {**params, 'loc': loc_remap[params['loc']]}
            for name, params in features_config.items()
            if name in selected_set
        }
    # ─────────────────────────────────────────────────────────────

    if 'method' not in config.keys():
        config['method'] = 'stitched'

    if config['method'] == 'parameterized':
        test_p0,  test_p1,  test_wcs  = parameterize_weights(test_coefs, config)
        train_p0, train_p1, train_wcs = parameterize_weights(train_coefs, config)
        test_feats  = torch.concatenate([test_feats,  test_wcs],  dim=1)
        train_feats = torch.concatenate([train_feats, train_wcs], dim=1)
    elif config['method'] == 'stitched':
        test_p0,  test_p1  = get_probabilities(test_coefs, config)
        train_p0, train_p1 = get_probabilities(train_coefs, config)
    elif config['method'] == 'alice':
        test_p0,  test_p1  = get_probabilities(test_coefs, config)
        train_p0, train_p1 = get_probabilities(train_coefs, config)

    print(test_feats.shape)
    print(test_coefs.shape)
    print(test_feats[:5])

    test_coefs  = None
    train_coefs = None

    print("[INFO] preparing training features...")
    train_means = train_feats.mean(0)
    train_stds  = train_feats.std(0)
    train_feats = (train_feats - train_means) / train_stds
    norm_test   = (test_feats - train_means) / train_stds

    batches   = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_feats, train_p0, train_p1), 
                                            batch_size=config['batchSize'], shuffle=True, num_workers=0)
    model     = Model(nFeatures=train_feats.shape[1], method=config['method'], device=config['device'], config=config['network'], seed=config['seed'])
    norm_test = norm_test.to(model.device)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=config['learningRate'])

    scheduler_type = config.get('scheduler', 'plateau')
    if scheduler_type == 'plateau':
        # ReduceLROnPlateau: steps LR down when val BCE stops improving.
        # Directly optimises the calibration signal that determines ratio quality.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=config.get('factor', 0.5),
            patience=config.get('lr_patience', 5),
        )
        print(f"[INFO] scheduler: ReduceLROnPlateau  factor={config.get('factor', 0.5)}  lr_patience={config.get('lr_patience', 5)}")
    elif scheduler_type == 'cosine':
        # Linear warmup → CosineAnnealingLR: smooth, deterministic, reproducible
        # across EFT scan points when you want identical training conditions.
        warmup = config.get('warmup_epochs', 5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config['epochs'] - warmup), eta_min=1e-6),
            ],
            milestones=[warmup],
        )
        print(f"[INFO] scheduler: cosine+warmup  warmup_epochs={warmup}  T_max={max(1, config['epochs'] - warmup)}")
    else:
        scheduler = None
        print("[INFO] scheduler: none")

    model.net.eval()
    with torch.no_grad():
        trainLoss = [model.loss(batches.dataset[:][0], batches.dataset[:][1], batches.dataset[:][2]).item()]
        testLoss  = [model.loss(norm_test, test_p0, test_p1).item()]
    model.net.train()
    lrHistory = [optimizer.param_groups[0]['lr']]

    use_wandb = config.get('wandb', True)
    if use_wandb:
        wandb.init(
            project=config.get('wandb_project', 'topsbi'),
            name=os.path.basename(config['name'].rstrip('/')),
            config=config,
        )
        wandb.log({'train_loss': trainLoss[0], 'test_loss': testLoss[0], 'lr': lrHistory[0]}, step=0)

    os.makedirs(f'{config["name"]}/complete/animations', exist_ok=True)
    os.makedirs(f'{config["name"]}/complete/kinematics', exist_ok=True)
    for feature in features_config.keys():
        os.makedirs(f'{config["name"]}/incomplete/kinematics/{feature}', exist_ok=True)

    # early stopping parameters
    patience      = config.get('patience', 10)
    best_test_loss = float('inf')
    best_epoch     = 0
    patience_count = 0
    best_state     = None

    print("[INFO] starting networkPlots for every 50 epochs...")
    for epoch in tqdm.tqdm(range(config['epochs'])):
        s  = model.net(norm_test).cpu().detach().numpy().flatten()
        noOnes = s != 1
        s = s[noOnes]
        lr = s / (1 - s)
        tlr = (test_p1/test_p0).detach().cpu().numpy().flatten()
        for feature, params in features_config.items():
            if epoch == 0:
                ylim = kinematic_histogram(test_feats[noOnes, params['loc']].cpu().numpy(), params, epoch, lr, tlr[noOnes], 
                                           f'{config["name"]}/incomplete/kinematics/{feature}/{epoch:04d}.png')
            else: 
                kinematic_histogram(test_feats[noOnes, params['loc']].cpu().numpy(), params, epoch, lr, tlr[noOnes], 
                                    f'{config["name"]}/incomplete/kinematics/{feature}/{epoch:04d}.png', ylim=ylim)
        model.net.eval()
        with torch.no_grad():
            trainLoss.append(model.loss(batches.dataset[:][0], batches.dataset[:][1], batches.dataset[:][2]).item())
        model.net.train()
        check_loss('train_loss', trainLoss[-1], epoch)
        if epoch%50 == 0:
            networkPlots(norm_test, test_p0, test_p1, model.net, trainLoss,
                         testLoss, f'{config["name"]}/incomplete/epoch_{epoch:04d}')
        for train_feats, train_p0, train_p1 in batches:
            optimizer.zero_grad()
            loss = model.loss(train_feats, train_p0, train_p1)
            loss.backward()
            optimizer.step()

        model.net.eval()
        with torch.no_grad():
            current_test_loss = model.loss(norm_test, test_p0, test_p1).item()
        model.net.train()
        testLoss.append(current_test_loss)
        check_loss('test_loss', current_test_loss, epoch)

        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(current_test_loss)
            else:
                scheduler.step()
        lrHistory.append(optimizer.param_groups[0]['lr'])

        # ── early stopping ──
        if current_test_loss < best_test_loss:
            best_test_loss = current_test_loss
            best_epoch     = epoch
            best_state     = {k: v.clone() for k, v in model.net.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if use_wandb:
            wandb.log({
                'train_loss': trainLoss[-1],
                'test_loss': current_test_loss,
                'best_test_loss': best_test_loss,
                'lr': lrHistory[-1],
            }, step=epoch + 1)

        if patience_count >= patience:
            print(f"[INFO] early stopping at epoch {epoch}, best epoch was {best_epoch} (test loss {best_test_loss:.4f})")
            break

    if best_state is not None:
        model.net.load_state_dict(best_state)
        print(f"[INFO] restored best checkpoint from epoch {best_epoch}")

    if use_wandb:
        wandb.summary['best_epoch']     = best_epoch
        wandb.summary['best_test_loss'] = best_test_loss

    networkPlots(norm_test, test_p0, test_p1, model.net, trainLoss, testLoss, f'{config["name"]}/complete', lr_history=lrHistory)

    if use_wandb:
        for plotName in ['loss.png', 'lossLog.png', 'roc.png', 'netOut.png']:
            path = f'{config["name"]}/complete/{plotName}'
            if os.path.exists(path):
                wandb.log({plotName: wandb.Image(path)})

    # keep the best model for validation
    torch.save(model.net.state_dict(), f'{config["name"]}/model.pt')

    s  = model.net(norm_test).cpu().detach().numpy().flatten()
    noOnes = s != 1
    s = s[noOnes]
    lr = s / (1 - s)
    tlr = (test_p1/test_p0).detach().cpu().numpy().flatten()
    for feature, params in features_config.items():
        kinematic_histogram(test_feats[noOnes, params['loc']].cpu().numpy(), params, epoch, lr, tlr[noOnes], 
                            f'{config["name"]}/incomplete/kinematics/{feature}/{epoch:04d}.png', ylim=ylim)
        kinematic_histogram(test_feats[noOnes, params['loc']].cpu().numpy(), params, epoch, lr, tlr[noOnes], 
                            f'{config["name"]}/complete/kinematics/{feature}.png', ylim=ylim, epoch_title=False)
        plots = sorted(glob.glob(f'{config["name"]}/incomplete/kinematics/{feature}/*.png'))
        animate_plots(plots, f'{config["name"]}/complete/animations/{feature}.gif')

    if use_wandb:
        wandb.finish()

    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='configuration yml file used for training')
    with open(parser.parse_args().config, 'r') as f:
        config = yaml.safe_load(f)
    config = main(config)

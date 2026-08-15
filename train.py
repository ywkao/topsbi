from topsbi.model.net import Model
from topsbi.tools.plots import networkPlots, kinematic_histogram, animate_plots
from topsbi.tools.data import parameterize_weights, prepare_features, get_probabilities

import argparse, glob, math, os, tqdm, torch, wandb, yaml

from .schema import FEATURE_NAMES

def check_loss(name, value, epoch):
    """Warn if a loss value is non-finite or non-positive (signals training instability)."""
    if not math.isfinite(value) or value <= 0:
        print(f"[WARNING] {name} at epoch {epoch} is non-finite or non-positive: {value}")

def sanitize_events(feats, p0, p1, config, split_name):
    """
    Reject unphysical / pathological events that come from morphing-fit artifacts.

    Three filters (all controllable via config['sanitize']):
      1. reject_negative (default True): drop events with p0 <= 0 or p1 <= 0.
         Negative weights are morphing numerical artifacts, not physical
         probabilities, and give ill-defined likelihood ratios.
      2. lr_cap (default 1000): drop events whose likelihood ratio p1/p0
         falls outside [1/lr_cap, lr_cap]. Extreme LR values are dominated
         by morphing artifacts (a handful of events with pr -> 0 in the
         reference hypothesis) rather than physics, and their contribution
         to BCE loss can be O(20x) larger than a typical event.
         Set to None or <= 0 to disable.
      3. weight_cap (default 100): drop events where max(p0, p1) > weight_cap.
         Since p0 and p1 are normalized to mean=1, a single event with
         p0=1000 contributes as much to loss/gradient as 1000 typical events.
         Even when its LR looks reasonable, such an event will dominate
         training statistics and create train/test loss asymmetry (train
         is more likely to sample these rare high-weight events).
         Set to None or <= 0 to disable.

    The mask is applied consistently to feats, p0, p1 so downstream
    normalization / DataLoader construction stays coherent.

    Args:
        feats: [N, F] tensor of per-event features
        p0:    [N]    tensor of event weights under hypothesis c0
        p1:    [N]    tensor of event weights under hypothesis c1
        config: main config dict; reads config['sanitize']
        split_name: 'train' or 'test', used only for log messages
    Returns:
        (feats, p0, p1, mask) — filtered tensors and the boolean mask used
    """
    sanitize_cfg    = config.get('sanitize', {}) or {}
    reject_negative = sanitize_cfg.get('reject_negative', True)
    lr_cap          = sanitize_cfg.get('lr_cap', 10000)
    weight_cap      = sanitize_cfg.get('weight_cap', 100)

    n_before = p0.shape[0]
    mask     = torch.ones(n_before, dtype=torch.bool, device=p0.device)

    # --- filter 1: non-positive weights ---------------------------------
    if reject_negative:
        physical = (p0 > 0) & (p1 > 0)
        n_bad    = int((~physical).sum().item())
        if n_bad > 0:
            n_p0_bad = int((p0 <= 0).sum().item())
            n_p1_bad = int((p1 <= 0).sum().item())
            print(f"[SANITIZE-{split_name}] rejecting {n_bad}/{n_before} events "
                  f"with non-positive weights (p0<=0: {n_p0_bad}, p1<=0: {n_p1_bad})")
        mask &= physical

    # --- filter 2: extreme likelihood ratio -----------------------------
    if lr_cap is not None and lr_cap > 0:
        # only compute LR on events already passing filter 1 to avoid div-by-zero
        safe_p0 = torch.where(p0 > 0, p0, torch.ones_like(p0))
        lr      = p1 / safe_p0
        sane_lr = (lr > 1.0 / lr_cap) & (lr < lr_cap)
        n_before_lr = int(mask.sum().item())
        combined    = mask & sane_lr
        n_lr_reject = n_before_lr - int(combined.sum().item())
        if n_lr_reject > 0:
            print(f"[SANITIZE-{split_name}] rejecting {n_lr_reject} additional events "
                  f"with LR = p1/p0 outside [{1.0/lr_cap:.2e}, {lr_cap:.2e}]")
        mask = combined

    # --- filter 3: absolute weight magnitude ----------------------------
    if weight_cap is not None and weight_cap > 0:
        sane_w      = (p0 < weight_cap) & (p1 < weight_cap)
        n_before_w  = int(mask.sum().item())
        combined    = mask & sane_w
        n_w_reject  = n_before_w - int(combined.sum().item())
        if n_w_reject > 0:
            print(f"[SANITIZE-{split_name}] rejecting {n_w_reject} additional events "
                  f"with max(p0, p1) > {weight_cap} "
                  f"(single-event weight dominates loss when p0, p1 are mean-normalized)")
        mask = combined

    n_after = int(mask.sum().item())
    frac    = 100.0 * n_after / max(n_before, 1)
    print(f"[SANITIZE-{split_name}] kept {n_after}/{n_before} events ({frac:.3f}%)")

    return feats[mask], p0[mask], p1[mask], mask

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

    # ── Fix 1: reject morphing-artifact events ─────────────────────────
    # Rare events with negative weights or extreme LR (from pr -> 0 in the
    # reference hypothesis) can dominate BCE loss and destabilize training.
    # This filters them out consistently across feats / p0 / p1.
    # Config knobs: sanitize.reject_negative (bool), sanitize.lr_cap (float).
    print("[INFO] sanitizing events (Fix 1: reject unphysical / extreme-LR events)...")
    train_feats, train_p0, train_p1, _ = sanitize_events(
        train_feats, train_p0, train_p1, config, 'train')
    test_feats,  test_p0,  test_p1,  _ = sanitize_events(
        test_feats,  test_p0,  test_p1,  config, 'test')
    # ───────────────────────────────────────────────────────────────────

    print("[INFO] preparing training features...")
    train_means = train_feats.mean(0)
    train_stds  = train_feats.std(0)
    train_feats = (train_feats - train_means) / train_stds
    norm_test   = (test_feats - train_means) / train_stds

    batches   = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_feats, train_p0, train_p1), 
                                            batch_size=config['batchSize'], shuffle=True, num_workers=0)
    model     = Model(nFeatures=train_feats.shape[1], method=config['method'], device=config['device'], config=config['network'], seed=config['seed'])
    norm_test = norm_test.to(model.device)
    opt_cls   = getattr(torch.optim, config.get('optimizer', 'Adam'))
    optimizer = opt_cls(model.net.parameters(),
                        lr=config['learningRate'],
                        weight_decay=config.get('weight_decay', 0.0))

    #----------------------------------------------------------------------------------------------------
    # DEBUG
    #----------------------------------------------------------------------------------------------------
    import numpy as np
    train_p0_np = batches.dataset[:][1].cpu().numpy()
    train_p1_np = batches.dataset[:][2].cpu().numpy()
    test_p0_np  = test_p0.cpu().numpy()
    test_p1_np  = test_p1.cpu().numpy()
    
    for name, p0, p1 in [('train', train_p0_np, train_p1_np),
                         ('test',  test_p0_np,  test_p1_np)]:
        lr = p1 / (p0 + 1e-10)
        print(f"\n=== {name} (N={len(p0)}) ===")
        print(f"p0:  min={p0.min():.3e}  max={p0.max():.3e}  "
              f"p99={np.percentile(p0,99):.3e}  p99.99={np.percentile(p0,99.99):.3e}")
        print(f"p1:  min={p1.min():.3e}  max={p1.max():.3e}  "
              f"p99={np.percentile(p1,99):.3e}  p99.99={np.percentile(p1,99.99):.3e}")
        print(f"lr:  min={lr.min():.3e}  max={lr.max():.3e}  "
              f"p99={np.percentile(lr,99):.3e}  p99.99={np.percentile(lr,99.99):.3e}")
        # 看有多少 event 的 weight 是「clamp-affected」的
        # 也就是 clamp 前 pr 應該很小的那批
        heavy = (p0 > np.percentile(p0, 99.9)) | (p1 > np.percentile(p1, 99.9))
        print(f"top 0.1% weight events: {heavy.sum()}  "
              f"貢獻 sum(p0)={p0[heavy].sum()/p0.sum()*100:.1f}%, "
              f"sum(p1)={p1[heavy].sum()/p1.sum()*100:.1f}%")
    #----------------------------------------------------------------------------------------------------

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

    use_wandb  = config.get('wandb', True)
    skip_plots = config.get('skipPlots', False)  # ponytail: kill per-epoch PNGs for HP tuning
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
        if not skip_plots:
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
        for train_feats, train_p0, train_p1 in batches:
            optimizer.zero_grad()
            loss = model.loss(train_feats, train_p0, train_p1)
            loss.backward()
            optimizer.step()

        model.net.eval()
        with torch.no_grad():
            trainLoss.append(model.loss(batches.dataset[:][0], batches.dataset[:][1], batches.dataset[:][2]).item())
            current_test_loss = model.loss(norm_test, test_p0, test_p1).item()
        model.net.train()
        testLoss.append(current_test_loss)
        check_loss('train_loss', trainLoss[-1], epoch)
        check_loss('test_loss', current_test_loss, epoch)
        if not skip_plots and epoch % 50 == 0:
            networkPlots(norm_test, test_p0, test_p1, model.net, trainLoss,
                         testLoss, f'{config["name"]}/incomplete/epoch_{epoch:04d}')

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

    if not skip_plots:
        networkPlots(norm_test, test_p0, test_p1, model.net, trainLoss, testLoss, f'{config["name"]}/complete', lr_history=lrHistory)

    if use_wandb:
        for plotName in ['loss.png', 'lossLog.png', 'roc.png', 'netOut.png']:
            path = f'{config["name"]}/complete/{plotName}'
            if os.path.exists(path):
                wandb.log({plotName: wandb.Image(path)})

    # keep the best model for validation
    torch.save(model.net.state_dict(), f'{config["name"]}/model.pt')

    if not skip_plots:
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

    return config, best_test_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='configuration yml file used for training')
    with open(parser.parse_args().config, 'r') as f:
        config = yaml.safe_load(f)
    main(config)

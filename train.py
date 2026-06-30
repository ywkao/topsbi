from topsbi.model.net import Model
from topsbi.tools.plots import networkPlots
from topsbi.tools.data import parameterize_weights, prepare_features, get_probabilities

import argparse, tqdm, torch, yaml

from .schema import FEATURE_NAMES

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
        test_p0 /= 2; test_p1 /= 2; train_p0 /= 2; train_p1 /= 2

    print(test_feats.shape)
    print(test_coefs.shape)
    print(test_feats[:5])

    test_coefs  = None
    train_coefs = None

    print("[INFO] preparing training features...")
    test_feats  = prepare_features(test_feats)
    train_feats = prepare_features(train_feats)

    batches   = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_feats, train_p0, train_p1),
                                            batch_size=config['batchSize'], shuffle=True, num_workers=0)
    model     = Model(nFeatures=test_feats.shape[1], method=config['method'], device=config['device'], config=config['network'], seed=config['seed'])
    optimizer = torch.optim.Adam(model.net.parameters(), lr=config['learningRate'])
    trainLoss = [model.loss(batches.dataset[:][0], batches.dataset[:][1], batches.dataset[:][2]).item()]
    testLoss  = [model.loss(test_feats, test_p0, test_p1).item()]

    # early stopping parameters
    patience      = config.get('patience', 10)
    best_test_loss = float('inf')
    best_epoch     = 0
    patience_count = 0
    best_state     = None

    print("[INFO] starting networkPlots for every 50 epochs...")
    for epoch in tqdm.tqdm(range(config['epochs'])):
        if epoch % 50 == 0:
            networkPlots(test_feats, test_p0, test_p1, model.net, trainLoss,
                         testLoss, f'{config["name"]}/incomplete/epoch_{epoch:04d}')
        for train_feats, train_p0, train_p1 in batches:
            optimizer.zero_grad()
            loss = model.loss(train_feats, train_p0, train_p1)
            loss.backward()
            optimizer.step()

            ### # ── debug: print gradient norm ──
            ### total_norm = sum(p.grad.norm().item()**2 for p in model.net.parameters() if p.grad is not None) ** 0.5
            ### print(f"grad norm: {total_norm:.6e}")
            ### print(train_p0[:10])
            ### print(train_p1[:10])
            ### print((trainLoss[0] - trainLoss[-1]) / trainLoss[0])

        trainLoss.append(model.loss(batches.dataset[:][0], batches.dataset[:][1], batches.dataset[:][2]).item())
        current_test_loss = model.loss(test_feats, test_p0, test_p1).item()
        testLoss.append(current_test_loss)

        # ── early stopping ──
        if current_test_loss < best_test_loss:
            best_test_loss = current_test_loss
            best_epoch     = epoch
            best_state     = {k: v.clone() for k, v in model.net.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"[INFO] early stopping at epoch {epoch}, best epoch was {best_epoch} (test loss {best_test_loss:.4f})")
                break

    if best_state is not None:
        model.net.load_state_dict(best_state)
        print(f"[INFO] restored best checkpoint from epoch {best_epoch}")

    networkPlots(test_feats, test_p0, test_p1, model.net, trainLoss, testLoss, f'{config["name"]}/complete')

    # keep the best model for validation
    torch.save(model.net.state_dict(), f'{config["name"]}/model.pt')

    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='configuration yml file used for training')
    with open(parser.parse_args().config, 'r') as f:
        config = yaml.safe_load(f)
    config = main(config)

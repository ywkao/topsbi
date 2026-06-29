from topsbi.model.net import Model
from topsbi.tools.plots import networkPlots
from topsbi.tools.data import parameterize_weights, prepare_features, get_probabilities

import argparse, tqdm, torch, yaml

FEATURE_NAMES = [
    "lep_pt", "lep_eta", "lep_phi", "lep_mass",
    "met_pt", "met_phi",
    "jet0_pt", "jet0_eta", "jet0_phi", "jet0_mass",
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_mass",
    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_mass",
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_mass",
    "njets", "HT", "mT_W",
    "jet0_flav", "jet1_flav", "jet2_flav", "jet3_flav",
    "lep_top_pt", "lep_top_eta", "lep_top_phi", "lep_top_mass",
    "had_top_pt", "had_top_eta", "had_top_phi", "had_top_mass",
    "dr_tt", "dr_lep_had", "m_ttbar", "cos_theta_star",
    "dy_tt", "dphi_tt", "pt_tt", "y_tt",
    "cos_theta_l", "cos_theta_had", "dphi_l_had",
    "cos_lep_n", "cos_lep_r", "cos_lep_k",
    "cos_had_n", "cos_had_r", "cos_had_k",
    "beta_t_star", "c_hel", "c_han",
    "dr_l_j0", "dr_l_j1", "dr_l_j2", "dr_l_j3",
    "dr_j01", "dr_j02", "dr_j03", "dr_j12", "dr_j13", "dr_j23",
    "m_j01", "m_j02", "m_j03", "m_j12", "m_j13", "m_j23",
    "m_lb_min",
]

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
        trainLoss.append(model.loss(batches.dataset[:][0], batches.dataset[:][1], batches.dataset[:][2]).item())
        testLoss.append(model.loss(test_feats, test_p0, test_p1).item())
    networkPlots(test_feats, test_p0, test_p1, model.net, trainLoss, testLoss, f'{config["name"]}/complete')

    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='configuration yml file used for training')
    with open(parser.parse_args().config, 'r') as f:
        config = yaml.safe_load(f)
    config = main(config)

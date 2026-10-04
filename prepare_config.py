import yaml
from pathlib import Path
from copy import deepcopy

with open("examples/training/stitched.yml") as f:
    base = yaml.safe_load(f)

wcs = base['wcs']          # 16 BSM names
n   = len(base['c0'])      # 17 = 1 SM + 16 BSM

FEATURE_SETS = {
    # -------------------------------------------------------------
    # Baseline: all 74 features
    # -------------------------------------------------------------
    "all": "all",

    # -------------------------------------------------------------
    # pt_mass: energy-scale handles only (existing)
    # -------------------------------------------------------------
    "pt_mass": [
        "lep_pt", "met_pt",
        "jet0_pt", "jet1_pt", "jet2_pt", "jet3_pt",
        "HT", "mT_W",
        "lep_top_pt", "lep_top_mass",
        "had_top_pt", "had_top_mass",
        "m_ttbar", "pt_tt",
        "m_j01", "m_j02", "m_j03",
        "m_j12", "m_j13", "m_j23",
        "m_lb_min",
    ],

    # -------------------------------------------------------------
    # top30: per-WC top-ranked features, hand-tuned (existing)
    # -------------------------------------------------------------
    "top30": [
        # Tier 0: dominant kinematics
        "m_ttbar", "had_top_pt", "lep_top_pt", "HT",
        "lep_pt", "met_pt",
        "jet0_pt", "jet1_pt", "jet2_pt", "jet3_pt",
        # ttbar system kinematics
        "pt_tt", "dy_tt", "dphi_tt", "cos_theta_star",
        # spin correlation / angular
        "c_hel", "c_han", "beta_t_star",
        "cos_theta_l", "cos_theta_had", "dphi_l_had",
        "dr_lep_had", "dr_tt",
        # jet masses
        "jet0_mass", "jet1_mass", "jet2_mass",
        # dijet / lepton-jet geometry
        "dr_j01", "dr_j02", "dr_l_j1",
        "m_j12", "m_j13",
    ],

    # -------------------------------------------------------------
    # compact: physics-motivated, 26 features
    #   per-WC chi2 top-10 union + spin observables + lep_top reco
    # -------------------------------------------------------------
    "compact": [
        # Energy-scale
        "m_ttbar", "HT",
        "jet0_pt", "jet1_pt", "jet2_pt", "jet3_pt",
        "met_pt", "lep_pt", "pt_tt",
        # Leptonic top reco (encodes neutrino pz)
        "lep_top_pt", "lep_top_eta", "lep_top_phi", "lep_top_mass",
        # Hadronic top pT (top chi2 handle)
        "had_top_pt",
        # ttbar system kinematics
        "beta_t_star", "y_tt", "dr_tt", "njets",
        # Spin / angular
        "c_hel", "c_han",
        "cos_theta_star", "cos_theta_l", "cos_theta_had",
        # Azimuthal correlations
        "dphi_tt", "dphi_l_had",
        # Top-mass proxy
        "m_lb_min",
    ],

    # -------------------------------------------------------------
    # lowlevel: quasi-4-momentum, 31 features
    #   raw 4-momenta of 6 final-state objects + lep_top reco
    #   + njets + b-tag info
    # -------------------------------------------------------------
    "lowlevel": [
        # Lepton 4-momentum
        "lep_pt", "lep_eta", "lep_phi", "lep_mass",
        # MET (transverse only)
        "met_pt", "met_phi",
        # 4 leading jets 4-momentum
        "jet0_pt", "jet0_eta", "jet0_phi", "jet0_mass",
        "jet1_pt", "jet1_eta", "jet1_phi", "jet1_mass",
        "jet2_pt", "jet2_eta", "jet2_phi", "jet2_mass",
        "jet3_pt", "jet3_eta", "jet3_phi", "jet3_mass",
        # Leptonic top reco (encodes neutrino pz via W-mass constraint)
        "lep_top_pt", "lep_top_eta", "lep_top_phi", "lep_top_mass",
        # Auxiliary
        "njets",
        "jet0_flav", "jet1_flav", "jet2_flav", "jet3_flav",
    ],

    # -------------------------------------------------------------
    # highlevel: reconstruction-level observables, 13 features
    #   spin correlations + ttbar system kinematics
    # -------------------------------------------------------------
    "highlevel": [
        # Spin correlation coefficients
        "c_hel", "c_han",
        # Angular observables in top rest frames
        "cos_theta_l", "cos_theta_had", "cos_theta_star",
        # ttbar system kinematics
        "m_ttbar", "pt_tt", "y_tt", "beta_t_star",
        # Reconstructed top pTs
        "lep_top_pt", "had_top_pt",
        # ttbar angular separation
        "dr_tt", "dphi_tt",
    ],
}

def make_network(nodes, dropout=0.20):
    return [
        {"type": "Linear", "out": nodes, "activation": "LeakyReLU"},
        {"type": "Linear", "in": nodes, "out": nodes, "activation": "LeakyReLU", "dropout": dropout},
        {"type": "Linear", "in": nodes, "out": nodes, "activation": "LeakyReLU", "dropout": dropout},
        {"type": "Linear", "in": nodes, "out": 1, "activation": "Sigmoid", "dropout": dropout},
    ]

NETWORK_SETS = {
    "n256_dropout": make_network(256),
    "n128_dropout": make_network(128),
    "n64_dropout":  make_network(64),
    "n32_dropout":  make_network(32),
}

SCHEDULER_SETS = {
    "plateau": {
        "scheduler": "plateau",
        "factor": 0.5,
        "lr_patience": 5,
    },
    "cosine": {
        "scheduler": "cosine",
        "warmup_epochs": 5,
    },
}

outdir = Path("condor/configs")
outdir.mkdir(parents=True, exist_ok=True)


WC_TRAINING_VALUES = {
    # WC          [ pone,  pfive,  pten,  nfive ]  <- constructive x3 + destructive x1 (nfive)
    'ctGIm':      [ 0.40,   0.90,  1.28, -0.91],
    'ctGRe':      [-0.03,  -0.15, -0.30,  4.82],
    'cQj38':      [ 1.50,   3.56,  5.12, -3.94],
    'cQj18':      [ 0.78,   2.60,  4.08, -5.39],
    'cQu8':       [ 1.06,   3.27,  5.02, -5.78],
    'cQd8':       [ 1.40,   4.26,  6.50, -7.27],
    'ctj8':       [ 0.63,   2.17,  3.44, -4.90],
    'ctu8':       [ 1.12,   3.45,  5.30, -6.10],
    'ctd8':       [ 1.44,   4.40,  6.73, -7.58],
    'cQj31':      [ 0.65,   1.48,  2.10, -1.51],
    'cQj11':      [-0.63,  -1.46, -2.08,  1.53],
    'cQu1':       [ 0.79,   1.85,  2.64, -1.99],
    'cQd1':       [ 1.06,   2.41,  3.42, -2.47],
    'ctj1':       [ 0.59,   1.38,  1.97, -1.49],
    'ctu1':       [-0.80,  -1.82, -2.59,  1.87],
    'ctd1':       [-0.99,  -2.28, -3.25,  2.38],
}

# Which WC_TRAINING_VALUES column each basis point uses.
# (i,0) and (i,i) both switch on only c_i, so they need different values,
# otherwise the morphing matrix is singular.
LIN_IDX   = 1   # pfive -> (i,0) linear term
QUAD_IDX  = 3   # nfive -> (i,i) quadratic term
CROSS_IDX = 1   # pfive -> (i,j) cross term, both c_i and c_j on

def basis_points():
    """One c1 per lower-triangular term (i,j), i>=j, skipping SM*SM: 153-1 = 152."""
    names = ['SM'] + wcs
    for i in range(1, n):
        for j in range(i + 1):
            c1 = [0.0]*n; c1[0] = 1.0
            if j == 0:
                c1[i] = WC_TRAINING_VALUES[names[i]][LIN_IDX]
            elif j == i:
                c1[i] = WC_TRAINING_VALUES[names[i]][QUAD_IDX]
            else:
                c1[i] = WC_TRAINING_VALUES[names[i]][CROSS_IDX]
                c1[j] = WC_TRAINING_VALUES[names[j]][CROSS_IDX]
            yield f"{names[j]}_{names[i]}", c1

seeds = [2]

# main
feat_tags = ["highlevel"]
sched_tag = "cosine"
sched_cfg = SCHEDULER_SETS[sched_tag]
net_tag = "n256_dropout"
net_val = NETWORK_SETS[net_tag]

c0 = [0.0]*n; c0[0] = 1.0
count = 0
for feat_tag in feat_tags:
    feat_val = FEATURE_SETS[feat_tag]
    for term, c1 in basis_points():
        for seed in seeds:
            cfg = deepcopy(base)
            cfg['c0'], cfg['c1'] = c0, c1

            cfg['features'] = feat_val
            cfg['network']  = net_val
            cfg['patience'] = 20
            cfg['seed']     = seed
            cfg.update(sched_cfg)
            cfg['name'] = (
                f"{base['name'].rsplit('/', 2)[0]}"
                f"/{feat_tag}_{net_tag}_{sched_tag}/{term}/seed_{seed}"
            )

            out = outdir / (
                f"{term}_feature_{feat_tag}"
                f"_net_{net_tag}_sched_{sched_tag}_seed_{seed}.yml"
            )
            print(f"{term:>14}  c1={[v for v in c1[1:] if v]}  seed={seed}  -> {out.name}")
            with open(out, 'w') as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            count += 1

print(f"\nwrote {count} configs to {outdir}")

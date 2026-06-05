#!/usr/bin/env python3
"""
Plot basic kinematic distributions for a NanoGEN ttbar semi-leptonic sample.

Mirrors the feature set used by colleague's SemiLepProcessor.calc_features:
  lepton (pt, eta, phi, mass), GenMET (pt, phi),
  4 leading jets (pt, eta, phi, mass), nJets
Plus a few sanity quantities (HT, mT_W).

The object/event selection here approximates genObjectSelection /
genEventSelection -- swap to the real analysis_tools functions when
you want exact agreement with the processor.

Usage:
    python plot_nanogen_features.py \
        --file root://cmseos.fnal.gov//store/user/honor/.../nanogen_modCentral_1.root \
        --outdir ./plots
"""

import os
import argparse
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

NanoAODSchema.warn_missing_crossrefs = False
plt.style.use(hep.style.CMS)


# ----------------------------- selection -----------------------------

def gen_object_selection(events):
    """Approximate genObjectSelection: clean prompt-lepton & jets at gen level."""
    leps = events.GenDressedLepton
    lep_mask = (
        (leps.pt > 25.0)
        & (abs(leps.eta) < 2.4)
        & (abs(leps.pdgId) != 15)        # GenDressedLepton already excludes tau, but be explicit
        & (~leps.hasTauAnc)              # drop leptons from tau decays
    )
    cleanleps = leps[lep_mask]

    jets = events.GenJet
    jet_kin = (jets.pt > 30.0) & (abs(jets.eta) < 2.4)

    # dR cleaning: jet must be > 0.4 from any selected lepton.
    # Build per-jet sublist of dRs to all (selected) leptons, then take min.
    pairs = ak.cartesian({"j": jets, "l": cleanleps}, nested=True)
    dRs = pairs.j.delta_r(pairs.l)
    min_dR = ak.fill_none(ak.min(dRs, axis=-1), 999.0)
    jet_mask = jet_kin & (min_dR > 0.4)
    cleanjets = jets[jet_mask]

    return cleanleps, cleanjets


def gen_event_selection(leps, jets):
    """Semi-leptonic ttbar: exactly 1 lepton + >=4 jets."""
    return (ak.num(leps) == 1) & (ak.num(jets) >= 4)


# ----------------------------- features -----------------------------

def build_features(events, mask):
    """Return dict of {feature_name: 1D numpy array} for selected events."""
    leps, jets = gen_object_selection(events)
    sel_leps = ak.flatten(leps[mask])
    sel_jets = jets[mask]
    sel_met  = events.GenMET[mask]

    feats = {
        "lep_pt":   ak.to_numpy(sel_leps.pt),
        "lep_eta":  ak.to_numpy(sel_leps.eta),
        "lep_phi":  ak.to_numpy(sel_leps.phi),
        "lep_mass": ak.to_numpy(sel_leps.mass),
        "met_pt":   ak.to_numpy(sel_met.pt),
        "met_phi":  ak.to_numpy(sel_met.phi),
        "njets":    ak.to_numpy(ak.num(sel_jets)),
        "HT":       ak.to_numpy(ak.sum(sel_jets.pt, axis=1)),
    }
    for i in range(4):
        feats[f"jet{i}_pt"]   = ak.to_numpy(sel_jets.pt[:, i])
        feats[f"jet{i}_eta"]  = ak.to_numpy(sel_jets.eta[:, i])
        feats[f"jet{i}_phi"]  = ak.to_numpy(sel_jets.phi[:, i])
        feats[f"jet{i}_mass"] = ak.to_numpy(sel_jets.mass[:, i])

    # mT(lep, MET) -- physical sanity check
    dphi_cos = np.cos(feats["lep_phi"] - feats["met_phi"])
    feats["mT_W"] = np.sqrt(2 * feats["lep_pt"] * feats["met_pt"] * (1 - dphi_cos))

    return feats


# ----------------------------- binning / labels -----------------------------

BINNING = {
    "lep_pt":   (50, 0, 300),
    "lep_eta":  (50, -3, 3),
    "lep_phi":  (40, -np.pi, np.pi),
    "lep_mass": (50, 0, 0.2),
    "met_pt":   (50, 0, 300),
    "met_phi":  (40, -np.pi, np.pi),
    "njets":    (10, 3.5, 13.5),
    "HT":       (60, 0, 1500),
    "mT_W":     (50, 0, 250),
}
XLABEL = {
    "lep_pt":   r"lepton $p_T$ [GeV]",
    "lep_eta":  r"lepton $\eta$",
    "lep_phi":  r"lepton $\phi$",
    "lep_mass": r"lepton mass [GeV]",
    "met_pt":   r"GenMET $p_T$ [GeV]",
    "met_phi":  r"GenMET $\phi$",
    "njets":    r"$N_{jets}$",
    "HT":       r"$H_T$ [GeV]",
    "mT_W":     r"$m_T(\ell, \mathrm{MET})$ [GeV]",
}
for i in range(4):
    BINNING[f"jet{i}_pt"]   = (50, 0, 500 if i == 0 else 300)
    BINNING[f"jet{i}_eta"]  = (50, -3, 3)
    BINNING[f"jet{i}_phi"]  = (40, -np.pi, np.pi)
    BINNING[f"jet{i}_mass"] = (50, 0, 60)
    XLABEL[f"jet{i}_pt"]    = rf"jet$_{{{i}}}$ $p_T$ [GeV]"
    XLABEL[f"jet{i}_eta"]   = rf"jet$_{{{i}}}$ $\eta$"
    XLABEL[f"jet{i}_phi"]   = rf"jet$_{{{i}}}$ $\phi$"
    XLABEL[f"jet{i}_mass"]  = rf"jet$_{{{i}}}$ mass [GeV]"


# ----------------------------- plotting -----------------------------

def plot_one(name, values, weights, outdir, label="ttbar semi-lep (NanoGEN)"):
    nb, lo, hi = BINNING[name]
    h = hist.Hist(hist.axis.Regular(nb, lo, hi, name=name), storage="weight")
    h.fill(values, weight=weights)

    fig, ax = plt.subplots(figsize=(8, 6))
    hep.histplot(h, ax=ax, histtype="step", linewidth=1.5, label=label)
    hep.cms.label(ax=ax, text="Simulation", data=False, com=13)
    ax.set_xlabel(XLABEL[name])
    ax.set_ylabel("Events (weighted)")
    ax.legend(loc="best")
    ax.set_xlim(lo, hi)

    out = os.path.join(outdir, f"{name}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default="/eos/uscms/store/user/honor/TTbarSemileptonic/modCentral/251114_001833/0000/nanogen_modCentral_1.root",
    )
    parser.add_argument("--outdir", default="./plots")
    parser.add_argument("--nevents", type=int, default=None,
                        help="Limit number of events (for quick tests)")
    parser.add_argument("--weight", choices=["genWeight", "sm_point", "none"],
                        default="genWeight",
                        help="Which event weight to use")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Opening {args.file}")
    events = NanoEventsFactory.from_root(
        {args.file: "Events"},
        schemaclass=NanoAODSchema,
        metadata={"dataset": "TTbarSemileptonic_modCentral"},
    ).events()

    if args.nevents:
        events = events[: args.nevents]
    n_in = len(events)
    print(f"  loaded {n_in} events")

    # ----- Selection -----
    leps, jets = gen_object_selection(events)
    mask = gen_event_selection(leps, jets)
    n_sel = int(ak.sum(mask))
    print(f"  passing selection: {n_sel} / {n_in} ({100*n_sel/n_in:.1f}%)")

    # ----- Weights -----
    if args.weight == "genWeight":
        w = ak.to_numpy(events.genWeight[mask])
    elif args.weight == "sm_point":
        w = ak.to_numpy(events.LHEWeight["sm_point"][mask])
    else:
        w = np.ones(n_sel)
    print(f"  using weight = '{args.weight}', sum = {w.sum():.3e}")

    # ----- Features -----
    feats = build_features(events, mask)

    # ----- Plot all features -----
    print(f"\nWriting plots to {args.outdir}/")
    for name, values in feats.items():
        out = plot_one(name, values, w, args.outdir)
        print(f"  -> {os.path.basename(out)}   "
              f"mean={values.mean():.3g}  std={values.std():.3g}")

    # ----- Summary -----
    print("\n=== Summary ===")
    print(f"  events read       : {n_in}")
    print(f"  events selected   : {n_sel}")
    print(f"  sum(genWeight)    : {float(ak.sum(events.genWeight)):.3e}")
    if "sm_point" in events.LHEWeight.fields:
        print(f"  sum(LHE sm_point) : "
              f"{float(ak.sum(events.LHEWeight['sm_point'])):.3e}")


if __name__ == "__main__":
    main()

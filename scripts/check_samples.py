import torch
import uproot

def check_nanoAOD():
    path = "/eos/uscms/store/user/honor/TTbarSemileptonic/modCentral/251114_001833/0000/nanogen_modCentral_1.root"
    f = uproot.open(path)
    
    print("=== Top-level keys ===")
    print(f.keys())
    print(f.classnames())
    
    tree = f["Events"]
    print(f"\n=== Events tree: {tree.num_entries} entries ===")
    
    branches = tree.keys()
    print(f"\nTotal branches: {len(branches)}")
    for b in branches[:30]:
        print(f"  {b:40s}  {tree[b].typename}")

def check_trainingSamples():
    path = "/uscms/home/honor/nobackup/Outputs_sbi/pretraining/modCentral_total/"

    for f in ["train.p", "test.p", "validation.p"]:
        fname = path + f
        d = torch.load(fname, map_location='cpu', weights_only=False)
        print(f"Sample {f}:")
        for i, t in enumerate(d.tensors):
            print(f'  - tensor[{i}]: shape={t.shape}, dtype={t.dtype}')

if __name__ == "__main__":
    # check_nanoAOD()
    check_trainingSamples()

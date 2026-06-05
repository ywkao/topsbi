import uproot

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

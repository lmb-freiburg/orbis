import json
import random

with open("DoTA_prepared/manifest.json") as f:
    manifest = json.load(f)

MANIFEST_LIMIT = 1500
SEED = 42

calib_clips = [c for c in manifest if c["non_ood_split"] == "calib"]
heldout_clips = [c for c in manifest if c["non_ood_split"] == "heldout"]

# keep the same ratio as your full split (e.g. ~3600:729 ≈ 83:17)
total_full = len(calib_clips) + len(heldout_clips)
calib_ratio = len(calib_clips) / total_full

n_calib = round(MANIFEST_LIMIT * calib_ratio)
n_heldout = MANIFEST_LIMIT - n_calib

random.seed(SEED)
random.shuffle(calib_clips)
random.shuffle(heldout_clips)

subset = calib_clips[:n_calib] + heldout_clips[:n_heldout]
random.shuffle(subset) 

print(f"Subset: {n_calib} calib, {n_heldout} heldout, {len(subset)} total (each also contributes 1 OOD window)")

with open("DoTA_prepared/manifest_subset1500.json", "w") as f:
    json.dump(subset, f, indent=2)
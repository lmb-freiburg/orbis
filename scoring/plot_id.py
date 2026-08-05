"""
Generates, for clip h55PiQMnlJY_003552:
  - Unnormalized: detailed, semantic, combined  (one figure, 3 rows)
  - Normalized:   detailed, semantic, combined  (one figure, 3 rows;
                  z-score vs calib_stats_combined.pt, sliced per head)
for both the non-ood and ood folders (skips whichever doesn't exist).

Requires: surprise_scorer_unified.py in the same directory / on sys.path.
"""

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

import torch
from omegaconf import OmegaConf

# ---- adjust to your actual project root ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util import instantiate_from_config
from surprise_scorer_unified import (
    HEADS,
    get_device,
    find_clip_folder,
    generate_all_variants,
)

CLIP_ID = "h55PiQMnlJY_003552"
T_GRID = [0.2, 0.4, 0.6, 0.8]
N_NOISE_SAMPLES = 2

EXP_DIR = PROJECT_ROOT / "logs_wm" / "orbis_288x512"
CALIB_STATS_COMBINED_PATH = PROJECT_ROOT / "results_pt" / "calib_stats_combined.pt"  # n=3000
OUTPUT_DIR = PROJECT_ROOT / "results" / "single_clip_variants" / CLIP_ID


def main():
    device = get_device()
    print(f"Using device: {device}")

    if not CALIB_STATS_COMBINED_PATH.exists():
        raise FileNotFoundError(
            f"{CALIB_STATS_COMBINED_PATH} not found. Generate it first with "
            f"scorer_batch_norm_combined.py (HEADS=['combined']) over your 3000 calib clips."
        )

    print("Loading model...")
    cfg = OmegaConf.load(EXP_DIR / "config.yaml")
    model = instantiate_from_config(cfg.model)
    state = torch.load(EXP_DIR / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    for split, sample_label in [("ood", "Anomaly")]:
        try:
            folder = find_clip_folder(PROJECT_ROOT, CLIP_ID, split)
        except FileNotFoundError as e:
            print(f"[Skipping] {e}")
            continue

        print(f"Processing {CLIP_ID} | {split} -> {folder}")
        generate_all_variants(
            model=model,
            clip_id=CLIP_ID,
            folder_path=str(folder),
            sample_label=sample_label,
            t_grid=T_GRID,
            calib_stats_combined_path=CALIB_STATS_COMBINED_PATH,
            device=device,
            output_dir=OUTPUT_DIR,
            heads=HEADS,
            n_noise_samples=N_NOISE_SAMPLES,
        )

    print(f"\nDone. Plots saved under: {OUTPUT_DIR}")
    print("  overlay_normal_unnormalized.png   / overlay_normal_normalized.png")


if __name__ == "__main__":
    main()
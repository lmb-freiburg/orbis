import json
import os
import warnings
from pathlib import Path
from collections import defaultdict

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# Import multi-head helper functions and models
from plotter_batch_norm import plot_and_overlay, get_device, instantiate_from_config

# --- Config & Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORBIS_ROOT = PROJECT_ROOT
MANIFEST_PATH = ORBIS_ROOT / "DoTA_class" / "manifest_dota_classes.json"
OUTPUT_BASE_DIR = ORBIS_ROOT / "results" / "unnormalized" / "classes"
CALIB_STATS_PATH = ORBIS_ROOT / "results" / "calib_stats_combined.pt"
EXP_DIR = ORBIS_ROOT / "logs_wm" / "orbis_288x512"

T_GRID = [0.2, 0.4, 0.6, 0.8]
HEADS = ["detailed", "semantic"]
MAX_CLIPS_PER_CLASS = 10


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Declare CALIB_STATS_PATH as global since we reassign it during fallback
    global CALIB_STATS_PATH

    # 1. Read Class Manifest & Select 10 Clips Per Class
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}. Run sampling script first.")

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    # Group clips by class and pick top 10 per class
    clips_by_class = defaultdict(list)
    for sample in manifest:
        clips_by_class[sample["anomaly_name"]].append(sample)

    selected_manifest = []
    for class_name, clips in clips_by_class.items():
        selected_manifest.extend(clips[:MAX_CLIPS_PER_CLASS])

    print(f"Selected {len(selected_manifest)} total clips across {len(clips_by_class)} classes.")

    # 2. Load Model
    print("Loading model...")
    cfg = OmegaConf.load(EXP_DIR / "config.yaml")
    model = instantiate_from_config(cfg.model)
    checkpoint_path = EXP_DIR / "checkpoints" / "last.ckpt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    # 3. Load Calibration Stats
    if not CALIB_STATS_PATH.exists():
        fallback_path = ORBIS_ROOT / "results" / "calib_stats.pt"
        if fallback_path.exists():
            CALIB_STATS_PATH = fallback_path
        else:
            raise FileNotFoundError(f"Stats file not found at {CALIB_STATS_PATH}")

    print(f"Loading calibration stats from: {CALIB_STATS_PATH}")
    calib_stats = torch.load(CALIB_STATS_PATH, weights_only=True)

    # Prepare all execution tasks
    tasks = []
    for sample in selected_manifest:
        for split in ["ood", "non-ood"]:
            tasks.append((sample, split))

    # 4. Processing Loop with tqdm
    pbar = tqdm(tasks, desc="Processing clips", unit="clip")
    
    for sample, split in pbar:
        clip_id = sample["clip_id"]
        class_name = sample["anomaly_name"]
        split_folder = ORBIS_ROOT / "DoTA_class" / class_name / clip_id / split

        # Update progress bar description with current item info
        pbar.set_postfix({"class": class_name, "clip": clip_id, "split": split})

        if not split_folder.exists():
            pbar.write(f"Skipping missing directory: {split_folder}")
            continue

        save_dir = OUTPUT_BASE_DIR / class_name / clip_id / split
        save_dir.mkdir(parents=True, exist_ok=True)

        sample_label = f"{class_name}_{split.upper()}"
        
        # Expected Z-Score file path
        expected_filename = f"overlay_{sample_label.lower()}_combined_zscore.png"
        expected_filepath = save_dir / expected_filename

        if expected_filepath.exists():
            pbar.write(f"[Skipping - Exists] {class_name} -> {clip_id} | {split.upper()} | ZScore")
            continue

        try:
            plot_and_overlay(
                model=model,
                clip_id=clip_id,
                folder_path=str(split_folder),
                sample_label=sample_label,
                t_grid=T_GRID,
                device=device,
                heads=HEADS,
                n_noise_samples=2,
                output_dir_override=str(save_dir)
            )
        except Exception as e:
            pbar.write(f"  [Error] Failed to process {clip_id} ({split}, ZScore): {e}")

    print("\n" + "=" * 60)
    print(f"Execution complete! All outputs saved to: {OUTPUT_BASE_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
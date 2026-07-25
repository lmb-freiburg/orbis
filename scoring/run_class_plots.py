import json
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path
import torch
from omegaconf import OmegaConf
from collections import defaultdict
import warnings


# Import plotter function from plotter.py
from plotter_batch_norm import plot_and_overlay, get_device, instantiate_from_config

# --- Config & Paths ---

PROJECT_ROOT = Path(__file__).resolve().parent
ORBIS_ROOT = PROJECT_ROOT.parent
MANIFEST_PATH = ORBIS_ROOT / "DoTA_class" / "manifest_dota_classes.json"
OUTPUT_BASE_DIR = ORBIS_ROOT / "results" / "batch_norm" / "classes"
CALIB_STATS_PATH = ORBIS_ROOT / "results" / "calib_stats.pt"
EXP_DIR = ORBIS_ROOT / "logs_wm" / "orbis_288x512"

T_GRID = [0.2, 0.4, 0.6, 0.8]
MAX_CLIPS_PER_CLASS = 1 


def main():
    device = get_device()
    print(f"Using device: {device}")

    # 1. Read Class Manifest & Select 1 Clip Per Class
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}. Run sampling script first.")

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    # Group clips by class and select top 1 per class
    clips_by_class = defaultdict(list)
    for sample in manifest:
        clips_by_class[sample["anomaly_name"]].append(sample)

    selected_manifest = []
    for class_name, clips in clips_by_class.items():
        selected_manifest.extend(clips[:MAX_CLIPS_PER_CLASS])

    # 2. Check if model load can be deferred or needed
    print("Loading model...")
    cfg = OmegaConf.load(EXP_DIR / "config.yaml")
    model = instantiate_from_config(cfg.model)
    checkpoint_path = EXP_DIR / "checkpoints" / "last.ckpt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    # 3. Load Calibration Stats
    print(f"Loading calibration stats from: {CALIB_STATS_PATH}")
    calib_stats = torch.load(CALIB_STATS_PATH, weights_only=True)

    print(f"\nFast run: Processing {len(selected_manifest)} clips (1 per class) x 2 splits (OOD + Non-OOD) x 2 modes (Z-Score + Min-Max).\n")

    processed_count = 0
    total_tasks = len(selected_manifest) * 2 * 2  # 1 clip * 2 splits * 2 modes

    for sample in selected_manifest:
        clip_id = sample["clip_id"]
        class_name = sample["anomaly_name"]

        for split in ["ood", "non-ood"]:
            split_folder = ORBIS_ROOT / "DoTA_class" / class_name / clip_id / split

            if not split_folder.exists():
                print(f"Skipping missing directory: {split_folder}")
                continue

            save_dir = OUTPUT_BASE_DIR / class_name / clip_id / split
            save_dir.mkdir(parents=True, exist_ok=True)

            for use_minmax in [False, True]:
                processed_count += 1
                mode_label = "MinMax" if use_minmax else "ZScore"
                sample_label = f"{class_name}_{split.upper()}"
                
                # Check if target image already exists
                expected_filename = f"overlay_{sample_label.lower()}_{mode_label.lower()}.png"
                expected_filepath = save_dir / expected_filename

                if expected_filepath.exists():
                    print(f"[{processed_count}/{total_tasks}] [Skipping - Exists] {class_name} -> {clip_id} | {split.upper()} | {mode_label}")
                    continue

                print(f"[{processed_count}/{total_tasks}] Processing {class_name} -> {clip_id} | {split.upper()} | {mode_label}")

                try:
                    plot_and_overlay(
                        model=model,
                        clip_id=clip_id,
                        folder_path=str(split_folder),
                        sample_label=sample_label,
                        t_grid=T_GRID,
                        calib_stats=calib_stats,
                        device=device,
                        n_noise_samples=4,
                        z_vmax=3.0,
                        z_threshold=1.0,
                        use_minmax=use_minmax,
                        output_dir_override=str(save_dir)
                    )
                except Exception as e:
                    print(f"  [Error] Failed to process {clip_id} ({split}, {mode_label}): {e}")

    print("\n" + "=" * 60)
    print(f"Execution complete! All outputs ready in: {OUTPUT_BASE_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
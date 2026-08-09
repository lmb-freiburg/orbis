import json
import os
import warnings
from pathlib import Path
from collections import defaultdict

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

import torch
from omegaconf import OmegaConf

import sys
SCORING_DIR = Path(__file__).resolve().parent
DIAGNOSTIC_PROBES_DIR = SCORING_DIR.parent
ORBIS_ROOT = DIAGNOSTIC_PROBES_DIR.parent

for p_add in [str(ORBIS_ROOT), str(DIAGNOSTIC_PROBES_DIR), str(SCORING_DIR)]:
    if p_add not in sys.path:
        sys.path.insert(0, p_add)

try:
    from scoring.plotter_batch_norm import plot_and_overlay, get_device, instantiate_from_config
except ImportError:
    from plotter_batch_norm import plot_and_overlay, get_device, instantiate_from_config

MANIFEST_PATH = ORBIS_ROOT / "DoTA_class" / "manifest_dota_classes.json"
OUTPUT_BASE_DIR = ORBIS_ROOT / "results" / "batch_norm2" / "classes_combined"
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

    # Total tasks: clips * 2 splits (OOD / Non-OOD)
    total_tasks = len(selected_manifest) * 2
    processed_count = 0

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

            processed_count += 1
            sample_label = f"{class_name}_{split.upper()}"
            
            # Expected Z-Score file path
            expected_filename = f"overlay_{sample_label.lower()}_combined_zscore.png"
            expected_filepath = save_dir / expected_filename

            if expected_filepath.exists():
                print(f"[{processed_count}/{total_tasks}] [Skipping - Exists] {class_name} -> {clip_id} | {split.upper()} | ZScore")
                continue

            print(f"[{processed_count}/{total_tasks}] Processing {class_name} -> {clip_id} | {split.upper()} | ZScore")

            try:
                plot_and_overlay(
                    model=model,
                    clip_id=clip_id,
                    folder_path=str(split_folder),
                    sample_label=sample_label,
                    t_grid=T_GRID,
                    calib_stats=calib_stats,
                    device=device,
                    heads=HEADS,
                    n_noise_samples=2,
                    z_vmax=3.0,
                    z_threshold=1.0,
                    use_minmax=False,
                    output_dir_override=str(save_dir)
                )
            except Exception as e:
                print(f"  [Error] Failed to process {clip_id} ({split}, ZScore): {e}")

    print("\n" + "=" * 60)
    print(f"Execution complete! All outputs saved to: {OUTPUT_BASE_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
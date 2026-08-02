import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

# --- Configuration & Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"
FRAMES_ROOT = Path("/Volumes/maccbeast")
OUTPUT_DIR = PROJECT_ROOT / "DoTA_class"

NUM_FRAMES = 11
FRAMES_INTO_ANOMALY = 11  # Target frame lands 3 frames after onset
CLIPS_PER_CLASS_MIN = 5
CLIPS_PER_CLASS_MAX = 6

# Explicit list of excluded anomaly names (Class 9 / Unknown / Unclassified)
EXCLUDED_ANOMALY_NAMES = {
    "unknown",
    "uk",
    "other",
    "unclassified",
    "9",
    "class_9",
}


def get_early_ood_window_start(anomaly_start, num_frames=11, frames_into_anomaly=3):
    """
    Window ends FRAMES_INTO_ANOMALY frames after onset, not centered/deep into the anomaly.
    Most of the window is still pre-anomaly context; only the tail shows the object emerging.
    """
    target_frame_idx = anomaly_start + frames_into_anomaly
    window_start = target_frame_idx - (num_frames - 1)
    return window_start if window_start >= 0 else None


def copy_window(clip_id, frame_indices, dest_dir):
    """Copies sequence of frames to destination directory."""
    src_dir = FRAMES_ROOT / "frames" / clip_id / "images"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for new_idx, frame_id in enumerate(frame_indices):
        src = src_dir / f"{frame_id:06d}.jpg"
        if not src.exists():
            return False
        shutil.copy2(src, dest_dir / f"{new_idx:06d}.jpg")
    return True


def main():
    # Set random seed for reproducible sampling across runs
    random.seed(42)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_files = sorted(ANNOTATIONS_DIR.glob("*.json"))

    # Dictionary to group candidate clips by class: { class_name: [ clip_metadata_dict, ... ] }
    candidates_by_class = defaultdict(list)

    stats = {
        "skipped_ignored": 0,
        "skipped_excluded_class": 0,
        "skipped_night": 0,
        "skipped_window": 0,
    }

    # 1. Parse and Filter Metadata
    for jf in json_files:
        with open(jf, "r") as f:
            meta = json.load(f)

        # Condition 1: Check ignore flag
        if meta.get("ignore", False):
            stats["skipped_ignored"] += 1
            continue

        accident_name = str(meta.get("accident_name", "")).strip().lower()

        # Condition 2: Exclude Class 9 / Unknown / Other
        if not accident_name or accident_name in EXCLUDED_ANOMALY_NAMES:
            stats["skipped_excluded_class"] += 1
            continue

        # Condition 3: Exclude Night scenes
        if meta.get("night", False):
            stats["skipped_night"] += 1
            continue

        clip_id = meta["video_name"]
        anomaly_start = meta["anomaly_start"]
        anomaly_end = meta["anomaly_end"]

        # Condition 4: Frame window validation
        window_start = get_early_ood_window_start(
            anomaly_start, NUM_FRAMES, FRAMES_INTO_ANOMALY
        )
        if window_start is None or window_start + NUM_FRAMES - 1 >= anomaly_end:
            stats["skipped_window"] += 1
            continue
        if anomaly_start < NUM_FRAMES:  # Not enough pre-anomaly frames for non-ood window
            stats["skipped_window"] += 1
            continue

        non_ood_indices = list(range(0, NUM_FRAMES))
        ood_indices = list(range(window_start, window_start + NUM_FRAMES))

        candidates_by_class[accident_name].append(
            {
                "clip_id": clip_id,
                "anomaly_name": accident_name,
                "night": False,
                "non_ood_indices": non_ood_indices,
                "ood_indices": ood_indices,
            }
        )

    print("\n" + "=" * 60)
    print(" Filtering Summary ")
    print("=" * 60)
    print(f"Skipped (Ignored):          {stats['skipped_ignored']}")
    print(f"Skipped (Class 9/Unknown):  {stats['skipped_excluded_class']}")
    print(f"Skipped (Night Clips):      {stats['skipped_night']}")
    print(f"Skipped (Invalid Window):   {stats['skipped_window']}")
    print("=" * 60 + "\n")

    manifest = []
    total_copied_clips = 0

    # 2. Sample 5-6 Clips Per Class and Copy Frames
    for class_name, clips in candidates_by_class.items():
        # Shuffle candidates randomly to get diverse sampling
        random.shuffle(clips)

        # Select between 5 and 6 clips
        sample_count = min(len(clips), CLIPS_PER_CLASS_MAX)
        if sample_count < CLIPS_PER_CLASS_MIN:
            print(
                f"[Warning] Class '{class_name}' only has {sample_count} valid candidate(s)."
            )

        selected_clips = clips[:sample_count]
        copied_in_class = 0

        for candidate in selected_clips:
            clip_id = candidate["clip_id"]

            # Structure: DoTA_class / <class_name> / <clip_id> / [non-ood | ood]
            clip_dest_dir = OUTPUT_DIR / class_name / clip_id

            ok1 = copy_window(clip_id, candidate["non_ood_indices"], clip_dest_dir / "non-ood")
            ok2 = copy_window(clip_id, candidate["ood_indices"], clip_dest_dir / "ood")

            if not (ok1 and ok2):
                # Cleanup incomplete copies
                shutil.rmtree(clip_dest_dir, ignore_errors=True)
                print(f"  [Error] Missing frames for clip '{clip_id}'. Skipped.")
                continue

            copied_in_class += 1
            manifest.append(
                {
                    "clip_id": clip_id,
                    "anomaly_name": class_name,
                    "night": False,
                    "class_folder": f"DoTA_class/{class_name}/{clip_id}",
                    "non_ood_split": f"{class_name}_test",
                    "ood_split": f"{class_name}_test",
                }
            )

        print(f"Class '{class_name}': Successfully copied {copied_in_class} clips.")
        total_copied_clips += copied_in_class

    # 3. Save Overall Manifest
    manifest_path = OUTPUT_DIR / "manifest_dota_classes.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Sampling Complete! Total Clips Extracted: {total_copied_clips}")
    print(f"Manifest saved to: {manifest_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
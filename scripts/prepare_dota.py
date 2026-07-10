import json
import os
import shutil
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"
FRAMES_ROOT = "/Volumes/maccbeast"
OUTPUT_DIR = PROJECT_ROOT / "DoTA_prepared"
NUM_FRAMES = 11

#CALIB_COUNT = 3600
#HELDOUT_COUNT = 1000

# new ratio
CALIB_COUNT = 3300
HELDOUT_COUNT = 1029

SEED = 42


def get_ood_window_start(anomaly_start, anomaly_end, num_frames=11, mode="center"):
    anomaly_len = anomaly_end - anomaly_start
    if anomaly_len < num_frames:
        return None
    if mode == "center":
        return anomaly_start + (anomaly_len - num_frames) // 2
    elif mode == "onset":
        return anomaly_start
    else:
        raise ValueError(mode)


def copy_window(clip_id, frame_indices, dest_subdir):
    src_dir = Path(FRAMES_ROOT) / "frames" / clip_id / "images"
    dest_dir = Path(OUTPUT_DIR) / clip_id / dest_subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    for new_idx, frame_id in enumerate(frame_indices):
        src = src_dir / f"{frame_id:06d}.jpg"
        if not src.exists():
            return False
        dst = dest_dir / f"{new_idx:06d}.jpg"
        shutil.copy2(src, dst)
    return True


def main():
    json_files = sorted(Path(ANNOTATIONS_DIR).glob("*.json"))
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Found {len(json_files)} annotation files")

    valid_clips = []
    skipped_ignore, skipped_short_normal, skipped_short_anomaly, skipped_missing_frames = 0, 0, 0, 0

    for jf in json_files:
        with open(jf) as f:
            meta = json.load(f)

        clip_id = meta["video_name"]
        if meta.get("ignore", False):
            skipped_ignore += 1
            continue

        anomaly_start = meta["anomaly_start"]
        anomaly_end = meta["anomaly_end"]

        if anomaly_start < NUM_FRAMES:
            skipped_short_normal += 1
            continue

        ood_start = get_ood_window_start(anomaly_start, anomaly_end, NUM_FRAMES, mode="center")
        if ood_start is None:
            skipped_short_anomaly += 1
            continue

        non_ood_indices = list(range(0, NUM_FRAMES))
        ood_indices = list(range(ood_start, ood_start + NUM_FRAMES))

        ok_non_ood = copy_window(clip_id, non_ood_indices, "non-ood")
        ok_ood = copy_window(clip_id, ood_indices, "ood")

        if not (ok_non_ood and ok_ood):
            skipped_missing_frames += 1
            # clean up partial copy if one side failed
            shutil.rmtree(Path(OUTPUT_DIR) / clip_id, ignore_errors=True)
            continue

        valid_clips.append({
            "clip_id": clip_id,
            "anomaly_name": meta.get("accident_name", "unknown") if False else meta["labels"][ood_start]["accident_name"],
            "night": meta.get("night", False),
            "ego_involve": meta.get("ego_involve", False),
            "num_frames": meta["num_frames"],
        })

        if len(valid_clips) % 200 == 0:
            print(f"Processed {len(valid_clips)} valid clips so far...")

    print(f"\nDone. Valid clips: {len(valid_clips)}")
    print(f"Skipped — ignore flag: {skipped_ignore}, "
          f"not enough pre-anomaly frames: {skipped_short_normal}, "
          f"anomaly segment too short: {skipped_short_anomaly}, "
          f"missing frame files: {skipped_missing_frames}")

    # ---- split assignment ----
    random.seed(SEED)
    shuffled = valid_clips.copy()
    random.shuffle(shuffled)

    n_calib = min(CALIB_COUNT, len(shuffled))
    n_heldout = min(HELDOUT_COUNT, max(0, len(shuffled) - n_calib))

    for i, clip in enumerate(shuffled):
        if i < n_calib:
            clip["non_ood_split"] = "calib"
        elif i < n_calib + n_heldout:
            clip["non_ood_split"] = "heldout"
        else:
            clip["non_ood_split"] = "unused_non_ood"   # extra clips beyond calib+heldout quota
        clip["ood_split"] = "ood"                        # every valid clip's OOD window is used

    manifest_path = Path(OUTPUT_DIR) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(shuffled, f, indent=2)

    print(f"\nSplit: {n_calib} calib, {n_heldout} heldout, "
          f"{len(shuffled) - n_calib - n_heldout} unused_non_ood, "
          f"{len(shuffled)} total ood")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
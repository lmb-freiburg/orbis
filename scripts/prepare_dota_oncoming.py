import json
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"
FRAMES_ROOT = "/Volumes/maccbeast"
OUTPUT_DIR = PROJECT_ROOT / "DoTA_oncoming"
NUM_FRAMES = 11
TARGET_ANOMALY_NAME = "oncoming"
FRAMES_INTO_ANOMALY = 3   # target frame lands just 3 frames after anomaly onset — object visible, not yet crashed


def get_early_ood_window_start(anomaly_start, num_frames=11, frames_into_anomaly=3):
    """
    Window ends FRAMES_INTO_ANOMALY frames after onset, not centered/deep into the anomaly.
    Most of the window is still pre-anomaly context; only the tail shows the object emerging.
    """
    target_frame_idx = anomaly_start + frames_into_anomaly
    window_start = target_frame_idx - (num_frames - 1)
    return window_start if window_start >= 0 else None


def copy_window(clip_id, frame_indices, dest_subdir):
    src_dir = Path(FRAMES_ROOT) / "frames" / clip_id / "images"
    dest_dir = Path(OUTPUT_DIR) / clip_id / dest_subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    for new_idx, frame_id in enumerate(frame_indices):
        src = src_dir / f"{frame_id:06d}.jpg"
        if not src.exists():
            return False
        shutil.copy2(src, dest_dir / f"{new_idx:06d}.jpg")
    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_files = sorted(ANNOTATIONS_DIR.glob("*.json"))

    valid_clips = []
    skipped_not_oncoming, skipped_window, skipped_missing = 0, 0, 0

    for jf in json_files:
        with open(jf) as f:
            meta = json.load(f)

        if meta.get("ignore", False):
            continue
        if meta.get("accident_name") != TARGET_ANOMALY_NAME:
            skipped_not_oncoming += 1
            continue

        clip_id = meta["video_name"]
        anomaly_start = meta["anomaly_start"]
        anomaly_end = meta["anomaly_end"]

        window_start = get_early_ood_window_start(anomaly_start, NUM_FRAMES, FRAMES_INTO_ANOMALY)
        if window_start is None or window_start + NUM_FRAMES - 1 >= anomaly_end:
            skipped_window += 1
            continue
        if anomaly_start < NUM_FRAMES:  # not enough pre-anomaly frames for a normal window
            skipped_window += 1
            continue

        non_ood_indices = list(range(0, NUM_FRAMES))
        ood_indices = list(range(window_start, window_start + NUM_FRAMES))

        ok1 = copy_window(clip_id, non_ood_indices, "non-ood")
        ok2 = copy_window(clip_id, ood_indices, "ood")
        if not (ok1 and ok2):
            skipped_missing += 1
            shutil.rmtree(Path(OUTPUT_DIR) / clip_id, ignore_errors=True)
            continue

        valid_clips.append({
            "clip_id": clip_id,
            "anomaly_name": meta["accident_name"],
            "night": meta.get("night", False),
            "non_ood_split": "oncoming_test",   # reuse existing calibration, no new calib/heldout needed
            "ood_split": "oncoming_test",
        })

    print(f"Valid oncoming clips: {len(valid_clips)}")
    print(f"Skipped — not oncoming: {skipped_not_oncoming}, window issues: {skipped_window}, "
          f"missing frames: {skipped_missing}")

    with open(OUTPUT_DIR / "manifest_oncoming.json", "w") as f:
        json.dump(valid_clips, f, indent=2)


if __name__ == "__main__":
    main()
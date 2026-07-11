import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from util import instantiate_from_config

RESULTS_DIR = "results"
os.makedirs(f"{RESULTS_DIR}/error_maps", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/scores", exist_ok=True)
SCORES_CSV = f"{RESULTS_DIR}/scores/scores.csv"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------- loader ----------
def get_sorted_frame_paths(folder):
    paths = sorted(glob.glob(f"{folder}/*.jpg"))
    assert len(paths) == 11, f"expected 11 frames in {folder}, found {len(paths)}"
    return paths


def load_clip_as_window(frame_paths, size=(512, 288), num_frames=6):
    idxs = [0, 2, 4, 6, 8, 10]  # 11 frames @10fps -> 6 frames @5fps
    assert len(frame_paths) == 11, f"expected 11 frames, got {len(frame_paths)}"
    frames = []
    for i in idxs:
        img = cv2.cvtColor(cv2.imread(frame_paths[i]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
        frames.append(t)
    return torch.stack(frames)

@torch.no_grad()
def surprise_score(model, images, frame_rate, t_grid, n_noise_samples=3, use_ema=False):
    net = model.ema_vit if use_ema else model.vit # the STDiT model that predicts the velocity of the target frame given the context frames
    net.eval()
    # calls self.ae.encode(images)['continuous']
    x = model.encode_frames(images) # get the latent representation of the frames from the orbis tokenizer [1, 6, 3, 288, 512] -> [1, 6, 16, 16, 16]
    context, target = x[:, :-1].clone(), x[:, -1:] # separate the context frames and the target frame
    b = x.shape[0] # [1, 6, 16, 16, 16]
    n_channels = target.shape[2]
    half = n_channels // 2
    per_t_scalar = []   # list of [B] tensors, one per t
    per_t_map = {}       # {t_val: [B, 16, 16] tensor}
    per_t_detail, per_t_semantic = [], []
    for t_val in t_grid: # t_grid = [0.3, 0.5, 0.7, 0.9] — the four different time steps to evaluate the model's prediction error
        errs = []
        for _ in range(n_noise_samples): # n_noise_samples = 3 — for each t_val, we sample 3 different noise vectors to compute the prediction error
            t = torch.full((b,), t_val, device=x.device)
            target_t, noise = model.add_noise(target, t) 
            pred = net(target_t, context, t, frame_rate=frame_rate) # the STDiT outputs a prediction the same shape as target_t — [1, 1, 16, 16, 16], 
            # one predicted velocity value per channel per spatial cell.
            true_v = model.A(t) * target + model.B(t) * noise # the true velocity is a linear combination of the target frame and the noise vector, 
            # weighted by A(t) and B(t) respectively.
            errs.append((pred.float() - true_v.float()) ** 2)   # [B, 1, C, 16, 16]
        err_avg = torch.stack(errs).mean(0)
        per_t_scalar.append(err_avg.mean(dim=[1, 2, 3, 4]))       # [B] collapses everything but batch → one scalar per clip — score_mean
        per_t_map[t_val] = err_avg.mean(dim=2).squeeze(1)          # [B, 16, 16]  collapses only the channel dimension → [1, 1, 16, 16] → one error value per 16×16 grid cell, still tied to where in the frame it came from.

    raw_score = torch.stack(per_t_scalar, dim=1).mean(dim=1)      # [B] raw_score averages across the 4 t-level scores too, giving one final number per clip.
    return raw_score, per_t_scalar, per_t_map


def log_sample_result(sample_id, split, label, per_t_scalar_values, t_grid, mean_score, error_map=None):
    """
    per_t_scalar_values: list of floats, one per t
    error_map: optional dict {t_val: [16,16] numpy array} for ONE sample (pass only for illustrative examples)
    """
    row = {"sample_id": sample_id, "split": split, "label": label, "score_mean": mean_score}
    for t_val, score in zip(t_grid, per_t_scalar_values):
        row[f"score_t{t_val}"] = score

    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(SCORES_CSV)
    df_row.to_csv(SCORES_CSV, mode="a", header=write_header, index=False)

    if error_map is not None:
        save_dict = {f"t{t_val}": arr for t_val, arr in error_map.items()}
        np.savez_compressed(f"{RESULTS_DIR}/error_maps/{sample_id}.npz", **save_dict)

    print(f"Logged {sample_id} ({split}/{label}) -> mean_score={mean_score:.5f}")


def process_and_log(model, folder, sample_id, split, label, t_grid, n_noise_samples=3, save_map=False, device=None):
    """One-stop: load frames from folder, score, log. Call this in a loop for many samples later."""
    if device is None:
        device = get_device()

    paths = get_sorted_frame_paths(folder)
    window = load_clip_as_window(paths).unsqueeze(0).to(device)  # [1, 6, C, H, W]
    frame_rate = torch.full((1,), 5.0).to(device)

    raw_score, per_t_scalar, per_t_map = surprise_score(model, window, frame_rate, t_grid, n_noise_samples)

    per_t_values = [s[0].item() for s in per_t_scalar]
    error_map_np = {t: per_t_map[t][0].cpu().numpy() for t in t_grid} if save_map else None

    log_sample_result(sample_id, split, label, per_t_values, t_grid, raw_score[0].item(), error_map_np)


@torch.no_grad()
def compute_and_save_heatmap(model, folder, out_sample_id, t_grid, n_noise_samples, device):
    """
    Runs surprise_score on ONE clip window and saves its spatial error map to disk.
    Does NOT touch scores.csv — this is purely for visualization, kept separate
    from your real calibration/eval results.
    """
    paths = get_sorted_frame_paths(folder)
    window = load_clip_as_window(paths).unsqueeze(0).to(device)
    frame_rate = torch.full((1,), 5.0).to(device)

    raw_score, per_t_scalar, per_t_map = surprise_score(model, window, frame_rate, t_grid, n_noise_samples)

    mean_score = raw_score[0].item()
    error_map_np = {t: per_t_map[t][0].cpu().numpy() for t in t_grid}
    save_dict = {f"t{t_val}": arr for t_val, arr in error_map_np.items()}
    np.savez_compressed(f"{RESULTS_DIR}/error_maps/{out_sample_id}.npz", **save_dict)

    print(f"Saved heatmap for {out_sample_id} -> mean_score={mean_score:.5f}")
    return mean_score


HEATMAP_CLIP_IDS = [
    "4wKjxDXnmYs_003798",   
    "fdvMUP8qvzw_000969",   
    "L334aqEJxys_001608",  
    "xpOyD-qrQUw_004160",    
    "3tEZvtQZ18Q_004890",  
]

def generate_heatmaps_for_clips(model, clip_ids, t_grid, n_noise_samples=3,
                                 base_dir="DoTA_prepared", device=None):
    """
    For each clip_id, generates heatmaps for BOTH its non-ood and ood windows,
    so you can show "expected" vs "surprised" side by side for the same clip.
    """
    if device is None:
        device = get_device()

    for clip_id in clip_ids:
        clip_dir = Path(base_dir) / clip_id

        compute_and_save_heatmap(
            model, str(clip_dir / "non-ood"),
            out_sample_id=f"{clip_id}_nonood_heatmap",
            t_grid=t_grid, n_noise_samples=n_noise_samples, device=device,
        )
        compute_and_save_heatmap(
            model, str(clip_dir / "ood"),
            out_sample_id=f"{clip_id}_ood_heatmap",
            t_grid=t_grid, n_noise_samples=n_noise_samples, device=device,
        )

def fit_calibration(t_grid, min_calib_samples=30):
    """
    Reads scores.csv, computes mean/std per t-column using ONLY rows with split == 'calib'.
    Returns dict: {t_val: {"mean": ..., "std": ...}}
    Also returns overall mean/std on score_mean, for the simple aggregate z-score.
    """
    df = pd.read_csv(SCORES_CSV)
    calib_df = df[df["split"] == "calib"]

    if len(calib_df) < min_calib_samples:
        print(f"WARNING: only {len(calib_df)} calibration samples found "
              f"(recommend >= {min_calib_samples} before trusting these stats). "
              f"Proceeding anyway since you're still in pilot mode.")

    stats = {}
    for t_val in t_grid:
        col = f"score_t{t_val}"
        stats[t_val] = {"mean": float(calib_df[col].mean()), "std": float(calib_df[col].std())}

    stats["overall"] = {
        "mean": float(calib_df["score_mean"].mean()),
        "std": float(calib_df["score_mean"].std()),
    }

    stats_path = f"{RESULTS_DIR}/scores/calibration_stats.json"
    import json
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Calibration stats fit on {len(calib_df)} samples, saved to {stats_path}")
    return stats


def apply_calibration(stats, t_grid):
    """
    Reads scores.csv, computes z-scores for 'heldout' and 'ood' rows using calib stats.
    Returns a DataFrame with added z-score columns. Never touches 'calib' rows for fitting again.
    """
    df = pd.read_csv(SCORES_CSV)
    eval_df = df[df["split"].isin(["heldout", "ood"])].copy()

    for t_val in t_grid:
        col = f"score_t{t_val}"
        m, s = stats[t_val]["mean"], stats[t_val]["std"]
        eval_df[f"z_t{t_val}"] = (eval_df[col] - m) / (s + 1e-8)

    m, s = stats["overall"]["mean"], stats["overall"]["std"]
    eval_df["z_overall"] = (eval_df["score_mean"] - m) / (s + 1e-8)

    out_path = f"{RESULTS_DIR}/scores/calibrated_scores.csv"
    eval_df.to_csv(out_path, index=False)
    print(f"Calibrated {len(eval_df)} rows, saved to {out_path}")
    return eval_df

def compute_split_stats(eval_df):
    """Compute summary stats for score_mean and z_overall."""
    return {
        "count": len(eval_df),
        "score_mean": {
            "mean": float(eval_df["score_mean"].mean()),
            "std": float(eval_df["score_mean"].std()),
        },
        "z_overall": {
            "mean": float(eval_df["z_overall"].mean()),
            "std": float(eval_df["z_overall"].std()),
        },
    }

def compute_auroc_by_category(eval_df, manifest_path="DoTA_prepared/manifest_subset1500.json"):
    """
    Breaks down AUROC by DoTA anomaly_name, joining eval_df's OOD rows back to the
    manifest (since anomaly_name wasn't logged directly into scores.csv this run).
    Uses ALL heldout-normal rows as the negative class for every category comparison.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    clip_to_anomaly = {c["clip_id"]: c.get("anomaly_name", "unknown") for c in manifest}

    df = eval_df.copy()
    df["clip_id"] = df.apply(
        lambda row: row["sample_id"].rsplit("_ood", 1)[0] if row["label"] == "anomaly"
        else row["sample_id"].rsplit("_nonood", 1)[0],
        axis=1
    )
    df["anomaly_name"] = df["clip_id"].map(clip_to_anomaly)

    normal_df = df[df["label"] == "normal"]
    anomaly_df = df[df["label"] == "anomaly"]

    print("\nPer-category AUROC (this category's anomalies vs all heldout-normal):")
    results = []
    for name, group in anomaly_df.groupby("anomaly_name"):
        if len(group) < 5:
            continue 
        combined = pd.concat([normal_df, group])
        y_true = (combined["label"] == "anomaly").astype(int)
        y_score = combined["z_overall"]
        try:
            auc = roc_auc_score(y_true, y_score)
            results.append((name, len(group), auc))
        except ValueError:
            continue

    results.sort(key=lambda x: x[2], reverse=True)
    for name, n, auc in results:
        print(f"  {name:25s} n={n:4d}  AUROC={auc:.4f}")

    return results

def split_calibrated_scores(calibrated_scores_path=f"{RESULTS_DIR}/scores/calibrated_scores.csv", out_dir=f"{RESULTS_DIR}/scores"):
    """Split calibrated_scores into heldout/ood CSVs and write stats JSON files."""
    df = pd.read_csv(calibrated_scores_path)
    results = {}

    for split in ["heldout", "ood"]:
        subset = df[df["split"] == split].copy()
        subset_path = os.path.join(out_dir, f"{split}_calibrated_scores.csv")
        stats_path = os.path.join(out_dir, f"{split}_stats.json")

        subset.to_csv(subset_path, index=False)
        stats = compute_split_stats(subset)

        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        results[split] = {
            "csv_path": subset_path,
            "stats_path": stats_path,
            "stats": stats,
        }
        print(f"Saved {len(subset)} {split} rows to {subset_path} and stats to {stats_path}")

    return results


def compute_auroc(eval_df):
    heldout_ood_df = eval_df[eval_df["split"].isin(["heldout", "ood"])]
    y_true = (heldout_ood_df["label"] == "anomaly").astype(int)
    y_score = heldout_ood_df["z_overall"]
    auc = roc_auc_score(y_true, y_score)
    print(f"AUROC (heldout-normal vs ood): {auc:.4f}")
    return auc

if __name__ == "__main__":

    exp_dir = "logs_wm/orbis_288x512"
    cfg = OmegaConf.load(f"{exp_dir}/config.yaml")
    model = instantiate_from_config(cfg.model)
    state = torch.load(f"{exp_dir}/checkpoints/last.ckpt", map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to("cuda").eval()  

    t_grid = [0.3, 0.5, 0.7, 0.9]

    generate_heatmaps_for_clips(model, HEATMAP_CLIP_IDS, t_grid, n_noise_samples=4, device="cuda")

    with open("DoTA_prepared/manifest_subset1500.json") as f:
        manifest = json.load(f)

    completed = set()
    if os.path.exists(SCORES_CSV):
        completed = set(pd.read_csv(SCORES_CSV)["sample_id"])

    for i, clip in enumerate(manifest):
        clip_id = clip["clip_id"]
        clip_dir = Path("DoTA_prepared") / clip_id

        nonood_id = f"{clip_id}_nonood"
        ood_id = f"{clip_id}_ood"

        if clip["non_ood_split"] != "unused_non_ood" and nonood_id not in completed:
            process_and_log(model, str(clip_dir / "non-ood"), sample_id=nonood_id,
                             split=clip["non_ood_split"], label="normal",
                             t_grid=t_grid, n_noise_samples=3, save_map=False)

        if ood_id not in completed:
            process_and_log(model, str(clip_dir / "ood"), sample_id=ood_id,
                             split=clip["ood_split"], label="anomaly",
                             t_grid=t_grid, n_noise_samples=3, save_map=False)

        if (i + 1) % 100 == 0:
            print(f"--- Progress: {i + 1}/{len(manifest)} clips ---")

    stats = fit_calibration(t_grid)
    eval_df= apply_calibration(stats, t_grid)
    split_calibrated_scores()
    compute_auroc(apply_calibration(stats, t_grid))

    compute_auroc_by_category(eval_df, manifest_path="DoTA_prepared/manifest_subset1500.json")
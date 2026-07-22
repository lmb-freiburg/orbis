import glob
import json
import os
import sys
from pathlib import Path

# Enable CPU fallback for unsupported MPS operators before torch imports.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

PYTORCH_ENABLE_MPS_FALLBACK=1

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


def load_clip_as_window(frame_paths, size=(512, 288)):
    idxs = [0, 2, 4, 6, 8, 10]  # 11 frames @10fps -> 6 frames @5fps
    assert len(frame_paths) == 11, f"expected 11 frames, got {len(frame_paths)}"
    frames = []
    for i in idxs:
        img = cv2.cvtColor(cv2.imread(frame_paths[i]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
        frames.append(t)
    return torch.stack(frames)


# ---------- scorer (factorized: detail + semantic streams) ----------
@torch.no_grad()
def surprise_score_factorized(model, images, frame_rate, t_grid, n_noise_samples=8, use_ema=False):
    net = model.ema_vit if use_ema else model.vit
    net.eval()
    x = model.encode_frames(images) 
    context, target = x[:, :-1].clone(), x[:, -1:]
    # context - [1, 5, 32, 18, 32] target - [1, 1, 32, 18, 32]
    b = x.shape[0]
    n_channels = target.shape[2]
    half = n_channels // 2

    per_t_detail, per_t_semantic = [], []
    per_t_map_detail, per_t_map_semantic = {}, {}

    for t_val in t_grid:
        errs = []
        for _ in range(n_noise_samples):
            t = torch.full((b,), t_val, device=x.device)
            target_t, noise = model.add_noise(target, t) # CHECK - 1
            pred = net(target_t, context, t, frame_rate=frame_rate)
            # pred shape torch.Size([1, 1, 32, 18, 32])
            true_v = model.A(t) * target + model.B(t) * noise # CHECK -2
            # true_v shape torch.Size([1, 1, 32, 18, 32])
            errs.append((pred.float() - true_v.float()) ** 2)  
            
        err_avg = torch.stack(errs).mean(0) # err_avg shape: torch.Size([1, 1, 32, 18, 32])
        print(f"err_avg shape: {err_avg.shape}, half: {half}")
        detail_err = err_avg[:, :, :half]
        semantic_err = err_avg[:, :, half:]
        # detail_err shape: torch.Size([1, 1, 16, 18, 32]), semantic_err shape: torch.Size([1, 1, 16, 18, 32])

        per_t_detail.append(detail_err.mean(dim=[1, 2, 3, 4]))
        per_t_semantic.append(semantic_err.mean(dim=[1, 2, 3, 4]))  
        per_t_map_detail[t_val] = detail_err.mean(dim=2).squeeze(1)   
        per_t_map_semantic[t_val] = semantic_err.mean(dim=2).squeeze(1)

    raw_detail = torch.stack(per_t_detail, dim=1).mean(dim=1)
    raw_semantic = torch.stack(per_t_semantic, dim=1).mean(dim=1)

    return raw_detail, raw_semantic, per_t_detail, per_t_semantic, per_t_map_detail, per_t_map_semantic


# ---------- logging ----------
def log_sample_result(sample_id, split, label, per_t_detail_values, per_t_semantic_values,
                       t_grid, detail_mean, semantic_mean, error_map_detail=None, error_map_semantic=None):
    row = {"sample_id": sample_id, "split": split, "label": label,
           "score_detail_mean": detail_mean, "score_semantic_mean": semantic_mean}
    for t_val, d_score, s_score in zip(t_grid, per_t_detail_values, per_t_semantic_values):
        row[f"score_detail_t{t_val}"] = d_score
        row[f"score_semantic_t{t_val}"] = s_score

    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(SCORES_CSV)
    df_row.to_csv(SCORES_CSV, mode="a", header=write_header, index=False)

    if error_map_detail is not None:
        save_dict = {f"detail_t{t}": arr for t, arr in error_map_detail.items()}
        save_dict.update({f"semantic_t{t}": arr for t, arr in error_map_semantic.items()})
        np.savez_compressed(f"{RESULTS_DIR}/error_maps/{sample_id}.npz", **save_dict)

    print(f"Logged {sample_id} ({split}/{label}) -> detail={detail_mean:.5f} semantic={semantic_mean:.5f}")


def process_and_log(model, folder, sample_id, split, label, t_grid, n_noise_samples=8, save_map=False, device=None):
    if device is None:
        device = get_device()

    paths = get_sorted_frame_paths(folder)
    window = load_clip_as_window(paths).unsqueeze(0).to(device)
    frame_rate = torch.full((1,), 5.0).to(device)

    raw_detail, raw_semantic, per_t_detail, per_t_semantic, per_t_map_d, per_t_map_s = \
        surprise_score_factorized(model, window, frame_rate, t_grid, n_noise_samples)

    per_t_detail_vals = [s[0].item() for s in per_t_detail]
    per_t_semantic_vals = [s[0].item() for s in per_t_semantic]

    error_map_d = {t: per_t_map_d[t][0].cpu().numpy() for t in t_grid} if save_map else None
    error_map_s = {t: per_t_map_s[t][0].cpu().numpy() for t in t_grid} if save_map else None

    log_sample_result(sample_id, split, label, per_t_detail_vals, per_t_semantic_vals,
                       t_grid, raw_detail[0].item(), raw_semantic[0].item(),
                       error_map_d, error_map_s)


@torch.no_grad()
def compute_and_save_heatmap(model, folder, out_sample_id, t_grid, n_noise_samples, device):
    """Scores ONE clip window and saves BOTH streams' spatial error maps. Does NOT touch scores.csv."""
    paths = get_sorted_frame_paths(folder)
    window = load_clip_as_window(paths).unsqueeze(0).to(device)
    frame_rate = torch.full((1,), 5.0).to(device)

    raw_detail, raw_semantic, per_t_detail, per_t_semantic, per_t_map_d, per_t_map_s = \
        surprise_score_factorized(model, window, frame_rate, t_grid, n_noise_samples)

    save_dict = {f"detail_t{t}": per_t_map_d[t][0].cpu().numpy() for t in t_grid}
    save_dict.update({f"semantic_t{t}": per_t_map_s[t][0].cpu().numpy() for t in t_grid})
    np.savez_compressed(f"{RESULTS_DIR}/error_maps/{out_sample_id}.npz", **save_dict)

    print(f"Saved heatmap for {out_sample_id} -> detail={raw_detail[0].item():.5f} "
          f"semantic={raw_semantic[0].item():.5f}")
    return raw_detail[0].item(), raw_semantic[0].item()


def generate_heatmaps_for_clips(model, clip_ids, t_grid, n_noise_samples=3,
                                 base_dir="DoTA_oncoming", device=None):
    if device is None:
        device = get_device()
    for clip_id in clip_ids:
        clip_dir = Path(base_dir) / clip_id
        compute_and_save_heatmap(model, str(clip_dir / "non-ood"), f"{clip_id}_nonood_heatmap",
                                  t_grid, n_noise_samples, device)
        compute_and_save_heatmap(model, str(clip_dir / "ood"), f"{clip_id}_ood_heatmap",
                                  t_grid, n_noise_samples, device)


# ---------- Stage 1: calibration ----------
def fit_calibration(t_grid, min_calib_samples=30):
    df = pd.read_csv(SCORES_CSV)
    calib_df = df[df["split"] == "calib"]

    if len(calib_df) < min_calib_samples:
        print(f"WARNING: only {len(calib_df)} calibration samples found "
              f"(recommend >= {min_calib_samples}).")

    stats = {"detail": {}, "semantic": {}}
    for stream in ["detail", "semantic"]:
        for t_val in t_grid:
            col = f"score_{stream}_t{t_val}"
            stats[stream][str(t_val)] = {"mean": float(calib_df[col].mean()), "std": float(calib_df[col].std())}
        stats[stream]["overall"] = {
            "mean": float(calib_df[f"score_{stream}_mean"].mean()),
            "std": float(calib_df[f"score_{stream}_mean"].std()),
        }

    with open(f"{RESULTS_DIR}/scores/calibration_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Calibration stats fit on {len(calib_df)} samples (detail + semantic streams)")
    return stats


# ---------- Stage 2: apply calibration ----------
def apply_calibration(stats, t_grid):
    df = pd.read_csv(SCORES_CSV)
    eval_df = df[df["split"].isin(["heldout", "ood"])].copy()

    for stream in ["detail", "semantic"]:
        for t_val in t_grid:
            col = f"score_{stream}_t{t_val}"
            m, s = stats[stream][str(t_val)]["mean"], stats[stream][str(t_val)]["std"]
            eval_df[f"z_{stream}_t{t_val}"] = (eval_df[col] - m) / (s + 1e-8)
        m, s = stats[stream]["overall"]["mean"], stats[stream]["overall"]["std"]
        eval_df[f"z_{stream}_overall"] = (eval_df[f"score_{stream}_mean"] - m) / (s + 1e-8)

    eval_df.to_csv(f"{RESULTS_DIR}/scores/calibrated_scores.csv", index=False)
    print(f"Calibrated {len(eval_df)} rows (detail + semantic)")
    return eval_df


def compute_split_stats(eval_df, stream):
    return {
        "count": len(eval_df),
        "score_mean": {
            "mean": float(eval_df[f"score_{stream}_mean"].mean()),
            "std": float(eval_df[f"score_{stream}_mean"].std()),
        },
        "z_overall": {
            "mean": float(eval_df[f"z_{stream}_overall"].mean()),
            "std": float(eval_df[f"z_{stream}_overall"].std()),
        },
    }


def split_calibrated_scores(calibrated_scores_path=f"{RESULTS_DIR}/scores/calibrated_scores.csv",
                             out_dir=f"{RESULTS_DIR}/scores"):
    df = pd.read_csv(calibrated_scores_path)
    results = {}
    for split in ["heldout", "ood"]:
        subset = df[df["split"] == split].copy()
        subset_path = os.path.join(out_dir, f"{split}_calibrated_scores.csv")
        subset.to_csv(subset_path, index=False)

        stream_stats = {}
        for stream in ["detail", "semantic"]:
            stream_stats[stream] = compute_split_stats(subset, stream)

        stats_path = os.path.join(out_dir, f"{split}_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stream_stats, f, indent=2)

        results[split] = {"csv_path": subset_path, "stats_path": stats_path, "stats": stream_stats}
        print(f"Saved {len(subset)} {split} rows to {subset_path} and stats to {stats_path}")
    return results


def compute_auroc(eval_df, stream="semantic"):
    y_true = (eval_df["label"] == "anomaly").astype(int)
    y_score = eval_df[f"z_{stream}_overall"]
    auc = roc_auc_score(y_true, y_score)
    print(f"AUROC [{stream}] (heldout-normal vs ood): {auc:.4f}")
    return auc


def compute_auroc_by_category(eval_df, manifest_path, stream="semantic"):
    with open(manifest_path) as f:
        manifest = json.load(f)
    clip_to_anomaly = {c["clip_id"]: c.get("anomaly_name", "unknown") for c in manifest}

    df = eval_df.copy()
    df["clip_id"] = df.apply(
        lambda row: row["sample_id"].rsplit("_ood", 1)[0] if row["label"] == "anomaly"
        else row["sample_id"].rsplit("_nonood", 1)[0], axis=1)
    df["anomaly_name"] = df["clip_id"].map(clip_to_anomaly)

    normal_df = df[df["label"] == "normal"]
    anomaly_df = df[df["label"] == "anomaly"]

    print(f"\nPer-category AUROC [{stream}]:")
    results = []
    for name, group in anomaly_df.groupby("anomaly_name"):
        if len(group) < 5:
            continue
        combined = pd.concat([normal_df, group])
        y_true = (combined["label"] == "anomaly").astype(int)
        y_score = combined[f"z_{stream}_overall"]
        try:
            auc = roc_auc_score(y_true, y_score)
            results.append((name, len(group), auc))
        except ValueError:
            continue
    results.sort(key=lambda x: x[2], reverse=True)
    for name, n, auc in results:
        print(f"  {name:25s} n={n:4d}  AUROC={auc:.4f}")
    return results

# --------- test main ------------
if __name__ == "__main__":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    exp_dir = "logs_wm/orbis_288x512"
    device = get_device()
    print(f"Using device: {device}")

    cfg = OmegaConf.load(f"{exp_dir}/config.yaml")
    model = instantiate_from_config(cfg.model)
    state = torch.load(f"{exp_dir}/checkpoints/last.ckpt", map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    t_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # clip 1 - uO2zGO5ydBc_001446
    # clip 2 - 3u_CIo9IaWo_002136
    test_clip_id = "uO2zGO5ydBc_001446"
    generate_heatmaps_for_clips(model, [test_clip_id], t_grid, n_noise_samples=3, device=device)

# ---------- main ----------
# if __name__ == "__main__":
#     exp_dir = "logs_wm/orbis_288x512"
#     cfg = OmegaConf.load(f"{exp_dir}/config.yaml")
#     model = instantiate_from_config(cfg.model)
#     state = torch.load(f"{exp_dir}/checkpoints/last.ckpt", map_location="cpu", weights_only=True)["state_dict"]
#     model.load_state_dict(state, strict=True)
#     device = get_device()   # instead of hardcoding "cuda"
#     model = model.to(device).eval()

#     t_grid = [0.3, 0.5, 0.7, 0.9]

#     generate_heatmaps_for_clips(model, HEATMAP_CLIP_IDS[:1], t_grid, n_noise_samples=4, device=device)

#     with open("DoTA_prepared/manifest_subset1500.json") as f:
#         manifest = json.load(f)

#     completed = set()
#     if os.path.exists(SCORES_CSV):
#         completed = set(pd.read_csv(SCORES_CSV)["sample_id"])

#     for i, clip in enumerate(manifest):
#         clip_id = clip["clip_id"]
#         clip_dir = Path("DoTA_prepared") / clip_id

#         nonood_id = f"{clip_id}_nonood"
#         ood_id = f"{clip_id}_ood"

#         if clip["non_ood_split"] != "unused_non_ood" and nonood_id not in completed:
#             process_and_log(model, str(clip_dir / "non-ood"), sample_id=nonood_id,
#                              split=clip["non_ood_split"], label="normal",
#                              t_grid=t_grid, n_noise_samples=3, save_map=False)

#         if ood_id not in completed:
#             process_and_log(model, str(clip_dir / "ood"), sample_id=ood_id,
#                              split=clip["ood_split"], label="anomaly",
#                              t_grid=t_grid, n_noise_samples=3, save_map=False)

#         if (i + 1) % 100 == 0:
#             print(f"--- Progress: {i + 1}/{len(manifest)} clips ---")

#     stats = fit_calibration(t_grid)
#     eval_df = apply_calibration(stats, t_grid)
#     split_calibrated_scores()

#     for stream in ["detail", "semantic"]:
#         compute_auroc(eval_df, stream=stream)
#         compute_auroc_by_category(eval_df, manifest_path="DoTA_prepared/manifest_subset1500.json", stream=stream)
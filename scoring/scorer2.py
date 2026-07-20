import glob
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from util import instantiate_from_config

RESULTS_DIR = "results_pt"
os.makedirs(f"{RESULTS_DIR}/samples", exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_sorted_frame_paths(folder):
    paths = sorted(glob.glob(f"{folder}/*.jpg"))
    assert len(paths) == 11, f"expected 11 frames in {folder}, found {len(paths)}"
    return paths


def load_clip_as_window(frame_paths, size=(512, 288)):
    idxs = [0, 2, 4, 6, 8, 10]
    frames = []
    for i in idxs:
        img = cv2.cvtColor(cv2.imread(frame_paths[i]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
        frames.append(t)
    return torch.stack(frames)


# ---------- scorer: semantic stream only, keeps full per-token map ----------
@torch.no_grad()
def surprise_score_semantic(model, images, frame_rate, t_grid, n_noise_samples=8, use_ema=False):
    net = model.ema_vit if use_ema else model.vit
    net.eval()
    x = model.encode_frames(images)
    context, target = x[:, :-1].clone(), x[:, -1:]
    b = x.shape[0]
    n_channels = target.shape[2]
    half = n_channels // 2

    per_t_map = {}     # {t_val: [H, W] tensor} — semantic only, channel-collapsed
    per_t_scalar = {}   # {t_val: scalar float}

    for t_val in t_grid:
        errs = []
        for _ in range(n_noise_samples):
            t = torch.full((b,), t_val, device=x.device)
            target_t, noise = model.add_noise(target, t)
            pred = net(target_t, context, t, frame_rate=frame_rate)
            true_v = model.A(t) * target + model.B(t) * noise
            errs.append((pred.float() - true_v.float()) ** 2)  # [B, 1, C, H, W]
        err_avg = torch.stack(errs).mean(0)
        semantic_err = err_avg[:, :, half:]  # [B, 1, C/2, H, W]

        semantic_map = semantic_err.mean(dim=2).squeeze(0).squeeze(0)  # [H, W], B=1 assumed
        per_t_map[t_val] = semantic_map.cpu()
        per_t_scalar[t_val] = semantic_map.mean().item()

    return per_t_map, per_t_scalar


def process_and_save_pt(model, folder, sample_id, split, label, t_grid, n_noise_samples, device):
    """Runs scoring for one clip window, saves full per-token maps + scalars to a .pt file.
    
    If output file already exists, skips model inference and loads existing payload.
    """
    out_path = Path(RESULTS_DIR) / "samples" / f"{sample_id}.pt"
    
    # Check if the sample has already been processed and saved
    if out_path.exists():
        return torch.load(out_path, map_location="cpu", weights_only=True)

    paths = get_sorted_frame_paths(folder)
    window = load_clip_as_window(paths).unsqueeze(0).to(device)
    frame_rate = torch.full((1,), 5.0).to(device)

    per_t_map, per_t_scalar = surprise_score_semantic(model, window, frame_rate, t_grid, n_noise_samples)

    payload = {
        "sample_id": sample_id, "split": split, "label": label,
        "per_t_map": per_t_map,       # dict {t_val: [H, W] CPU tensor}
        "per_t_scalar": per_t_scalar,  # dict {t_val: float}
        "score_mean": sum(per_t_scalar.values()) / len(per_t_scalar),
    }
    torch.save(payload, out_path)
    return payload


# ---------- Welford running mean/variance, per-token, per t ----------
class WelfordAccumulator:
    def __init__(self, shape, t_grid):
        self.t_grid = t_grid
        self.n = {t: 0 for t in t_grid}
        self.mean = {t: torch.zeros(shape) for t in t_grid}
        self.M2 = {t: torch.zeros(shape) for t in t_grid}

    def update(self, t_val, new_map):
        """new_map: [H, W] tensor for this one sample, this one t."""
        self.n[t_val] += 1
        n = self.n[t_val]
        delta = new_map - self.mean[t_val]
        self.mean[t_val] += delta / n
        delta2 = new_map - self.mean[t_val]
        self.M2[t_val] += delta * delta2

    def finalize(self, eps=1e-8):
        """Returns {t_val: {"mean": [H,W], "std": [H,W], "n": int}}"""
        stats = {}
        for t_val in self.t_grid:
            n = self.n[t_val]
            var = self.M2[t_val] / max(n - 1, 1)  # sample variance
            stats[t_val] = {
                "mean": self.mean[t_val],
                "std": torch.sqrt(var + eps),
                "n": n,
            }
        return stats


if __name__ == "__main__":
    exp_dir = "logs_wm/orbis_288x512"
    device = get_device()
    print(f"Using device: {device}")

    cfg = OmegaConf.load(f"{exp_dir}/config.yaml")
    model = instantiate_from_config(cfg.model)
    state = torch.load(f"{exp_dir}/checkpoints/last.ckpt", map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    t_grid = [0.2, 0.4, 0.6, 0.8]
    N_NOISE_SAMPLES = 3

    with open("DoTA_prepared/manifest_subset1500.json") as f:
        manifest = json.load(f)

    # infer map shape from one sample first, so the accumulator is the right size
    first_clip = manifest[0]["clip_id"]
    first_folder = str(Path("DoTA_prepared") / first_clip / "non-ood")
    probe_map, _ = surprise_score_semantic(
        model,
        load_clip_as_window(get_sorted_frame_paths(first_folder)).unsqueeze(0).to(device),
        torch.full((1,), 5.0).to(device), t_grid, n_noise_samples=1,
    )
    map_shape = probe_map[t_grid[0]].shape  # (H, W)
    print(f"Per-token map shape: {map_shape}")

    accumulator = WelfordAccumulator(map_shape, t_grid)

    for clip in tqdm(manifest, desc="Processing clips"):
        clip_id = clip["clip_id"]
        clip_dir = Path("DoTA_prepared") / clip_id
        nonood_id = f"{clip_id}_nonood"
        ood_id = f"{clip_id}_ood"

        if clip["non_ood_split"] != "unused_non_ood":
            payload = process_and_save_pt(model, str(clip_dir / "non-ood"), nonood_id,
                                           clip["non_ood_split"], "normal", t_grid,
                                           N_NOISE_SAMPLES, device)
            if clip["non_ood_split"] == "calib":
                for t_val in t_grid:
                    accumulator.update(t_val, payload["per_t_map"][t_val])

        process_and_save_pt(model, str(clip_dir / "ood"), ood_id, clip["ood_split"],
                             "anomaly", t_grid, N_NOISE_SAMPLES, device)

    calib_stats = accumulator.finalize()
    torch.save(calib_stats, f"{RESULTS_DIR}/calib_stats.pt")
    print(f"Saved per-token calibration stats (n={calib_stats[t_grid[0]]['n']} calib samples)")
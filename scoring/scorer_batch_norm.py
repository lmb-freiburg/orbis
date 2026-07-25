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
os.makedirs(f"{RESULTS_DIR}", exist_ok=True)


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


# ---------- scorer: detailed and combined heads only ----------
@torch.no_grad()
def surprise_score_detailed_combined(model, images, frame_rate, t_grid, n_noise_samples=2, use_ema=False):
    net = model.ema_vit if use_ema else model.vit
    net.eval()
    
    x = model.encode_frames(images)
    context, target = x[:, :-1].clone(), x[:, -1:]
    b = x.shape[0]
    n_channels = target.shape[2]
    half = n_channels // 2

    context_exp = context.repeat_interleave(n_noise_samples, dim=0)
    target_exp = target.repeat_interleave(n_noise_samples, dim=0)

    per_t_maps = {
        "detailed": {},
        "combined": {}
    }

    for t_val in t_grid:
        t = torch.full((b * n_noise_samples,), t_val, device=x.device)
        
        target_t, noise = model.add_noise(target_exp, t)
        
        fr = frame_rate.repeat_interleave(n_noise_samples, dim=0) if frame_rate.numel() > 1 else frame_rate
        pred = net(target_t, context_exp, t, frame_rate=fr)
        
        true_v = model.A(t) * target_exp + model.B(t) * noise
        
        err = (pred.float() - true_v.float()) ** 2  # [B*N, 1, C, H, W]
        
        # Reshape and average across noise samples: [B, 1, C, H, W]
        err_avg = err.view(b, n_noise_samples, 1, n_channels, err.shape[-2], err.shape[-1]).mean(dim=1)
        
        detailed_err = err_avg[:, :, :half]
        
        # Mean across channels -> squeeze to [H, W] (B=1)
        detailed_map = detailed_err.mean(dim=2).squeeze(0).squeeze(0)
        combined_map = err_avg.mean(dim=2).squeeze(0).squeeze(0)

        per_t_maps["detailed"][t_val] = detailed_map
        per_t_maps["combined"][t_val] = combined_map

    return per_t_maps


# ---------- Welford running mean/variance for multi-head metrics ----------
class MultiHeadWelfordAccumulator:
    def __init__(self, shape, t_grid, heads, device):
        self.shape = shape
        self.t_grid = t_grid
        self.heads = heads
        self.device = device
        self.n = {t: 0 for t in t_grid}
        
        self.mean = {h: {t: torch.zeros(shape, device=device) for t in t_grid} for h in heads}
        self.M2 = {h: {t: torch.zeros(shape, device=device) for t in t_grid} for h in heads}

    def update(self, t_val, head_maps):
        """head_maps: dict mapping head_name -> [H, W] tensor for this sample and t."""
        self.n[t_val] += 1
        n = self.n[t_val]
        
        for h in self.heads:
            new_map = head_maps[h]
            delta = new_map - self.mean[h][t_val]
            self.mean[h][t_val] += delta / n
            delta2 = new_map - self.mean[h][t_val]
            self.M2[h][t_val] += delta * delta2

    def state_dict(self):
        return {
            "n": self.n,
            "mean": {h: {t: self.mean[h][t].cpu() for t in self.t_grid} for h in self.heads},
            "M2": {h: {t: self.M2[h][t].cpu() for t in self.t_grid} for h in self.heads},
        }

    def load_state_dict(self, state):
        self.n = state["n"]
        for h in self.heads:
            for t in self.t_grid:
                self.mean[h][t] = state["mean"][h][t].to(self.device)
                self.M2[h][t] = state["M2"][h][t].to(self.device)

    def finalize(self, eps=1e-8):
        stats = {h: {} for h in self.heads}
        for h in self.heads:
            for t_val in self.t_grid:
                n = self.n[t_val]
                var = self.M2[h][t_val] / max(n - 1, 1)
                stats[h][t_val] = {
                    "mean": self.mean[h][t_val].cpu(),
                    "std": torch.sqrt(var + eps).cpu(),
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
    N_NOISE_SAMPLES = 2
    HEADS = ["detailed", "combined"]
    
    CKPT_PATH = f"{RESULTS_DIR}/calib_checkpoint_det_comb.pt"

    with open("DoTA_prepared/manifest_subset1500.json") as f:
        manifest = json.load(f)

    # Infer map shape from first sample
    first_clip = manifest[0]["clip_id"]
    first_folder = str(Path("DoTA_prepared") / first_clip / "non-ood")
    probe_maps = surprise_score_detailed_combined(
        model,
        load_clip_as_window(get_sorted_frame_paths(first_folder)).unsqueeze(0).to(device),
        torch.full((1,), 5.0).to(device), t_grid, n_noise_samples=1,
    )
    map_shape = probe_maps["detailed"][t_grid[0]].shape  # (H, W)
    print(f"Per-token map shape: {map_shape}")

    accumulator = MultiHeadWelfordAccumulator(map_shape, t_grid, HEADS, device)
    processed_clips = set()

    # Resume from checkpoint if it exists
    if os.path.exists(CKPT_PATH):
        print(f"Found checkpoint at {CKPT_PATH}! Resuming...")
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        accumulator.load_state_dict(ckpt["accumulator"])
        processed_clips = set(ckpt["processed_clips"])
        print(f"Resuming from clip {len(processed_clips)} / {len(manifest)}")

    for clip in tqdm(manifest, desc="Processing calib clips"):
        clip_id = clip["clip_id"]
        if clip.get("non_ood_split") == "calib":
            if clip_id in processed_clips:
                continue

            clip_dir = Path("DoTA_prepared") / clip_id
            folder = str(clip_dir / "non-ood")
            
            paths = get_sorted_frame_paths(folder)
            window = load_clip_as_window(paths).unsqueeze(0).to(device)
            frame_rate = torch.full((1,), 5.0).to(device)
            
            per_t_maps = surprise_score_detailed_combined(model, window, frame_rate, t_grid, N_NOISE_SAMPLES)
            
            for t_val in t_grid:
                head_maps_at_t = {h: per_t_maps[h][t_val] for h in HEADS}
                accumulator.update(t_val, head_maps_at_t)

            processed_clips.add(clip_id)

            # Periodic checkpointing
            if len(processed_clips) % 50 == 0:
                torch.save({
                    "accumulator": accumulator.state_dict(),
                    "processed_clips": list(processed_clips)
                }, CKPT_PATH)

    calib_stats = accumulator.finalize()
    
    # Save detailed and combined separately
    torch.save(calib_stats, f"{RESULTS_DIR}/calib_stats_detailed_combined.pt")
    
    # Remove checkpoint on success
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)

    print(f"Saved detailed and combined calibration stats (n={calib_stats['detailed'][t_grid[0]]['n']} samples)")
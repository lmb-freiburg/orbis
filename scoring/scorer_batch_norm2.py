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


# ---------- parameterized scorer: runs only specified inference passes ----------
@torch.no_grad()
def surprise_score(model, images, frame_rate, t_grid, heads, n_noise_samples=2, use_ema=False):
    net = model.ema_vit if use_ema else model.vit
    net.eval()
    
    x = model.encode_frames(images)
    context, target = x[:, :-1].clone(), x[:, -1:]
    b = x.shape[0]
    n_channels = target.shape[2]
    half = n_channels // 2

    context_exp = context.repeat_interleave(n_noise_samples, dim=0)
    target_exp = target.repeat_interleave(n_noise_samples, dim=0)

    # Dynamically initialize output based on requested heads
    per_t_maps = {h: {} for h in heads}
    is_mps = x.device.type == "mps"

    for t_val in t_grid:
        t = torch.full((b * n_noise_samples,), t_val, device=x.device)
        fr = frame_rate.repeat_interleave(n_noise_samples, dim=0) if frame_rate.numel() > 1 else frame_rate

        # Helper function to run a single isolated pass
        def run_inference_pass(ctx, tgt, err_start_idx, err_end_idx):
            tgt_t, noise = model.add_noise(tgt, t)
            
            if is_mps:
                # with torch.autocast(device_type="mps", dtype=torch.float16):
                pred = net(tgt_t, ctx, t, frame_rate=fr)
            else:
                pred = net(tgt_t, ctx, t, frame_rate=fr)
            
            true_v = model.A(t) * tgt + model.B(t) * noise
            err = (pred.float() - true_v.float()) ** 2
            
            # Reshape: [B, N_noise, 1, C, H, W] -> average across noise samples
            err_avg = err.view(b, n_noise_samples, 1, n_channels, err.shape[-2], err.shape[-1]).mean(dim=1)
            
            # Extract MSE only for the active channels we are evaluating
            active_err = err_avg[:, :, err_start_idx:err_end_idx]
            
            # Channel mean -> squeeze to [H, W]
            return active_err.mean(dim=2).squeeze(0).squeeze(0)

        # ---------------------------------------------------------
        # PASS 1: DETAILED ONLY
        # ---------------------------------------------------------
        if "detailed" in heads:
            ctx_detailed = context_exp.clone()
            tgt_detailed = target_exp.clone()
            ctx_detailed[:, :, half:, :, :] = 0  
            tgt_detailed[:, :, half:, :, :] = 0  
            
            per_t_maps["detailed"][t_val] = run_inference_pass(
                ctx_detailed, tgt_detailed, err_start_idx=0, err_end_idx=half
            )

        # ---------------------------------------------------------
        # PASS 2: SEMANTIC ONLY
        # ---------------------------------------------------------
        if "semantic" in heads:
            ctx_semantic = context_exp.clone()
            tgt_semantic = target_exp.clone()
            ctx_semantic[:, :, :half, :, :] = 0
            tgt_semantic[:, :, :half, :, :] = 0
            
            per_t_maps["semantic"][t_val] = run_inference_pass(
                ctx_semantic, tgt_semantic, err_start_idx=half, err_end_idx=n_channels
            )

        # ---------------------------------------------------------
        # PASS 3: COMBINED
        # ---------------------------------------------------------
        if "combined" in heads:
            per_t_maps["combined"][t_val] = run_inference_pass(
                context_exp, target_exp, err_start_idx=0, err_end_idx=n_channels
            )

    return per_t_maps


# ---------- Welford running mean/variance ----------
class MultiHeadWelfordAccumulator:
    def __init__(self, shape, t_grid, heads, device):
        self.t_grid = t_grid
        self.heads = heads
        self.n = {t: 0 for t in t_grid}
        
        self.mean = {h: {t: torch.zeros(shape, device=device) for t in t_grid} for h in heads}
        self.M2 = {h: {t: torch.zeros(shape, device=device) for t in t_grid} for h in heads}

    def update(self, t_val, head_maps):
        self.n[t_val] += 1
        n = self.n[t_val]
        
        for h in self.heads:
            new_map = head_maps[h]
            delta = new_map - self.mean[h][t_val]
            self.mean[h][t_val] += delta / n
            delta2 = new_map - self.mean[h][t_val]
            self.M2[h][t_val] += delta * delta2

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
    
    # Configure your variants here. The calculation loop will dynamically adjust.
    HEADS = ["detailed", "semantic"]

    with open("DoTA_prepared/manifest_subset1500.json") as f:
        manifest = json.load(f)

    # Infer map shape from first sample
    first_clip = manifest[0]["clip_id"]
    first_folder = str(Path("DoTA_prepared") / first_clip / "non-ood")
    probe_maps = surprise_score(
        model,
        load_clip_as_window(get_sorted_frame_paths(first_folder)).unsqueeze(0).to(device),
        torch.full((1,), 5.0).to(device), 
        t_grid,
        heads=HEADS,
        n_noise_samples=1,
    )
    
    # Use the first requested head to grab the shape map dynamically
    map_shape = probe_maps[HEADS[0]][t_grid[0]].shape
    print(f"Per-token map shape: {map_shape}")

    accumulator = MultiHeadWelfordAccumulator(map_shape, t_grid, HEADS, device)

    for i, clip in enumerate(tqdm(manifest, desc="Processing calib clips")):
        if clip.get("non_ood_split") == "calib":
            clip_dir = Path("DoTA_prepared") / clip["clip_id"]
            folder = str(clip_dir / "non-ood")
            
            paths = get_sorted_frame_paths(folder)
            window = load_clip_as_window(paths).unsqueeze(0).to(device)
            frame_rate = torch.full((1,), 5.0).to(device)
            
            per_t_maps = surprise_score(model, window, frame_rate, t_grid, heads=HEADS, n_noise_samples=N_NOISE_SAMPLES)
            
            for t_val in t_grid:
                head_maps_at_t = {h: per_t_maps[h][t_val] for h in HEADS}
                accumulator.update(t_val, head_maps_at_t)

            if device.type == "mps" and i % 50 == 0:
                torch.mps.empty_cache()

    # Finalize and write
    calib_stats = accumulator.finalize()
    
    # Generate dynamic filename based on chosen heads
    filename_suffix = "_".join(HEADS)
    save_path = f"{RESULTS_DIR}/calib_stats_{filename_suffix}.pt"
    
    torch.save(calib_stats, save_path)
    print(f"Saved stats to {save_path} (n={calib_stats[HEADS[0]][t_grid[0]]['n']})")
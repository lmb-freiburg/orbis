import glob
import os
import random
import sys
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from util import instantiate_from_config

RESULTS_DIR = "results_pt"
os.makedirs(f"{RESULTS_DIR}", exist_ok=True)


# ---------- PyTorch Parallel Dataset ----------
class DoTACalibDataset(Dataset):
    """
    Dataset that randomly samples clips strictly from 'non-ood' folders in DoTA_prepared
    without using a manifest file.
    """
    def __init__(self, base_dir="DoTA_prepared", num_samples=3000, size=(512, 288), seed=42):
        self.base_dir = Path(base_dir)
        self.size = size
        self.idxs = [0, 2, 4, 6, 8, 10]

        print(f"Scanning for 'non-ood' clip folders in '{self.base_dir}'...")
        all_folders = [p for p in self.base_dir.glob("*/non-ood") if p.is_dir()]
        
        # Pre-cache frame paths during validation to eliminate duplicate glob/sort calls in __getitem__
        valid_clips = []
        for folder in tqdm(all_folders, desc="Validating clip folders"):
            paths = sorted(glob.glob(f"{folder}/*.jpg"))
            if len(paths) == 11:
                valid_clips.append((folder, paths))

        print(f"Found {len(valid_clips)} valid 'non-ood' clip folders.")

        random.seed(seed)
        num_to_sample = min(num_samples, len(valid_clips))
        self.clips = random.sample(valid_clips, num_to_sample)
        print(f"Randomly selected {len(self.clips)} non-ood samples for calibration.")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        folder, paths = self.clips[idx]

        frames = []
        for i in self.idxs:
            img = cv2.cvtColor(cv2.imread(paths[i]), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size)
            # Normalize to [-1, 1]
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
            frames.append(t)
            
        clip_id = f"{folder.parent.name}/{folder.name}"
        return torch.stack(frames), clip_id


# ---------- Optimized Scorer ----------
@torch.no_grad()
def surprise_score(model, images, frame_rate, t_grid, heads, n_noise_samples=2):
    """
    CUDA-optimized inference block with Mixed Precision support.
    """
    net = model.vit
    net.eval()
    
    # Modern AMP API (replaces deprecated torch.cuda.amp.autocast)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        x = model.encode_frames(images)
        context, target = x[:, :-1], x[:, -1:]
        b, _, n_channels, h, w = target.shape
        half = n_channels // 2

        context_exp = context.repeat_interleave(n_noise_samples, dim=0)
        target_exp = target.repeat_interleave(n_noise_samples, dim=0)

        per_t_maps = {h_name: {} for h_name in heads}

        for t_val in t_grid:
            t = torch.full((b * n_noise_samples,), t_val, device=x.device)
            fr = frame_rate.repeat_interleave(n_noise_samples, dim=0) if frame_rate.numel() > 1 else frame_rate

            def run_inference_pass(ctx, tgt, err_start_idx, err_end_idx):
                tgt_t, noise = model.add_noise(tgt, t)
                pred = net(tgt_t, ctx, t, frame_rate=fr)
                
                true_v = model.A(t) * tgt + model.B(t) * noise
                err = (pred.float() - true_v.float()).pow(2)
                
                err_avg = err.view(b, n_noise_samples, 1, n_channels, h, w).mean(dim=1)
                return err_avg[:, :, err_start_idx:err_end_idx]

            # Pass 1: DETAILED
            if "detailed" in heads:
                zero_indices = torch.arange(half, n_channels, device=x.device)
                ctx_det = context_exp.index_fill(2, zero_indices, 0)
                tgt_det = target_exp.index_fill(2, zero_indices, 0)
                
                per_t_maps["detailed"][t_val] = run_inference_pass(
                    ctx_det, tgt_det, err_start_idx=0, err_end_idx=half
                )

            # Pass 2: SEMANTIC
            if "semantic" in heads:
                zero_indices = torch.arange(0, half, device=x.device)
                ctx_sem = context_exp.index_fill(2, zero_indices, 0)
                tgt_sem = target_exp.index_fill(2, zero_indices, 0)
                
                per_t_maps["semantic"][t_val] = run_inference_pass(
                    ctx_sem, tgt_sem, err_start_idx=half, err_end_idx=n_channels
                )

            # Pass 3: COMBINED
            if "combined" in heads:
                per_t_maps["combined"][t_val] = run_inference_pass(
                    context_exp, target_exp, err_start_idx=0, err_end_idx=n_channels
                )

    return per_t_maps


# ---------- Vectorized Welford Running Mean/Variance ----------
class MultiHeadWelfordAccumulator:
    def __init__(self, shape, t_grid, heads, device):
        self.t_grid = t_grid
        self.heads = heads
        self.n = {t: 0 for t in t_grid}
        
        # Accumulate stats directly on CUDA
        self.mean = {h: {t: torch.zeros(shape, device=device, dtype=torch.float32) for t in t_grid} for h in heads}
        self.M2 = {h: {t: torch.zeros(shape, device=device, dtype=torch.float32) for t in t_grid} for h in heads}

    def update_batch(self, t_val, head_maps):
        """
        FULLY VECTORIZED BATCH WELFORD UPDATE (CUDA Fast-Path).
        Eliminates Python element-wise loops for maximum speed.
        """
        for h in self.heads:
            batch_maps = head_maps[h].squeeze(1)  # Shape: [B, C, H, W]
            batch_size = batch_maps.shape[0]

            # Batch mean and variance calculation (Parallelized across CUDA threads)
            batch_mean = batch_maps.mean(dim=0)
            batch_m2 = ((batch_maps - batch_mean) ** 2).sum(dim=0)

            n_old = self.n[t_val]
            n_new = n_old + batch_size
            self.n[t_val] = n_new

            if n_old == 0:
                self.mean[h][t_val] = batch_mean
                self.M2[h][t_val] = batch_m2
            else:
                delta = batch_mean - self.mean[h][t_val]
                self.mean[h][t_val] += delta * (batch_size / n_new)
                self.M2[h][t_val] += batch_m2 + (delta ** 2) * (n_old * batch_size / n_new)

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
    # 1. CUDA Performance Configurations
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 2. Model Setup
    exp_dir = "logs_wm/orbis_288x512"
    cfg = OmegaConf.load(f"{exp_dir}/config.yaml")
    model = instantiate_from_config(cfg.model)
    state = torch.load(f"{exp_dir}/checkpoints/last.ckpt", map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"Skipping torch.compile due to: {e}")

    # 3. Execution Parameters
    t_grid = [0.2, 0.4, 0.6, 0.8]
    N_NOISE_SAMPLES = 2
    HEADS = ["combined"]
    BATCH_SIZE = 4  
    NUM_SAMPLES = 3000

    # 4. Data Pipeline Setup
    dataset = DoTACalibDataset(base_dir="DoTA_prepared", num_samples=NUM_SAMPLES, seed=42)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,       
        pin_memory=True,     
        drop_last=False,
    )

    # 5. Dynamic Shape Inference
    probe_batch, _ = next(iter(loader))
    probe_batch = probe_batch.to(device, non_blocking=True)
    probe_fr = torch.full((probe_batch.shape[0],), 5.0, device=device)

    probe_maps = surprise_score(
        model,
        probe_batch,
        probe_fr,
        t_grid,
        heads=HEADS,
        n_noise_samples=1,
    )
    
    map_shape = probe_maps[HEADS[0]][t_grid[0]][0].squeeze(0).shape
    print(f"Per-token map shape: {map_shape}")

    # 6. Main Accumulation Loop
    accumulator = MultiHeadWelfordAccumulator(map_shape, t_grid, HEADS, device)

    pbar = tqdm(loader, desc="Processing calib clips", unit="batch")
    for windows, _ in pbar:
        windows = windows.to(device, non_blocking=True)
        current_b = windows.shape[0]
        frame_rate = torch.full((current_b,), 5.0, device=device)

        per_t_maps = surprise_score(
            model,
            windows,
            frame_rate,
            t_grid,
            heads=HEADS,
            n_noise_samples=N_NOISE_SAMPLES,
        )

        for t_val in t_grid:
            head_maps_at_t = {h: per_t_maps[h][t_val] for h in HEADS}
            accumulator.update_batch(t_val, head_maps_at_t)

    pbar.close()

    # 7. Finalize and Save
    calib_stats = accumulator.finalize()
    filename_suffix = "_".join(HEADS)
    save_path = f"{RESULTS_DIR}/calib_stats_{filename_suffix}.pt"

    torch.save(calib_stats, save_path)
    print(f"Saved stats to {save_path} (n={calib_stats[HEADS[0]][t_grid[0]]['n']})")
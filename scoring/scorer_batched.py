import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
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
os.makedirs(RESULTS_DIR, exist_ok=True)


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


def _read_and_prep_img(path, size=(512, 288)):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0


# Multithreaded fast loader across CPU threads
def load_clip_as_window_fast(frame_paths, size=(512, 288), executor=None):
    idxs = [0, 2, 4, 6, 8, 10]
    selected_paths = [frame_paths[i] for i in idxs]
    
    if executor is not None:
        frames = list(executor.map(lambda p: _read_and_prep_img(p, size), selected_paths))
    else:
        frames = [_read_and_prep_img(p, size) for p in selected_paths]
        
    return torch.stack(frames)


# ---------- Batched Scorer ----------
@torch.no_grad()
def surprise_score_semantic(model, images, frame_rate, t_grid, n_noise_samples=8, use_ema=False):
    net = model.ema_vit if use_ema else model.vit
    net.eval()
    
    # Forward pass frame encoding
    x = model.encode_frames(images)
    context, target = x[:, :-1], x[:, -1:]
    
    n_channels = target.shape[2]
    half = n_channels // 2

    # Virtual batch expansion along N dimension
    target_exp = target.expand(n_noise_samples, -1, -1, -1, -1)   # [N, 1, C, H, W]
    context_exp = context.expand(n_noise_samples, -1, -1, -1, -1) # [N, T-1, C, H, W]
    frame_rate_exp = frame_rate.expand(n_noise_samples)            # [N]

    per_t_map = {}

    for t_val in t_grid:
        t = torch.full((n_noise_samples,), t_val, device=x.device)
        
        # Single vectorized noise forward pass
        target_t, noise = model.add_noise(target_exp, t)
        pred = net(target_t, context_exp, t, frame_rate=frame_rate_exp)
        true_v = model.A(t) * target_exp + model.B(t) * noise
        
        err = (pred.float() - true_v.float()) ** 2                # [N, 1, C, H, W]
        err_avg = err.mean(dim=0)                                  # [1, C, H, W]
        
        semantic_map = err_avg[:, half:].mean(dim=1).squeeze(0)
        per_t_map[t_val] = semantic_map.cpu()

    return per_t_map


# ---------- Welford Accumulator ----------
class WelfordAccumulator:
    def __init__(self, shape, t_grid):
        self.t_grid = t_grid
        self.n = {t: 0 for t in t_grid}
        self.mean = {t: torch.zeros(shape) for t in t_grid}
        self.M2 = {t: torch.zeros(shape) for t in t_grid}

    def update(self, t_val, new_map):
        self.n[t_val] += 1
        n = self.n[t_val]
        delta = new_map - self.mean[t_val]
        self.mean[t_val] += delta / n
        delta2 = new_map - self.mean[t_val]
        self.M2[t_val] += delta * delta2

    def update_all_t(self, per_t_map):
        for t_val, new_map in per_t_map.items():
            self.update(t_val, new_map)

    def finalize(self, eps=1e-8):
        stats = {}
        for t_val in self.t_grid:
            n = self.n[t_val]
            var = self.M2[t_val] / max(n - 1, 1)
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

    # PyTorch performance flags
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    t_grid = [0.2, 0.4, 0.6, 0.8]
    N_NOISE_SAMPLES = 3

    with open("DoTA_prepared/manifest_subset1500.json") as f:
        manifest = json.load(f)

    # Reusable CPU ThreadPool for fast parallel image reading
    io_executor = ThreadPoolExecutor(max_workers=8)

    # Infer spatial shape dynamically
    first_clip = manifest[0]["clip_id"]
    first_folder = str(Path("DoTA_prepared") / first_clip / "non-ood")
    probe_window = load_clip_as_window_fast(get_sorted_frame_paths(first_folder), executor=io_executor).unsqueeze(0).to(device)
    probe_map = surprise_score_semantic(
        model,
        probe_window,
        torch.full((1,), 5.0).to(device), 
        t_grid=[0.2], 
        n_noise_samples=1,
    )
    map_shape = probe_map[0.2].shape
    print(f"Per-token map shape: {map_shape}")

    accumulators = {
        "calib": WelfordAccumulator(map_shape, t_grid),
        "heldout": WelfordAccumulator(map_shape, t_grid),
        "ood": WelfordAccumulator(map_shape, t_grid),
    }

    all_sample_results = []
    frame_rate = torch.full((1,), 5.0, device=device)

    for clip in tqdm(manifest, desc="Processing clips", unit="clip"):
        clip_id = clip["clip_id"]
        clip_dir = Path("DoTA_prepared") / clip_id

        # 1. Non-OOD
        non_ood_split = clip.get("non_ood_split")
        if non_ood_split != "unused_non_ood" and non_ood_split in accumulators:
            paths = get_sorted_frame_paths(str(clip_dir / "non-ood"))
            window = load_clip_as_window_fast(paths, executor=io_executor).unsqueeze(0).to(device, non_blocking=True)
            
            per_t_map = surprise_score_semantic(
                model, window, frame_rate, t_grid, N_NOISE_SAMPLES
            )
            accumulators[non_ood_split].update_all_t(per_t_map)

            per_t_scalar = {t: float(per_t_map[t].mean()) for t in t_grid}
            all_sample_results.append({
                "sample_id": f"{clip_id}_nonood",
                "split": non_ood_split,
                "label": "normal",
                "per_t_scalar": per_t_scalar,
                "score_mean": sum(per_t_scalar.values()) / len(t_grid),
            })

        # 2. OOD
        ood_split = clip.get("ood_split")
        target_ood_key = ood_split if ood_split in accumulators else "ood"
        paths_ood = get_sorted_frame_paths(str(clip_dir / "ood"))
        window_ood = load_clip_as_window_fast(paths_ood, executor=io_executor).unsqueeze(0).to(device, non_blocking=True)
        
        per_t_map_ood = surprise_score_semantic(
            model, window_ood, frame_rate, t_grid, N_NOISE_SAMPLES
        )
        accumulators[target_ood_key].update_all_t(per_t_map_ood)

        per_t_scalar_ood = {t: float(per_t_map_ood[t].mean()) for t in t_grid}
        all_sample_results.append({
            "sample_id": f"{clip_id}_ood",
            "split": ood_split,
            "label": "anomaly",
            "per_t_scalar": per_t_scalar_ood,
            "score_mean": sum(per_t_scalar_ood.values()) / len(t_grid),
        })

    io_executor.shutdown()

    # Save outputs
    for split_name, acc in accumulators.items():
        stats = acc.finalize()
        save_path = f"{RESULTS_DIR}/{split_name}_stats.pt"
        torch.save(stats, save_path)
        sample_count = stats[t_grid[0]]["n"]
        tqdm.write(f"Saved {split_name} statistics (n={sample_count}) -> {save_path}")

    json_path = f"{RESULTS_DIR}/all_sample_scores.json"
    with open(json_path, "w") as f:
        json.dump(all_sample_results, f, indent=2)

    tqdm.write(f"Saved score summary for {len(all_sample_results)} samples to {json_path}")
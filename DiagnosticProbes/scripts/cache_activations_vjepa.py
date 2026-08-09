import os
import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# CUDA & CUDNN Optimizations for Cloud GCE (NVIDIA T4 GPU + 8 vCPUs)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

from dota import get_dota_dataloaders
from src.models.vision_transformer import vit_base

# ============================================================
# V-JEPA CONSTANTS & STANDARD TORCHVISION PREPROCESSING
# ============================================================
NUM_FRAMES = 5  # V-JEPA takes 5 frames (no target frame concept)
IMG_SIZE = 384
TARGET_TUBELET = 5

# Preprocess transform: Direct resize to (384, 384) without cropping to preserve 100% of original scene FOV
preprocess = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_clip_tensor(clip_tensor, device="cuda"):
    """
    Un-normalizes clip_tensor from [-1, 1] to [0, 1] and applies torchvision Resize((384, 384)) + ImageNet Normalize:
    Input: clip_tensor [B, 3, T=5, H=288, W=512] in [-1, 1]
    Output: Normalized tensor [B, 3, T=5, 384, 384] with zero spatial cropping.
    """
    B, C, T_frames, H, W = clip_tensor.shape

    # Un-normalize from [-1, 1] range (produced by dota dataloader) to [0, 1] range
    clip_tensor = clip_tensor * 0.5 + 0.5

    # Reshape to [B * T, 3, H, W] for 2D image transforms
    x = clip_tensor.permute(0, 2, 1, 3, 4).reshape(B * T_frames, C, H, W).to(device, non_blocking=True)

    # Preprocess pipeline (Direct Resize to 384x384 without cropping + ImageNet Normalize)
    x = preprocess(x)

    # Reshape back to [B, 3, T=5, 384, 384]
    x = x.view(B, T_frames, C, IMG_SIZE, IMG_SIZE).permute(0, 2, 1, 3, 4)
    return x


def load_vjepa_model(ckpt_path="vjepa_ckpt/vjepa2_1_vitb_dist_vitG_384.pt", device="cuda"):
    print("\n=======================================================")
    print(f"Loading V-JEPA 2.1 Base Model (384x384, {NUM_FRAMES} frames)...")
    print("=======================================================")

    model = vit_base(
        img_size=(IMG_SIZE, IMG_SIZE), num_frames=NUM_FRAMES, use_rope=True
    )

    if not os.path.exists(ckpt_path):
        alt_path = os.path.join("..", ckpt_path)
        if os.path.exists(alt_path):
            ckpt_path = alt_path

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"V-JEPA checkpoint not found at '{ckpt_path}'")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt["encoder"]
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}

    # Interpolate patch embedding temporal kernel for target tubelet size = 5
    old_weight = state["patch_embed.proj.weight"]
    if old_weight.shape[2] != TARGET_TUBELET:
        print(f"Interpolating temporal kernel from {old_weight.shape[2]} -> {TARGET_TUBELET}")
        out_c, in_c, old_t, h, w = old_weight.shape
        weight = old_weight.permute(0, 1, 3, 4, 2).reshape(-1, old_t)
        weight = F.interpolate(
            weight.unsqueeze(1),
            size=TARGET_TUBELET,
            mode="linear",
            align_corners=False
        )
        weight = weight.squeeze(1).reshape(out_c, in_c, h, w, TARGET_TUBELET)
        state["patch_embed.proj.weight"] = weight.permute(0, 1, 4, 2, 3)

    msg = model.load_state_dict(state, strict=False)
    print(f"Model load status: {msg}")
    model.eval()
    return model.to(device)


def cache_vjepa_features(model, dataloader, device, save_dir, split_name, checkpoint_interval=100):
    model.eval()

    all_features = []
    all_labels = []
    all_mc_labels = []
    all_source_mc_labels = []
    all_ego_labels = []
    all_target_frame_ids = []
    all_video_ids = []

    os.makedirs(save_dir, exist_ok=True)
    print(f"\nExtracting final V-JEPA activations for '{split_name}' split...")

    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(dataloader, desc=f"V-JEPA Caching ({split_name})")):
            clip_tensor = batch_data[0]
            labels = batch_data[1]

            if len(batch_data) == 7:
                mc_labels, source_mc_labels, ego_labels, clip_ids, target_frame_ids = batch_data[2:]
            elif len(batch_data) == 6:
                mc_labels, source_mc_labels, ego_labels = batch_data[2], batch_data[3], None
                clip_ids, target_frame_ids = batch_data[4], batch_data[5]
            elif len(batch_data) == 5:
                mc_labels, source_mc_labels, ego_labels = batch_data[2], None, None
                clip_ids, target_frame_ids = batch_data[3], batch_data[4]
            else:
                mc_labels, source_mc_labels, ego_labels = None, None, None
                clip_ids, target_frame_ids = batch_data[2], [None] * len(batch_data[2])

            # Preprocess 5-frame clip batch to [B, 3, 5, 384, 384] using standard torchvision transforms
            x = preprocess_clip_tensor(clip_tensor, device=device)

            # FP16 Automatic Mixed Precision inference for T4 GPU Tensor Cores
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(x)
            else:
                outputs = model(x)

            # Store only final output activations in FP16 CPU format
            all_features.append(outputs.cpu().half())
            all_labels.append(labels.cpu())

            if mc_labels is not None:
                all_mc_labels.append(mc_labels.cpu())
            if source_mc_labels is not None:
                all_source_mc_labels.append(source_mc_labels.cpu())
            if ego_labels is not None:
                all_ego_labels.append(ego_labels.cpu())
            all_video_ids.extend(clip_ids)
            if isinstance(target_frame_ids, (list, tuple)):
                all_target_frame_ids.extend(target_frame_ids)

            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Incremental partial saving
            if (i + 1) % checkpoint_interval == 0:
                partial_save_path = os.path.join(save_dir, f"{split_name}_vjepa_final_partial_mc.pt")
                torch.save({
                    'features': torch.cat(all_features, dim=0),
                    'labels': torch.cat(all_labels, dim=0),
                    'mc_labels': torch.cat(all_mc_labels, dim=0) if len(all_mc_labels) > 0 else None,
                    'source_mc_labels': torch.cat(all_source_mc_labels, dim=0) if len(all_source_mc_labels) > 0 else None,
                    'ego_labels': torch.cat(all_ego_labels, dim=0) if len(all_ego_labels) > 0 else None,
                    'video_ids': all_video_ids,
                    'target_frame_ids': all_target_frame_ids
                }, partial_save_path)
                print(f"\nSaved partial checkpoint ({i+1} batches) to '{partial_save_path}'")

    final_save_path = os.path.join(save_dir, f"{split_name}_vjepa_final_mc.pt")
    partial_save_path = os.path.join(save_dir, f"{split_name}_vjepa_final_partial_mc.pt")

    tensor_features = torch.cat(all_features, dim=0)
    tensor_labels = torch.cat(all_labels, dim=0)

    save_dict = {
        'features': tensor_features,
        'labels': tensor_labels,
        'mc_labels': torch.cat(all_mc_labels, dim=0) if len(all_mc_labels) > 0 else None,
        'source_mc_labels': torch.cat(all_source_mc_labels, dim=0) if len(all_source_mc_labels) > 0 else None,
        'ego_labels': torch.cat(all_ego_labels, dim=0) if len(all_ego_labels) > 0 else None,
        'video_ids': all_video_ids,
        'target_frame_ids': all_target_frame_ids
    }

    torch.save(save_dict, final_save_path)
    if os.path.exists(partial_save_path):
        os.remove(partial_save_path)

    print(f"Saved complete V-JEPA activations to '{final_save_path}': {tensor_features.shape[0]} clips with shape {tensor_features.shape[1:]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Cache final V-JEPA 2.1 activations using DoTA dataloaders.")
    parser.add_argument("--ckpt", type=str, default="vjepa_ckpt/vjepa2_1_vitb_dist_vitG_384.pt")
    parser.add_argument("--seq_dir", type=str, default="/Volumes/maccbeast/frames/")
    parser.add_argument("--anno_dir", type=str, default="annotations/")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=6, help="Optimal worker count for 8 vCPUs")
    parser.add_argument("--save_dir", type=str, default="./cached_features")
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=900, help="Cap dataset samples before 80/20 split")
    parser.add_argument("--cloud_dir", type=str, default="DOTA_training")
    parser.add_argument("--cloud_file", type=str, default="DoTA_training.pt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA Acceleration Backend (NVIDIA T4 GPU)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS Acceleration Backend")
    else:
        device = torch.device("cpu")
        print("Using CPU Backend")

    train_loader, val_loader = get_dota_dataloaders(
        args.seq_dir,
        args.anno_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        return_multiclass_labels=True,
        num_frames_per_clip=NUM_FRAMES,  # 5 frames for V-JEPA
        use_cloud_dataset=True,
        cloud_dir=args.cloud_dir,
        cloud_file=args.cloud_file
    )

    model = load_vjepa_model(ckpt_path=args.ckpt, device=device)

    cache_vjepa_features(model, train_loader, device, save_dir=args.save_dir, split_name="train", checkpoint_interval=args.checkpoint_interval)
    cache_vjepa_features(model, val_loader, device, save_dir=args.save_dir, split_name="val", checkpoint_interval=args.checkpoint_interval)

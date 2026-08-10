import os
import sys
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torchvision import transforms
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_PROBES_DIR = PROJECT_ROOT / "DiagnosticProbes"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf
from util import instantiate_from_config
from DiagnosticProbes.scripts.dota import DOTA_CLASS_NAMES
from DiagnosticProbes.Activations.linear_attention_probe_binary import AttentionProbe

def resolve_path(p):
    if p is None: return None
    p_path = Path(p)
    if p_path.is_absolute(): return str(p_path)
    p1 = PROJECT_ROOT / p
    if p1.exists(): return str(p1)
    p2 = DIAGNOSTIC_PROBES_DIR / p
    if p2.exists(): return str(p2)
    return str(p1)

def process_attention_map(attn_1d, img_bgr, W, H):
    """Helper function to normalize, resize, and overlay a single attention tensor."""
    attn_2d = attn_1d.reshape(18, 32).numpy()
    
    # Normalize attention weights to [0, 1]
    attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
    
    # Resize and Color
    attn_resized = cv2.resize(attn_2d, (W, H), interpolation=cv2.INTER_CUBIC)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
    
    # Overlay with alpha 0.5
    overlay_bgr = cv2.addWeighted(img_bgr, 0.5, heatmap_bgr, 0.5, 0)
    
    # Convert BGR to RGB for matplotlib
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

def generate_live_heatmap(target_video_id, specific_target_frame=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    orbis_exp_dir = resolve_path("./logs_wm/orbis_288x512")
    if not orbis_exp_dir or not os.path.exists(orbis_exp_dir):
        orbis_exp_dir = resolve_path("./logs_tk/tokenizer_288x512")
        
    config_file = (Path(orbis_exp_dir) / "config.yaml").resolve()
    ckpt_file = (Path(orbis_exp_dir) / "checkpoints/last.ckpt").resolve()
    probe_weights = resolve_path("checkpoints/binary/best_binary_attention_probe.pt")
    dota_pt_path = resolve_path("DOTA_training/DoTA_training.pt")
    output_dir = resolve_path("DiagnosticProbes/heatmaps/live_binary")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Orbis Base Model from {ckpt_file}...")
    cfg = OmegaConf.load(config_file)
    orbis_model = instantiate_from_config(cfg.model)
    state_dict = torch.load(str(ckpt_file), map_location="cpu")["state_dict"]
    orbis_model.load_state_dict(state_dict, strict=True)
    orbis_model.to(device).eval()

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output[0] if isinstance(output, tuple) else output
        return hook
    
    backbone = getattr(orbis_model, "vit", orbis_model)
    hook_handle = backbone.blocks[17].register_forward_hook(get_activation('block18'))

    print(f"Loading Attention Probe from {probe_weights}...")
    probe_model = AttentionProbe(input_dim=768, num_classes=2, num_heads=8).to(device) 
    probe_model.load_state_dict(torch.load(probe_weights, map_location=device))
    probe_model.eval()

    print(f"Loading DOTA metadata from {dota_pt_path}...")
    data = torch.load(dota_pt_path, map_location="cpu")
    clip_paths = data["clip_paths"]
    video_ids = data.get("video_ids", [])
    target_frame_ids = data.get("target_frame_ids", [])
    labels = data["labels"]
    mc_labels = data.get("mc_labels")
    
    transform = transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop((288, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    matches = 0
    for i, vid in enumerate(video_ids):
        if target_video_id not in vid:
            continue
            
        target_frame_id = target_frame_ids[i]
        
        # If a specific target frame is requested, skip if it's not in the ID
        if specific_target_frame and specific_target_frame not in str(target_frame_id):
            continue
            
        matches += 1
        c_paths = clip_paths[i]
        binary_label = int(labels[i].item())
        mc_label = int(mc_labels[i].item()) if mc_labels is not None else -1
        class_label = DOTA_CLASS_NAMES.get(mc_label, f"Class_{mc_label}")

        print(f"\nProcessing matching sequence: {vid} | Target Frame: {target_frame_id}")
        
        frames = []
        for p in c_paths:
            # Resolve path relative to the root where this script runs
            resolved_p = p.replace('../', '') if p.startswith('../') else p
            if not os.path.exists(resolved_p):
                resolved_p = os.path.join("DoTA_sequences", resolved_p.split("DoTA_sequences/")[-1] if "DoTA_sequences/" in resolved_p else resolved_p)
            
            img = cv2.imread(resolved_p)
            if img is None:
                print(f"Warning: Could not load {resolved_p}")
                continue
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            frames.append(transform(img))
            
        if len(frames) != 6:
            print("Warning: Not enough valid frames. Skipping.")
            continue
            
        clip_tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(device)

        with torch.no_grad():
            if hasattr(orbis_model, "encode_frames") and hasattr(orbis_model, "vit"):
                clips = clip_tensor.permute(0, 2, 1, 3, 4).contiguous()
                latents = orbis_model.encode_frames(clips)
                context = latents[:, :-1].contiguous() if latents.size(1) > 1 else None
                target = latents[:, -1:].contiguous()
                t = torch.zeros(clips.shape[0], dtype=torch.float32, device=device)
                frame_rate = torch.full((clips.shape[0],), 5.0, dtype=torch.float32, device=device)
                _ = orbis_model.vit(target, context, t, frame_rate=frame_rate)
            else:
                t = torch.zeros(clip_tensor.shape[0], dtype=torch.float32, device=device)
                _ = orbis_model(clip_tensor, t)
            
            features = activation['block18']
            if features.dim() == 4:
                features = features[:, -1, :, :] 
            
            logits, attn_weights = probe_model(features)
            probs = torch.nn.functional.softmax(logits, dim=1)[:, 1]
            prob_anom = probs.item()
            final_attn_tensor = attn_weights.squeeze(0).squeeze(1).cpu() 

        # Find target frame
        target_img_path = None
        for p in c_paths:
            if target_frame_id in p:
                target_img_path = p.replace('../', '') if p.startswith('../') else p
                break
        if not target_img_path:
            target_img_path = c_paths[-1].replace('../', '') if c_paths[-1].startswith('../') else c_paths[-1]

        target_frame_name = os.path.basename(target_img_path)
        img = cv2.imread(target_img_path)
        if img is None:
            # fallback relative path
            target_img_path = os.path.join("DoTA_sequences", target_img_path.split("DoTA_sequences/")[-1] if "DoTA_sequences/" in target_img_path else target_img_path)
            img = cv2.imread(target_img_path)

        H, W, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig = plt.figure(figsize=(25, 10))
        gs = gridspec.GridSpec(2, 5, figure=fig)
        
        # Row 0, Col 0: Raw RGB Image
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(img_rgb)
        ax.set_title(f"Raw Image ({target_frame_name})", fontsize=16, pad=12, fontweight='bold')
        ax.axis('off')

        num_heads = final_attn_tensor.shape[0] if final_attn_tensor.dim() > 1 else 8
        for h in range(min(4, num_heads)):
            head_attn = final_attn_tensor[h] if final_attn_tensor.dim() > 1 else final_attn_tensor
            _, overlay = process_attention_map(head_attn, img, W, H)
            ax = fig.add_subplot(gs[0, h + 1])
            ax.imshow(overlay)
            ax.set_title(f"Head {h + 1}", fontsize=16, pad=12)
            ax.axis('off')

        mean_attn = final_attn_tensor.mean(dim=0) if final_attn_tensor.dim() > 1 else final_attn_tensor
        _, mean_overlay = process_attention_map(mean_attn, img, W, H)
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(mean_overlay)
        ax.set_title(f"Mean Attention ({target_frame_name})", fontsize=16, pad=12, fontweight='bold')
        ax.axis('off')

        for h in range(4, min(8, num_heads)):
            head_attn = final_attn_tensor[h]
            _, overlay = process_attention_map(head_attn, img, W, H)
            ax = fig.add_subplot(gs[1, h - 3])
            ax.imshow(overlay)
            ax.set_title(f"Head {h + 1}", fontsize=16, pad=12)
            ax.axis('off')

        header_str = f"[LIVE INFERENCE] Video: {vid} | Target: {target_frame_name} | Class: {class_label} | True: {binary_label} | P(Anomalous): {prob_anom:.4f}"
        fig.suptitle(header_str, fontsize=18, y=1.02, fontweight='bold')
        plt.tight_layout()

        frame_stem = os.path.splitext(target_frame_name)[0]
        save_filename = f"LIVE_{vid}_frame_{frame_stem}_heatmap.jpg"
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved LIVE heatmap to {save_path}")

    hook_handle.remove()
    print(f"\nLive generation complete. {matches} matching sequences found and processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_id", type=str, default="d2SCftR5sWc_002095")
    parser.add_argument("--target_frame", type=str, default=None, help="Specific target frame ID (e.g. 073 or 000073) to filter by")
    args = parser.parse_args()
    generate_live_heatmap(args.sequence_id, args.target_frame)

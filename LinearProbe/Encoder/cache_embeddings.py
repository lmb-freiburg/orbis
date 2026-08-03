import os
import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from LinearProbe.dota import get_dota_dataloaders
from util import instantiate_from_config


def load_model_from_config(exp_dir, config_path, ckpt_path, device):
    config_file = (Path(exp_dir) / config_path).resolve()
    ckpt_file = (Path(exp_dir) / ckpt_path).resolve()

    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    cfg = OmegaConf.load(config_file)
    model = instantiate_from_config(cfg.model)
    state_dict = torch.load(str(ckpt_file), map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def cache_features(model, dataloader, device, save_dir, split_name):
    model.eval()
    
    # Storage arrays
    all_features = []
    all_labels = []
    all_ids = []
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Extracting unpooled encoder embeddings for '{split_name}' split...")
    
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(dataloader)):
            clips = batch_data[0].to(device)       # Shape: [B, C, T, H, W]
            labels = batch_data[1]
            clip_ids = batch_data[-1]

            # Extract the target frame (the final temporal frame of the clip)
            target_frame = clips[:, :, -1, :, :]   # Shape: [B, C, H, W]

            # Direct autoencoder continuous latent extraction
            ret = model.ae.encode(target_frame)
            h, h2 = ret["continuous"]              # Dual streams, each [B, C_stream, H_latent, W_latent]

            # Unpooled combination: concatenate along the channel dimension (dim=1)
            combined_vec = torch.cat([h, h2], dim=1)  # Shape: [B, C_total, H_latent, W_latent]

            # Move off device to prevent VRAM accumulation
            all_features.append(combined_vec.cpu())
            all_labels.append(labels.cpu())
            all_ids.extend(clip_ids)

    # Concatenate results into a single tensor
    tensor_features = torch.cat(all_features, dim=0)
    tensor_labels = torch.cat(all_labels, dim=0)
    
    # Save unpooled, combined embeddings
    final_save_path = os.path.join(save_dir, f"{split_name}_unpooled_embeddings.pt")
    torch.save({
        'features': tensor_features, 
        'labels': tensor_labels, 
        'ids': all_ids
    }, final_save_path)
    
    # Print the shape and total clips stored
    print(f"Saved completed dataset to {final_save_path}")
    print(f"Total samples: {tensor_features.shape[0]}")
    print(f"Embedded tensor shape: {list(tensor_features.shape)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Cache unpooled ORBIS encoder embeddings.")
    parser.add_argument("--exp_dir", type=str, default="./logs_wm/orbis_288x512")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.ckpt")
    parser.add_argument("--seq_dir", type=str, default="../DoTA_sequences")
    parser.add_argument("--anno_dir", type=str, default="../DOTA_annotations")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./cached_features")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon Hardware Acceleration Backend: MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA Acceleration Backend")
    else:
        device = torch.device("cpu")
        print("Using CPU Extension Only")

    train_loader, val_loader = get_dota_dataloaders(
        args.seq_dir,
        args.anno_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = load_model_from_config(args.exp_dir, args.config, args.ckpt, device)

    # Cache Train split
    cache_features(
        model, 
        train_loader, 
        device, 
        save_dir=args.save_dir,
        split_name="train"
    )
    
    # Cache Val split
    cache_features(
        model, 
        val_loader, 
        device, 
        save_dir=args.save_dir,
        split_name="val"
    )
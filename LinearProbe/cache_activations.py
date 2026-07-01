import os
import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.dota import get_dota_dataloaders
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
def cache_features(model, dataloader, device, save_path):
    model.eval()
    
    # Dictionary to hold the intercepted feature map

    backbone = getattr(model, "vit", model)
    if not hasattr(backbone, "blocks"):
        raise AttributeError("Expected the loaded model to expose a `blocks` module list.")
    activation = {}
    
    # 1. Define the forward hook
    def get_activation(name):
        def hook(model, input, output):
            # Some STDiT blocks return a tuple (hidden_states, context). We just want the hidden states.
            activation[name] = output[0] if isinstance(output, tuple) else output
        return hook

    # 2. Register the hook to the 20th block (index 19)
    hook_handle = backbone.blocks[19].register_forward_hook(get_activation('block_20'))
    
    all_features = []
    all_labels = []

    print(f"Extracting features to {save_path}...")
    with torch.no_grad():
        for clips, labels in tqdm(dataloader):
            clips = clips.to(device)

            if hasattr(model, "encode_frames") and hasattr(model, "vit"):
                # DoTA clips arrive as [B, C, T, H, W]; the second-stage model expects [B, T, C, H, W].
                clips = clips.permute(0, 2, 1, 3, 4).contiguous()
                latents = model.encode_frames(clips)
                context = latents[:, :-1].contiguous() if latents.size(1) > 1 else None
                target = latents[:, -1:].contiguous()
                t = torch.zeros(clips.shape[0], device=device)
                frame_rate = torch.full((clips.shape[0],), 5.0, device=device)
                _ = model.vit(target, context, t, frame_rate=frame_rate)
            else:
                t = torch.zeros(clips.shape[0], device=device)
                _ = model(clips, t)
            
            # 3. Retrieve the intercepted features from Block 20
            # Expected shape: [Batch, Sequence_Length (Tokens), Hidden_Dim]
            features = activation['block_20']
            
            # 4. Spatio-Temporal Pooling
            # We average-pool across tokens and frames to get one global feature vector per clip
            pooled_features = features.mean(dim=(1, 2)) if features.dim() == 4 else features.mean(dim=1)
            
            all_features.append(pooled_features.cpu())
            all_labels.append(labels.cpu())

    # Clean up the hook
    hook_handle.remove()
    
    # Concatenate and save to disk
    tensor_features = torch.cat(all_features, dim=0)
    tensor_labels = torch.cat(all_labels, dim=0)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'features': tensor_features, 'labels': tensor_labels}, save_path)
    print(f"Saved {tensor_features.shape[0]} clips with dimension {tensor_features.shape[1]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Cache linear-probe features from a trained orbis checkpoint.")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dota_dataloaders(
        args.seq_dir,
        args.anno_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = load_model_from_config(args.exp_dir, args.config, args.ckpt, device)

    cache_features(model, train_loader, device, os.path.join(args.save_dir, "train_block20.pt"))
    cache_features(model, val_loader, device, os.path.join(args.save_dir, "val_block20.pt"))
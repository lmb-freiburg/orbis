import os
import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def cache_features(model, dataloader, device, save_path, checkpoint_interval=100):
    model.eval()
    
    backbone = getattr(model, "vit", model)
    if not hasattr(backbone, "blocks"):
        raise AttributeError("Expected the loaded model to expose a `blocks` module list.")
    activation = {}
    
    # 1. Define the forward hook
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output[0] if isinstance(output, tuple) else output
        return hook

    # 2. Register the hook to the 20th block (index 19)
    # hook_handle = backbone.blocks[19].register_forward_hook(get_activation('block_20'))
    hook_handle = backbone.blocks[17].register_forward_hook(get_activation('block_18'))
    
    all_features = []
    all_labels = []
    
    # Setup paths for partial backups
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    partial_save_path = save_path.replace(".pt", "_partial.pt")

    print(f"Extracting features to {save_path} (saving backups every {checkpoint_interval} batches)...")
    
    with torch.no_grad():
        # print ('------------',len(dataloader), '---', len(dataloader.dataset))
        for i, (clips, labels) in enumerate(tqdm(dataloader)):
            # print (labels, labels.shape)
            clips = clips.to(device)

            if hasattr(model, "encode_frames") and hasattr(model, "vit"):
                # DoTA clips arrive as [B, C, T, H, W]; the second-stage model expects [B, T, C, H, W].
                clips = clips.permute(0, 2, 1, 3, 4).contiguous()
                latents = model.encode_frames(clips)
                context = latents[:, :-1].contiguous() if latents.size(1) > 1 else None
                print('----------', context.shape, '----------')
                target = latents[:, -1:].contiguous()
                
                # MPS optimization: explicit float32 constraints prevent internal ops mismatch
                t = torch.zeros(clips.shape[0], dtype=torch.float32, device=device)
                frame_rate = torch.full((clips.shape[0],), 5.0, dtype=torch.float32, device=device)
                _ = model.vit(target, context, t, frame_rate=frame_rate)
            else:
                t = torch.zeros(clips.shape[0], dtype=torch.float32, device=device)
                _ = model(clips, t)
            
            # # 3. Retrieve the intercepted features from Block 20
            # features = activation['block_20']
            # 3. Retrieve the intercepted features from Block 10
            features = activation['block_18']

            print('---------- Activations Shape - ', features.shape, '----------')
            
            # 4. Spatio-Temporal Pooling
            # print ("Pooled features shape - ", features.shape)
            # features = features.mean(dim=(1, 2)) if features.dim() == 4 else features.mean(dim=1)
            
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

            # 5. Incremental Caching checkpointing logic
            if (i + 1) % checkpoint_interval == 0:
                temp_features = torch.cat(all_features, dim=0)
                temp_labels = torch.cat(all_labels, dim=0)
                torch.save({'features': temp_features, 'labels': temp_labels}, partial_save_path)
                del temp_features, temp_labels  # Memory flush

    # Clean up the hook
    hook_handle.remove()
    
    # Concatenate and save absolute final output to disk
    tensor_features = torch.cat(all_features, dim=0)
    tensor_labels = torch.cat(all_labels, dim=0)
    
    torch.save({'features': tensor_features, 'labels': tensor_labels}, save_path)
    
    # Clean up partial file on complete run success
    if os.path.exists(partial_save_path):
        os.remove(partial_save_path)
        
    print(f"Saved completed dataset: {tensor_features.shape[0]} clips with dimension {tensor_features.shape[1]}")


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
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="Backup cache every N sequences")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Target MPS for local Apple Silicon M4 Pro, default back seamlessly to CUDA/CPU 
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

    cache_features(
        model, 
        train_loader, 
        device, 
        # os.path.join(args.save_dir, "train_block20.pt"),
        os.path.join(args.save_dir, "train_block18.pt"),
        checkpoint_interval=args.checkpoint_interval
    )
    cache_features(
        model, 
        val_loader, 
        device, 
        # os.path.join(args.save_dir, "val_block20.pt"),
        os.path.join(args.save_dir, "val_block18.pt"),
        checkpoint_interval=args.checkpoint_interval
    )
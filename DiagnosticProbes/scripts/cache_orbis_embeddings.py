import os
import argparse
import sys
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Optimize CUDA kernel selection for fixed image shapes
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

from dota import get_dota_dataloaders
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


def cache_features(model, dataloader, device, save_dir, split_name, checkpoint_interval=100):
    model.eval()
    
    # Storage arrays
    all_features = []
    all_labels = []
    all_mc_labels = []
    all_source_mc_labels = []
    all_ego_labels = []
    all_target_frame_ids = []
    all_video_ids = []
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Extracting embeddings for '{split_name}' split (saving backups every {checkpoint_interval} batches)...")
    
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(dataloader)):
            clips = batch_data[0].to(device, non_blocking=True)
            labels = batch_data[1]
            if len(batch_data) == 7:
                mc_labels = batch_data[2]
                source_mc_labels = batch_data[3]
                ego_labels = batch_data[4]
                clip_ids = batch_data[5]
                target_frame_ids = batch_data[6]
            elif len(batch_data) == 6:
                mc_labels = batch_data[2]
                source_mc_labels = batch_data[3]
                ego_labels = None
                clip_ids = batch_data[4]
                target_frame_ids = batch_data[5]
            elif len(batch_data) == 5:
                mc_labels = batch_data[2]
                source_mc_labels = None
                ego_labels = None
                clip_ids = batch_data[3]
                target_frame_ids = batch_data[4]
            else:
                mc_labels = None
                source_mc_labels = None
                ego_labels = None
                clip_ids = batch_data[2]
                target_frame_ids = [None] * len(clip_ids)

            # Extract the target frame (the final temporal frame of the clip)
            target_frame = clips[:, :, -1, :, :]   # Shape: [B, C, H, W]

            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    ret = model.ae.encode(target_frame)
                    h, h2 = ret["continuous"]              # Dual streams
                    combined_vec = torch.cat([h, h2], dim=1)
            else:
                ret = model.ae.encode(target_frame)
                h, h2 = ret["continuous"]              # Dual streams
                combined_vec = torch.cat([h, h2], dim=1)
            
            # Move off device to prevent VRAM accumulation, save as float16
            all_features.append(combined_vec.cpu().half())
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

            # Clean up memory
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

            # Incremental Caching checkpointing logic
            if (i + 1) % checkpoint_interval == 0:
                print('Caching Checkpoint')
                temp_labels = torch.cat(all_labels, dim=0)
                temp_mc_labels = torch.cat(all_mc_labels, dim=0) if len(all_mc_labels) > 0 else None
                temp_source_mc_labels = torch.cat(all_source_mc_labels, dim=0) if len(all_source_mc_labels) > 0 else None
                temp_ego_labels = torch.cat(all_ego_labels, dim=0) if len(all_ego_labels) > 0 else None
                
                partial_save_path = os.path.join(save_dir, f"{split_name}_all_unpooled_embeddings_partial.pt")
                temp_features = torch.cat(all_features, dim=0)
                
                save_dict = {
                    'features': temp_features,
                    'labels': temp_labels,
                    'video_ids': all_video_ids,
                    'target_frame_ids': all_target_frame_ids
                }
                if temp_mc_labels is not None:
                    save_dict['mc_labels'] = temp_mc_labels
                if temp_source_mc_labels is not None:
                    save_dict['source_mc_labels'] = temp_source_mc_labels
                if temp_ego_labels is not None:
                    save_dict['ego_labels'] = temp_ego_labels

                torch.save(save_dict, partial_save_path)
                print(f'Saved checkpoint {i+1} in {partial_save_path}')
                
                del temp_features
                del temp_labels
                if temp_mc_labels is not None:
                    del temp_mc_labels
                if temp_source_mc_labels is not None:
                    del temp_source_mc_labels
                if temp_ego_labels is not None:
                    del temp_ego_labels

    # Concatenate and save absolute final outputs
    tensor_labels = torch.cat(all_labels, dim=0)
    tensor_mc_labels = torch.cat(all_mc_labels, dim=0) if len(all_mc_labels) > 0 else None
    tensor_source_mc_labels = torch.cat(all_source_mc_labels, dim=0) if len(all_source_mc_labels) > 0 else None
    tensor_ego_labels = torch.cat(all_ego_labels, dim=0) if len(all_ego_labels) > 0 else None
    
    final_save_path = os.path.join(save_dir, f"{split_name}_all_unpooled_embeddings.pt")
    partial_save_path = os.path.join(save_dir, f"{split_name}_all_unpooled_embeddings_partial.pt")
    
    tensor_features = torch.cat(all_features, dim=0)
    
    save_dict = {
        'features': tensor_features,
        'labels': tensor_labels,
        'video_ids': all_video_ids,
        'target_frame_ids': all_target_frame_ids
    }
    if tensor_mc_labels is not None:
        save_dict['mc_labels'] = tensor_mc_labels
    if tensor_source_mc_labels is not None:
        save_dict['source_mc_labels'] = tensor_source_mc_labels
    if tensor_ego_labels is not None:
        save_dict['ego_labels'] = tensor_ego_labels

    torch.save(save_dict, final_save_path)
    
    # Clean up partial file on complete run success
    if os.path.exists(partial_save_path):
        os.remove(partial_save_path)
        
    print(f"Saved completed dataset to {final_save_path}: {tensor_features.shape[0]} clips with features {tensor_features.shape[1:]} and {len(all_target_frame_ids)} target_frame_ids")


def parse_args():
    parser = argparse.ArgumentParser(description="Cache AE embeddings from a trained orbis checkpoint.")
    parser.add_argument("--exp_dir", type=str, default="./logs_wm/orbis_288x512")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.ckpt")
    parser.add_argument("--seq_dir", type=str, default="../DoTA_sequences")
    parser.add_argument("--anno_dir", type=str, default="annotations/")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size optimized for NVIDIA T4 GPU")
    parser.add_argument("--num_workers", type=int, default=6, help="Optimal CPU worker threads for n1-standard-8 (8 vCPUs).")
    parser.add_argument("--save_dir", type=str, default="./cached_features")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="Backup cache every N sequences")
    parser.add_argument("--max_samples", type=int, default=3000, help="Cap dataset to this many clips before 80/20 train/val split.")
    parser.add_argument("--cloud_dir", type=str, default="DOTA_training")
    parser.add_argument("--cloud_file", type=str, default="DoTA_training.pt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    MULTI_CLASS = True
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA Acceleration Backend (NVIDIA T4 GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon Hardware Acceleration Backend: MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU Extension Only")

    train_loader, val_loader = get_dota_dataloaders(
        args.seq_dir,
        args.anno_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        return_multiclass_labels=MULTI_CLASS,
        num_frames_per_clip=6,
        use_cloud_dataset=True,
        cloud_dir=args.cloud_dir,
        cloud_file=args.cloud_file
    )

    model = load_model_from_config(args.exp_dir, args.config, args.ckpt, device)

    # Note: We now pass save_dir and split_name separately
    cache_features(
        model, 
        train_loader, 
        device, 
        save_dir=args.save_dir,
        split_name="train",
        checkpoint_interval=args.checkpoint_interval
    )
    
    cache_features(
        model, 
        val_loader, 
        device, 
        save_dir=args.save_dir,
        split_name="val",
        checkpoint_interval=args.checkpoint_interval
    )

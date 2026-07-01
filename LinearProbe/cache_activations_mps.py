import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.dota import get_dota_dataloaders
from util import instantiate_from_config


def get_autocast_context(device):
    if device.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16)
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


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
    if device.type == "mps":
        model = model.to(device=device, dtype=torch.float16)
    else:
        model = model.to(device)
    return model.eval()


def save_cache_shard(shard_path, feature_rows, label_rows):
    torch.save(
        {
            "features": torch.stack(feature_rows, dim=0),
            "labels": torch.stack(label_rows, dim=0),
        },
        shard_path,
    )


def merge_cache_shards(shard_paths, save_path):
    features = []
    labels = []
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu")
        features.append(shard["features"])
        labels.append(shard["labels"])

    torch.save(
        {
            "features": torch.cat(features, dim=0),
            "labels": torch.cat(labels, dim=0),
        },
        save_path,
    )


def cache_features(model, dataloader, device, save_path, checkpoint_interval=100):
    model.eval()
    model_dtype = next(model.parameters()).dtype
    
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
    
    save_path = Path(save_path)
    shard_paths = []
    pending_features = []
    pending_labels = []
    cached_sequences = 0
    shard_index = 0

    def flush_pending():
        nonlocal pending_features, pending_labels, shard_index
        if not pending_features:
            return

        shard_path = save_path.with_name(f"{save_path.stem}.part{shard_index:04d}{save_path.suffix}")
        save_cache_shard(shard_path, pending_features, pending_labels)
        shard_paths.append(shard_path)
        pending_features = []
        pending_labels = []
        shard_index += 1

    print(f"Extracting features to {save_path}...")
    batch_iterator = dataloader
    microbatch_size = 1 if device.type == "mps" else None

    with torch.inference_mode():
        for clips, labels in tqdm(batch_iterator):
            chunks = clips.split(microbatch_size, dim=0) if microbatch_size is not None else (clips,)
            label_chunks = labels.split(microbatch_size, dim=0) if microbatch_size is not None else (labels,)

            for chunk_clips, chunk_labels in zip(chunks, label_chunks):
                chunk_clips = chunk_clips.to(
                    device=device,
                    dtype=model_dtype,
                    non_blocking=device.type != "mps",
                )

                if hasattr(model, "encode_frames") and hasattr(model, "vit"):
                    chunk_clips = chunk_clips.permute(0, 2, 1, 3, 4).contiguous()
                    latents = model.encode_frames(chunk_clips)
                    context = latents[:, :-1].contiguous() if latents.size(1) > 1 else None
                    target = latents[:, -1:].contiguous()
                    t = torch.zeros(chunk_clips.shape[0], device=device)
                    frame_rate = torch.full((chunk_clips.shape[0],), 5.0, device=device)
                    _ = model.vit(target, context, t, frame_rate=frame_rate)
                else:
                    t = torch.zeros(chunk_clips.shape[0], device=device)
                    _ = model(chunk_clips, t)
            
                # 3. Retrieve the intercepted features from Block 20
                # Expected shape: [Batch, Sequence_Length (Tokens), Hidden_Dim]
                features = activation['block_20']

                # 4. Spatio-Temporal Pooling
                # We average-pool across tokens and frames to get one global feature vector per clip
                pooled_features = features.mean(dim=(1, 2)) if features.dim() == 4 else features.mean(dim=1)

                for sample_idx in range(pooled_features.size(0)):
                    pending_features.append(pooled_features[sample_idx].detach().cpu())
                    pending_labels.append(chunk_labels[sample_idx].detach().cpu())
                    cached_sequences += 1

                    if checkpoint_interval > 0 and cached_sequences % checkpoint_interval == 0:
                        flush_pending()

                activation.pop('block_20', None)

            if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

    # Clean up the hook
    hook_handle.remove()

    flush_pending()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    merge_cache_shards(shard_paths, save_path)
    final_cache = torch.load(save_path, map_location="cpu")
    print(f"Saved {final_cache['features'].shape[0]} clips with dimension {final_cache['features'].shape[1]}")


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
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

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
        str(Path(args.save_dir) / "train_block20.pt"),
        checkpoint_interval=args.checkpoint_interval,
    )
    cache_features(
        model,
        val_loader,
        device,
        str(Path(args.save_dir) / "val_block20.pt"),
        checkpoint_interval=args.checkpoint_interval,
    )
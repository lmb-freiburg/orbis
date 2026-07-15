import glob
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from util import instantiate_from_config

RESULTS_DIR = "results"
os.makedirs(f"{RESULTS_DIR}/probe", exist_ok=True)

# ---------------------------------------------------------
# 1. Device Setup (Prioritizing macOS MPS)
# ---------------------------------------------------------

def get_device():
    # Force check for Apple Silicon GPU (MPS) first, then CUDA, then CPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("-> Utilizing Apple Silicon GPU (MPS) for acceleration.")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("-> Utilizing NVIDIA CUDA GPU.")
        return torch.device("cuda")
    print("-> No GPU found. Defaulting to CPU.")
    return torch.device("cpu")

# ---------------------------------------------------------
# 2. Frame Processing & Feature Extraction
# ---------------------------------------------------------

def get_sorted_frame_paths(folder):
    paths = sorted(glob.glob(f"{folder}/*.jpg"))
    assert len(paths) == 11, f"Expected 11 frames in {folder}, found {len(paths)}"
    return paths

def load_clip_as_window(frame_paths, size=(512, 288)):
    # Slice the temporal frames down to a structured pattern
    idxs = [0, 2, 4, 6, 8, 10]
    frames = []
    for i in idxs:
        img = cv2.cvtColor(cv2.imread(frame_paths[i]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
        frames.append(t)
    return torch.stack(frames)

@torch.no_grad()
def extract_embedding(model, folder, device, use_max_pool=True):
    """
    Passes only the TARGET frame (the final frame of the 11-frame window) 
    through the Orbis encoder to generate semantic/detail representations.
    """
    paths = get_sorted_frame_paths(folder)
    window = load_clip_as_window(paths).unsqueeze(0).to(device)  # Shape: [1, 6, C, H, W]
    target_frame = window[:, -1]  # Target frame raw pixels: [1, C, H, W]

    ret = model.ae.encode(target_frame)
    h, h2 = ret["continuous"]  # Dual-stream latents, each [1, C_stream, 16, 16]

    # Pool spatial dimensions to obtain 1D vector representations
    if use_max_pool:
        # Global Max Pooling: takes the highest activation in the grid
        detail_vec = torch.amax(h, dim=[2, 3]).squeeze(0).cpu().numpy()
        semantic_vec = torch.amax(h2, dim=[2, 3]).squeeze(0).cpu().numpy()
    else:
        # Global Average Pooling: takes the average activation
        detail_vec = h.mean(dim=[2, 3]).squeeze(0).cpu().numpy()
        semantic_vec = h2.mean(dim=[2, 3]).squeeze(0).cpu().numpy()
    combined_vec = np.concatenate([detail_vec, semantic_vec])

    return detail_vec, semantic_vec, combined_vec

def extract_and_cache_all(model, clip_ids, folder_name, device, cache_path):
    """
    Extracts embeddings for ALL clips and caches them in a single .npz file.
    This allows us to split/re-split data in memory later instantly.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}...")
        data = np.load(cache_path)
        return data["detail"], data["semantic"], data["combined"]

    print(f"No cache found at {cache_path}. Running Orbis encoder over {len(clip_ids)} items on {device}...")
    detail_list, semantic_list, combined_list = [], [], []
    
    for i, clip_id in enumerate(clip_ids):
        folder = Path("DoTA_prepared") / clip_id / folder_name
        d, s, c = extract_embedding(model, str(folder), device)
        detail_list.append(d)
        semantic_list.append(s)
        combined_list.append(c)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(clip_ids)} clips...")

    detail_arr = np.stack(detail_list)
    semantic_arr = np.stack(semantic_list)
    combined_arr = np.stack(combined_list)
    
    np.savez_compressed(cache_path, detail=detail_arr, semantic=semantic_arr, combined=combined_arr)
    print(f"Successfully cached embeddings to {cache_path}!")
    return detail_arr, semantic_arr, combined_arr

# ---------------------------------------------------------
# 3. Dynamic 70:30 Clip-Level Splitting (No Data Leakage)
# ---------------------------------------------------------

def build_split_indices(manifest, seed=42, train_ratio=0.70):
    """
    Generates train/test indices using clip IDs to prevent target leakage.
    Returns:
        train_indices: list of indices corresponding to the train split
        test_indices: list of indices corresponding to the test split
    """
    all_clips = [c["clip_id"] for c in manifest]
    
    random.seed(seed)
    shuffled_clips = all_clips.copy()
    random.shuffle(shuffled_clips)
    
    n_train = round(len(shuffled_clips) * train_ratio)
    train_clips = set(shuffled_clips[:n_train])
    test_clips = set(shuffled_clips[n_train:])
    
    train_indices = [i for i, c in enumerate(manifest) if c["clip_id"] in train_clips]
    test_indices = [i for i, c in enumerate(manifest) if c["clip_id"] in test_clips]
    
    print("\n--- Unified 70:30 Clip-Level Split ---")
    print(f"Total Unique Clips: {len(shuffled_clips)}")
    print(f"Train Set: {len(train_indices)} clips ({len(train_indices) * 2} samples total)")
    print(f"Test Set : {len(test_indices)} clips ({len(test_indices) * 2} samples total)")
    print("--------------------------------------")
    
    return train_indices, test_indices

# ---------------------------------------------------------
# 4. PyTorch Model Architecture & Training Routine
# ---------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

def train_and_eval_probe(X_train, y_train, X_test, y_test, stream_name, device):
    print(f"\n--- Training PyTorch Probe on '{stream_name}' stream ---")
    
    # Standardize features (essential for stable gradient descent)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Mini-batch Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 40

    # Dataset & DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model initialization
    input_dim = X_train_scaled.shape[1]
    probe = LinearProbe(input_dim).to(device)
    
    # Loss & Adam Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=LEARNING_RATE)

    # Training Loop
    probe.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = probe(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}")

    # Add these lines right before the confusion matrix calculation
    print(f"Debug: X_test_t shape: {X_test_t.shape}")
    print(f"Debug: y_test shape: {y_test.shape}")
    

    # Evaluation Phase
    probe.eval()
    with torch.no_grad():
        X_test_t = X_test_t.to(device)
        logits = probe(X_test_t)
        # Apply sigmoid to convert logits into probabilities
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()
        preds = (probs >= 0.5).astype(float)
        print(f"Debug: len(preds): {len(preds)}")

    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    # Print a detailed report
    print(f"\n--- Detailed Report for {stream_name} ---")
    print(f"AUROC    : {auc:.4f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"Confusion Matrix:")
    print(f"          Predicted Normal | Predicted OOD")
    print(f"Actual Normal  {tn:^16} | {fp:^13}")
    print(f"Actual OOD     {fn:^16} | {tp:^13}")
    print("------------------------------------------\n")
    
    return {
        "stream": stream_name, 
        "auroc": auc, 
        "accuracy": acc, 
        "precision": prec, 
        "recall": rec
    }

# ---------------------------------------------------------
# 5. Pipeline Execution
# ---------------------------------------------------------

if __name__ == "__main__":
    exp_dir = "logs_wm/orbis_288x512"
    device = get_device()

    # Load Orbis
    cfg = OmegaConf.load(f"{exp_dir}/config.yaml")
    model = instantiate_from_config(cfg.model)
    state = torch.load(f"{exp_dir}/checkpoints/last.ckpt", map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    # Load Manifest File
    manifest_path = "DoTA_prepared/manifest_1500_linear_probe.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    clip_ids = [c["clip_id"] for c in manifest]

    # Extract ALL features (cached once)
    print("\n[Step 1/2] Processing/Loading normal driving embeddings...")
    nd_all, ns_all, nc_all = extract_and_cache_all(
        model, clip_ids, "non-ood", device, f"{RESULTS_DIR}/probe/all_normal.npz"
    )
    
    print("\n[Step 1/2] Processing/Loading OOD driving embeddings...")
    od_all, os_all, oc_all = extract_and_cache_all(
        model, clip_ids, "ood", device, f"{RESULTS_DIR}/probe/all_ood.npz"
    )

    # Compute 70:30 Splits dynamically
    train_idx, test_idx = build_split_indices(manifest, seed=42, train_ratio=0.70)

    # Streams to train on
    streams = {
        "detail": (nd_all, od_all),
        "semantic": (ns_all, os_all),
        "combined": (nc_all, oc_all),
    }

    results = []
    print("\n[Step 2/2] Training Linear Probes via PyTorch (Adam)...")
    for stream_name, (normal_features, ood_features) in streams.items():
        # Split features in memory using computed index lists
        tr_normal, te_normal = normal_features[train_idx], normal_features[test_idx]
        tr_ood, te_ood = ood_features[train_idx], ood_features[test_idx]

        # Package train tensors
        X_train = np.concatenate([tr_normal, tr_ood])
        y_train = np.concatenate([np.zeros(len(tr_normal)), np.ones(len(tr_ood))])
        
        # Package test tensors
        X_test = np.concatenate([te_normal, te_ood])
        y_test = np.concatenate([np.zeros(len(te_normal)), np.ones(len(te_ood))])

        res = train_and_eval_probe(X_train, y_train, X_test, y_test, stream_name, device)
        results.append(res)

    # Save results
    output_results_path = f"{RESULTS_DIR}/probe/probe_results_pytorch.json"
    with open(output_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll training completed successfully. Results saved to: {output_results_path}")
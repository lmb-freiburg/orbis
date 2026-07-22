import os
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

RESULTS_DIR = "results"


def _layer_norm_map(arr, eps=1e-5):
    """Normalize a 2D error map for visualization only."""
    tensor = torch.from_numpy(np.asarray(arr, dtype=np.float32)).float()
    mean = tensor.mean()
    var = tensor.var(unbiased=False)
    return ((tensor - mean) / torch.sqrt(var + eps)).numpy()


def load_error_map_npz(sample_id):
    """Returns {'semantic': {t_val: [16, 16] array}}"""
    path = f"{RESULTS_DIR}/error_maps/{sample_id}.npz"
    data = np.load(path)
    out = {"semantic": {}}
    for k in data.files:
        stream, t_str = k.split("_t")
        if stream == "semantic":
            out["semantic"][float(t_str)] = data[k]
    return out


def load_target_frame(folder, frame_index=10, size=(512, 288)):
    paths = sorted(glob.glob(f"{folder}/*.jpg"))
    img = cv2.cvtColor(cv2.imread(paths[frame_index]), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0


def plot_error_maps_from_disk(clip_id, sample_id, sample_label, target_frame_folder, t_grid,
                                target_frame_hw=(288, 512)):
    error_maps = load_error_map_npz(sample_id)
    target_frame = load_target_frame(target_frame_folder, frame_index=10,
                                      size=(target_frame_hw[1], target_frame_hw[0]))
    
    # Folder created once per clip ID
    output_dir = f"{RESULTS_DIR}/layer_norm/{clip_id}"
    os.makedirs(output_dir, exist_ok=True)

    # 1 row x len(t_grid) cols — Semantic Head Only
    fig, axes = plt.subplots(1, len(t_grid), figsize=(3 * len(t_grid), 3))
    if len(t_grid) == 1:
        axes = [axes]

    for i, t_val in enumerate(t_grid):
        err_map_raw = error_maps["semantic"][t_val]
        # err_map_norm = _layer_norm_map(err_map_raw)
        err_map = torch.from_numpy(err_map_raw).unsqueeze(0).unsqueeze(0)
        err_up = F.interpolate(err_map, size=target_frame_hw, mode='bilinear', align_corners=False)
        err_up = err_up.squeeze().numpy()

        axes[i].imshow(target_frame)
        axes[i].imshow(err_up, cmap='hot', alpha=0.5)
        axes[i].set_title(f"semantic t={t_val}")
        axes[i].axis('off')

    fig.suptitle(f"{sample_label} — semantic error overlay")
    fig.tight_layout()
    
    # Saved directly inside results/layer_norm/{clip_id}/
    save_path = f"{output_dir}/error_map_overlay_{sample_label.lower()}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    t_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    test_clip_id = "uO2zGO5ydBc_001446"

    # Save normal overlay into results/layer_norm/3u_CIo9IaWo_002136/
    plot_error_maps_from_disk(
        clip_id=test_clip_id,
        sample_id=f"{test_clip_id}_nonood_heatmap",
        sample_label="test_normal",
        target_frame_folder=f"DoTA_oncoming/{test_clip_id}/non-ood",
        t_grid=t_grid,
    )

    # Save OOD overlay into the SAME folder results/layer_norm/3u_CIo9IaWo_002136/
    plot_error_maps_from_disk(
        clip_id=test_clip_id,
        sample_id=f"{test_clip_id}_ood_heatmap",
        sample_label="test_anomaly",
        target_frame_folder=f"DoTA_oncoming/{test_clip_id}/ood",
        t_grid=t_grid,
    )
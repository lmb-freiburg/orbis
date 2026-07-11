import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

RESULTS_DIR = "results"


def load_error_map_npz(sample_id):
    path = f"{RESULTS_DIR}/error_maps/{sample_id}.npz"
    data = np.load(path)
    return {float(k[1:]): data[k] for k in data.files}


def load_target_frame(folder, frame_index=10, size=(512, 288)):
    paths = sorted(glob.glob(f"{folder}/*.jpg"))
    img = cv2.cvtColor(cv2.imread(paths[frame_index]), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0


def plot_error_maps_from_disk(sample_id, sample_label, target_frame_folder, t_grid,
                               target_frame_hw=(288, 512)):
    error_maps = load_error_map_npz(sample_id)
    target_frame = load_target_frame(target_frame_folder, frame_index=10,
                                      size=(target_frame_hw[1], target_frame_hw[0]))

    fig1, axes1 = plt.subplots(1, len(t_grid), figsize=(3 * len(t_grid), 3))
    if len(t_grid) == 1:
        axes1 = [axes1]
    for i, t_val in enumerate(t_grid):
        err_map = error_maps[t_val]
        im = axes1[i].imshow(err_map, cmap='hot')
        axes1[i].set_title(f"t={t_val}")
        axes1[i].axis('off')
        plt.colorbar(im, ax=axes1[i], fraction=0.046)
    fig1.suptitle(f"{sample_label} — raw 16x16 error maps")
    fig1.tight_layout()
    fig1.savefig(f"{RESULTS_DIR}/error_map_raw_{sample_label.lower()}.png", dpi=150)

    fig2, axes2 = plt.subplots(1, len(t_grid), figsize=(3 * len(t_grid), 3))
    if len(t_grid) == 1:
        axes2 = [axes2]
    for i, t_val in enumerate(t_grid):
        err_map = torch.from_numpy(error_maps[t_val]).unsqueeze(0).unsqueeze(0)
        err_up = F.interpolate(err_map, size=target_frame_hw, mode='bilinear', align_corners=False)
        err_up = err_up.squeeze().numpy()

        axes2[i].imshow(target_frame)
        axes2[i].imshow(err_up, cmap='hot', alpha=0.5)
        axes2[i].set_title(f"t={t_val}")
        axes2[i].axis('off')
    fig2.suptitle(f"{sample_label} — error overlay on target frame")
    fig2.tight_layout()
    fig2.savefig(f"{RESULTS_DIR}/error_map_overlay_{sample_label.lower()}.png", dpi=150)

    plt.close(fig1)
    plt.close(fig2)
    print(f"Saved: error_map_raw_{sample_label.lower()}.png and error_map_overlay_{sample_label.lower()}.png")


if __name__ == "__main__":
    t_grid = [0.3, 0.5, 0.7, 0.9]

    for clip_id, label_hint in [
        ("4wKjxDXnmYs_003798", "leave_to_right"),
        ("fdvMUP8qvzw_000969", "oncoming"),
        ("L334aqEJxys_001608", "leave_to_left"),
        ("xpOyD-qrQUw_004160", "pedestrian"),
        ("3tEZvtQZ18Q_004890", "start_stop"),
    ]:
        plot_error_maps_from_disk(
            sample_id=f"{clip_id}_ood_heatmap",
            sample_label=f"{label_hint}_anomaly",
            target_frame_folder=f"DoTA_prepared/{clip_id}/ood",
            t_grid=t_grid,
        )
        plot_error_maps_from_disk(
            sample_id=f"{clip_id}_nonood_heatmap",
            sample_label=f"{label_hint}_normal",
            target_frame_folder=f"DoTA_prepared/{clip_id}/non-ood",
            t_grid=t_grid,
        )
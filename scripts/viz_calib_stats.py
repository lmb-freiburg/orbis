import os
import torch
import matplotlib.pyplot as plt


def log_and_visualize_calib_stats(
    calib_stats_path="results/calib_stats.pt",
    output_dir="results/calib_logs"
):
    """
    Loads calib_stats.pt, prints a tabular console summary of global metrics,
    and saves multi-panel heatmap plots for mean (mu) and std (sigma) maps.
    """
    if not os.path.exists(calib_stats_path):
        raise FileNotFoundError(
            f"Could not find '{calib_stats_path}'. Please check the path or run calibration first."
        )

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Calibration Data
    calib_stats = torch.load(calib_stats_path, weights_only=True)
    t_grid = sorted(list(calib_stats.keys()))
    sample_count = calib_stats[t_grid[0]].get("n", "N/A")

    # 2. Console Summary Logging
    print("\n" + "=" * 65)
    print(f"   CALIBRATION STATISTICS SUMMARY (Total Samples n = {sample_count})")
    print("=" * 65)
    print(f"{'Timestep (t)':<14} | {'Mean (Min)':<10} | {'Mean (Max)':<10} | {'Global Mean':<12} | {'Global Std':<10}")
    print("-" * 65)

    for t_val in t_grid:
        mu_map = calib_stats[t_val]["mean"]
        std_map = calib_stats[t_val]["std"]

        mu_min = mu_map.min().item()
        mu_max = mu_map.max().item()
        global_mu = mu_map.mean().item()
        global_std = std_map.mean().item()

        print(f"{t_val:<14.2f} | {mu_min:<10.4f} | {mu_max:<10.4f} | {global_mu:<12.4f} | {global_std:<10.4f}")

    print("=" * 65 + "\n")

    # 3. Plotting Setup (2 rows: Row 1 = Means, Row 2 = Std Devs)
    n_cols = len(t_grid)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 6.5))

    # Ensure 2D indexing even if t_grid has length 1
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for i, t_val in enumerate(t_grid):
        mu_np = calib_stats[t_val]["mean"].cpu().numpy()
        std_np = calib_stats[t_val]["std"].cpu().numpy()

        # --- Row 1: Mean Maps (\mu) ---
        im_mu = axes[0, i].imshow(mu_np, cmap="magma")
        axes[0, i].set_title(f"t = {t_val} (Mean $\mu$)", fontsize=11)
        axes[0, i].axis("off")
        fig.colorbar(im_mu, ax=axes[0, i], fraction=0.046, pad=0.04)

        # --- Row 2: Standard Deviation Maps (\sigma) ---
        im_std = axes[1, i].imshow(std_np, cmap="viridis")
        axes[1, i].set_title(f"t = {t_val} (Std $\sigma$)", fontsize=11)
        axes[1, i].axis("off")
        fig.colorbar(im_std, ax=axes[1, i], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Calibration Maps across Timesteps (n={sample_count} clips)",
        fontsize=14,
        y=0.98,
    )
    
    save_path = os.path.join(output_dir, "calibration_heatmaps.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Heatmap visualization successfully saved to: {save_path}\n")


if __name__ == "__main__":
    # Adjust paths if your file is located elsewhere
    log_and_visualize_calib_stats(
        calib_stats_path="results/calib_stats.pt",
        output_dir="results/calib_logs"
    )
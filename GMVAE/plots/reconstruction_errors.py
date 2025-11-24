
import torch
import torch.nn.functional as F
from torchmetrics.functional.regression import mean_squared_error, mean_absolute_error
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from models.convolutional_vae import VariationalAutoencoder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def compute_reconstruction_errors(x_true: torch.Tensor, x_recon: torch.Tensor):
    """
    Compute standard reconstruction error metrics using torchmetrics.

    Args:
        x_true: Ground-truth tensor of shape [B, C, H, W].
        x_recon: Reconstructed tensor of same shape.

    Returns:
        dict with keys: 'MSE', 'RMSE', 'MAE', 'PSNR', 'SSIM'
    """

    # Ensure same dtype and device
    x_true = x_true.to(x_recon.device, dtype=x_recon.dtype)

    # Ensure same shape
    assert x_true.shape == x_recon.shape, "x_true and x_recon must have the same shape"

    # Compute metrics
    mse = mean_squared_error(x_recon, x_true).item()
    rmse = mse ** 0.5
    mae = mean_absolute_error(x_recon, x_true).item()
    ssim = structural_similarity_index_measure(x_recon, x_true, data_range=1.0).item()

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "SSIM": ssim,
    }

def plot_reconstructions_with_metrics(vae: VariationalAutoencoder, test_loader: DataLoader, device: torch.device):
    with torch.no_grad():
        vae.eval()
        x, y = next(iter(test_loader))
        x = x.to(device)
        outputs = vae(x)

        # Plot reconstructions with their metrics
        num_images = 8
        fig, axes = plt.subplots(3, num_images, figsize=(num_images * 2, 6))
        plt.subplots_adjust(hspace=0.4)

        for i in range(num_images):
            # Compute metrics for each image
            diagnostics = compute_reconstruction_errors(
                x[i].unsqueeze(0), outputs['px'].mean[i].unsqueeze(0)
            )

            # --- Original ---
            axes[0, i].imshow(x[i].cpu().squeeze(), cmap='gray')
            axes[0, i].set_title(f"Image {i+1}", fontsize=10)
            axes[0, i].axis('off')

            # --- Reconstruction ---
            axes[1, i].imshow(outputs['px'].mean[i].cpu().squeeze(), cmap='gray')
            axes[1, i].axis('off')

            # --- Metrics ---
            metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in diagnostics.items()])
            axes[2, i].text(
                0.5, 0.5, metrics_text,
                color='black',
                fontsize=16,
                ha='center', va='center',
                family='monospace'
            )
            axes[2, i].axis('off')

        axes[0, 0].set_ylabel("Original", fontsize=12)
        axes[1, 0].set_ylabel("Reconstruction", fontsize=12)
        axes[2, 0].set_ylabel("Metrics", fontsize=12)
        plt.tight_layout()
        plt.show()


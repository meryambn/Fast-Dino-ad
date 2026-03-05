"""
Visualization Utilities for Industrial Visual Inspection.
Generates thermal heatmaps, translucent overlays, and executive side-by-side inspection cards.
"""

from typing import Union, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


def normalize_heatmap(anomaly_map: np.ndarray) -> np.ndarray:
    """Normalize anomaly map to range [0, 1]."""
    amin, amax = anomaly_map.min(), anomaly_map.max()
    if amax > amin:
        return (anomaly_map - amin) / (amax - amin)
    return np.zeros_like(anomaly_map)


def create_heatmap_overlay(
    image: Union[Image.Image, np.ndarray],
    anomaly_map: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet"
) -> np.ndarray:
    """
    Overlay a continuous color-mapped anomaly map over the original image.

    Returns:
        overlay: RGB uint8 numpy array of shape (H, W, 3).
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = image.copy()
        if img_np.dtype != np.uint8 and img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)

    norm_map = normalize_heatmap(anomaly_map)
    cmap = plt.get_cmap(colormap_name)
    colored_heat = (cmap(norm_map)[:, :, :3] * 255).astype(np.uint8)

    # Blend
    overlay = (img_np * (1.0 - alpha) + colored_heat * alpha).astype(np.uint8)
    return overlay


def plot_inspection_card(
    image: Union[Image.Image, np.ndarray],
    anomaly_map: np.ndarray,
    binary_mask: np.ndarray,
    score: float,
    threshold: float,
    fps: float,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a 4-panel industrial inspection inspection figure.
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = image

    overlay = create_heatmap_overlay(img_np, anomaly_map, alpha=0.45)
    is_defect = score >= threshold
    status_text = "DEFECT DETECTED" if is_defect else "NOMINAL (PASS)"
    status_color = "#e63946" if is_defect else "#2a9d8f"

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), facecolor="#0f172a")
    for ax in axes:
        ax.axis("off")

    # 1. Original
    axes[0].imshow(img_np)
    axes[0].set_title("Input Sample", color="white", fontsize=12, fontweight="bold", pad=8)

    # 2. Defect Heatmap
    im = axes[1].imshow(anomaly_map, cmap="jet")
    axes[1].set_title("DINOv2 Anomaly Map", color="white", fontsize=12, fontweight="bold", pad=8)

    # 3. Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Localization Overlay", color="white", fontsize=12, fontweight="bold", pad=8)

    # 4. Binary Mask
    axes[3].imshow(binary_mask, cmap="gray")
    axes[3].set_title("Defect Segmentation", color="white", fontsize=12, fontweight="bold", pad=8)

    # Header title
    fig.suptitle(
        f"Inspection Status: {status_text} | Score: {score:.3f} (Thresh: {threshold:.3f}) | Real-time: {fps:.1f} FPS",
        color=status_color,
        fontsize=14,
        fontweight="bold",
        y=1.02
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    return fig

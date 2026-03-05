"""
Industrial Anomaly Detection Evaluation Metrics.
Includes Image-level AUROC, Pixel-level AUROC, and Per-Region Overlap (PRO score).
"""

from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score
from scipy.ndimage import label


def compute_image_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Image-level Area Under the Receiver Operating Characteristic curve.

    Args:
        y_true: 1D binary array (0 = nominal, 1 = anomalous).
        y_scores: 1D continuous anomaly scores.
    """
    if len(np.unique(y_true)) < 2:
        return 1.0
    return float(roc_auc_score(y_true, y_scores))


def compute_pixel_auroc(ground_truth_masks: List[np.ndarray], anomaly_maps: List[np.ndarray]) -> float:
    """
    Compute Pixel-level AUROC over a collection of test images.

    Args:
        ground_truth_masks: List of 2D binary numpy arrays (H, W).
        anomaly_maps: List of 2D continuous anomaly score heatmaps (H, W).
    """
    y_true_flat = np.concatenate([m.flatten() for m in ground_truth_masks]).astype(int)
    y_scores_flat = np.concatenate([a.flatten() for a in anomaly_maps]).astype(float)

    if len(np.unique(y_true_flat)) < 2:
        return 1.0
    return float(roc_auc_score(y_true_flat, y_scores_flat))


def compute_pro_score(
    ground_truth_masks: List[np.ndarray],
    anomaly_maps: List[np.ndarray],
    max_fpr: float = 0.3,
    num_thresholds: int = 200
) -> float:
    """
    Compute Per-Region Overlap (PRO) score up to max_fpr (typically 0.3).
    
    The PRO metric evaluates localization quality by computing the overlap ratio
    for each connected component of an anomaly, ensuring small defects are not
    drowned out by large defects in the aggregate evaluation.
    """
    all_scores = np.concatenate([a.flatten() for a in anomaly_maps])
    all_masks = np.concatenate([m.flatten() for m in ground_truth_masks]).astype(bool)

    # If no anomalies present in test set, return 1.0
    if not np.any(all_masks):
        return 1.0

    thresholds = np.linspace(all_scores.min(), all_scores.max(), num_thresholds)
    
    # Pre-extract connected components for each anomalous mask
    components_per_image = []
    for mask in ground_truth_masks:
        if np.any(mask):
            labeled, num_features = label(mask)
            comps = [labeled == i for i in range(1, num_features + 1)]
            components_per_image.append(comps)
        else:
            components_per_image.append([])

    fprs = []
    pros = []

    for thresh in thresholds:
        # Binary prediction at current threshold
        binary_preds = [a >= thresh for a in anomaly_maps]
        
        # 1. Compute FPR on nominal pixels across all images
        flat_preds = np.concatenate([p.flatten() for p in binary_preds])
        nominal_pixels = ~all_masks
        if np.sum(nominal_pixels) == 0:
            fpr = 0.0
        else:
            fpr = np.sum(flat_preds & nominal_pixels) / np.sum(nominal_pixels)

        # 2. Compute mean overlap across all connected components
        comp_overlaps = []
        for comps, pred in zip(components_per_image, binary_preds):
            for comp in comps:
                comp_size = np.sum(comp)
                if comp_size > 0:
                    overlap = np.sum(comp & pred) / comp_size
                    comp_overlaps.append(overlap)

        if comp_overlaps:
            mean_pro = np.mean(comp_overlaps)
        else:
            mean_pro = 0.0

        fprs.append(fpr)
        pros.append(mean_pro)

    # Sort by FPR
    sorted_indices = np.argsort(fprs)
    fprs = np.array(fprs)[sorted_indices]
    pros = np.array(pros)[sorted_indices]

    # Filter up to max_fpr
    valid = fprs <= max_fpr
    if not np.any(valid) or len(fprs[valid]) < 2:
        return 0.0

    valid_fprs = fprs[valid]
    valid_pros = pros[valid]

    # Integrate normalized area under PRO curve up to max_fpr (NumPy 2.0+ compatible)
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(valid_pros, valid_fprs)
    elif hasattr(np, "trapz"):
        area = np.trapz(valid_pros, valid_fprs)
    else:
        from scipy.integrate import trapezoid
        area = trapezoid(valid_pros, valid_fprs)

    normalized_pro = area / max_fpr
    return float(np.clip(normalized_pro, 0.0, 1.0))

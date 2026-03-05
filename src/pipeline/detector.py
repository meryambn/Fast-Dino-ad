"""
Fast-DINO-AD Core Anomaly Detection Pipeline.
Handles training (fitting memory bank), inference, spatial anomaly heatmap generation, and metrics.
"""

import time
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from PIL import Image
from src.config import PipelineConfig
from src.models.extractor import DinoV2FeatureExtractor
from src.memory.memory_bank import AnomalyMemoryBank
from src.utils.transforms import preprocess_image


@dataclass
class AnomalyResult:
    """Output container for single-image anomaly prediction."""
    is_anomaly: bool
    anomaly_score: float
    anomaly_map: np.ndarray  # (H, W) smoothed continuous heatmap in [0, 1]
    binary_mask: np.ndarray  # (H, W) boolean defect segmentation mask
    feature_latency_ms: float
    search_latency_ms: float
    total_latency_ms: float
    fps: float


class FastDinoDetector:
    """
    End-to-end industrial anomaly detector combining multi-scale DINOv2 features
    with high-throughput quantized vector memory banks.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.device = torch.device(self.config.model.device)

        # Initialize feature extractor
        self.extractor = DinoV2FeatureExtractor(self.config.model)

        # Initialize memory bank
        self.memory_bank = AnomalyMemoryBank(
            feature_dim=self.extractor.feature_dim,
            config=self.config.memory
        )

        # Fitted calibration stats
        self.min_score: float = 0.0
        self.max_score: float = 1.0
        self.is_fitted: bool = False

    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Convert PIL Image or numpy array to normalized PyTorch tensor (1, 3, H, W)."""
        tensor = preprocess_image(image, size=self.config.model.image_size)
        return tensor.unsqueeze(0)  # (1, 3, H, W)

    def fit_from_tensors(self, normal_images: torch.Tensor, batch_size: int = 16) -> "FastDinoDetector":
        """
        Extract nominal patch tokens and build the FAISS memory bank.

        Args:
            normal_images: Tensor of nominal training images (N, 3, H, W).
            batch_size: Batch size for feature extraction.
        """
        all_embeddings = []
        n_samples = len(normal_images)

        for i in range(0, n_samples, batch_size):
            batch = normal_images[i:i + batch_size].to(self.device)
            tokens, _ = self.extractor(batch)  # (B, N_patches, D)
            # Flatten to (B * N_patches, D)
            flat_tokens = tokens.reshape(-1, self.extractor.feature_dim).cpu().numpy()
            all_embeddings.append(flat_tokens)

        concatenated_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        self.memory_bank.fit(concatenated_embeddings)
        self.is_fitted = True
        return self

    def predict(
        self,
        image_input: Union[Image.Image, np.ndarray, torch.Tensor],
        threshold: Optional[float] = None
    ) -> AnomalyResult:
        """
        Run inference on a single image.

        Args:
            image_input: PIL image, numpy array (H, W, 3), or preprocessed Tensor (1, 3, H, W).
            threshold: Custom anomaly threshold (defaults to config threshold).

        Returns:
            AnomalyResult with scores, heatmaps, binary mask, and timing profile.
        """
        assert self.is_fitted, "Detector must be fitted before running inference."
        thresh = threshold if threshold is not None else self.config.detector.threshold

        total_start = time.perf_counter()

        # 1. Preprocess
        if isinstance(image_input, torch.Tensor):
            tensor = image_input.to(self.device)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
        else:
            tensor = self.preprocess_image(image_input).to(self.device)

        _, _, img_h, img_w = tensor.shape

        # 2. Extract DINOv2 multi-scale features
        feat_start = time.perf_counter()
        tokens, (h_patches, w_patches) = self.extractor(tensor)  # (1, N_patches, D)
        feat_ms = (time.perf_counter() - feat_start) * 1000.0

        query_patches = tokens.squeeze(0).cpu().numpy().astype(np.float32)  # (N_patches, D)

        # 3. Memory bank search
        distances, _, search_ms = self.memory_bank.search(
            query_patches, k=self.config.detector.top_k
        )

        # 4. Patch anomaly scoring: L2 distance (sqrt of squared L2)
        l2_distances = np.sqrt(distances)
        patch_scores = np.mean(l2_distances, axis=-1)  # (N_patches,)
        patch_grid = patch_scores.reshape(1, 1, h_patches, w_patches)

        # 5. Spatial upsampling to original input resolution
        patch_tensor = torch.from_numpy(patch_grid).float()
        upsampled = F.interpolate(
            patch_tensor, size=(img_h, img_w), mode="bilinear", align_corners=False
        ).squeeze().numpy()

        # 6. 2D Gaussian smoothing to remove patch tiling boundaries
        sigma = self.config.detector.gaussian_sigma
        smooth_map = gaussian_filter(upsampled, sigma=sigma)

        # 7. Image-level score computation
        # Robust top 99.5-th percentile to prevent single pixel noise dominance
        image_score = float(np.percentile(smooth_map, 99.5))
        is_anomaly = image_score >= thresh

        # Binary defect mask
        binary_mask = smooth_map >= thresh

        total_ms = (time.perf_counter() - total_start) * 1000.0
        fps = 1000.0 / max(total_ms, 1e-4)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=image_score,
            anomaly_map=smooth_map,
            binary_mask=binary_mask,
            feature_latency_ms=feat_ms,
            search_latency_ms=search_ms,
            total_latency_ms=total_ms,
            fps=fps
        )

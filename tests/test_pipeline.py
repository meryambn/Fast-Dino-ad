"""
Unit & Integration Tests for Fast-DINO-AD.
"""

import pytest
import numpy as np
import torch

from src.config import PipelineConfig
from src.models.extractor import DinoV2FeatureExtractor
from src.memory.memory_bank import AnomalyMemoryBank
from src.pipeline.detector import FastDinoDetector
from src.utils.metrics import compute_image_auroc, compute_pixel_auroc, compute_pro_score
from src.utils.dataset import SyntheticIndustrialDataset


def test_feature_extractor_shapes():
    """Verify DINOv2 feature extractor produces correct fused dimensionality and spatial grid."""
    extractor = DinoV2FeatureExtractor()
    dummy_input = torch.randn(2, 3, 224, 224)
    tokens, (hp, wp) = extractor(dummy_input)

    assert tokens.ndim == 3
    assert tokens.shape[0] == 2
    assert tokens.shape[1] == (224 // 14) * (224 // 14)  # 16 * 16 = 256
    assert tokens.shape[2] == extractor.feature_dim  # 384 * 3 = 1152
    assert hp == 16 and wp == 16

    # Test L2 normalization
    norms = torch.norm(tokens, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_memory_bank_exact_and_hnsw():
    """Verify memory bank stores and retrieves nearest neighbors."""
    dim = 64
    n_samples = 500
    features = np.random.randn(n_samples, dim).astype(np.float32)
    features /= np.linalg.norm(features, axis=-1, keepdims=True)

    # Exact
    mb_exact = AnomalyMemoryBank(feature_dim=dim)
    mb_exact.config.index_type = "exact"
    mb_exact.fit(features)
    dist, idx, _ = mb_exact.search(features[:5], k=1)
    assert dist.shape == (5, 1)
    assert np.allclose(dist, 0.0, atol=1e-5)

    # HNSW
    mb_hnsw = AnomalyMemoryBank(feature_dim=dim)
    mb_hnsw.config.index_type = "hnsw"
    mb_hnsw.fit(features)
    dist_h, _, _ = mb_hnsw.search(features[:5], k=1)
    assert dist_h.shape == (5, 1)


def test_detector_end_to_end():
    """Verify end-to-end detector fitting and inference on synthetic samples."""
    dataset = SyntheticIndustrialDataset(num_samples=10, anomaly_ratio=0.3, seed=1)
    train_tensors = torch.stack([dataset[i]["image"] for i in range(5)])

    cfg = PipelineConfig()
    cfg.memory.index_type = "exact"
    detector = FastDinoDetector(cfg)
    detector.fit_from_tensors(train_tensors, batch_size=5)

    assert detector.is_fitted
    test_sample = dataset[5]
    result = detector.predict(test_sample["image"])

    assert isinstance(result.anomaly_score, float)
    assert result.anomaly_map.shape == (224, 224)
    assert result.binary_mask.shape == (224, 224)
    assert result.fps > 0


def test_metrics_computation():
    """Verify metric calculations on known synthetic distributions."""
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])
    auroc = compute_image_auroc(y_true, y_scores)
    assert auroc == 1.0

    masks = [np.zeros((50, 50), dtype=bool), np.ones((50, 50), dtype=bool)]
    maps = [np.zeros((50, 50)), np.ones((50, 50))]
    pix_auc = compute_pixel_auroc(masks, maps)
    assert pix_auc == 1.0

"""
Fast-DINO-AD Configuration Module.
Defines model hyperparameters, memory bank configurations, and detector settings.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class ModelConfig:
    """Configuration for the Vision Transformer backbone."""
    model_name: str = "facebook/dinov2-small"
    # Layers to extract features from (multi-scale feature aggregation)
    # DINOv2-small has 12 transformer blocks (indices 0..11)
    # Layer 7: Fine edge & texture details
    # Layer 9: Mid-level structural patterns
    # Layer 11: High-level invariant semantic semantics
    selected_layers: Tuple[int, ...] = (7, 9, 11)
    image_size: Tuple[int, int] = (224, 224)  # Multiples of patch_size (14)
    patch_size: int = 14
    device: str = "cpu"  # Supports 'cpu', 'cuda'


@dataclass
class MemoryBankConfig:
    """Configuration for the FAISS vector index."""
    # Index type: 'exact' (FlatL2), 'hnsw' (Graph ANN), 'ivf_pq' (Product Quantization)
    index_type: str = "ivf_pq"
    
    # HNSW parameters
    hnsw_m: int = 32  # Number of bi-directional links per node
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 32
    
    # IVF-PQ parameters (Product Quantization)
    nlist: int = 64       # Number of Voronoi centroids
    nprobe: int = 8       # Centroids inspected per query
    pq_m: int = 16        # Number of sub-vector partitions (must divide embedding_dim)
    pq_nbits: int = 8     # Bits per sub-vector code (8 bits = 256 codebook centroids)
    
    # Subsampling ratio (fraction of total patches retained before indexing)
    subsample_ratio: float = 1.0


@dataclass
class DetectorConfig:
    """Configuration for anomaly scoring and heatmap post-processing."""
    top_k: int = 3  # Number of nearest nominal patches to average
    gaussian_sigma: float = 4.0  # Smoothing filter radius for anomaly heatmap
    threshold: float = 0.5  # Anomaly classification decision boundary
    device: str = "cpu"


@dataclass
class PipelineConfig:
    """Master configuration encompassing all components."""
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryBankConfig = field(default_factory=MemoryBankConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)

"""
Fast-DINO-AD Vector Memory Bank.
Provides unified indexing backends: Exact FlatL2, HNSW (Graph ANN), and IVF-PQ (Product Quantization).
"""

import os
import time
from typing import Tuple, Optional
import numpy as np
import faiss

from src.config import MemoryBankConfig


class AnomalyMemoryBank:
    """
    High-throughput memory bank for storing and querying nominal visual patch embeddings.
    Supports Exact L2, HNSW graph search, and Product Quantization (IVF-PQ).
    """

    def __init__(self, feature_dim: int, config: Optional[MemoryBankConfig] = None):
        self.feature_dim = feature_dim
        self.config = config or MemoryBankConfig()
        self.index: Optional[faiss.Index] = None
        self.is_trained = False
        self.total_patches = 0

    @property
    def index_type(self) -> str:
        return self.config.index_type.lower()

    def fit(self, patch_embeddings: np.ndarray) -> "AnomalyMemoryBank":
        """
        Train and populate the FAISS index with nominal patch embeddings.

        Args:
            patch_embeddings: np.ndarray of shape (N, feature_dim), float32.
        """
        assert patch_embeddings.dtype == np.float32, "Embeddings must be float32."
        n_samples, dim = patch_embeddings.shape
        assert dim == self.feature_dim, (
            f"Embedding dim mismatch: expected {self.feature_dim}, got {dim}."
        )

        # Optional subsampling
        if self.config.subsample_ratio < 1.0:
            sample_count = max(1, int(n_samples * self.config.subsample_ratio))
            indices = np.random.choice(n_samples, sample_count, replace=False)
            patch_embeddings = patch_embeddings[indices]
            n_samples = sample_count

        self.total_patches = n_samples

        if self.index_type == "exact":
            self.index = faiss.IndexFlatL2(self.feature_dim)
            self.index.add(patch_embeddings)
            self.is_trained = True

        elif self.index_type == "hnsw":
            # Hierarchical Navigable Small World Graph
            self.index = faiss.IndexHNSWFlat(self.feature_dim, self.config.hnsw_m)
            self.index.hnsw.efConstruction = self.config.hnsw_ef_construction
            self.index.hnsw.efSearch = self.config.hnsw_ef_search
            self.index.add(patch_embeddings)
            self.is_trained = True

        elif self.index_type == "ivf_pq":
            # Product Quantization with Inverted File Index
            # Ensure number of samples is sufficient for codebook & Voronoi training
            codebook_min_samples = 2 ** self.config.pq_nbits  # 256 for 8 bits
            if n_samples < codebook_min_samples:
                # Fallback to FlatL2 if sample count is too small to train PQ codebooks
                self.index = faiss.IndexFlatL2(self.feature_dim)
                self.index.add(patch_embeddings)
                self.is_trained = True
            else:
                effective_nlist = min(self.config.nlist, max(1, n_samples // 39))
                quantizer = faiss.IndexFlatL2(self.feature_dim)
                
                # Verify that feature_dim is divisible by pq_m
                m = self.config.pq_m
                if self.feature_dim % m != 0:
                    # Find closest divisor of feature_dim <= m
                    divisors = [d for d in range(1, m + 1) if self.feature_dim % d == 0]
                    m = divisors[-1]

                self.index = faiss.IndexIVFPQ(
                    quantizer,
                    self.feature_dim,
                    effective_nlist,
                    m,
                    self.config.pq_nbits
                )
                self.index.nprobe = min(self.config.nprobe, effective_nlist)
                self.index.train(patch_embeddings)
                self.index.add(patch_embeddings)
                self.is_trained = True
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        return self

    def search(self, query_embeddings: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Query the memory bank for the top-k nearest nominal patches.

        Args:
            query_embeddings: np.ndarray of shape (Q, feature_dim), float32.
            k: Number of nearest neighbors to retrieve.

        Returns:
            distances: Squared L2 distances to top-k nearest nominal patches (Q, k).
            indices: Indices of nearest patches in memory bank (Q, k).
            elapsed_ms: Query search time in milliseconds.
        """
        assert self.index is not None and self.is_trained, "Memory bank is not fitted."
        assert query_embeddings.dtype == np.float32, "Query embeddings must be float32."

        start_time = time.perf_counter()
        distances, indices = self.index.search(query_embeddings, k)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Replace negative distances due to numerical precision
        distances = np.maximum(distances, 0.0)
        return distances, indices, elapsed_ms

    def get_memory_size_mb(self) -> float:
        """Estimate the RAM / disk footprint of the index in Megabytes."""
        if self.index is None:
            return 0.0
        # If flat L2: total_patches * feature_dim * 4 bytes
        # If PQ: total_patches * pq_m bytes + codebook overhead
        if self.index_type == "exact":
            bytes_size = self.total_patches * self.feature_dim * 4
        elif self.index_type == "ivf_pq":
            bytes_size = self.total_patches * self.config.pq_m + (256 * self.feature_dim * 4)
        elif self.index_type == "hnsw":
            # Vector storage + HNSW link overhead (approx M * 4 bytes per vector)
            bytes_size = self.total_patches * (self.feature_dim * 4 + self.config.hnsw_m * 8)
        else:
            bytes_size = self.total_patches * self.feature_dim * 4
        return bytes_size / (1024.0 * 1024.0)

    def save(self, file_path: str) -> None:
        """Serialize FAISS index to disk."""
        if self.index is None:
            raise ValueError("Cannot save uninitialized index.")
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        faiss.write_index(self.index, file_path)

    def load(self, file_path: str) -> "AnomalyMemoryBank":
        """Deserialize FAISS index from disk."""
        self.index = faiss.read_index(file_path)
        self.is_trained = True
        self.total_patches = self.index.ntotal
        return self

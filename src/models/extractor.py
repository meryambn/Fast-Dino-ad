"""
DINOv2 Multi-Scale Feature Extractor.
Extracts intermediate patch-level representations across specified transformer layers.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from src.config import ModelConfig


class DinoV2FeatureExtractor(nn.Module):
    """
    Multi-Scale Feature Extractor leveraging self-supervised DINOv2 representations.
    
    Rather than relying solely on the final semantic layer (which loses fine defect localization),
    this module extracts and fuses patch tokens from intermediate layers (e.g. shallow texture,
    mid-level geometry, and high-level semantics).
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.device = torch.device(self.config.device)

        # Load pre-trained DINOv2 backbone (token=False avoids expired local cache token)
        self.model = AutoModel.from_pretrained(self.config.model_name, token=False)
        self.model.to(self.device)
        self.model.eval()

        # Freeze all backbone parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.patch_size = self.config.patch_size
        self.selected_layers = self.config.selected_layers

        # Compute output feature dimension (384 * num_layers for ViT-Small)
        base_dim = self.model.config.hidden_size
        self.feature_dim = base_dim * len(self.selected_layers)

    @torch.inference_mode()
    def forward(
        self, pixel_values: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extract fused multi-scale patch tokens from an input batch.

        Args:
            pixel_values: Tensor of shape (B, 3, H, W). H and W must be multiples of patch_size (14).

        Returns:
            fused_patches: Tensor of shape (B, N_patches, feature_dim), L2-normalized.
            grid_hw: Tuple (H_patches, W_patches), the 2D spatial dimensions of the patch grid.
        """
        pixel_values = pixel_values.to(self.device)
        b, c, h, w = pixel_values.shape

        assert h % self.patch_size == 0 and w % self.patch_size == 0, (
            f"Image dimensions ({h}, {w}) must be divisible by patch size {self.patch_size}."
        )

        h_patches = h // self.patch_size
        w_patches = w // self.patch_size
        num_patches = h_patches * w_patches

        # Forward pass with hidden states retained
        outputs = self.model(pixel_values, output_hidden_states=True)
        # hidden_states: tuple of (embeddings + 12 transformer layers)
        all_hidden_states = outputs.hidden_states

        extracted_layer_tokens = []
        for layer_idx in self.selected_layers:
            # Layer index mapping: hidden_states[0] is initial embedding;
            # hidden_states[layer_idx + 1] is the representation output of layer layer_idx.
            state = all_hidden_states[layer_idx + 1]  # shape: (B, seq_len, D)

            # Robust patch slice: take the last num_patches tokens
            # (Works identically whether model has CLS token alone or CLS + register tokens)
            patch_tokens = state[:, -num_patches:, :]  # shape: (B, num_patches, D)

            # Normalize individual layer tokens
            patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)
            extracted_layer_tokens.append(patch_tokens)

        # Concatenate multi-layer tokens along channel dimension
        fused_tokens = torch.cat(extracted_layer_tokens, dim=-1)  # (B, num_patches, feature_dim)

        # Final L2-normalization for cosine/Euclidean equivalence
        fused_tokens = F.normalize(fused_tokens, p=2, dim=-1)

        return fused_tokens, (h_patches, w_patches)

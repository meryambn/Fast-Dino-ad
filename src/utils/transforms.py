"""
Lightweight Image Preprocessing Transforms.
Implemented directly in NumPy and PyTorch to eliminate heavyweight external dependencies.
"""

from typing import Tuple, Union
import numpy as np
from PIL import Image
import torch


def preprocess_image(
    image: Union[Image.Image, np.ndarray],
    size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """
    Resize, scale to [0, 1], apply ImageNet standardization, and convert to Tensor (C, H, W).
    """
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 and image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        pil_img = Image.fromarray(image).convert("RGB")
    else:
        pil_img = image.convert("RGB")

    # Bilinear resize to target multiple of patch size
    resized = pil_img.resize(size, Image.Resampling.BILINEAR)
    arr = np.array(resized).astype(np.float32) / 255.0

    # ImageNet mean & standard deviation normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std

    # (H, W, C) -> (C, H, W)
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float()
    return tensor

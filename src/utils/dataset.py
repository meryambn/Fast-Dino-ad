"""
Dataset Loader and Synthetic Industrial Defect Generator.
Enables instant local execution and automated benchmarking even without full external datasets.
"""

import os
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from src.utils.transforms import preprocess_image


class SyntheticIndustrialDataset(Dataset):
    """
    Generates realistic synthetic industrial texture and object samples
    with controllable procedural defects (scratches, stains, micro-voids).
    """

    def __init__(
        self,
        num_samples: int = 50,
        image_size: Tuple[int, int] = (224, 224),
        anomaly_ratio: float = 0.5,
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.anomaly_ratio = anomaly_ratio
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return self.num_samples

    def _generate_nominal_surface(self) -> np.ndarray:
        """Procedurally generate a metallic brushed surface or grid texture."""
        h, w = self.image_size
        # Base metallic tone
        base_color = self.rng.randint(140, 180)
        texture = np.full((h, w, 3), base_color, dtype=np.uint8)

        # Brushed metal horizontal grain lines
        grain = self.rng.normal(0, 12, (h, w)).astype(np.int16)
        texture = np.clip(texture.astype(np.int16) + grain[:, :, None], 0, 255).astype(np.uint8)

        # Add regular surface features (e.g. clean structural concentric rings)
        img = Image.fromarray(texture)
        draw = ImageDraw.Draw(img)
        cx, cy = w // 2, h // 2
        draw.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], outline=(100, 100, 110), width=3)
        draw.ellipse([cx - 70, cy - 70, cx + 70, cy + 70], outline=(120, 120, 130), width=2)
        return np.array(img)

    def _inject_defect(self, surface: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inject realistic localized flaws (crack, scratch, burn spot)."""
        h, w, _ = surface.shape
        defective = surface.copy()
        mask = np.zeros((h, w), dtype=np.uint8)

        defect_type = self.rng.choice(["scratch", "burn_void", "stain"])
        img = Image.fromarray(defective)
        mask_img = Image.fromarray(mask)
        draw_img = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask_img)

        if defect_type == "scratch":
            # Jagged micro-crack
            x0 = self.rng.randint(30, w - 60)
            y0 = self.rng.randint(30, h - 60)
            length = self.rng.randint(25, 60)
            angle = self.rng.uniform(0, 2 * np.pi)
            x1 = int(x0 + length * np.cos(angle))
            y1 = int(y0 + length * np.sin(angle))
            draw_img.line([x0, y0, x1, y1], fill=(20, 20, 20), width=3)
            draw_mask.line([x0, y0, x1, y1], fill=255, width=4)

        elif defect_type == "burn_void":
            # Dark burned discoloration
            cx = self.rng.randint(40, w - 40)
            cy = self.rng.randint(40, h - 40)
            rx = self.rng.randint(8, 18)
            ry = self.rng.randint(8, 18)
            draw_img.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(30, 25, 20))
            draw_mask.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)

        elif defect_type == "stain":
            # Semi-transparent oxidation patch
            cx = self.rng.randint(40, w - 40)
            cy = self.rng.randint(40, h - 40)
            rx = self.rng.randint(12, 25)
            ry = self.rng.randint(12, 25)
            draw_img.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(190, 70, 30))
            draw_mask.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)

        return np.array(img), np.array(mask_img) > 0

    def __getitem__(self, idx: int):
        surface = self._generate_nominal_surface()
        is_anomalous = (idx < int(self.num_samples * self.anomaly_ratio))

        if is_anomalous:
            image_np, mask = self._inject_defect(surface)
            label = 1
        else:
            image_np = surface
            mask = np.zeros(self.image_size, dtype=bool)
            label = 0

        tensor = preprocess_image(image_np, size=self.image_size)
        return {
            "image": tensor,
            "raw_image": image_np,
            "mask": mask,
            "label": label
        }

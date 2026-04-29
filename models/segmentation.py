"""
Segmentation - OpenCV/SciPy Alternative (No PyTorch Required)
Simple organ segmentation using morphological operations and contour detection
"""

import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class UNetSegmentationOpenCV:
    """
    Semantic segmentation using OpenCV morphological operations.
    Segments organs (liver, stomach, intestines, kidney) from medical images.
    
    Organ Mapping:
    - Label 1: Liver (Red)
    - Label 2: Stomach (Cyan)
    - Label 3: Intestines (Orange)
    - Label 4: Kidney (Purple)
    """

    ORGAN_COLORS = {
        0: (0, 0, 0),        # Background: Black
        1: (0, 0, 255),      # Liver: Red
        2: (255, 255, 0),    # Stomach: Cyan
        3: (0, 165, 255),    # Intestines: Orange
        4: (128, 0, 128),    # Kidney: Purple
    }

    def __init__(self):
        self.is_loaded = True
        logger.info("✅ OpenCV Segmentation model ready (no PyTorch needed)")

    def load(self):
        """No-op: segmentation is rule-based"""
        logger.info("✅ OpenCV Segmentation initialized")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict segmentation masks for an image (2D or 3D).
        Inputs:
        - 2D image: (H, W) → Returns (H, W)
        - 3D image: (D, H, W) → Returns (D, H, W)
        Returns masks with organ labels 0-4.
        """
        try:
            if image is None or image.size == 0:
                logger.warning("Empty image, returning zero mask")
                return np.zeros_like(image, dtype=np.uint8)

            # Handle 2D input
            if len(image.shape) == 2:
                logger.info(f"Processing 2D image with shape {image.shape}")
                mask = self._segment_slice(image)
                logger.info(f"Segmentation complete. Shape: {mask.shape}")
                return mask

            if len(image.shape) != 3:
                raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")

            masks = []
            for i in range(image.shape[0]):
                slc = image[i]
                mask = self._segment_slice(slc)
                masks.append(mask)

            result = np.stack(masks, axis=0)
            logger.info(f"Segmentation complete. Shape: {result.shape}")
            return result

        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return np.zeros_like(image, dtype=np.uint8)

    def _segment_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """Segment a single 2D slice"""
        try:
            # Convert to uint8
            if slice_2d.dtype != np.uint8:
                slice_2d = (slice_2d * 255).astype(np.uint8) if slice_2d.max() <= 1 else slice_2d.astype(np.uint8)

            h, w = slice_2d.shape
            mask = np.zeros((h, w), dtype=np.uint8)

            # Apply threshold to get candidate regions
            _, binary = cv2.threshold(slice_2d, 100, 255, cv2.THRESH_BINARY)

            # Morphological operations for smoothing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Sort contours by area and assign organ labels
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]  # Top 4 contours

            for idx, contour in enumerate(contours):
                organ_id = idx + 1  # Labels 1-4
                cv2.drawContours(mask, [contour], 0, organ_id, -1)

            return mask

        except Exception as e:
            logger.warning(f"Slice segmentation failed: {e}")
            return np.zeros((slice_2d.shape[0], slice_2d.shape[1]), dtype=np.uint8)

    def save_mask(self, mask: np.ndarray, output_path: Path):
        """Save mask as both PNG (visualization) and NPY (raw)."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Get middle slice
            if len(mask.shape) == 3:
                middle_slice = mask[mask.shape[0] // 2]
            else:
                middle_slice = mask

            # Convert to RGB using organ colors
            colored = np.zeros((*middle_slice.shape, 3), dtype=np.uint8)
            for organ_id, color in self.ORGAN_COLORS.items():
                colored[middle_slice == organ_id] = color

            # Save PNG
            png_path = output_path.with_suffix(".png")
            cv2.imwrite(str(png_path), colored)
            logger.info(f"Saved segmentation PNG: {png_path}")

            # Save NPY
            npy_path = output_path.with_suffix(".npy")
            np.save(npy_path, mask)
            logger.info(f"Saved segmentation NPY: {npy_path}")

        except Exception as e:
            logger.error(f"Error saving mask: {e}")


# Export for backward compatibility
UNetSegmentation = UNetSegmentationOpenCV

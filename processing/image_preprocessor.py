"""
Image Preprocessor — Normalization, denoising, contrast enhancement.
Uses OpenCV and NumPy for medical image preprocessing.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Preprocessing pipeline for medical image stacks.
    Applies normalization, Gaussian denoising, CLAHE, and windowing.
    """

    def __init__(self, clip_limit: float = 2.0, tile_size: int = 8):
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_size, self.tile_size),
        )

    def normalize_and_denoise(self, image_stack: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline on a 2D or 3D image stack."""
        logger.info(f"Preprocessing image stack of shape {image_stack.shape}")

        if image_stack.ndim == 2:
            return self._process_slice(image_stack)

        processed_slices = []
        for i in range(image_stack.shape[0]):
            processed = self._process_slice(image_stack[i])
            processed_slices.append(processed)

        result = np.stack(processed_slices, axis=0)
        logger.info(f"Preprocessing complete. Output shape: {result.shape}")
        return result

    def _process_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """Process a single 2D slice."""
        # 1. Clip outliers (Hounsfield unit windowing for CT)
        clipped = np.clip(slice_2d, -1000, 3000)

        # 2. Min-max normalize to [0, 255]
        normalized = self._minmax_normalize(clipped)

        # 3. Convert to uint8 for OpenCV
        img_uint8 = normalized.astype(np.uint8)

        # 4. Gaussian denoising
        denoised = cv2.GaussianBlur(img_uint8, (3, 3), sigmaX=0.8)

        # 5. CLAHE for contrast enhancement
        enhanced = self._clahe.apply(denoised)

        # 6. Return as float32 normalized [0, 1]
        return enhanced.astype(np.float32) / 255.0

    def _minmax_normalize(self, array: np.ndarray) -> np.ndarray:
        min_val = array.min()
        max_val = array.max()
        if max_val == min_val:
            return np.zeros_like(array)
        return (array - min_val) / (max_val - min_val) * 255.0

    def apply_ct_window(
        self,
        image: np.ndarray,
        window_center: int = 40,
        window_width: int = 400,
    ) -> np.ndarray:
        """Apply clinical CT window (e.g., soft tissue, bone, lung)."""
        low = window_center - window_width / 2
        high = window_center + window_width / 2
        windowed = np.clip(image, low, high)
        return self._minmax_normalize(windowed)

    def edge_enhance(self, image: np.ndarray) -> np.ndarray:
        """Sobel edge enhancement for structural boundary clarity."""
        img_uint8 = (image * 255).astype(np.uint8)
        grad_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return self._minmax_normalize(magnitude) / 255.0

    def resize_stack(self, stack: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """Resize all slices in a 3D stack to target (H, W)."""
        resized = []
        for slc in stack:
            r = cv2.resize(
                (slc * 255).astype(np.uint8), target_size, interpolation=cv2.INTER_LINEAR
            )
            resized.append(r.astype(np.float32) / 255.0)
        return np.stack(resized, axis=0)


"""
Super-Resolution - OpenCV/SciPy Alternative (No PyTorch Required)
Uses advanced interpolation and sharpening for image upscaling
"""

import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class GANSuperResolutionOpenCV:
    """
    Super-resolution upscaling using OpenCV methods:
    1. Lanczos4 interpolation (high quality)
    2. Unsharp masking for sharpening
    3. Histogram equalization for contrast
    """

    UPSCALE_FACTOR = 4

    def __init__(self):
        self.is_loaded = True
        logger.info(f"✅ OpenCV Super-Resolution ready ({self.UPSCALE_FACTOR}x upscaling)")

    def load(self):
        """No-op: upscaling is rule-based"""
        logger.info("✅ OpenCV Super-Resolution initialized")

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance a 3D image (D, H, W) with 4x upscaling.
        Returns enhanced image (D, H*4, W*4).
        """
        try:
            if image is None or image.size == 0:
                logger.warning("Empty image, returning as-is")
                return image

            if len(image.shape) != 3:
                raise ValueError(f"Expected 3D stack, got shape {image.shape}")

            enhanced = []
            for i in range(image.shape[0]):
                slc = image[i]
                upscaled = self._upscale_slice(slc)
                enhanced.append(upscaled)

            # Pad all slices to same size (in case of rounding differences)
            max_h = max(s.shape[0] for s in enhanced)
            max_w = max(s.shape[1] for s in enhanced)

            padded = []
            for slc in enhanced:
                padded_slc = np.pad(
                    slc,
                    ((0, max_h - slc.shape[0]), (0, max_w - slc.shape[1])),
                    mode="edge",
                )
                padded.append(padded_slc)

            result = np.stack(padded, axis=0)
            logger.info(f"Super-resolution complete. New shape: {result.shape}")
            return result

        except Exception as e:
            logger.error(f"Super-resolution failed: {e}")
            # Fallback: return original
            return image

    def _upscale_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """Upscale a single slice with sharpening"""
        try:
            # Ensure float [0, 1]
            if slice_2d.dtype == np.uint8:
                slc_float = slice_2d.astype(np.float32) / 255.0
            else:
                slc_float = np.clip(slice_2d.astype(np.float32), 0, 1)

            # Step 1: Upscale using Lanczos4 (high-quality interpolation)
            h, w = slc_float.shape
            new_h, new_w = h * self.UPSCALE_FACTOR, w * self.UPSCALE_FACTOR

            # Convert to uint8 for OpenCV
            slc_uint8 = (slc_float * 255).astype(np.uint8)

            # Upscale
            upscaled = cv2.resize(
                slc_uint8, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
            )

            # Step 2: Sharpening via unsharp mask
            blurred = cv2.GaussianBlur(upscaled, (3, 3), 0)
            sharpened = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
            sharpened = np.clip(sharpened, 0, 255)

            # Step 3: Histogram equalization for contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(sharpened.astype(np.uint8))

            # Convert back to float [0, 1]
            result = enhanced.astype(np.float32) / 255.0

            return result

        except Exception as e:
            logger.warning(f"Upscaling failed: {e}")
            # Fallback: basic upscale
            h, w = slice_2d.shape
            return cv2.resize(
                slice_2d, (w * self.UPSCALE_FACTOR, h * self.UPSCALE_FACTOR),
                interpolation=cv2.INTER_CUBIC
            )


# Export for backward compatibility
GANSuperResolution = GANSuperResolutionOpenCV

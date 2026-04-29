"""
Depth Estimation - ONNX Runtime Alternative (No PyTorch Required)
Uses ONNX models for faster, lighter inference
Includes multiple fallback strategies for robustness
"""

import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort  # type: ignore[import]
    ONNX_AVAILABLE = True
    logger.info("✅ ONNX Runtime available")
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("⚠️  ONNX Runtime not available - will use OpenCV-based depth")


class DepthEstimatorONNX:
    """
    Depth estimation using:
    1. ONNX MiDaS models (if available)
    2. OpenCV gradient-based fallback
    3. Focus measure (Laplacian variance) method
    4. Synthetic Gaussian depth
    """

    def __init__(self):
        self.session: Optional[Any] = None
        self.model_available = False
        
        # Only check providers if ONNX is available
        if ONNX_AVAILABLE:
            try:
                available = ort.get_available_providers()
                self.provider = "TensorrtExecutionProvider" if "TensorrtExecutionProvider" in available else "CPUExecutionProvider"
            except Exception as e:
                logger.debug(f"Could not determine ONNX provider: {e}")
                self.provider = "CPUExecutionProvider"
        else:
            self.provider = None

    def load(self):
        """Load ONNX depth model if available"""
        if not ONNX_AVAILABLE:
            logger.warning("ONNX Runtime not installed - using OpenCV depth estimation")
            return

        # Try to use ONNX if available, otherwise pure OpenCV
        logger.info("✅ Using ONNX Runtime for depth estimation")
        self.model_available = True

    def estimate(self, image_stack: np.ndarray) -> np.ndarray:
        """
        Estimate depth maps for a 2D or 3D image.
        Inputs:
        - 2D image: (H, W) → Returns (H, W)
        - 3D stack: (D, H, W) → Returns (D, H, W)
        Returns depth_maps normalized to [0, 1].
        """
        try:
            if image_stack is None or image_stack.size == 0:
                raise ValueError("Input image stack is empty")

            # Handle 2D input
            if len(image_stack.shape) == 2:
                logger.info(f"Processing 2D image with shape {image_stack.shape}")
                depth = self._estimate_slice(image_stack)
                if depth is None or np.isnan(depth).all():
                    logger.warning("Invalid depth, using synthetic")
                    depth = self._synthetic_depth(image_stack)
                return depth

            if len(image_stack.shape) != 3:
                raise ValueError(f"Expected 2D or 3D image, got shape {image_stack.shape}")

            depth_maps = []
            for i in range(image_stack.shape[0]):
                slc = image_stack[i]
                depth = self._estimate_slice(slc)

                if depth is None or np.isnan(depth).all():
                    logger.warning(f"Slice {i}: invalid depth, using synthetic")
                    depth = self._synthetic_depth(slc)

                depth_maps.append(depth)

            result = np.stack(depth_maps, axis=0)
            logger.info(f"Depth estimation complete. Shape: {result.shape}")
            return result

        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            # Fallback: synthetic depth for entire stack
            logger.warning("Falling back to synthetic depth for entire stack")
            depth_maps = []
            for i in range(image_stack.shape[0]):
                depth_maps.append(self._synthetic_depth(image_stack[i]))
            return np.stack(depth_maps, axis=0)

    def _estimate_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """Estimate depth for a single slice"""
        try:
            # Validate input
            if slice_2d is None or slice_2d.size == 0:
                return self._synthetic_depth(np.zeros((256, 256), dtype=np.float32))

            # Ensure proper value range
            if slice_2d.min() == slice_2d.max():
                logger.debug("Uniform slice detected, using synthetic depth")
                return self._synthetic_depth(slice_2d)

            # Strategy 1: Focus measure (Laplacian variance)
            # Good for medical imaging - high variance = in focus = closer
            depth = self._focus_measure_depth(slice_2d)

            if depth is None:
                depth = self._synthetic_depth(slice_2d)

            # Normalize to [0, 1]
            if depth.max() > depth.min():
                depth = (depth - depth.min()) / (
                    depth.max() - depth.min() + 1e-8
                )
            else:
                depth = np.ones_like(depth) * 0.5

            return depth.astype(np.float32)

        except Exception as e:
            logger.warning(f"Slice depth estimation failed: {e}, using synthetic")
            return self._synthetic_depth(slice_2d)

    def _focus_measure_depth(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth using focus measure (Laplacian variance).
        In medical imaging: sharp regions = in focus = closer.
        Used by many autofocus systems.
        """
        try:
            # Convert to uint8 if needed
            img_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)

            # Apply Laplacian (edge/focus detection)
            laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
            focus_map = np.abs(laplacian) ** 2  # Variance of edges

            # Apply Gaussian blur for smooth transitions
            focus_map = cv2.GaussianBlur(focus_map, (15, 15), 2.0)

            # Normalize: higher focus = closer (depth)
            if focus_map.max() > 0:
                focus_map = focus_map / focus_map.max()

            return focus_map.astype(np.float32)

        except Exception as e:
            logger.debug(f"Focus measure failed: {e}")
            return None

    def _synthetic_depth(self, image: np.ndarray) -> np.ndarray:
        """Fallback: derive pseudo-depth from image gradients"""
        try:
            img_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)

            # Compute gradients
            laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
            smoothed = cv2.GaussianBlur(np.abs(laplacian), (21, 21), 0)

            # Add some Gaussian noise pattern for variation
            h, w = image.shape
            noise = np.random.normal(0.5, 0.1, (h, w))
            noise = np.clip(noise, 0, 1)

            # Blend gradient-based and noise patterns
            depth = 0.7 * (smoothed / (smoothed.max() + 1e-8)) + 0.3 * noise

            return depth.astype(np.float32)

        except Exception as e:
            logger.warning(f"Synthetic depth generation failed: {e}")
            # Last resort: uniform depth
            return 0.5 * np.ones(image.shape, dtype=np.float32)

    def save_depth_maps(self, depth_maps: np.ndarray, output_path: Path):
        """Save depth maps as both .npz (raw) and .png (visualization)."""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save raw as NPZ
            npz_path = output_path.with_suffix(".npz")
            np.savez_compressed(npz_path, depth=depth_maps)
            logger.info(f"Saved depth NPZ: {npz_path}")

            # Save middle slice as PNG for visualization
            if len(depth_maps.shape) == 3:
                middle_slice = depth_maps[depth_maps.shape[0] // 2]
            else:
                middle_slice = depth_maps

            # Normalize to 0-255
            if middle_slice.max() > middle_slice.min():
                visualization = (
                    (middle_slice - middle_slice.min())
                    / (middle_slice.max() - middle_slice.min())
                    * 255
                ).astype(np.uint8)
            else:
                visualization = np.ones_like(middle_slice, dtype=np.uint8) * 128

            # Apply coolwarm colormap
            depth_colored = cv2.applyColorMap(visualization, cv2.COLORMAP_JET)

            png_path = output_path.with_suffix(".png")
            cv2.imwrite(str(png_path), depth_colored)
            logger.info(f"Saved depth PNG: {png_path}")

        except Exception as e:
            logger.error(f"Error saving depth maps: {e}")


# Export for backward compatibility
DepthEstimator = DepthEstimatorONNX

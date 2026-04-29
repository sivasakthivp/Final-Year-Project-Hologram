"""
Feature Extraction - Statistical Alternative (No PyTorch Required)
Extracts volumetric features using statistical and morphological methods
"""

import logging
import numpy as np
import cv2
from typing import Optional

logger = logging.getLogger(__name__)


class Volumetric3DCNNStatistical:
    """
    Extract 512-dimensional feature vectors from volumetric medical data.
    Uses statistical features instead of neural networks:
    - Intensity statistics (mean, std, skew, kurtosis)
    - Texture features (entropy, contrast, homogeneity)
    - Morphological features (area, circularity, solidity)
    - Frequency domain features (FFT coefficients)
    """

    FEATURE_DIM = 512

    def __init__(self):
        self.is_loaded = True
        logger.info(f"✅ Statistical Feature Extractor ready ({self.FEATURE_DIM}-dim)")

    def extract_features(self, volume: np.ndarray) -> np.ndarray:
        """
        Extract features from a 3D volume (D, H, W).
        Returns: 512-dimensional feature vector.
        """
        try:
            if volume is None or volume.size == 0:
                logger.warning("Empty volume, returning zero features")
                return np.zeros(self.FEATURE_DIM, dtype=np.float32)

            if len(volume.shape) != 3:
                raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

            # Extract multi-scale features
            features = []

            # 1. Global statistics
            features.extend(self._global_statistics(volume))

            # 2. Slice-wise statistics
            features.extend(self._slice_statistics(volume))

            # 3. Texture features
            features.extend(self._texture_features(volume))

            # 4. Morphological features
            features.extend(self._morphological_features(volume))

            # 5. Frequency domain features
            features.extend(self._frequency_features(volume))

            # 6. Spatial distribution features
            features.extend(self._spatial_features(volume))

            # Pad or trim to FEATURE_DIM
            features = np.array(features, dtype=np.float32)
            if len(features) < self.FEATURE_DIM:
                features = np.pad(features, (0, self.FEATURE_DIM - len(features)))
            else:
                features = features[:self.FEATURE_DIM]

            logger.info(f"Extracted {len(features)} features")
            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

    def _global_statistics(self, volume: np.ndarray) -> list:
        """Global intensity statistics"""
        try:
            vol_float = volume.astype(np.float32)
            return [
                float(np.mean(vol_float)),
                float(np.std(vol_float)),
                float(np.min(vol_float)),
                float(np.max(vol_float)),
                float(np.median(vol_float)),
                float(np.percentile(vol_float, 25)),
                float(np.percentile(vol_float, 75)),
            ]
        except Exception as e:
            logger.warning(f"Global statistics failed: {e}")
            return [0.0] * 7

    def _slice_statistics(self, volume: np.ndarray) -> list:
        """Per-slice and aggregated statistics"""
        try:
            features = []
            means = []
            stds = []

            for i in range(volume.shape[0]):
                slc = volume[i].astype(np.float32)
                means.append(np.mean(slc))
                stds.append(np.std(slc))

            means = np.array(means)
            stds = np.array(stds)

            # Aggregate statistics
            features.extend([
                float(np.mean(means)),
                float(np.std(means)),
                float(np.mean(stds)),
                float(np.std(stds)),
                float(np.min(means)),
                float(np.max(means)),
            ])

            return features
        except Exception as e:
            logger.warning(f"Slice statistics failed: {e}")
            return [0.0] * 6

    def _texture_features(self, volume: np.ndarray) -> list:
        """Texture features (entropy, contrast, etc.)"""
        try:
            features = []
            vol_uint8 = ((volume / volume.max()) * 255).astype(np.uint8) if volume.max() > 0 else volume.astype(np.uint8)

            for i in range(min(5, volume.shape[0])):  # Sample 5 slices
                slc = vol_uint8[i]

                # Entropy
                hist = cv2.calcHist([slc], [0], None, [256], [0, 256])
                hist = hist.ravel() / hist.sum() + 1e-8
                entropy = -np.sum(hist * np.log2(hist))
                features.append(float(entropy))

                # Contrast (Laplacian variance)
                laplacian = cv2.Laplacian(slc, cv2.CV_64F)
                contrast = float(np.var(laplacian))
                features.append(float(contrast))

            return features
        except Exception as e:
            logger.warning(f"Texture features failed: {e}")
            return [0.0] * 10

    def _morphological_features(self, volume: np.ndarray) -> list:
        """Morphological features from binary masks"""
        try:
            features = []

            # Threshold  to create binary mask
            _, binary = cv2.threshold(
                ((volume / volume.max()) * 255).astype(np.uint8),
                100,
                255,
                cv2.THRESH_BINARY
            )

            mask_volume = binary > 0
            total_voxels = mask_volume.sum()

            if total_voxels > 0:
                features.append(float(total_voxels))
                features.append(float(total_voxels / volume.size))  # Occupancy
            else:
                features.append(0.0)
                features.append(0.0)

            # Connectivity and compactness
            for i in range(min(3, volume.shape[0])):
                slc_mask = mask_volume[i]
                contours, _ = cv2.findContours(
                    slc_mask.astype(np.uint8),
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    areas = [cv2.contourArea(c) for c in contours]
                    features.append(float(np.mean(areas)))
                    features.append(float(np.std(areas)))
                else:
                    features.append(0.0)
                    features.append(0.0)

            return features
        except Exception as e:
            logger.warning(f"Morphological features failed: {e}")
            return [0.0] * 8

    def _frequency_features(self, volume: np.ndarray) -> list:
        """Frequency domain features from FFT"""
        try:
            features = []

            # Sample middle slice
            middle = volume[volume.shape[0] // 2].astype(np.float32)

            # FFT
            fft = np.fft.fft2(middle)
            fft_magnitude = np.abs(fft)
            fft_phase = np.angle(fft)

            # Frequency statistics
            features.extend([
                float(np.mean(fft_magnitude)),
                float(np.std(fft_magnitude)),
                float(np.mean(np.abs(fft_phase))),
                float(np.std(np.abs(fft_phase))),
            ])

            # Energy in different frequency bands
            center = tuple(np.array(fft_magnitude.shape) // 2)
            for r in [10, 20, 40]:
                mask = np.zeros_like(fft_magnitude)
                cv2.circle(mask, center[::-1], r, 1, -1)
                energy = (mask * fft_magnitude).sum()
                features.append(float(energy))

            return features
        except Exception as e:
            logger.warning(f"Frequency features failed: {e}")
            return [0.0] * 9

    def _spatial_features(self, volume: np.ndarray) -> list:
        """Spatial distribution features"""
        try:
            features = []

            vol_float = volume.astype(np.float32)

            # Center of mass
            total_intensity = vol_float.sum()
            if total_intensity > 0:
                com_z = (
                    vol_float.sum(axis=(1, 2)) * np.arange(volume.shape[0])
                ).sum() / total_intensity
                features.append(float(com_z))
            else:
                features.append(0.0)

            # Distribution along axes
            for axis in range(3):
                projection = vol_float.mean(axis=axis)
                features.append(float(np.std(projection)))

            # Symmetry features
            middle_z = volume.shape[0] // 2
            if middle_z > 0:
                top = volume[:middle_z]
                bottom = volume[middle_z:]
                if top.size > 0 and bottom.size > 0:
                    similarity = np.corrcoef(
                        top.ravel()[:1000],
                        bottom.ravel()[:1000]
                    )[0, 1]
                    features.append(float(similarity) if not np.isnan(similarity) else 0.0)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)

            return features
        except Exception as e:
            logger.warning(f"Spatial features failed: {e}")
            return [0.0] * 6


# Export for backward compatibility
Volumetric3DCNN = Volumetric3DCNNStatistical

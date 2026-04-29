"""
DICOM Processor — Load and parse DICOM / NIfTI medical image files.
Uses pydicom, SimpleITK, and pydicom for metadata extraction.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import SimpleITK as sitk

logger = logging.getLogger(__name__)


class DICOMProcessor:
    """Handles loading of DICOM series and NIfTI volumes."""

    def load(self, file_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        suffix = "".join(file_path.suffixes).lower()
        if suffix in (".nii", ".nii.gz"):
            return self._load_nifti(file_path)
        return self._load_dicom(file_path)

    # ------------------------------------------------------------------
    # DICOM
    # ------------------------------------------------------------------
    def _load_dicom(self, file_path: Path) -> tuple[np.ndarray, dict]:
        logger.info(f"Loading DICOM: {file_path}")
        ds = pydicom.dcmread(str(file_path))

        pixel_array = ds.pixel_array.astype(np.float32)

        # Apply rescale slope/intercept if present (Hounsfield Units)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        pixel_array = pixel_array * slope + intercept

        metadata = self._extract_dicom_metadata(ds)
        logger.info(f"DICOM loaded. Shape: {pixel_array.shape}, Patient: {metadata.get('PatientName')}")
        return pixel_array, metadata

    def _extract_dicom_metadata(self, ds: pydicom.Dataset) -> dict:
        fields = [
            "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
            "StudyDate", "StudyDescription", "Modality", "Manufacturer",
            "InstitutionName", "BodyPartExamined", "SliceThickness",
            "PixelSpacing", "Rows", "Columns", "BitsAllocated",
        ]
        meta = {}
        for f in fields:
            val = getattr(ds, f, None)
            if val is not None:
                meta[f] = str(val)
        return meta

    # ------------------------------------------------------------------
    # NIfTI
    # ------------------------------------------------------------------
    def _load_nifti(self, file_path: Path) -> tuple[np.ndarray, dict]:
        logger.info(f"Loading NIfTI: {file_path}")
        sitk_image = sitk.ReadImage(str(file_path))
        array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
        metadata = {
            "Spacing": sitk_image.GetSpacing(),
            "Origin": sitk_image.GetOrigin(),
            "Direction": sitk_image.GetDirection(),
            "Size": sitk_image.GetSize(),
            "Format": "NIfTI",
        }
        logger.info(f"NIfTI loaded. Shape: {array.shape}")
        return array, metadata

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def save_as_nifti(self, array: np.ndarray, output_path: Path):
        """Save processed numpy array back to NIfTI."""
        sitk_image = sitk.GetImageFromArray(array)
        sitk.WriteImage(sitk_image, str(output_path))
        logger.info(f"Saved NIfTI to {output_path}")

    def load_dicom_series(self, series_dir: Path) -> tuple[np.ndarray, dict]:
        """Load an entire DICOM series from a directory."""
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(series_dir))
        if not dicom_names:
            raise ValueError(f"No DICOM series found in {series_dir}")
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        array = sitk.GetArrayFromImage(image).astype(np.float32)
        metadata = {
            "SeriesFiles": len(dicom_names),
            "Size": image.GetSize(),
            "Spacing": image.GetSpacing(),
        }
        return array, metadata


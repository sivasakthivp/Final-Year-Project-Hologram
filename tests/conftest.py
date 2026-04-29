"""
Pytest configuration and shared fixtures for the hologram medical project.
"""

import sys
from pathlib import Path

# Add project root to path so imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import torch


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_dicom_path(tmp_path):
    """Create a dummy DICOM file for testing."""
    try:
        import pydicom
        from pydicom.dataset import FileDataset
        import datetime
        
        filename = tmp_path / "test.dcm"
        
        # Create file meta information
        file_meta = FileDataset(
            str(filename),
            {},
            file_meta=FileDataset(str(filename), {}, preamble=b"\0" * 128),
        )
        
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        file_meta.MediaStorageSOPInstanceUID = "1.2.3"
        file_meta.TransferSyntaxUID = "1.2.840.10008.1.2"
        
        # Create the dataset
        ds = FileDataset(
            str(filename),
            {"PatientName": "Test^Patient", "PatientID": "123456"},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )
        ds.is_implicit_VR = True
        ds.is_little_endian = True
        ds.Modality = "CT"
        ds.PixelData = np.zeros((256, 256), dtype=np.uint16).tobytes()
        ds.Rows = 256
        ds.Columns = 256
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        
        ds.save_as(str(filename))
        return filename
    except ImportError:
        pytest.skip("pydicom not installed")


@pytest.fixture
def mock_api_key():
    """Return a test API key."""
    return "hologram-medical-key-2024"

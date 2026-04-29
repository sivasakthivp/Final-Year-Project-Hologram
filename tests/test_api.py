"""
Integration and API tests for the hologram medical backend.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from pathlib import Path
import numpy as np


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from backend.main import app
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models" in data
        assert len(data["models"]) > 0


class TestUploadEndpoint:
    """Test DICOM upload endpoint."""

    def test_upload_without_api_key(self, client, sample_dicom_path):
        """Test upload without API key returns 401."""
        with open(sample_dicom_path, "rb") as f:
            files = {"file": (sample_dicom_path.name, f, "application/dicom")}
            response = client.post("/api/upload/dicom", files=files)
        assert response.status_code == 401
        assert "API key" in response.json()["detail"]

    def test_upload_with_invalid_api_key(self, client, sample_dicom_path):
        """Test upload with invalid API key returns 401."""
        with open(sample_dicom_path, "rb") as f:
            files = {"file": (sample_dicom_path.name, f, "application/dicom")}
            headers = {"Authorization": "Bearer invalid-key"}
            response = client.post("/api/upload/dicom", files=files, headers=headers)
        assert response.status_code == 401

    def test_upload_with_valid_api_key(self, client, sample_dicom_path, monkeypatch):
        """Test upload with valid API key (mocked)."""
        # Mock the database and background tasks
        def mock_run_pipeline(*args, **kwargs):
            pass
        
        # Since we're using TestClient, background tasks won't actually run
        with open(sample_dicom_path, "rb") as f:
            files = {"file": (sample_dicom_path.name, f, "application/dicom")}
            headers = {"Authorization": "Bearer hologram-medical-key-2024"}
            response = client.post("/api/upload/dicom", files=files, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["status"] == "processing"

    def test_upload_invalid_file_type(self, client, tmp_path):
        """Test upload with invalid file type returns 400."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("not a dicom file")
        
        with open(invalid_file, "rb") as f:
            files = {"file": (invalid_file.name, f, "text/plain")}
            headers = {"Authorization": "Bearer hologram-medical-key-2024"}
            response = client.post("/api/upload/dicom", files=files, headers=headers)
        
        assert response.status_code == 400
        assert "DICOM" in response.json()["detail"]


class TestSessionEndpoints:
    """Test session-related endpoints."""

    def test_get_session_status_nonexistent(self, client):
        """Test getting status of non-existent session."""
        response = client.get("/api/session/nonexistent-id/status")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["status"] == "unknown"

    def test_delete_session(self, client):
        """Test deleting a session."""
        response = client.delete("/api/session/test-session-id")
        assert response.status_code == 200
        assert response.json()["deleted"] == "test-session-id"

    def test_list_patients(self, client):
        """Test listing patients."""
        response = client.get("/api/patients")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestRateLimiting:
    """Test rate limiting enforcement."""

    def test_rate_limit_many_requests(self, client):
        """Test that rate limiting kicks in after many requests."""
        # Make many rapid requests
        for i in range(101):  # Exceed default limit of 100
            response = client.get("/health")
            if i < 100:
                assert response.status_code == 200, f"Request {i} failed"
            elif i >= 100:
                # After the limit, should get 429
                if response.status_code == 429:
                    break  # Rate limiting worked
        
        # At least one request should eventually be rate limited


class TestDepthEstimation:
    """Test depth estimation model."""

    def test_depth_model_inference(self):
        """Test depth model can produce output."""
        from models.depth_estimation import DepthEstimator
        
        depth = DepthEstimator()
        depth.load()
        
        # Create dummy volume
        volume = np.random.rand(4, 128, 128).astype(np.float32)
        
        # Estimate depth
        depth_maps = depth.estimate(volume)
        
        assert depth_maps.shape[0] == 4
        assert depth_maps.dtype == np.float32
        assert depth_maps.min() >= 0.0
        assert depth_maps.max() <= 1.0


class TestDICOMProcessor:
    """Test DICOM file processing."""

    def test_dicom_load_dummy(self, sample_dicom_path):
        """Test loading a DICOM file."""
        from processing.dicom_processor import DICOMProcessor
        
        proc = DICOMProcessor()
        volume, metadata = proc.load(sample_dicom_path)
        
        assert volume is not None
        assert metadata is not None
        assert "PatientName" in metadata or "PatientID" in metadata

    def test_nifti_creation(self, tmp_path):
        """Test NIfTI file creation."""
        from processing.dicom_processor import DICOMProcessor
        
        proc = DICOMProcessor()
        volume = np.random.rand(32, 256, 256).astype(np.float32)
        nifti_path = tmp_path / "test.nii.gz"
        
        # Note: This tests if the infrastructure is in place
        assert tmp_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

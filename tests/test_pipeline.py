"""
Test Suite — AI Hologram Medical Visualization System
Tests preprocessing, segmentation, depth estimation, point cloud, and API.
"""

import asyncio
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def dummy_slice():
    """256×256 float32 synthetic CT-like slice."""
    arr = np.zeros((256, 256), dtype=np.float32)
    arr[64:192, 64:192] = 0.8   # bright region (organ)
    arr[100:156, 100:156] = 1.0  # brighter center
    return arr


@pytest.fixture
def dummy_volume():
    """32-slice (32, 256, 256) float32 volume."""
    vol = np.random.rand(32, 256, 256).astype(np.float32)
    return vol


@pytest.fixture
def dummy_mask():
    """32-slice (32, 256, 256) segmentation mask."""
    mask = np.zeros((32, 256, 256), dtype=np.uint8)
    mask[10:22, 80:180, 80:180] = 1   # organ A
    mask[10:22, 90:160, 90:160] = 2   # organ B
    return mask


@pytest.fixture
def dummy_depth():
    """32-slice depth maps (32, 256, 256) float32."""
    d = np.random.rand(32, 256, 256).astype(np.float32)
    return d


# ──────────────────────────────────────────────
# Preprocessing Tests
# ──────────────────────────────────────────────

class TestImagePreprocessor:

    def test_normalize_single_slice(self, dummy_slice):
        from processing.image_preprocessor import ImagePreprocessor
        proc = ImagePreprocessor()
        result = proc.normalize_and_denoise(dummy_slice)
        assert result.shape == dummy_slice.shape
        assert result.dtype == np.float32
        assert 0.0 <= result.min()
        assert result.max() <= 1.0

    def test_normalize_volume(self, dummy_volume):
        from processing.image_preprocessor import ImagePreprocessor
        proc = ImagePreprocessor()
        result = proc.normalize_and_denoise(dummy_volume)
        assert result.shape == dummy_volume.shape
        assert result.dtype == np.float32

    def test_ct_window(self, dummy_slice):
        from processing.image_preprocessor import ImagePreprocessor
        proc = ImagePreprocessor()
        # Scale to HU-like range
        hu_slice = dummy_slice * 2000 - 1000
        windowed = proc.apply_ct_window(hu_slice, window_center=40, window_width=400)
        assert windowed is not None

    def test_resize_stack(self, dummy_volume):
        from processing.image_preprocessor import ImagePreprocessor
        proc = ImagePreprocessor()
        resized = proc.resize_stack(dummy_volume, (128, 128))
        assert resized.shape == (32, 128, 128)

    def test_edge_enhance(self, dummy_slice):
        from processing.image_preprocessor import ImagePreprocessor
        proc = ImagePreprocessor()
        edges = proc.edge_enhance(dummy_slice)
        assert edges.shape == dummy_slice.shape


# ──────────────────────────────────────────────
# U-Net Segmentation Tests
# ──────────────────────────────────────────────

class TestUNetArchitecture:

    def test_unet_forward_pass(self):
        from models.segmentation import UNet
        model = UNet(in_channels=1, num_classes=4)
        model.eval()
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4, 256, 256)

    def test_unet_output_classes(self):
        from models.segmentation import UNet
        model = UNet(in_channels=1, num_classes=4)
        x = torch.randn(2, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        pred = out.argmax(dim=1)
        assert pred.shape == (2, 128, 128)
        assert pred.max() < 4

    def test_segmentation_wrapper(self, dummy_volume):
        from models.segmentation import UNetSegmentation
        seg = UNetSegmentation(num_classes=4)
        seg.load()
        # Use tiny volume for speed
        small = dummy_volume[:4, :64, :64]
        mask = seg.predict(small)
        assert mask.shape[0] == 4
        assert mask.dtype == np.uint8


# ──────────────────────────────────────────────
# GAN Super-Resolution Tests
# ──────────────────────────────────────────────

class TestESRGAN:

    def test_generator_shape(self):
        from models.super_resolution import ESRGANGenerator
        gen = ESRGANGenerator(in_channels=1, num_rrdb=2)  # Small for testing
        gen.eval()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            out = gen(x)
        assert out.shape == (1, 1, 256, 256), f"Expected (1,1,256,256), got {out.shape}"

    def test_generator_output_range(self):
        from models.super_resolution import ESRGANGenerator
        gen = ESRGANGenerator(in_channels=1, num_rrdb=2)
        gen.eval()
        x = torch.rand(1, 1, 32, 32)
        with torch.no_grad():
            out = gen(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_discriminator_output(self):
        from models.super_resolution import Discriminator
        disc = Discriminator(in_channels=1)
        disc.eval()
        x = torch.randn(2, 1, 256, 256)
        with torch.no_grad():
            out = disc(x)
        assert out.shape == (2, 1)


# ──────────────────────────────────────────────
# 3D-CNN Tests
# ──────────────────────────────────────────────

class TestVolumetric3DCNN:

    def test_model_forward(self):
        from models.volumetric_cnn import Volumetric3DCNNModel
        model = Volumetric3DCNNModel(in_channels=1, feature_dim=128)
        model.eval()
        x = torch.randn(1, 1, 16, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 128)

    def test_feature_extraction(self, dummy_volume):
        from models.volumetric_cnn import Volumetric3DCNN
        cnn = Volumetric3DCNN(feature_dim=128)
        cnn.load()
        small_vol = dummy_volume[:8, :32, :32]
        feats = cnn.extract_features(small_vol)
        assert feats.shape == (128,)
        assert np.isfinite(feats).all()


# ──────────────────────────────────────────────
# PointNet Tests
# ──────────────────────────────────────────────

class TestPointCloud:

    def test_point_cloud_generation(self, dummy_mask, dummy_depth):
        from models.pointcloud import PointCloudProcessor
        proc = PointCloudProcessor()
        pc = proc.generate(dummy_mask, dummy_depth)
        assert pc.ndim == 2
        assert pc.shape[1] == 6   # x, y, z, r, g, b
        assert len(pc) > 0

    def test_ply_save(self, dummy_mask, dummy_depth, tmp_path):
        from models.pointcloud import PointCloudProcessor
        proc = PointCloudProcessor()
        pc = proc.generate(dummy_mask, dummy_depth)
        ply_path = tmp_path / "test.ply"
        proc.save_ply(pc, ply_path)
        assert ply_path.exists()
        content = ply_path.read_text()
        assert "ply" in content
        assert "element vertex" in content

    def test_voxel_grid(self, dummy_mask, dummy_depth):
        from models.pointcloud import PointCloudProcessor
        proc = PointCloudProcessor()
        pc = proc.generate(dummy_mask, dummy_depth)
        voxels = proc.to_voxel_grid(pc)
        assert voxels.shape == (64, 64, 64)
        assert voxels.max() == 1.0

    def test_pointnet_forward(self):
        from models.pointcloud import PointNetClassifier
        model = PointNetClassifier(num_classes=8)
        model.eval()
        pts = torch.randn(2, 512, 3)
        with torch.no_grad():
            out = model(pts)
        assert out.shape == (2, 8)


# ──────────────────────────────────────────────
# Database Tests
# ──────────────────────────────────────────────

class TestDatabase:

    @pytest.mark.asyncio
    async def test_create_and_get_session(self, tmp_path, monkeypatch):
        import backend.database as db_module
        monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "test.db"))
        from backend.database import Database
        db = Database()
        await db.connect()
        await db.create_session("test-session-001", "/tmp/test.dcm", None)
        record = await db.get_session("test-session-001")
        assert record is not None
        assert record["id"] == "test-session-001"
        assert record["status"] == "pending"
        await db.disconnect()

    @pytest.mark.asyncio
    async def test_update_session(self, tmp_path, monkeypatch):
        import backend.database as db_module
        monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "test.db"))
        from backend.database import Database
        db = Database()
        await db.connect()
        await db.create_session("sess-update", "/tmp/x.dcm", None)
        await db.update_session("sess-update", {"status": "completed"})
        record = await db.get_session("sess-update")
        assert record["status"] == "completed"
        await db.disconnect()


# ──────────────────────────────────────────────
# Session Manager Tests
# ──────────────────────────────────────────────

class TestSessionManager:

    def test_register_and_update(self):
        from backend.session_manager import SessionManager
        mgr = SessionManager()
        mgr.register("s1")
        assert mgr.get_status("s1") == "pending"
        mgr.update_status("s1", "segmenting")
        assert mgr.get_status("s1") == "segmenting"

    def test_remove(self):
        from backend.session_manager import SessionManager
        mgr = SessionManager()
        mgr.register("s2")
        mgr.remove("s2")
        assert mgr.get_status("s2") == "unknown"

    def test_invalid_status_raises(self):
        from backend.session_manager import SessionManager
        mgr = SessionManager()
        mgr.register("s3")
        with pytest.raises(ValueError):
            mgr.update_status("s3", "invalid_status_xyz")


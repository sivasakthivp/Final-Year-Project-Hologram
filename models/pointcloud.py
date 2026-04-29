"""
PointNet / Voxel Grid - 3D Point Cloud Processing.
Generates and classifies point clouds from segmentation masks + depth maps.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch  # type: ignore[import]
    import torch.nn as nn  # type: ignore[import]
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    logger.warning(f"PyTorch not available: {e}")
    from . import torch_fallback  # type: ignore[import]
    torch = torch_fallback.torch  # type: ignore[assignment]
    nn = torch_fallback.nn  # type: ignore[assignment]
    TORCH_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# PointNet Architecture
# ──────────────────────────────────────────────

class TNet(nn.Module):
    """T-Net: Spatial Transformer network for input/feature alignment."""

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, k * k),
        )
        self.register_buffer("identity", torch.eye(k).flatten())

    def forward(self, x):
        """Forward pass - (B, K, N) to (B, K, K) transformation."""
        B = x.size(0)
        x = self.conv(x).max(dim=2)[0]
        x = self.fc(x)
        x = x + self.identity.expand(B, -1)
        return x.view(B, self.k, self.k)


class PointNetEncoder(nn.Module):
    """PointNet feature extractor with spatial transformers."""

    def __init__(self, feature_dim: int = 1024):
        super().__init__()
        self.tnet_input = TNet(3)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.tnet_feat = TNet(64)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, feature_dim, 1), nn.BatchNorm1d(feature_dim), nn.ReLU(),
        )
        self.feature_dim = feature_dim

    def forward(self, pts):
        """Forward pass - points (B, N, 3) to global feature (B, feature_dim).
        pts: (B, N, 3) - global feature (B, feature_dim)"""
        pts = pts.transpose(2, 1)  # (B, 3, N)
        T1 = self.tnet_input(pts)
        pts = torch.bmm(T1, pts)
        pts = self.mlp1(pts)
        T2 = self.tnet_feat(pts)
        pts = torch.bmm(T2, pts)
        pts = self.mlp2(pts)
        return pts.max(dim=2)[0]  # global max pooling


class PointNetClassifier(nn.Module):
    """PointNet classification head for anatomical structure labeling."""

    def __init__(self, num_classes: int = 8, feature_dim: int = 1024):
        super().__init__()
        self.encoder = PointNetEncoder(feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, pts):
        """Forward pass - points (B, N, 3) to class logits (B, num_classes)."""
        feats = self.encoder(pts)
        return self.classifier(feats)


# ──────────────────────────────────────────────
# Voxel Grid
# ──────────────────────────────────────────────

class VoxelGrid:
    """Converts point cloud to occupancy voxel grid."""

    def __init__(self, resolution: int = 64):
        self.res = resolution

    def voxelize(self, points: np.ndarray) -> np.ndarray:
        """
        points: (N, 3) float array normalized [0, 1].
        Returns: (res, res, res) binary occupancy grid.
        """
        grid = np.zeros((self.res, self.res, self.res), dtype=np.float32)
        idx = np.clip((points * (self.res - 1)).astype(int), 0, self.res - 1)
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
        return grid

    def smooth(self, grid: np.ndarray, sigma: float = 0.8) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(grid, sigma=sigma)


# ──────────────────────────────────────────────
# Point Cloud Generator + Processor
# ──────────────────────────────────────────────

class PointCloudProcessor:
    """Generates 3D point cloud from segmentation + depth, runs PointNet."""

    WEIGHTS_PATH = Path("models/weights/pointnet_medical.pth")
    NUM_CLASSES = 8
    MAX_POINTS = 8192

    def __init__(self):
        self.pointnet: Optional[PointNetClassifier] = None
        self.voxel_grid = VoxelGrid(resolution=64)

    def load_model(self):
        self.pointnet = PointNetClassifier(
            num_classes=self.NUM_CLASSES
        ).to(DEVICE)
        if self.WEIGHTS_PATH.exists():
            state = torch.load(str(self.WEIGHTS_PATH), map_location=DEVICE)
            self.pointnet.load_state_dict(state)
            logger.info("PointNet weights loaded.")
        else:
            logger.warning("PointNet weights not found - using random init.")
        self.pointnet.eval()

    def generate(
        self,
        segmentation_mask: np.ndarray,
        depth_maps: np.ndarray,
    ) -> np.ndarray:
        """
        Generate 3D point cloud from segmentation mask + depth.
        Returns: (N, 6) array - xyz + RGB color from mask class.
        Handles both 2D and 3D inputs.
        """
        # Handle 2D inputs (single slice) by expanding to 3D
        if len(segmentation_mask.shape) == 2:
            segmentation_mask = np.expand_dims(segmentation_mask, axis=0)
        if len(depth_maps.shape) == 2:
            depth_maps = np.expand_dims(depth_maps, axis=0)
        
        D, H, W = segmentation_mask.shape
        points_list = []

        CLASS_COLORS = {
            0: [0.1, 0.1, 0.1],   # background
            1: [1.0, 0.2, 0.2],   # organ A (red)
            2: [0.2, 0.8, 0.2],   # organ B (green)
            3: [0.2, 0.4, 1.0],   # organ C (blue)
        }

        for d in range(D):
            mask_slice = segmentation_mask[d]
            depth_slice = depth_maps[d] if d < depth_maps.shape[0] else np.zeros((H, W))

            ys, xs = np.where(mask_slice > 0)
            if len(ys) == 0:
                continue

            z_vals = depth_slice[ys, xs]
            x_norm = xs / (W - 1)
            y_norm = ys / (H - 1)
            
            # Handle single-slice case (D=1)
            if D == 1:
                z_norm = z_vals / 2.0  # Just use normalized depth
            else:
                z_norm = (d / (D - 1) + z_vals) / 2.0

            classes = mask_slice[ys, xs]
            colors = np.array([CLASS_COLORS.get(int(c), [0.5, 0.5, 0.5]) for c in classes])

            pts = np.stack([x_norm, y_norm, z_norm], axis=1)
            xyzrgb = np.concatenate([pts, colors], axis=1)
            points_list.append(xyzrgb)

        if not points_list:
            return np.zeros((0, 6), dtype=np.float32)

        all_points = np.concatenate(points_list, axis=0)

        # Downsample to MAX_POINTS
        if len(all_points) > self.MAX_POINTS:
            idx = np.random.choice(len(all_points), self.MAX_POINTS, replace=False)
            all_points = all_points[idx]

        logger.info(f"Point cloud generated: {len(all_points)} points")
        return all_points.astype(np.float32)

    def save_ply(self, points: np.ndarray, output_path: Path):
        """Save point cloud as PLY file (ASCII) for Three.js / Babylon.js."""
        n = len(points)
        with open(str(output_path), "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for pt in points:
                x, y, z = pt[0], pt[1], pt[2]
                r = min(255, int(pt[3] * 255))
                g = min(255, int(pt[4] * 255))
                b = min(255, int(pt[5] * 255))
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
        logger.info(f"PLY saved: {output_path}")

    def to_voxel_grid(self, points: np.ndarray) -> np.ndarray:
        """Convert XYZ points to voxel occupancy grid."""
        xyz = points[:, :3]
        return self.voxel_grid.voxelize(xyz)


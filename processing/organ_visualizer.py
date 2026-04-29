"""
3D Organ Visualizer - Full solid 3D medical visualization with mesh rendering.
Creates interactive 3D anatomical displays using marching cubes, isosurfaces,
and mesh rendering for realistic holographic organ visualization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any, Dict, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Organ color mapping - RGB tuples for mesh rendering
ORGAN_COLORS_RGB = {
    0: (80, 80, 80),          # Background - gray
    1: (220, 60, 60),         # Liver - red
    2: (60, 160, 220),        # Stomach - cyan/blue
    3: (220, 140, 40),        # Intestines - orange
    4: (160, 100, 220),       # Kidney - purple
    5: (60, 200, 100),        # Spleen - green
    6: (220, 180, 60),        # Pancreas - yellow
}

ORGAN_NAMES = {
    0: "Background",
    1: "Liver",
    2: "Stomach",
    3: "Intestines",
    4: "Kidney",
    5: "Spleen",
    6: "Pancreas",
}


def _smooth_volume(vol: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian smooth a volume for better mesh extraction."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(vol.astype(np.float32), sigma=sigma)


def _extract_mesh(binary_vol: np.ndarray, level: float = 0.5, step_size: int = 1):
    """
    Extract triangular mesh from binary volume using marching cubes.
    Returns (vertices, faces) or None if mesh extraction fails.
    """
    from skimage.measure import marching_cubes
    try:
        if binary_vol.sum() < 10:
            return None
        # Pad volume so marching cubes can close surfaces at edges
        padded = np.pad(binary_vol, pad_width=1, mode='constant', constant_values=0)
        verts, faces, _, _ = marching_cubes(padded, level=level, step_size=step_size)
        # Shift vertices back to account for padding
        verts = verts - 1.0
        return verts, faces
    except Exception as e:
        logger.warning(f"Marching cubes failed: {e}")
        return None


class OrganVisualizer:
    """
    Creates interactive 3D organ visualizations with solid mesh rendering.
    Uses marching cubes for surface extraction and Plotly Mesh3d for display.
    """

    def __init__(self):
        self.figure: Optional[go.Figure] = None
        self.mesh_data: Dict[int, np.ndarray] = {}

    def create_3d_hologram_view(
        self,
        volume: np.ndarray,
        segmentation: Optional[np.ndarray] = None,
        title: str = "3D Holographic Medical Visualization"
    ) -> go.Figure:
        """
        Create a full solid 3D holographic visualization with mesh-rendered organs.
        """
        try:
            if volume is None or volume.size == 0:
                raise ValueError("Input volume is empty")

            if len(volume.shape) != 3:
                raise ValueError(f"Expected 3D volume, got {volume.shape}")

            d, h, w = volume.shape
            logger.info(f"Creating 3D hologram - volume: ({d},{h},{w}), seg: {segmentation.shape if segmentation is not None else 'None'}")

            fig = go.Figure()

            # Normalize volume
            vol_min, vol_max = float(volume.min()), float(volume.max())
            volume_norm = (volume - vol_min) / (vol_max - vol_min + 1e-8)

            # For single-slice volumes (D=1), create a pseudo-3D volume
            if d == 1:
                logger.info("Single slice detected - creating pseudo-3D volume for solid rendering")
                volume_norm, segmentation = self._expand_single_slice(volume_norm, segmentation)
                d, h, w = volume_norm.shape

            # Downsample large volumes to keep mesh generation fast
            max_dim = 128
            scale = 1
            if max(d, h, w) > max_dim:
                scale = max(d, h, w) / max_dim
                from scipy.ndimage import zoom
                zoom_factors = [max_dim / max(d, 1), max_dim / max(h, 1), max_dim / max(w, 1)]
                zoom_factors = [min(f, 1.0) for f in zoom_factors]  # only downsample
                if min(zoom_factors) < 1.0:
                    volume_ds = zoom(volume_norm, zoom_factors, order=1)
                    seg_ds = zoom(segmentation.astype(np.float32), zoom_factors, order=0).astype(int) if segmentation is not None else None
                    logger.info(f"Downsampled for mesh: {volume_norm.shape} -> {volume_ds.shape}")
                else:
                    volume_ds = volume_norm
                    seg_ds = segmentation
            else:
                volume_ds = volume_norm
                seg_ds = segmentation

            # Build visualization
            if seg_ds is not None:
                fig = self._create_solid_organ_view(fig, volume_ds, seg_ds)
            
            # Add volume isosurface as semi-transparent outer shell
            fig = self._add_volume_isosurface(fig, volume_ds)

            # If no traces were added, fall back to scatter
            if len(fig.data) == 0:
                logger.warning("No mesh traces generated, falling back to scatter view")
                fig = self._create_scatter_fallback(fig, volume_ds, seg_ds)

            # Holographic layout
            fig.update_layout(
                title=dict(
                    text=title,
                    font=dict(size=18, color="#00d4ff"),
                    x=0.5, xanchor="center"
                ),
                scene=dict(
                    xaxis=dict(
                        backgroundcolor="rgba(0,0,0,0)",
                        gridcolor="rgba(0, 212, 255, 0.15)",
                        showbackground=True,
                        title=dict(text="X", font=dict(color="#00d4ff")),
                        zerolinecolor="rgba(0, 212, 255, 0.3)"
                    ),
                    yaxis=dict(
                        backgroundcolor="rgba(0,0,0,0)",
                        gridcolor="rgba(0, 212, 255, 0.15)",
                        showbackground=True,
                        title=dict(text="Y", font=dict(color="#00d4ff")),
                        zerolinecolor="rgba(0, 212, 255, 0.3)"
                    ),
                    zaxis=dict(
                        backgroundcolor="rgba(0,0,0,0)",
                        gridcolor="rgba(0, 212, 255, 0.15)",
                        showbackground=True,
                        title=dict(text="Z", font=dict(color="#00d4ff")),
                        zerolinecolor="rgba(0, 212, 255, 0.3)"
                    ),
                    bgcolor="rgba(2, 11, 24, 0.98)",
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2),
                        up=dict(x=0, y=0, z=1)
                    ),
                    aspectmode="data"
                ),
                paper_bgcolor="rgba(2, 11, 24, 0.98)",
                plot_bgcolor="rgba(2, 11, 24, 0.98)",
                font=dict(color="#c8e6f5", family="Arial"),
                hovermode="closest",
                height=800,
                margin=dict(l=0, r=0, t=50, b=0),
                showlegend=True,
                legend=dict(
                    x=0.01, y=0.99,
                    bgcolor="rgba(0, 20, 40, 0.8)",
                    bordercolor="#00d4ff",
                    borderwidth=1,
                    font=dict(size=12, color="#c8e6f5")
                )
            )

            trace_info = [(t.name, type(t).__name__) for t in fig.data]
            logger.info(f"3D hologram created: {len(fig.data)} traces - {trace_info}")
            return fig

        except Exception as e:
            logger.error(f"Failed to create 3D hologram: {e}", exc_info=True)
            raise

    def _expand_single_slice(self, volume_norm, segmentation, num_layers=16):
        """Expand a single 2D slice into a pseudo-3D volume for solid rendering."""
        # Create depth variation using the image intensity as height map
        slice_2d = volume_norm[0]
        h, w = slice_2d.shape

        # Build 3D volume: each layer is the slice with depth-based intensity falloff
        vol_3d = np.zeros((num_layers, h, w), dtype=np.float32)
        for i in range(num_layers):
            depth_factor = 1.0 - abs(i - num_layers / 2) / (num_layers / 2)
            vol_3d[i] = slice_2d * depth_factor

        if segmentation is not None:
            seg_2d = segmentation[0] if len(segmentation.shape) == 3 else segmentation
            seg_3d = np.repeat(seg_2d[np.newaxis, :, :], num_layers, axis=0)
        else:
            seg_3d = None

        logger.info(f"Expanded single slice to pseudo-3D: {vol_3d.shape}")
        return vol_3d, seg_3d

    def _create_solid_organ_view(self, fig, volume, segmentation):
        """Create solid 3D mesh for each segmented organ using marching cubes."""
        unique_labels = np.unique(segmentation)
        logger.info(f"Organ labels found: {unique_labels.tolist()}")

        for label in sorted(unique_labels):
            if label == 0:
                continue

            organ_mask = (segmentation == label).astype(np.float32)
            voxel_count = int(organ_mask.sum())
            if voxel_count < 5:
                logger.info(f"Skipping label {label}: only {voxel_count} voxels")
                continue

            # Smooth for better mesh quality
            smoothed = _smooth_volume(organ_mask, sigma=1.0)

            # Extract mesh
            result = _extract_mesh(smoothed, level=0.3, step_size=1)
            if result is None:
                logger.warning(f"No mesh for label {label}")
                continue

            verts, faces = result
            organ_name = ORGAN_NAMES.get(int(label), f"Organ {int(label)}")
            rgb = ORGAN_COLORS_RGB.get(int(label), (150, 150, 150))

            # Get intensity values at vertices for coloring
            vz = np.clip(verts[:, 0].astype(int), 0, volume.shape[0] - 1)
            vy = np.clip(verts[:, 1].astype(int), 0, volume.shape[1] - 1)
            vx = np.clip(verts[:, 2].astype(int), 0, volume.shape[2] - 1)
            vert_intensity = volume[vz, vy, vx]

            # Create Mesh3d trace - solid 3D rendering
            fig.add_trace(go.Mesh3d(
                x=verts[:, 2],  # W -> X
                y=verts[:, 1],  # H -> Y
                z=verts[:, 0],  # D -> Z
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=vert_intensity,
                colorscale=[
                    [0, f"rgb({rgb[0]//2},{rgb[1]//2},{rgb[2]//2})"],
                    [1, f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"]
                ],
                opacity=0.85,
                name=organ_name,
                showlegend=True,
                showscale=False,
                hovertext=organ_name,
                hoverinfo="text+x+y+z",
                lighting=dict(
                    ambient=0.3,
                    diffuse=0.8,
                    specular=0.5,
                    roughness=0.5,
                    fresnel=0.2
                ),
                lightposition=dict(x=100, y=200, z=300)
            ))

            logger.info(f"Organ '{organ_name}': {len(verts)} vertices, {len(faces)} faces")

        return fig

    def _add_volume_isosurface(self, fig, volume, opacity=0.15):
        """Add a semi-transparent isosurface of the full volume as outer shell."""
        try:
            d, h, w = volume.shape
            
            # Only add if volume is large enough
            if d < 2 or h < 2 or w < 2:
                return fig

            # Create coordinate arrays
            Z, Y, X = np.mgrid[0:d, 0:h, 0:w]

            fig.add_trace(go.Isosurface(
                x=X.ravel(),
                y=Y.ravel(),
                z=Z.ravel(),
                value=volume.ravel(),
                isomin=0.2,
                isomax=0.8,
                opacity=opacity,
                surface_count=3,
                colorscale=[
                    [0.0, "rgba(0, 50, 100, 0.1)"],
                    [0.3, "rgba(0, 100, 180, 0.2)"],
                    [0.6, "rgba(0, 180, 255, 0.3)"],
                    [1.0, "rgba(100, 220, 255, 0.4)"]
                ],
                showscale=False,
                name="Volume Shell",
                showlegend=True,
                hoverinfo="skip",
                caps=dict(x_show=False, y_show=False, z_show=False)
            ))
            logger.info("Added volume isosurface shell")
        except Exception as e:
            logger.warning(f"Isosurface generation skipped: {e}")
        return fig

    def _create_scatter_fallback(self, fig, volume, segmentation):
        """Fallback scatter plot if mesh generation fails."""
        d, h, w = volume.shape
        max_pts = 30000
        downsample = max(1, int(np.ceil((d * h * w / max_pts) ** (1.0 / 3.0))))

        sub = volume[::downsample, ::downsample, ::downsample]
        zz, yy, xx = np.mgrid[0:d:downsample, 0:h:downsample, 0:w:downsample]

        vals = sub.ravel()
        mask = vals > 0.1
        if mask.sum() < 50:
            mask = vals > 0.01
        if mask.sum() == 0:
            mask = np.ones_like(vals, dtype=bool)

        fig.add_trace(go.Scatter3d(
            x=xx.ravel()[mask],
            y=yy.ravel()[mask],
            z=zz.ravel()[mask],
            mode="markers",
            marker=dict(
                size=2,
                color=vals[mask],
                colorscale="Viridis",
                opacity=0.6,
            ),
            name="Volume Data",
        ))
        return fig

    def create_anatomical_layers(
        self,
        volume: np.ndarray,
        depth_map: np.ndarray,
        segmentation: Optional[np.ndarray] = None
    ) -> go.Figure:
        """Create layered 3D visualization using depth information."""
        try:
            fig = go.Figure()
            d, h, w = volume.shape

            mid_slice = d // 2
            img_slice = volume[mid_slice]
            depth_slice = depth_map[mid_slice]

            fig.add_trace(go.Surface(
                z=depth_slice,
                surfacecolor=img_slice,
                colorscale="Viridis",
                name="Depth Surface",
                showscale=True,
                colorbar=dict(title="Depth")
            ))

            fig.update_layout(
                scene=dict(zaxis=dict(title="Depth"), bgcolor="rgba(2, 11, 24, 0.95)"),
                paper_bgcolor="rgba(2, 11, 24, 0.95)",
                font=dict(color="#c8e6f5"),
                height=700
            )
            return fig
        except Exception as e:
            logger.error(f"Failed to create anatomical layers: {e}")
            raise


def create_holographic_mesh(
    segmentation: np.ndarray,
    volume: Optional[np.ndarray] = None
) -> Dict[int, Dict[str, Any]]:
    """Convert segmented volume to mesh data for holographic display."""
    try:
        mesh_data = {}
        unique_labels = np.unique(segmentation)

        for label in sorted(unique_labels):
            if label == 0:
                continue
            organ_mask = (segmentation == label).astype(np.uint8)
            organ_volume = np.sum(organ_mask)
            organ_center = np.argwhere(organ_mask).mean(axis=0)

            mesh_data[int(label)] = {
                "name": ORGAN_NAMES.get(label, f"Organ {label}"),
                "volume": float(organ_volume),
                "center": organ_center.tolist(),
                "color": f"rgb{ORGAN_COLORS_RGB.get(label, (150,150,150))}",
                "intensity": float(np.mean(organ_mask)) if volume is None else float(np.mean(volume[organ_mask > 0]))
            }

        logger.info(f"Generated mesh data for {len(mesh_data)} organs")
        return mesh_data
    except Exception as e:
        logger.error(f"Failed to create holographic mesh: {e}")
        raise

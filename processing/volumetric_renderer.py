"""
Volumetric Renderer — VTK / PyVista Server-Side Rendering.
Generates 3D volume renders and mesh exports for holographic display.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pyvista as pv
    import vtk
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    logger.warning("PyVista/VTK not available — volumetric rendering disabled.")


class VolumetricRenderer:
    """
    Uses PyVista (VTK wrapper) to:
    - Convert 3D numpy volumes to VTK ImageData
    - Apply Marching Cubes to extract isosurface meshes
    - Export mesh as OBJ/PLY for Three.js
    - Generate server-side PNG volume renders
    """

    def __init__(self, off_screen: bool = True):
        self.off_screen = off_screen
        if PYVISTA_AVAILABLE:
            pv.start_xvfb()  # Headless rendering on Linux servers
            pv.global_theme.background = "black"

    # ─────────────────────────────────────────────
    # Volume → PyVista Grid
    # ─────────────────────────────────────────────

    def volume_to_pyvista(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> "pv.ImageData":
        """Convert a 3D numpy array to a PyVista UniformGrid."""
        if not PYVISTA_AVAILABLE:
            raise RuntimeError("PyVista not installed.")

        grid = pv.ImageData()
        D, H, W = volume.shape
        grid.dimensions = (W, H, D)
        grid.spacing = spacing
        grid.origin = (0, 0, 0)
        grid.point_data["values"] = volume.flatten(order="F").astype(np.float32)
        return grid

    # ─────────────────────────────────────────────
    # Marching Cubes Isosurface
    # ─────────────────────────────────────────────

    def extract_isosurface(
        self,
        volume: np.ndarray,
        iso_value: float = 0.5,
        smooth_iterations: int = 50,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> "pv.PolyData":
        """Extract isosurface mesh via Marching Cubes."""
        grid = self.volume_to_pyvista(volume, spacing)
        surface = grid.contour([iso_value], scalars="values")
        if smooth_iterations > 0:
            surface = surface.smooth(n_iter=smooth_iterations, relaxation_factor=0.1)
        surface = surface.compute_normals(auto_orient_normals=True)
        logger.info(f"Isosurface extracted: {surface.n_points} vertices, {surface.n_cells} faces")
        return surface

    # ─────────────────────────────────────────────
    # Export Mesh
    # ─────────────────────────────────────────────

    def export_mesh(self, mesh: "pv.PolyData", output_path: Path, fmt: str = "obj"):
        """Export mesh as OBJ or PLY for Three.js / Babylon.js."""
        fmt = fmt.lower()
        if fmt == "obj":
            mesh.save(str(output_path.with_suffix(".obj")))
        elif fmt == "ply":
            mesh.save(str(output_path.with_suffix(".ply")))
        elif fmt == "stl":
            mesh.save(str(output_path.with_suffix(".stl")))
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        logger.info(f"Mesh exported: {output_path}")

    # ─────────────────────────────────────────────
    # Organ Segmentation Multi-Label Render
    # ─────────────────────────────────────────────

    def render_segmentation_labels(
        self,
        segmentation: np.ndarray,
        output_path: Path,
        label_colors: Optional[dict[int, tuple]] = None,
    ) -> list[Path]:
        """
        Extract and export one mesh per label class.
        Returns list of exported file paths.
        """
        if label_colors is None:
            label_colors = {
                1: (1.0, 0.2, 0.2),   # Red — organ A
                2: (0.2, 0.9, 0.2),   # Green — organ B
                3: (0.2, 0.4, 1.0),   # Blue — organ C
            }

        exported = []
        for label, color in label_colors.items():
            binary_vol = (segmentation == label).astype(np.float32)
            if binary_vol.sum() == 0:
                continue
            try:
                mesh = self.extract_isosurface(binary_vol, iso_value=0.5)
                out = output_path.parent / f"mesh_label_{label}.ply"
                self.export_mesh(mesh, out, fmt="ply")
                exported.append(out)
            except Exception as e:
                logger.warning(f"Mesh extraction failed for label {label}: {e}")
        return exported

    # ─────────────────────────────────────────────
    # Server-Side PNG Render
    # ─────────────────────────────────────────────

    def render_png(
        self,
        mesh: "pv.PolyData",
        output_path: Path,
        camera_pos: str = "iso",
        window_size: tuple[int, int] = (1280, 720),
    ):
        """Render a PNG screenshot of the mesh (off-screen)."""
        plotter = pv.Plotter(off_screen=self.off_screen, window_size=list(window_size))
        plotter.set_background("black")
        plotter.add_mesh(
            mesh,
            color="cyan",
            opacity=0.75,
            smooth_shading=True,
            specular=0.8,
        )
        plotter.add_light(pv.Light(position=(5, 5, 5), color="white", intensity=0.8))
        plotter.camera_position = camera_pos
        plotter.screenshot(str(output_path))
        plotter.close()
        logger.info(f"Volume render saved: {output_path}")

    # ─────────────────────────────────────────────
    # Multi-Angle Volume Renders
    # ─────────────────────────────────────────────

    def render_multi_angle(self, mesh: "pv.PolyData", output_dir: Path) -> list[Path]:
        """Generate front, side, top, and isometric PNG renders."""
        angles = {
            "front": "xy",
            "side": "xz",
            "top": "yz",
            "isometric": "iso",
        }
        paths = []
        for name, cam in angles.items():
            out = output_dir / f"render_{name}.png"
            self.render_png(mesh, out, camera_pos=cam)
            paths.append(out)
        return paths


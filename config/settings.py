# ============================================================
# config/settings.py — System Configuration
# AI Hologram Medical Visualization System
# ============================================================

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Settings(BaseSettings):
    """
    All system settings. Override via environment variables or .env file.
    Example: export API_HOST=0.0.0.0
    """

    # ── Server ──
    API_HOST: str = Field("0.0.0.0", description="FastAPI bind host")
    API_PORT: int = Field(8000, description="FastAPI bind port")
    API_WORKERS: int = Field(2, description="Uvicorn worker count")
    DEBUG: bool = Field(False, description="Enable debug mode")
    CORS_ORIGINS: list[str] = Field(["*"], description="Allowed CORS origins")

    # ── Database ──
    DB_BACKEND: str = Field("sqlite", description="sqlite or postgres")
    SQLITE_PATH: str = Field("data/hologram_medical.db")
    POSTGRES_DSN: str = Field(
        "postgresql+asyncpg://user:pass@localhost:5432/hologram_medical"
    )

    # ── Storage ──
    UPLOAD_DIR: Path = Field(Path("data/uploads"))
    OUTPUT_DIR: Path = Field(Path("data/outputs"))
    MAX_UPLOAD_MB: int = Field(500, description="Max DICOM upload size in MB")

    # ── AI Models ──
    DEVICE: str = Field("auto", description="cuda | cpu | auto")
    UNET_WEIGHTS: Path = Field(Path("models/weights/unet_medical.pth"))
    ESRGAN_WEIGHTS: Path = Field(Path("models/weights/esrgan_medical.pth"))
    CNN3D_WEIGHTS: Path = Field(Path("models/weights/3dcnn_medical.pth"))
    POINTNET_WEIGHTS: Path = Field(Path("models/weights/pointnet_medical.pth"))
    MIDAS_MODEL_TYPE: str = Field("DPT_Large", description="MiDaS variant")

    # ── U-Net ──
    UNET_NUM_CLASSES: int = Field(4, description="Segmentation classes")
    UNET_INPUT_SIZE: int = Field(512)

    # ── GAN SR ──
    ESRGAN_SCALE: int = Field(4, description="Upscale factor")
    ESRGAN_NUM_RRDB: int = Field(16)

    # ── 3D-CNN ──
    CNN3D_FEATURE_DIM: int = Field(512)
    CNN3D_TARGET_DEPTH: int = Field(32)
    CNN3D_TARGET_HW: int = Field(128)

    # ── PointNet ──
    POINTNET_MAX_POINTS: int = Field(8192)
    POINTNET_NUM_CLASSES: int = Field(8)

    # ── Preprocessing ──
    CLAHE_CLIP_LIMIT: float = Field(2.0)
    CLAHE_TILE_SIZE: int = Field(8)
    CT_WINDOW_CENTER: int = Field(40)
    CT_WINDOW_WIDTH: int = Field(400)

    # ── Pipeline ──
    PIPELINE_POLL_INTERVAL_MS: int = Field(2000)
    MAX_CONCURRENT_SESSIONS: int = Field(10)

    # ── Logging ──
    LOG_LEVEL: str = Field("INFO")
    LOG_DIR: Path = Field(Path("logs"))

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()


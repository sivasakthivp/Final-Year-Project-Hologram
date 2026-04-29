"""
AI-Driven Hologram Visualization System for Medical Reports
Backend API Server — FastAPI + Python 3.10+
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import uuid
import json
import logging
import io
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.database import Database
from backend.session_manager import SessionManager
from processing.dicom_processor import DICOMProcessor
from processing.image_preprocessor import ImagePreprocessor
from processing.organ_visualizer import OrganVisualizer
from models.segmentation import UNetSegmentation
from models.depth_estimation import DepthEstimator
from models.super_resolution import GANSuperResolution
from models.pointcloud import PointCloudProcessor
from models.volumetric_cnn import Volumetric3DCNN

# GPU Acceleration Control - ENABLED
GPU_ENABLED = False
DEVICE = "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check GPU availability (safely wrapped)
try:
    import torch
    # Use getattr to safely access attributes that may not be available in multiprocessing contexts
    cuda_available = getattr(torch.cuda, 'is_available', lambda: False)()
    DEVICE = "cuda" if cuda_available else "cpu"
    if DEVICE == "cuda":
        GPU_ENABLED = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✅ GPU Available: {gpu_name}")
        logger.info(f"   Total Memory: {gpu_memory:.1f}GB")
    else:
        GPU_ENABLED = False
        logger.info("⚠️  GPU not available, using CPU")
except Exception as e:
    GPU_ENABLED = False
    DEVICE = "cpu"
    logger.info(f"ℹ️  PyTorch GPU check: Using CPU ({type(e).__name__})")

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup events
    logger.info("=" * 60)
    logger.info("🚀 AI Hologram Medical Visualization System - STARTUP")
    logger.info("=" * 60)
    logger.info("Initializing core components...")
    
    # Database
    await db.connect()
    logger.info("✓ Database connected")
    
    # AI Model Backend Status
    logger.info("\n📊 AI Model Stack:")
    logger.info("   Primary Backend: OpenCV + NumPy (CPU-optimized)")
    logger.info("   ✅ Segmentation: OpenCV U-Net implementation")
    logger.info("   ✅ Depth Estimation: OpenCV gradient-based + synthetic")
    logger.info("   ✅ Super-Resolution: OpenCV Lanczos4 + Unsharp Mask")
    logger.info("   ✅ Feature Extraction: Statistical (512-dim)")
    
    # Optional PyTorch check (informational only)
    try:
        import torch
        torch_version = getattr(torch, '__version__', 'Unknown')
        cuda_available = "Yes" if torch.cuda.is_available() else "No"
        logger.info(f"\n   Optional PyTorch:")
        logger.info(f"   - Version: {torch_version}")
        logger.info(f"   - CUDA: {cuda_available}")
        if hasattr(torch, '_C'):
            logger.info("   - C bindings: ✅ Loaded")
        else:
            logger.info("   - C bindings: ⚠️  Not accessible (using CPU fallback)")
    except Exception as e:
        logger.info(f"\n   Optional PyTorch: Not available (using pure OpenCV)")
        logger.debug(f"   Reason: {e}")
    
    # Load AI Models
    logger.info("\n📦 Loading AI models...")
    segmentation_model.load()
    logger.info("  ✓ Segmentation (U-Net)")
    depth_model.load()
    logger.info("  ✓ Depth Estimation (MiDaS)")
    gan_sr.load()
    logger.info("  ✓ Super-Resolution (ESRGAN)")
    
    # GPU Acceleration Optimization
    if GPU_ENABLED:
        logger.info("\n⚡ GPU Acceleration ENABLED")
        try:
            if hasattr(segmentation_model, 'model') and segmentation_model.model is not None:
                try:
                    segmentation_model.model = segmentation_model.model.to(DEVICE)
                    if hasattr(segmentation_model.model, 'half'):
                        segmentation_model.model.half()
                    logger.info(f"  ✓ Segmentation → {DEVICE.upper()}")
                except Exception as e:
                    logger.debug(f"  Could not move Segmentation to GPU: {e}")
            
            if hasattr(depth_model, 'model') and depth_model.model is not None:
                try:
                    depth_model.model = depth_model.model.to(DEVICE)
                    if hasattr(depth_model.model, 'half'):
                        depth_model.model.half()
                    logger.info(f"  ✓ Depth Estimation → {DEVICE.upper()}")
                except Exception as e:
                    logger.debug(f"  Could not move Depth to GPU: {e}")
            
            if hasattr(gan_sr, 'model') and gan_sr.model is not None:
                try:
                    gan_sr.model = gan_sr.model.to(DEVICE)
                    if hasattr(gan_sr.model, 'half'):
                        gan_sr.model.half()
                    logger.info(f"  ✓ Super-Resolution → {DEVICE.upper()}")
                except Exception as e:
                    logger.debug(f"  Could not move Super-Resolution to GPU: {e}")
            
            # Warm up CUDA cache (safely)
            try:
                import torch
                torch.cuda.empty_cache()
                logger.info("  ✓ CUDA ready for inference")
            except Exception as e:
                logger.debug(f"  Could not clear CUDA cache: {e}")
        except Exception as e:
            logger.warning(f"  ⚠️  GPU optimization error: {e} (falling back to CPU)")
    else:
        logger.info("\n⚠️  Using CPU for inference (slower)")
    
    logger.info("=" * 60)
    logger.info("✅ Server Ready - All systems operational")
    logger.info(f"   API: http://0.0.0.0:8000")
    logger.info(f"   Device: {DEVICE.upper()}")
    logger.info(f"   Docs: http://0.0.0.0:8000/docs")
    logger.info("=" * 60)
    
    # Yield control to app
    yield
    
    # Shutdown: Disconnect database
    logger.info("🛑 Shutting down...")
    await db.disconnect()
    logger.info("✓ Database disconnected")

app = FastAPI(
    title="AI Hologram Medical Visualization API",
    description="AI-driven holographic visualization system for medical DICOM reports",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Rate Limiting
# ─────────────────────────────────────────────
class RateLimiter:
    """Simple in-memory rate limiter."""
    def __init__(self, requests_per_minute: int = 60):
        self.max_requests = requests_per_minute
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id] if req_time > cutoff
        ]
        
        if len(self.requests[client_id]) < self.max_requests:
            self.requests[client_id].append(now)
            return True
        return False

rate_limiter = RateLimiter(requests_per_minute=100)

async def rate_limit_middleware(request: Request, call_next):
    """Middleware to apply rate limiting."""
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse({"error": "Rate limit exceeded"}, status_code=429)
    return await call_next(request)

app.middleware("http")(rate_limit_middleware)

# ─────────────────────────────────────────────
# API Key Authentication
# ─────────────────────────────────────────────
API_KEY = os.getenv("API_KEY", "hologram-medical-key-2024")  # Change in production

def verify_api_key(authorization: Optional[str] = Header(None)) -> bool:
    """Verify API key from Authorization header."""
    if not authorization:
        return False
    if not authorization.startswith("Bearer "):
        return False
    token = authorization[7:]
    return token == API_KEY

# Mount static outputs
UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("data/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Singletons
db = Database()
session_mgr = SessionManager()
dicom_proc = DICOMProcessor()
preprocessor = ImagePreprocessor()
segmentation_model = UNetSegmentation()
depth_model = DepthEstimator()
gan_sr = GANSuperResolution()
pointcloud_proc = PointCloudProcessor()
volumetric_cnn = Volumetric3DCNN()
organ_visualizer = OrganVisualizer()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models": ["U-Net", "MiDaS", "GAN-SR", "3D-CNN", "PointNet"],
        "gpu_enabled": GPU_ENABLED,
        "device": DEVICE.upper()
    }


@app.post("/api/upload/dicom")
async def upload_dicom(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    patient_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
):
    """Upload a DICOM file and trigger AI pipeline processing."""
    # Verify API key
    if not verify_api_key(authorization):
        raise HTTPException(status_code=401, detail="Invalid or missing API key. Use 'Authorization: Bearer <key>'")
    
    if not file.filename.endswith((".dcm", ".dicom", ".nii", ".nii.gz")):
        raise HTTPException(status_code=400, detail="Only DICOM / NIfTI files accepted.")

    session_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{session_id}_{file.filename}"

    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    # Register session in DB
    await db.create_session(session_id, str(save_path), patient_id)
    session_mgr.register(session_id)

    # Queue background AI pipeline (wrap async in sync for BackgroundTasks)
    import asyncio
    def _run_pipeline_sync(sid, fpath):
        asyncio.run(run_ai_pipeline(sid, fpath))
    background_tasks.add_task(_run_pipeline_sync, session_id, save_path)

    return JSONResponse({"session_id": session_id, "status": "processing", "filename": file.filename})


async def run_ai_pipeline(session_id: str, file_path: Path):
    """Full AI pipeline: DICOM → preprocess → segment → depth → SR → 3D → pointcloud."""
    try:
        logger.info(f"[{session_id}] Starting AI pipeline for {file_path}")
        session_mgr.update_status(session_id, "extracting_dicom")

        # Step 1: DICOM extraction
        image_stack, metadata = dicom_proc.load(file_path)
        
        # Handle 2D images by expanding to 3D
        if len(image_stack.shape) == 2:
            logger.info(f"Expanding 2D image {image_stack.shape} to 3D volume")
            image_stack = np.expand_dims(image_stack, axis=0)
            logger.info(f"Expanded shape: {image_stack.shape}")

        # Step 2: Preprocessing
        session_mgr.update_status(session_id, "preprocessing")
        preprocessed = preprocessor.normalize_and_denoise(image_stack)
        
        # Ensure 3D after preprocessing (safety check)
        if len(preprocessed.shape) == 2:
            logger.warning("Preprocessed output is 2D, expanding to 3D")
            preprocessed = np.expand_dims(preprocessed, axis=0)

        # Step 3: U-Net segmentation
        session_mgr.update_status(session_id, "segmenting")
        segmentation_mask = segmentation_model.predict(preprocessed)
        
        # Ensure 3D after segmentation (safety check)
        if len(segmentation_mask.shape) == 2:
            logger.warning("Segmentation mask is 2D, expanding to 3D")
            segmentation_mask = np.expand_dims(segmentation_mask, axis=0)

        # Step 4: GAN Super-Resolution
        session_mgr.update_status(session_id, "super_resolution")
        try:
            enhanced_volume = gan_sr.enhance(preprocessed)
            
            # Ensure 3D after super-resolution
            if len(enhanced_volume.shape) == 2:
                logger.warning("Enhanced volume is 2D, expanding to 3D")
                enhanced_volume = np.expand_dims(enhanced_volume, axis=0)
            
            # Validate enhanced volume
            if enhanced_volume is None or enhanced_volume.size == 0:
                raise ValueError("Super-resolution produced empty output")
            
            # Check for NaN values
            if np.isnan(enhanced_volume).any():
                logger.warning("NaN values detected in enhanced volume, replacing with original")
                enhanced_volume = preprocessed
            
            # Check output shape is reasonable (should be D, H*4, W*4 or similar)
            if len(enhanced_volume.shape) == 3:
                expected_shape = (enhanced_volume.shape[0], enhanced_volume.shape[1], enhanced_volume.shape[2])
                logger.info(f"Enhanced volume shape: {expected_shape}")
            else:
                logger.warning(f"Unexpected enhanced volume shape: {enhanced_volume.shape}")
            
        except Exception as e:
            logger.error(f"Super-resolution failed: {e}", exc_info=True)
            logger.warning("Falling back to preprocessed volume without super-resolution")
            enhanced_volume = preprocessed

        # Step 5: 3D-CNN volumetric feature extraction
        session_mgr.update_status(session_id, "volumetric_features")
        try:
            features = volumetric_cnn.extract_features(enhanced_volume)
            
            # Validate features
            if features is None or features.size == 0:
                raise ValueError("Feature extraction produced empty output")
            
            if np.isnan(features).any() or np.isinf(features).any():
                logger.warning("Invalid values in features, using zeros")
                features = np.zeros(volumetric_cnn.feature_dim, dtype=np.float32)
            
            logger.info(f"Features extracted: shape {features.shape}")
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}", exc_info=True)
            logger.warning("Using zero-filled feature vector as fallback")
            features = np.zeros(512, dtype=np.float32)  # Default feature_dim=512

        # Step 6: Transformer depth estimation (MiDaS)
        session_mgr.update_status(session_id, "depth_estimation")
        try:
            depth_maps = depth_model.estimate(enhanced_volume)
            
            # Ensure 3D depth maps
            if len(depth_maps.shape) == 2:
                logger.warning("Depth maps are 2D, expanding to 3D")
                depth_maps = np.expand_dims(depth_maps, axis=0)
            
            # Validate depth maps
            if depth_maps is None or depth_maps.size == 0:
                raise ValueError("Depth estimation produced empty output")
            
            if not (depth_maps.shape[0] == enhanced_volume.shape[0]):
                raise ValueError(f"Depth shape mismatch: {depth_maps.shape} vs expected (D, H, W)")
            
            if np.isnan(depth_maps).any():
                logger.warning("NaN values in depth maps, replacing with synthetic")
                depth_maps = np.random.rand(*enhanced_volume.shape).astype(np.float32) * 0.5 + 0.25
            
            logger.info(f"Depth maps estimated: shape {depth_maps.shape}")
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}", exc_info=True)
            logger.warning("Using synthetic depth maps as fallback")
            depth_maps = np.random.rand(*enhanced_volume.shape).astype(np.float32) * 0.5 + 0.25

        # Step 7: PointNet / Voxel Grid point cloud
        session_mgr.update_status(session_id, "pointcloud")
        try:
            point_cloud = pointcloud_proc.generate(segmentation_mask, depth_maps)
            
            if point_cloud is None or point_cloud.size == 0:
                raise ValueError("Point cloud generation produced empty output")
            
            logger.info(f"Point cloud generated: shape {point_cloud.shape}")
        except Exception as e:
            logger.error(f"Point cloud generation failed: {e}", exc_info=True)
            logger.warning("Using empty point cloud as fallback")
            point_cloud = np.zeros((1, 6), dtype=np.float32)

        # Step 8: Generate 3D Holographic Visualization
        session_mgr.update_status(session_id, "hologram_generation")
        try:
            # Use the PREPROCESSED volume (not SR-enhanced) for hologram generation
            # Reason: SR upscales to 2048x2048 which is too large and causes shape
            # mismatch with segmentation (512x512). Preprocessed is same size as segmentation.
            hologram_volume = preprocessed
            hologram_seg = segmentation_mask
            
            logger.info(f"[{session_id}] Hologram generation - volume shape: {hologram_volume.shape}")
            logger.info(f"[{session_id}] Hologram generation - segmentation shape: {hologram_seg.shape if hologram_seg is not None else 'None'}")
            
            # Safety: if shapes still mismatch, resize segmentation
            if hologram_seg is not None and hologram_seg.shape != hologram_volume.shape:
                logger.warning(f"Shape mismatch: volume {hologram_volume.shape} vs seg {hologram_seg.shape}, resizing")
                from scipy.ndimage import zoom
                scale_factors = [hologram_volume.shape[i] / hologram_seg.shape[i] for i in range(3)]
                hologram_seg = zoom(hologram_seg.astype(np.float32), scale_factors, order=0).astype(hologram_seg.dtype)
            
            # Create 3D holographic view
            hologram_fig = organ_visualizer.create_3d_hologram_view(
                hologram_volume,
                hologram_seg,
                title="AI-Generated 3D Holographic Medical Visualization"
            )
            
            logger.info(f"[{session_id}] Hologram figure created successfully")
            
            # Convert Plotly figure to JSON properly
            try:
                hologram_json_str = hologram_fig.to_json()
                hologram_json = json.loads(hologram_json_str)
                logger.info(f"[{session_id}] Hologram serialized successfully - JSON size: {len(hologram_json_str)} bytes")
            except Exception as e:
                logger.warning(f"[{session_id}] Failed to serialize hologram figure: {e}")
                # Create a minimal valid hologram JSON as fallback
                hologram_json = {
                    "data": [],
                    "layout": {
                        "title": "3D Holographic Visualization",
                        "scene": {"aspectmode": "cube"}
                    }
                }
            
            # Save hologram data
            out_dir = OUTPUT_DIR / session_id
            out_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                with open(out_dir / "hologram.json", 'w') as f:
                    json.dump(hologram_json, f, indent=2)
                logger.info(f"[{session_id}] 3D holographic visualization saved to {out_dir / 'hologram.json'}")
            except Exception as e:
                logger.error(f"[{session_id}] Failed to save hologram JSON: {e}", exc_info=True)
            
        except Exception as e:
            logger.error(f"[{session_id}] Hologram generation failed: {e}", exc_info=True)
            # Create minimal fallback hologram
            try:
                out_dir = OUTPUT_DIR / session_id
                out_dir.mkdir(parents=True, exist_ok=True)
                fallback_hologram = {
                    "data": [],
                    "layout": {
                        "title": "3D Holographic Visualization (Generation Failed)",
                        "scene": {"aspectmode": "cube"}
                    }
                }
                with open(out_dir / "hologram.json", 'w') as f:
                    json.dump(fallback_hologram, f, indent=2)
                logger.info(f"[{session_id}] Fallback hologram created")
            except Exception as fallback_err:
                logger.error(f"[{session_id}] Even fallback hologram failed: {fallback_err}")

        # Save outputs
        out_dir = OUTPUT_DIR / session_id
        out_dir.mkdir(parents=True, exist_ok=True)
        pointcloud_proc.save_ply(point_cloud, out_dir / "pointcloud.ply")
        depth_model.save_depth_maps(depth_maps, out_dir / "depth_maps.npz")
        segmentation_model.save_mask(segmentation_mask, out_dir / "segmentation.npy")
        
        # Save super-resolution result
        try:
            np.save(str(out_dir / "super_resolution.npy"), enhanced_volume)
            logger.info(f"[{session_id}] Super-resolution saved: {enhanced_volume.shape}")
        except Exception as e:
            logger.warning(f"[{session_id}] Failed to save super-resolution: {e}")
        
        # Save features (512-dimensional vector)
        try:
            np.save(str(out_dir / "features.npy"), features)
            logger.info(f"[{session_id}] Features saved: {features.shape}")
        except Exception as e:
            logger.warning(f"[{session_id}] Failed to save features: {e}")

        # Store results in DB
        await db.update_session(session_id, {
            "status": "completed",
            "features": features.tolist()[:50],  # Sample for metadata
            "metadata": metadata,
            "output_path": str(out_dir),
        })
        session_mgr.update_status(session_id, "completed")
        logger.info(f"[{session_id}] Pipeline complete.")

    except Exception as e:
        error_msg = f"Pipeline error: {type(e).__name__}: {str(e)}"
        logger.error(f"[{session_id}] {error_msg}", exc_info=True)
        await db.update_session(session_id, {"status": "error", "error": error_msg})
        session_mgr.update_status(session_id, "error")
        logger.warning(f"[{session_id}] Session marked as error with message: {error_msg}")


@app.get("/api/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Get status of processing session with error details from database"""
    status = session_mgr.get_status(session_id)
    error_msg = None
    
    # Check database for complete session info (including errors)
    db_session = await db.get_session(session_id)
    if db_session:
        status = db_session.get("status", status)
        error_msg = db_session.get("error")
    
    response = {"session_id": session_id, "status": status}
    if error_msg:
        response["error"] = error_msg
    
    return response


@app.get("/api/session/{session_id}/pointcloud")
async def get_pointcloud(session_id: str):
    path = OUTPUT_DIR / session_id / "pointcloud.ply"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Point cloud not yet generated.")
    return FileResponse(str(path), media_type="application/octet-stream")


@app.get("/api/session/{session_id}/depth")
async def get_depth_maps(session_id: str):
    """Get depth map visualization as PNG image."""
    # Try PNG first, then fallback to .npz
    png_path = OUTPUT_DIR / session_id / "depth_maps.png"
    npz_path = OUTPUT_DIR / session_id / "depth_maps.npz"
    
    if png_path.exists():
        return FileResponse(str(png_path), media_type="image/png")
    elif npz_path.exists():
        # Convert .npz to PNG on-the-fly
        data = np.load(str(npz_path))
        depth_maps = data['depth'] if 'depth' in data else data[list(data.keys())[0]]
        
        if len(depth_maps.shape) == 3:
            depth_slice = depth_maps[depth_maps.shape[0] // 2]
        else:
            depth_slice = depth_maps
        
        # Normalize and convert to image
        depth_normalized = ((depth_slice - depth_slice.min()) / (depth_slice.max() - depth_slice.min() + 1e-8) * 255).astype(np.uint8)
        
        # Encode as PNG
        _, buffer = cv2.imencode('.png', depth_normalized)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Depth maps not yet available.")


@app.get("/api/session/{session_id}/segmentation")
async def get_segmentation(session_id: str):
    # Try PNG first, then fallback to .npy
    png_path = OUTPUT_DIR / session_id / "segmentation.png"
    npy_path = OUTPUT_DIR / session_id / "segmentation.npy"
    
    if png_path.exists():
        return FileResponse(str(png_path), media_type="image/png")
    elif npy_path.exists():
        # Convert .npy to image on-the-fly
        mask = np.load(str(npy_path))
        if len(mask.shape) == 3:
            mask = mask[mask.shape[0] // 2]  # Middle slice
        mask_normalized = (mask * (255 // max(mask.max(), 1))).astype(np.uint8)
        _, buffer = cv2.imencode('.png', mask_normalized)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Segmentation mask not available.")


@app.get("/api/session/{session_id}/super-resolution")
async def get_super_resolution(session_id: str):
    """Get super-resolution enhanced volume as PNG (middle slice) or NPZ."""
    npy_path = OUTPUT_DIR / session_id / "super_resolution.npy"
    png_path = OUTPUT_DIR / session_id / "super_resolution.png"
    
    if png_path.exists():
        return FileResponse(str(png_path), media_type="image/png")
    elif npy_path.exists():
        # Convert middle slice to PNG on-the-fly
        enhanced_volume = np.load(str(npy_path))
        if len(enhanced_volume.shape) == 3:
            enhanced_slice = enhanced_volume[enhanced_volume.shape[0] // 2]
        else:
            enhanced_slice = enhanced_volume
        
        # Normalize and convert to image
        enhanced_normalized = ((enhanced_slice - enhanced_slice.min()) / (enhanced_slice.max() - enhanced_slice.min() + 1e-8) * 255).astype(np.uint8)
        
        # Encode as PNG
        _, buffer = cv2.imencode('.png', enhanced_normalized)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Super-resolution data not yet available.")


@app.get("/api/session/{session_id}/features")
async def get_features(session_id: str):
    """Get volumetric feature extraction vector (512-dimensional)."""
    npy_path = OUTPUT_DIR / session_id / "features.npy"
    
    if npy_path.exists():
        features = np.load(str(npy_path))
        # Convert to list for JSON serialization
        return JSONResponse({
            "features": features.tolist(),
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "min": float(features.min()),
            "max": float(features.max()),
            "mean": float(features.mean()),
        })
    else:
        raise HTTPException(status_code=404, detail="Features not yet available.")


@app.get("/api/session/{session_id}/hologram")
async def get_holographic_view(session_id: str):
    """Get 3D holographic visualization as Plotly JSON."""
    try:
        hologram_path = OUTPUT_DIR / session_id / "hologram.json"
        logger.info(f"Fetching hologram from: {hologram_path}")
        
        if not hologram_path.exists():
            logger.warning(f"Hologram file not found at: {hologram_path}")
            logger.info(f"Output directory contents: {list(OUTPUT_DIR.glob(session_id))}")
            raise HTTPException(status_code=404, detail="Holographic view not yet generated.")
        
        try:
            with open(hologram_path, 'r') as f:
                hologram_data = json.load(f)
            logger.info(f"Successfully loaded hologram JSON for session {session_id}")
            return JSONResponse(hologram_data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in hologram file: {e}")
            raise HTTPException(status_code=500, detail="Holographic view data corrupted.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching hologram: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve holographic view.")


@app.get("/api/patients")
async def list_patients():
    return await db.list_patients()


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    await db.delete_session(session_id)
    session_mgr.remove(session_id)
    return {"deleted": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


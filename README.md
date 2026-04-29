# AI Hologram Medical Visualization System

AI-driven holographic 3D visualization system for medical DICOM imaging.

## Features

- **DICOM Processing** - Load and process CT, MRI, X-Ray medical images
- **AI Segmentation** - U-Net organ/tissue segmentation (OpenCV)
- **Depth Estimation** - MiDaS-based 3D depth analysis
- **Super-Resolution** - GAN-based 4x image enhancement
- **3D Hologram** - Solid mesh organ rendering using marching cubes + isosurface
- **Point Cloud** - 3D point cloud generation from depth maps
- **Web Interface** - Streamlit frontend with real-time pipeline monitoring

## Quick Start (Windows)

1. Double-click `start.bat` or run:
   ```
   start.bat
   ```
2. Browser opens at http://localhost:8502
3. Upload a DICOM file and watch the AI pipeline process it

## Manual Start

```bash
# Activate virtual environment
myenv\Scripts\activate

# Start backend (terminal 1)
python run_system.py

# Start frontend (terminal 2)
streamlit run frontend/app_streamlit.py --server.port 8502
```

## Project Structure

```
hologram_medical/
├── start.bat                  # Windows launcher (starts everything)
├── run_system.py              # Backend entry point
├── requirements.txt           # Python dependencies
├── README.md
│
├── backend/                   # FastAPI backend server
│   ├── main.py                # API routes & pipeline orchestration
│   ├── database.py            # SQLite database layer
│   └── session_manager.py     # Processing session management
│
├── frontend/                  # Streamlit web interface
│   └── app_streamlit.py       # Main frontend application
│
├── models/                    # AI model implementations
│   ├── segmentation.py        # U-Net segmentation (primary)
│   ├── segmentation_opencv.py # OpenCV segmentation (fallback)
│   ├── depth_estimation.py    # MiDaS depth estimation
│   ├── depth_estimation_onnx.py
│   ├── super_resolution.py    # GAN super-resolution (primary)
│   ├── super_resolution_opencv.py  # OpenCV SR (fallback)
│   ├── volumetric_cnn.py      # 3D CNN feature extraction
│   ├── volumetric_cnn_statistical.py
│   ├── pointcloud.py          # Point cloud generation
│   └── torch_fallback.py      # PyTorch/CPU fallback utilities
│
├── processing/                # Image processing pipeline
│   ├── dicom_processor.py     # DICOM file parsing
│   ├── image_preprocessor.py  # Image normalization & prep
│   ├── organ_visualizer.py    # 3D hologram mesh generation
│   ├── gpu_processor.py       # GPU acceleration utilities
│   ├── gpu_renderer.py        # GPU rendering pipeline
│   └── volumetric_renderer.py # Volume rendering
│
├── config/
│   └── settings.py            # Application configuration
│
├── data/
│   ├── test_samples/          # Sample DICOM files for testing
│   ├── uploads/               # Uploaded DICOM files
│   └── outputs/               # Processing results per session
│
├── tests/                     # Test suite
│   ├── conftest.py
│   ├── test_api.py
│   └── test_pipeline.py
│
├── scripts/                   # Utility scripts
│   ├── generate_test_dicom.py
│   ├── train.py
│   ├── warmup_models.py
│   └── deploy.sh
│
├── gui/                       # Desktop GUI (optional)
│   └── main_gui.py
│
└── myenv/                     # Python virtual environment
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/api/upload/dicom` | POST | Upload DICOM file |
| `/api/session/{id}/status` | GET | Processing status |
| `/api/session/{id}/hologram` | GET | 3D holographic view |
| `/api/session/{id}/segmentation` | GET | Segmentation mask |
| `/api/session/{id}/depth` | GET | Depth map |
| `/api/session/{id}/super-resolution` | GET | Enhanced image |
| `/api/session/{id}/pointcloud` | GET | Point cloud data |
| `/api/session/{id}/features` | GET | Feature vector |
| `/api/patients` | GET | Patient list |

API documentation: http://localhost:8000/docs

## Tech Stack

- **Backend**: FastAPI, Python 3.10+
- **Frontend**: Streamlit, Plotly
- **AI/ML**: OpenCV, NumPy, scikit-image, PyTorch (optional)
- **3D Rendering**: Marching Cubes, Mesh3d, Isosurface
- **Database**: SQLite
- **Auth**: Bearer token authentication

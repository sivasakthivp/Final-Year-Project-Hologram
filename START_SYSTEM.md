# 🏥 AI Hologram Medical Visualization System - Startup Guide

## Complete System Startup Instructions

### Prerequisites
- ✅ Python 3.11+ installed
- ✅ Virtual environment created (`myenv`)
- ✅ All dependencies installed

---

## Step 1: Open Two Terminals

You need **2 separate terminal windows** to run backend and frontend simultaneously.

### Terminal 1 (Backend)
```powershell
cd "c:\Users\mrsiv\Documents\Final Year Project - Hologram"
```

### Terminal 2 (Frontend)
```powershell
cd "c:\Users\mrsiv\Documents\Final Year Project - Hologram"
```

---

## Step 2: Start Backend Server

**In Terminal 1, run:**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\myenv\Scripts\Activate.ps1
python backend/main.py
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:backend.main:✅ Server Ready - All systems operational
INFO:backend.main:   API: http://0.0.0.0:8000
INFO:backend.main:   Device: CPU
INFO:backend.main:   Docs: http://0.0.0.0:8000/docs
```

✅ **Backend is running** at `http://localhost:8000`

---

## Step 3: Start Frontend (Streamlit)

**In Terminal 2, run:**

```powershell
cd frontend
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
..\myenv\Scripts\Activate.ps1
streamlit run app_streamlit.py --server.port 8502
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8502
  Network URL: http://10.191.109.17:8502
```

✅ **Frontend is running** at `http://localhost:8502`

---

## Step 4: Access the Application

### Option A: Automatic (Frontend will open)
Streamlit may open automatically at `http://localhost:8502`

### Option B: Manual (Open in browser)
1. Open your web browser
2. Go to `http://localhost:8502`
3. You should see the **AI Hologram Medical Visualization** interface

---

## Complete Startup Sequence

```
Terminal 1                          Terminal 2
─────────────────────────────────────────────────────────────

$ Set-ExecutionPolicy...            (Wait ~2 seconds)
$ .\myenv\Scripts\Activate.ps1
(myenv) $                           
(myenv) $ python backend/main.py    

Backend Starting...
✅ Backend Ready ──────────────────→ NOW START TERMINAL 2

                                    $ cd frontend
                                    $ Set-ExecutionPolicy...
                                    $ ..\myenv\Scripts\Activate.ps1
                                    (myenv) $ streamlit run ...

                                    Frontend Starting...
                                    ✅ Frontend Ready

BOTH RUNNING:
✅ Backend  @ http://localhost:8000
✅ Frontend @ http://localhost:8502
```

---

## System Check

### Verify Backend is Online
Open your browser and go to:
```
http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "models": ["U-Net", "MiDaS", "GAN-SR", "3D-CNN", "PointNet"],
  "gpu_enabled": false,
  "device": "CPU"
}
```

### Verify Frontend is Online
Open your browser and go to:
```
http://localhost:8502
```

You should see the **Streamlit UI** with:
- 📤 Upload & Process tab
- 📊 Results tab
- 📈 History tab
- ⚙️ Settings tab

---

## Usage Guide

### Upload and Process DICOM Files

1. **Go to Frontend** → `http://localhost:8502`
2. **Click Tab:** 📤 Upload & Process
3. **Upload File:** 
   - Click "Drag and drop DICOM file"
   - Select a `.dcm` or `.dicom` file
4. **Enter Details:**
   - Patient ID (optional)
   - Modality (CT, MRI, X-Ray)
5. **Click:** 🚀 Upload & Process
6. **Watch Pipeline:** 
   - Progress bar shows all processing stages
   - Takes 30-60 seconds depending on image size
7. **View Results:** 📊 Results tab shows:
   - 🌀 3D Holographic visualization
   - 🎯 Segmentation mask
   - 📐 Depth map
   - 🔍 Super-resolution image
   - 📋 Processing details

---

## Test DICOM Files

Test samples are included in the project:

```
data/test_samples/
├── test_ct_head.dcm
├── test_mri_spine.dcm
├── test_xray_chest.dcm
└── Brain Scan/
```

**Try uploading one of these to test!**

---

## Troubleshooting

### Backend Won't Start

**Error:** `ModuleNotFoundError: No module named 'fastapi'`

**Solution:**
```powershell
.\myenv\Scripts\python.exe -m pip install -r requirements.txt
```

---

### Frontend Won't Start

**Error:** `Port 8502 already in use`

**Solution:**
```powershell
streamlit run app_streamlit.py --server.port 8503
# Use port 8503 instead
```

---

### PowerShell Execution Policy

**Error:** `File cannot be loaded because running scripts is disabled`

**Solution:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

### Backend Connection Error

**Error:** `Connection refused` or `Cannot reach backend`

**Check:**
1. Terminal 1 is still running (not closed)
2. Backend shows "Server Ready" message
3. Try: `http://localhost:8000/health`

---

## Quick Start Script

**Create file:** `START.bat`

```batch
@echo off
echo Starting AI Hologram Medical Visualization System...
echo.
echo Opening Terminal 1 - Backend Server
start cmd /k "cd /d %~dp0 && powershell -Command ""Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\myenv\Scripts\Activate.ps1; python backend/main.py"""

echo Waiting for backend to start...
timeout /t 3

echo.
echo Opening Terminal 2 - Frontend (Streamlit)
start cmd /k "cd /d %~dp0\frontend && powershell -Command ""Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; ..\myenv\Scripts\Activate.ps1; streamlit run app_streamlit.py --server.port 8502"""

echo.
echo ✅ System starting...
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:8502
echo.
pause
```

**Then just double-click `START.bat`** to launch everything!

---

## System Architecture

```
User Browser (http://localhost:8502)
        ↓
    Streamlit Frontend
        ↓
    REST API (http://localhost:8000)
        ↓
    FastAPI Backend
        ├─ DICOM Processor
        ├─ AI Models
        │  ├─ U-Net Segmentation
        │  ├─ MiDaS Depth Estimation
        │  ├─ GAN Super-Resolution
        │  └─ 3D-CNN Features
        ├─ Database (SQLite)
        └─ Session Manager
```

---

## API Documentation

Once backend is running, access Swagger UI:
```
http://localhost:8000/docs
```

View all available API endpoints:
- `POST /api/upload/dicom` - Upload DICOM file
- `GET /api/session/{session_id}/status` - Check processing status
- `GET /api/session/{session_id}/hologram` - Get 3D visualization
- `GET /api/session/{session_id}/segmentation` - Get segmentation mask
- `GET /api/session/{session_id}/depth` - Get depth map
- `GET /api/session/{session_id}/super-resolution` - Get enhanced image
- `GET /health` - Check backend health

---

## Common Ports

| Service | Port | URL |
|---------|------|-----|
| Backend API | 8000 | http://localhost:8000 |
| Frontend | 8502 | http://localhost:8502 |
| API Docs | 8000 | http://localhost:8000/docs |

---

## System Requirements

- **RAM:** 2GB minimum (4GB recommended)
- **GPU:** Optional (uses CPU if unavailable)
- **Storage:** 1GB for models and data
- **Python:** 3.11+
- **OS:** Windows 10/11, Linux, macOS

---

## Success Checklist ✅

- [ ] Backend terminal shows "Server Ready"
- [ ] Frontend terminal shows "Local URL: http://localhost:8502"
- [ ] Browser opens frontend at http://localhost:8502
- [ ] Health check passes at http://localhost:8000/health
- [ ] Can upload DICOM file
- [ ] Processing pipeline runs (progress bar)
- [ ] Results display in tabs (segmentation, depth, etc.)
- [ ] 3D hologram renders

---

## Need Help?

1. **Check terminal outputs** for error messages
2. **Verify ports** (8000, 8502) are not in use
3. **Restart both services** if something freezes
4. **Check internet connection** for initial model downloads
5. **Check disk space** for DICOM file processing

---

**System is ready when:**
- ✅ Backend running at `http://localhost:8000`
- ✅ Frontend running at `http://localhost:8502`
- ✅ Both show successful startup messages
- ✅ Browser can access frontend UI

🎉 **You're ready to process medical images!**

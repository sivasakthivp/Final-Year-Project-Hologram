@echo off
title AI Hologram Medical Visualization System
color 0B

echo ============================================================
echo    AI Hologram Medical Visualization System
echo    Starting Backend + Frontend...
echo ============================================================
echo.

:: Set script directory
setlocal enabledelayedexpansion
cd /d "%~dp0"

:: Find the right Python (prefer Python 3.11, fallback to python on PATH)
set PYTHON=
if exist "C:\Users\mrsiv\AppData\Local\Programs\Python\Python311\python.exe" (
    set PYTHON=C:\Users\mrsiv\AppData\Local\Programs\Python\Python311\python.exe
) else (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set PYTHON=python
    )
)

if "%PYTHON%"=="" (
    echo ERROR: Python not found! Install Python 3.11+ from python.org
    pause
    exit /b 1
)

echo Using Python: %PYTHON%
echo.

:: Install missing packages silently
echo Checking dependencies...
"%PYTHON%" -m pip install fastapi uvicorn python-multipart aiofiles aiosqlite pydicom Pillow numpy scipy scikit-image scikit-learn plyfile matplotlib plotly pydantic python-dotenv opencv-python-headless tqdm rich --quiet 2>nul
echo Dependencies OK.
echo.

:: Kill any leftover processes on ports 8000 and 8502
echo Cleaning up old processes...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8502 " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

:: Start Backend (FastAPI)
echo [1/2] Starting Backend API Server (port 8000)...
start "Backend - FastAPI" cmd /k ""%PYTHON%" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

:: Wait for backend to start
echo      Waiting for backend to initialize...
timeout /t 8 /nobreak

:: Check if streamlit is installed before starting frontend
"%PYTHON%" -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Streamlit for frontend...
    "%PYTHON%" -m pip install streamlit requests plotly --quiet 2>nul
)

:: Start Frontend (Streamlit)
echo [2/2] Starting Frontend (port 8502)...
start "Frontend - Streamlit" cmd /k ""%PYTHON%" -m streamlit run frontend/app_streamlit.py --server.port 8502"

:: Wait for frontend to start
timeout /t 5 /nobreak

:: Open browser
echo.
echo ============================================================
echo    System is running!
echo    Frontend:  http://localhost:8502
echo    Backend:   http://localhost:8000
echo    API Docs:  http://localhost:8000/docs
echo ============================================================
echo.
echo Opening browser...
start http://localhost:8502

echo.
echo Press any key to stop all services...
pause

:: Kill the servers
taskkill /FI "WINDOWTITLE eq Backend - FastAPI*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Frontend - Streamlit*" /F >nul 2>&1
:: Also kill by port in case window titles don't match
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8502 " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)
echo Services stopped.

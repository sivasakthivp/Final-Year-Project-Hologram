"""
Fast Startup Entry Points for GPU-Accelerated Medical Imaging
Choose how you want to run the system
"""

import sys
import os
import subprocess
import signal
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner"""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  🏥 GPU-ACCELERATED MEDICAL IMAGING SYSTEM                    ║
    ║  Powered by NVIDIA RTX GPU - Fast Real-Time Rendering         ║
    ║════════════════════════════════════════════════════════════════╝
    """)


def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU Ready: {device_name} ({memory_gb:.1f}GB)")
            return True
        else:
            print("⚠️  GPU not available - using CPU (slow)")
            return False
    except ImportError:
        print("❌ PyTorch not found - install with: pip install torch")
        return False


def run_desktop_gui():
    """Launch PyQt5 desktop GUI with GPU monitoring"""
    print("\n🖥️  Launching Desktop GUI...")
    logger.info("Starting PyQt5 desktop interface")
    
    try:
        from gui.main_gui import run_gui
        run_gui()
    except ImportError:
        print("❌ PyQt5 not installed")
        print("Install with: pip install PyQt5")
        sys.exit(1)


def run_fast_rendering():
    """Launch fast GPU rendering Streamlit app"""
    print("\n⚡ Launching GPU Rendering Dashboard...")
    logger.info("Starting Streamlit GPU rendering app")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "frontend/app_gpu_rendering.py",
            "--logger.level=info"
        ])
    except KeyboardInterrupt:
        print("\n✓ Stopped")


def run_backend_api():
    """Launch FastAPI backend server"""
    print("\n🚀 Launching Backend API Server...")
    logger.info("Starting FastAPI backend on http://localhost:8000")
    
    try:
        import uvicorn
        from backend.main import app
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            workers=2,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n✓ Server stopped")


def run_frontend_streamlit():
    """Launch main Streamlit frontend"""
    print("\n📊 Launching Frontend Dashboard...")
    logger.info("Starting Streamlit frontend on http://localhost:8501")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "frontend/app_streamlit.py",
            "--logger.level=info"
        ])
    except KeyboardInterrupt:
        print("\n✓ Stopped")


def run_full_stack():
    """Run entire stack: Backend + GPU Rendering Frontend"""
    print("\n🎯 Launching Full Stack (Backend + GPU Rendering)...")
    print("   Terminal 1: Backend API on http://localhost:8000")
    print("   Terminal 2: GPU Rendering on http://localhost:8501")
    
    processes = []
    
    try:
        # Start backend
        print("\n[1/2] Starting Backend API...")
        backend_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "backend.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--workers", "2"
        ])
        processes.append(("Backend", backend_process))
        time.sleep(2)
        
        # Start frontend
        print("[2/2] Starting GPU Rendering Frontend...")
        frontend_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            "frontend/app_gpu_rendering.py",
            "--logger.level=info"
        ])
        processes.append(("Frontend", frontend_process))
        
        print("\n✅ Full stack running!")
        print("   - API Docs: http://localhost:8000/docs")
        print("   - GPU Rendering: http://localhost:8501")
        print("\nPress Ctrl+C to stop all services")
        
        # Wait for all processes
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        for name, proc in processes:
            print(f"  Stopping {name}...", end="", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(" ✓")
        print("✓ Stopped")


def show_menu():
    """Display main menu"""
    print("""
    Choose how to run the system:
    
    1️⃣  Desktop GUI (PyQt5)
         - Professional desktop interface with GPU monitoring
         - Best for: Local machine development & analysis
    
    2️⃣  Fast GPU Rendering (Streamlit)
         - Real-time rendering dashboard with GPU stats
         - Best for: Quick rendering & visualization
    
    3️⃣  Backend API Only
         - FastAPI server on http://localhost:8000
         - Best for: Integration with external frontends
    
    4️⃣  Frontend Only (Streamlit)
         - Main Streamlit dashboard
         - Requires: Backend running separately
    
    5️⃣  Full Stack (Backend + GPU Rendering)
         - Complete system in background
         - Best for: Full-featured usage
    
    6️⃣  Quick Demo
         - Load sample DICOM and render
    
    0️⃣  Exit
    """)


def run_quick_demo():
    """Run quick demo with sample data"""
    print("\n🎬 Running Quick Demo...\n")
    
    try:
        import numpy as np
        from PIL import Image
        from processing.gpu_renderer import get_gpu_renderer
        
        renderer = get_gpu_renderer()
        
        # Create sample volume
        print("📊 Creating sample 3D volume...")
        volume = np.random.rand(64, 256, 256).astype(np.float32)
        
        # Add some structure
        center_x, center_y = 128, 128
        radius = 50
        for x in range(64):
            for y in range(256):
                for z in range(256):
                    dist = np.sqrt((y - center_y)**2 + (z - center_z)**2)
                    if dist < radius:
                        volume[x, y, z] += 0.5
        
        # Render
        print("🎨 Rendering with GPU...")
        rendered = renderer.render_volume(volume, brightness=1.2)
        
        print("✅ Render complete!")
        
        # Display stats
        gpu_stats = renderer.get_gpu_memory_usage()
        print(f"\n📈 GPU Stats:")
        print(f"   Allocated: {gpu_stats['allocated_mb']:.1f}MB")
        print(f"   Total GPU Memory: {gpu_stats['total_mb']:.1f}MB")
        
        # Save result
        output_path = Path("data/outputs/demo_render.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result_uint8 = (np.clip(rendered, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(result_uint8).save(output_path)
        
        print(f"💾 Saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point"""
    print_banner()
    
    # Check GPU
    gpu_available = check_gpu()
    print()
    
    while True:
        show_menu()
        
        choice = input("Enter your choice (0-6): ").strip()
        print()
        
        if choice == "0":
            print("👋 Goodbye!")
            sys.exit(0)
        
        elif choice == "1":
            try:
                run_desktop_gui()
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "2":
            try:
                run_fast_rendering()
            except KeyboardInterrupt:
                print("\n✓ GPU Rendering stopped")
        
        elif choice == "3":
            try:
                run_backend_api()
            except KeyboardInterrupt:
                print("\n✓ Backend stopped")
        
        elif choice == "4":
            try:
                run_frontend_streamlit()
            except KeyboardInterrupt:
                print("\n✓ Frontend stopped")
        
        elif choice == "5":
            try:
                run_full_stack()
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "6":
            try:
                run_quick_demo()
            except KeyboardInterrupt:
                pass
        
        else:
            print("❌ Invalid choice")
        
        print("\n")
        input("Press Enter to continue...")
        print("\n")


if __name__ == "__main__":
    main()

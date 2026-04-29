#!/usr/bin/env python
"""Pre-load all models at startup to prevent timeout on first use"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("AI Hologram Medical - Model Warmup")
print("=" * 60)
print()

try:
    print("[1/4] Loading U-Net Segmentation Model...")
    from models.segmentation import UNetSegmentation
    seg = UNetSegmentation()
    seg.load()
    print("      ✓ U-Net loaded successfully")
    
    print("[2/4] Loading Depth Estimator (MiDaS)...")
    from models.depth_estimation import DepthEstimator
    depth = DepthEstimator()
    depth.load()
    print("      ✓ Depth Estimator loaded successfully")
    
    print("[3/4] Loading GAN Super-Resolution...")
    from models.super_resolution import GANSuperResolution
    gan = GANSuperResolution()
    gan.load()
    print("      ✓ GAN Super-Resolution loaded successfully")
    
    print("[4/4] Loading 3D-CNN Volumetric Model...")
    from models.volumetric_cnn import Volumetric3DCNN
    cnn = Volumetric3DCNN()
    cnn.load()
    print("      ✓ 3D-CNN loaded successfully")
    
    print()
    print("=" * 60)
    print("SUCCESS! All models pre-loaded.")
    print("=" * 60)
    print()
    print("Your backend is now ready for fast inference!")
    print("Start backend with: python backend/main.py")
    print()
    
except Exception as e:
    print()
    print("=" * 60)
    print(f"ERROR: {e}")
    print("=" * 60)
    print()
    print("Troubleshooting:")
    print("1. Ensure all dependencies are installed:")
    print("   pip install -r requirements.txt")
    print()
    print("2. Check if models/weights exist:")
    print("   ls models/weights/")
    print()
    print("3. If models don't exist, they will be downloaded on first use")
    print()
    sys.exit(1)

# Debug script to check pixel coverage
import numpy as np
from PIL import Image
import base64
import io

def check_coverage(image_path):
    """Check pixel coverage of an image"""
    img = Image.open(image_path).convert('L')
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Calculate non-white pixel ratio
    pixel_ratio = np.mean(img_array < 0.9)
    total_pixels = img_array.size
    dark_pixels = np.sum(img_array < 0.9)
    
    print(f"Image: {image_path}")
    print(f"  Total pixels: {total_pixels}")
    print(f"  Dark pixels (< 0.9): {dark_pixels}")
    print(f"  Pixel ratio: {pixel_ratio:.4f} ({pixel_ratio*100:.2f}%)")
    print()
    
    return pixel_ratio

# Test with sample images
print("="*60)
print("PIXEL COVERAGE ANALYSIS")
print("="*60)
print()

# You can manually create test images or check existing ones
print("Instructions:")
print("1. Draw character '人' fully in web demo")
print("2. Open browser DevTools (F12)")
print("3. In Console, run:")
print("   document.getElementById('user-canvas').toDataURL()")
print("4. Save that base64 string to a file")
print("5. Or provide path to actual image files")
print()
print("Expected ratios:")
print("  - Reference (font): ~0.02-0.05 (2-5%)")
print("  - User (handwriting): ~0.01-0.04 (1-4%)")
print("  - Threshold: user < ref * 0.3 means user must have > 30% of ref's ink")

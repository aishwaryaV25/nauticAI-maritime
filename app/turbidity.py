"""
NautiCAI Turbidity Simulation & Visibility Enhancement
"""
import cv2
import numpy as np

def apply_turbidity(img_bgr, strength=0.6):
    """Simulate murky underwater green-water conditions."""
    s   = float(np.clip(strength, 0.0, 1.0))
    img = img_bgr.astype(np.float32)
    # Green-blue tint
    cast = np.array([1.0-0.15*s, 1.0+0.25*s, 1.0+0.05*s], dtype=np.float32)
    img  = img * cast
    # Contrast drop
    img  = (img - 128.0) * (1.0 - 0.35*s) + 128.0
    # Haze overlay
    fog  = np.array([60, 110, 80], dtype=np.float32)
    img  = img*(1.0-(0.10+0.35*s)) + fog*(0.10+0.35*s)
    # Blur + noise
    k    = int(1+4*s); k = k if k%2==1 else k+1
    img  = cv2.GaussianBlur(img, (k,k), 0)
    img += np.random.normal(0, 6+14*s, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)

def enhance_visibility(img_bgr, strength=0.6):
    """CLAHE-based green-water correction + contrast boost."""
    s   = float(np.clip(strength, 0.0, 1.0))
    img = img_bgr.astype(np.float32)
    # Reduce green dominance
    img[:,:,1] = img[:,:,1] * (1.0 - 0.20*s)
    # CLAHE on L channel
    lab   = cv2.cvtColor(np.clip(img,0,255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0+2.0*s, tileGridSize=(8,8))
    lab2  = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
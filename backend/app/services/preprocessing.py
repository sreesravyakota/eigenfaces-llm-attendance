# backend/app/services/preprocessing.py
from __future__ import annotations
from typing import Tuple, Optional, Union

import io
import numpy as np
import cv2

# If Pillow is installed, we'll use it to correct EXIF orientation.
try:
    from PIL import Image, ImageOps
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# ---- Config knobs (kept local for now; you can move to app.core.config later)
PCA_FACE_SIZE: Tuple[int, int] = (128, 128)   # H, W for Eigenfaces
NN_FACE_SIZE: Tuple[int, int] = (160, 160)    # common input size for FaceNet/ArcFace
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_CASCADE = cv2.CascadeClassifier(HAAR_PATH)


def _read_image_any(x: Union[bytes, np.ndarray]) -> np.ndarray:
    """
    Accept bytes (from UploadFile) or an already-loaded numpy image (H×W×C),
    and return an RGB uint8 numpy array with EXIF orientation fixed.
    Priority: Pillow → OpenCV fallback.
    """
    # 1) If the caller already passed a numpy image:
    if isinstance(x, np.ndarray):
        # Normalize to RGB uint8
        if x.ndim == 2:  # grayscale
            return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        if x.ndim == 3 and x.shape[2] == 3:
            # assume BGR (OpenCV) -> convert to RGB
            return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        if x.ndim == 3 and x.shape[2] == 4:
            # BGRA -> RGBA -> RGB
            rgba = cv2.cvtColor(x, cv2.COLOR_BGRA2RGBA)
            return np.asarray(Image.fromarray(rgba).convert("RGB"))
        return x

    # 2) If bytes, try Pillow first (best EXIF handling)
    if _HAS_PIL:
        with Image.open(io.BytesIO(x)) as im:
            # Fix orientation from EXIF and ensure RGB
            im = ImageOps.exif_transpose(im).convert("RGB")
            return np.asarray(im)

    # 3) Fallback to OpenCV if Pillow not present
    data = np.frombuffer(x, np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError("Could not decode image bytes")
    if bgr.ndim == 2:
        return cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    if bgr.shape[2] == 4:
        # BGRA -> convert via PIL to handle premultiplied alpha nicely
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGBA)
        return np.asarray(Image.fromarray(rgba).convert("RGB")) if _HAS_PIL else cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)



def _correct_orientation_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    If PIL is available, we already handled EXIF in _read_image_any. Otherwise, return as-is.
    """
    return rgb


def detect_and_align(rgb: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    """
    Detect the largest face using Haar cascade, crop with a margin, resize to out_size.
    Returns RGB uint8 image of shape out_size.
    If no face is found, a centered square crop is used (failsafe).
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Cascade works better on equalized grayscale
    gray_eq = cv2.equalizeHist(gray)
    faces = _CASCADE.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) == 0:
        # Fallback: center square crop
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        crop = rgb[top:top+side, left:left+side]
    else:
        # Take the largest detected face
        x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        # Add margin around face box (20%)
        m = int(0.2 * max(fw, fh))
        x0 = max(0, x - m)
        y0 = max(0, y - m)
        x1 = min(w, x + fw + m)
        y1 = min(h, y + fh + m)
        crop = rgb[y0:y1, x0:x1]

    # Resize to requested output size
    out = cv2.resize(crop, (out_size[1], out_size[0]), interpolation=cv2.INTER_AREA)
    return out


def normalize_01(img: np.ndarray) -> np.ndarray:
    """
    Convert uint8 [0,255] → float32 [0,1].
    """
    return (img.astype(np.float32) / 255.0).clip(0.0, 1.0)


def preprocess_for_pca(x: Union[bytes, np.ndarray]) -> np.ndarray:
    """
    Full pipeline for Eigenfaces:
      bytes/np → RGB → detect & align → to grayscale → resize 128x128 → normalize [0,1] → 1D vector
    Returns shape (H*W,) float32.
    """
    rgb = _read_image_any(x)
    rgb = _correct_orientation_rgb(rgb)
    face_rgb = detect_and_align(rgb, PCA_FACE_SIZE)

    # Eigenfaces typically uses grayscale
    face_gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)

    # Optional: CLAHE for lighting robustness
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_gray = clahe.apply(face_gray)

    face_norm = normalize_01(face_gray)  # H×W in [0,1]
    vec = face_norm.reshape(-1).astype(np.float32)
    return vec


def preprocess_for_nn(x: Union[bytes, np.ndarray]) -> np.ndarray:
    """
    Full pipeline for NN embeddings (FaceNet/ArcFace style):
      bytes/np → RGB → detect & align → RGB 160x160 → normalize to [0,1] float32 (model may later re-normalize)
    Returns shape (H,W,3) float32 in RGB.
    """
    rgb = _read_image_any(x)
    rgb = _correct_orientation_rgb(rgb)
    face_rgb = detect_and_align(rgb, NN_FACE_SIZE)
    face_f = normalize_01(face_rgb)  # (160,160,3) float32
    return face_f


def preprocess_dual(x: Union[bytes, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience helper: returns (pca_vector, nn_image)
    - pca_vector: (128*128,) float32
    - nn_image:   (160,160,3) float32
    """
    return preprocess_for_pca(x), preprocess_for_nn(x)

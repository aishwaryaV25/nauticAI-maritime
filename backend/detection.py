"""
NautiCAI — detection + annotation + heatmap engine.
Multi-model support: SubPipe, SubPipeMini, SubPipeMini2, Subsea1 4-class
"""
import math, os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from severity import (
    SEVERITY_MAP, CLASS_REMAP, DEFECT_CLASSES, SEV_WEIGHT,
    PIPELINE_DEFECTS, CABLE_DEFECTS, compute_risk, score_to_grade,
)

ROOT = Path(__file__).resolve().parent.parent

# ── Available Models ──────────────────────────────────────────────────────
AVAILABLE_MODELS = {
    "merged_original":  "weights/best_merged_original.pt",
    "subpipe_full":     "weights/best_subpipe_full.pt",
    "subpipemini":      "weights/best_subpipemini.pt",
    "subpipemini2":     "weights/best_subpipemini2.pt",
    "subsea1_4class":   "weights/best_subsea1_4class.pt",
    "archive":          "weights/best_archive.pt",
}

MODEL_DESCRIPTIONS = {
    "merged_original":  "NautiCAI Merged Dataset — Original Production Model — Multi-class",
    "subpipe_full":     "SubPipe Full 21K Images — Pipeline Segmentation",
    "subpipemini":      "SubPipeMini — Pipeline Segmentation — 99.5% mAP",
    "subpipemini2":     "SubPipeMini2 — Pipeline Segmentation — 99.5% mAP",
    "subsea1_4class":   "Subsea Pipeline v2 — Anode, Corner, Flange, Pipe — 99.5% mAP",
    "archive":          "SubPipe Archive — Pipeline Segmentation — 99.5% mAP",
}

_active_model_key = "merged_original"

def set_active_model(model_key: str):
    global _active_model_key, _model_cache
    if model_key in AVAILABLE_MODELS:
        _active_model_key = model_key
        _model_cache = {}

def get_active_model_key():
    return _active_model_key

def get_model_descriptions():
    return MODEL_DESCRIPTIONS

# ── Model loading ─────────────────────────────────────────────────────────
_model_cache = {}

def _find_model():
    active_path = ROOT / AVAILABLE_MODELS.get(_active_model_key, "")
    if active_path.exists():
        return active_path
    for key, rel_path in AVAILABLE_MODELS.items():
        p = ROOT / rel_path
        if p.exists():
            return p
    for n in ["best.pt", "yolov8s.pt", "yolov8n.pt"]:
        p = ROOT / n
        if p.exists():
            return p
    return None


def _auto_download_model():
    model_path = ROOT / "weights" / "best_archive.pt"
    if not model_path.exists():
        try:
            from huggingface_hub import hf_hub_download
            print("Downloading model from Hugging Face...")
            (ROOT / "weights").mkdir(exist_ok=True)
            hf_hub_download(
                repo_id="aishwarya252525/nauticai-yolov8",
                filename="best.pt",
                local_dir=str(ROOT / "weights"),
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            print(f"Model download failed: {e}")


def load_yolo():
    model_path = _find_model()
    if model_path is None:
        return None, None
    key = str(model_path)
    if key not in _model_cache:
        from ultralytics import YOLO
        print(f"Loading model: {model_path.name}")
        _model_cache[key] = YOLO(key)
    return _model_cache[key], model_path


def get_model_name():
    _, path = load_yolo()
    return path.name if path else "Demo"


# ── Detection ─────────────────────────────────────────────────────────────
def _pool_for_mode(mode: str):
    if mode == "pipeline":
        return PIPELINE_DEFECTS
    elif mode == "cable":
        return CABLE_DEFECTS
    return DEFECT_CLASSES


def _detect_real(img, conf_thr, iou_thr):
    model, _ = load_yolo()
    if model is None:
        return []
    results = model.predict(img, conf=0.5, iou=0.4, agnostic_nms=True, verbose=False)[0]
    dets = []
    det_id = 0
    img_w = img.width if hasattr(img, "width") else img.shape[1]
    img_h = img.height if hasattr(img, "height") else img.shape[0]
    img_area = img_w * img_h

    has_masks = hasattr(results, "masks") and results.masks is not None

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_i = int(box.cls[0])
        box_area = (x2 - x1) * (y2 - y1)
        coverage = box_area / img_area if img_area > 0 else 0
        if coverage > 0.50 or coverage < 0.005:
            continue
        cls_name = model.names.get(cls_i, DEFECT_CLASSES[cls_i % len(DEFECT_CLASSES)])
        cls = CLASS_REMAP.get(cls_name, CLASS_REMAP.get(cls_name.lower(), cls_name))
        sev = SEVERITY_MAP.get(cls, "Medium")
        det_id += 1

        det = dict(
            id=det_id,
            cls=cls,
            severity=sev,
            conf=conf,
            x1=x1, y1=y1, x2=x2, y2=y2,
            area=box_area,
            model=_active_model_key,
        )

        if has_masks and i < len(results.masks.data):
            mask = results.masks.data[i].cpu().numpy()
            mask_coverage = float(mask.sum()) / (mask.shape[0] * mask.shape[1])
            det["mask_coverage"] = round(mask_coverage * 100, 2)

        dets.append(det)
    return dets


def _detect_synthetic(img, conf_thr, pool):
    w, h = img.size
    rng = np.random.default_rng(sum(img.tobytes()[:64]))
    n = rng.integers(3, 9)
    dets = []
    for i in range(n):
        cx, cy = rng.integers(60, w - 60), rng.integers(60, h - 60)
        bw, bh = rng.integers(40, w // 4), rng.integers(30, h // 5)
        conf = float(rng.uniform(conf_thr, 0.98))
        cls = rng.choice(pool)
        sev = SEVERITY_MAP.get(cls, "Medium")
        x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
        x2, y2 = min(w, cx + bw // 2), min(h, cy + bh // 2)
        dets.append(dict(
            id=i + 1, cls=cls, severity=sev, conf=conf,
            x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
            area=int((x2 - x1) * (y2 - y1)),
            model="synthetic",
        ))
    return dets


def run_detection(img, conf_thr=0.25, iou_thr=0.45, mode="general"):
    pool = _pool_for_mode(mode)
    model, _ = load_yolo()
    if model is not None:
        dets = _detect_real(img, conf_thr, iou_thr)
        if dets:
            return dets
    return _detect_synthetic(img, conf_thr, pool)


# ── Annotation with Smart Label Placement ─────────────────────────────────

SEV_COLOR = {
    "Critical": (0,   0,   220),
    "High":     (0,  100,  255),
    "Medium":   (0,  180,  255),
    "Low":      (50, 200,   50),
    "Unknown":  (150, 150, 150),
}


def _rects_overlap(a, b, pad=4):
    """Check if two rects (x1,y1,x2,y2) overlap with padding"""
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or 
                a[3] + pad <= b[1] or b[3] + pad <= a[1])


def annotate_image(image, detections):
    """Annotate with collision-avoiding label placement"""
    if not detections:
        return image

    from PIL import Image as PILImage
    is_pil = isinstance(image, PILImage.Image)
    if is_pil:
        img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()

    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    placed_labels = []  # Track placed label rectangles

    for rank, d in enumerate(detections):
        bbox = d.get("bbox") or d.get("bounding_box") or {}
        x1 = int(bbox.get("xmin", d.get("x1", 0)))
        y1 = int(bbox.get("ymin", d.get("y1", 0)))
        x2 = int(bbox.get("xmax", d.get("x2", w)))
        y2 = int(bbox.get("ymax", d.get("y2", h)))
        sev = d.get("severity", "Unknown")
        color = SEV_COLOR.get(sev, SEV_COLOR["Unknown"])
        cls = d.get("cls") or d.get("class_name") or "unknown"
        conf = float(d.get("conf", 0))

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Prepare label
        box_w = max(x2 - x1, 1)
        label = f"[{str(rank+1).zfill(2)}] {cls} {conf*100:.0f}%"
        fs = min(0.5, max(0.3, box_w / 200.0))
        (lw, lh), _ = cv2.getTextSize(label, font, fs, 1)

        # Try candidate positions: above, below, above-right, below-right, right, left
        candidates = [
            (x1, y1 - lh - 8),           # above-left
            (x2 - lw - 6, y1 - lh - 8),  # above-right
            (x1, y2 + 4),                # below-left
            (x2 - lw - 6, y2 + 4),       # below-right
            (x2 + 4, y1),                # right-top
            (x2 + 4, y2 - lh - 6),       # right-bottom
            (x1 - lw - 10, y1),          # left-top
            (x1 - lw - 10, y2 - lh - 6), # left-bottom
        ]

        chosen = None
        for lx, ly in candidates:
            # Check bounds
            if lx < 0 or ly < 0 or lx + lw + 6 > w or ly + lh + 6 > h:
                continue
            
            # Label rect with padding
            label_rect = (lx, ly, lx + lw + 6, ly + lh + 6)
            
            # Check collision with existing labels
            if not any(_rects_overlap(label_rect, pr) for pr in placed_labels):
                chosen = (lx, ly)
                placed_labels.append(label_rect)
                break

        # Fallback: clamp to image (collision allowed)
        if chosen is None:
            lx = min(max(0, x1), w - lw - 8)
            ly = max(0, y1 - lh - 8) if y1 - lh - 8 >= 0 else y1 + lh + 4
            chosen = (lx, ly)

        lx, ly = chosen
        
        # Draw label background
        bg_y1, bg_y2 = ly, ly + lh + 6
        cv2.rectangle(img, (lx, bg_y1), (lx + lw + 6, bg_y2), color, -1)
        cv2.putText(img, label, (lx + 3, ly + lh + 2), font, fs, (255, 255, 255), 1, cv2.LINE_AA)

    if is_pil:
        return PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img


def build_heatmap(pil_img, dets):
    W, H = pil_img.size
    heat = np.zeros((H, W), dtype=np.float32)
    for d in dets:
        cx = min(W - 1, max(0, (d["x1"] + d["x2"]) // 2))
        cy = min(H - 1, max(0, (d["y1"] + d["y2"]) // 2))
        heat[cy, cx] += float(SEV_WEIGHT.get(d["severity"], 0))
    if heat.max() > 0:
        avg_area = np.mean([d.get("area", 3000) for d in dets]) if dets else 3000
        sig  = max(30, math.sqrt(avg_area) * 0.35)
        heat = gaussian_filter(heat, sigma=sig)
        heat = (heat / heat.max() * 255).astype(np.uint8)
    cmap = matplotlib.colormaps.get_cmap("plasma")
    hmap = (cmap(heat / 255.0)[:, :, :3] * 255).astype(np.uint8)
    dark = ImageEnhance.Brightness(pil_img).enhance(0.4)
    return Image.blend(dark, Image.fromarray(hmap).resize((W, H)), alpha=0.62)
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


app = FastAPI(title="FireWatch YOLO Inference", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/data/model_cache"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/firewatch_outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None

# Point these to the model repos/files that hold your weights.
# You can set these in Hugging Face Space Settings -> Variables / Secrets.
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "omerfarooq223/FireWatch-AI").strip()
HF_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "detector-best.pt").strip()
HF_SEG_MODEL_REPO = os.getenv("HF_SEG_MODEL_REPO", HF_MODEL_REPO).strip()
HF_SEG_MODEL_FILENAME = os.getenv("HF_SEG_MODEL_FILENAME", "seg-best.pt").strip()

_yolo_det: Optional[Any] = None
_yolo_seg: Optional[Any] = None


def _download_weight(repo_id: str, filename: str) -> Path:
    if not repo_id or not filename:
        raise FileNotFoundError("Missing Hugging Face model repo or filename.")
    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=HF_TOKEN,
                cache_dir=str(MODEL_CACHE_DIR),
            )
        )
    except Exception as exc:
        raise FileNotFoundError(f"Could not load {filename} from {repo_id}: {exc}") from exc


def get_yolo_det() -> YOLO:
    global _yolo_det
    if _yolo_det is None:
        _yolo_det = YOLO(str(_download_weight(HF_MODEL_REPO, HF_MODEL_FILENAME)))
    return _yolo_det


def get_yolo_seg() -> YOLO:
    global _yolo_seg
    if _yolo_seg is None:
        _yolo_seg = YOLO(str(_download_weight(HF_SEG_MODEL_REPO, HF_SEG_MODEL_FILENAME)))
    return _yolo_seg


def _boxes_from_result(result, im_w: int, im_h: int) -> list:
    out = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    for i in range(len(result.boxes)):
        xyxy = result.boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = map(float, xyxy)
        cls_id = int(result.boxes.cls[i]) if result.boxes.cls is not None else 0
        conf = float(result.boxes.conf[i]) if result.boxes.conf is not None else 0.0
        name = result.names.get(cls_id, str(cls_id)) if getattr(result, "names", None) else str(cls_id)
        out.append(
            {
                "bbox": {
                    "x": ((x1 + x2) / 2) / im_w,
                    "y": ((y1 + y2) / 2) / im_h,
                    "w": (x2 - x1) / im_w,
                    "h": (y2 - y1) / im_h,
                },
                "class_id": cls_id,
                "confidence": round(conf, 4),
                "class_name": name,
            }
        )
    return out


def _masks_from_result(result, im_w: int, im_h: int) -> tuple[list, float]:
    polygons = []
    area_px = 0.0
    if result.masks is None or result.masks.xy is None:
        return polygons, area_px
    for xy in result.masks.xy:
        if xy is None or len(xy) < 3:
            continue
        polygons.append([{"x": float(px) / im_w, "y": float(py) / im_h} for px, py in xy])
        area_px += float(cv2.contourArea(np.array(xy, dtype=np.float32)))
    return polygons, area_px


def _largest_bbox(entries: list) -> Optional[dict]:
    if not entries:
        return None
    return max(entries, key=lambda e: e["bbox"]["w"] * e["bbox"]["h"])["bbox"]


def _bbox_from_polygon(poly: list) -> dict:
    xs = [p["x"] for p in poly]
    ys = [p["y"] for p in poly]
    return {
        "x": (min(xs) + max(xs)) / 2,
        "y": (min(ys) + max(ys)) / 2,
        "w": max(xs) - min(xs),
        "h": max(ys) - min(ys),
    }


def infer_frame_yolo(frame_bgr: np.ndarray, conf: float = 0.25) -> dict:
    if frame_bgr is None or frame_bgr.size == 0:
        return {
            "fire": False,
            "smoke": False,
            "bbox": None,
            "fire_masks": [],
            "fire_segment_area_pixels": 0.0,
            "fire_boxes": [],
            "smoke_boxes": [],
            "raw_detections": [],
            "raw_segmentation_detections": [],
        }

    im_h, im_w = frame_bgr.shape[:2]
    det_result = get_yolo_det()(frame_bgr, conf=conf, verbose=False)[0]
    seg_result = get_yolo_seg()(frame_bgr, conf=conf, verbose=False)[0]

    det_entries = _boxes_from_result(det_result, im_w, im_h)
    seg_entries = _boxes_from_result(seg_result, im_w, im_h)
    fire_masks, fire_area_px = _masks_from_result(seg_result, im_w, im_h)

    fire_boxes = [d for d in det_entries if str(d.get("class_name", "")).lower() == "fire"]
    smoke_boxes = [d for d in det_entries if str(d.get("class_name", "")).lower() == "smoke"]

    fire = bool(fire_boxes) or bool(seg_entries) or bool(fire_masks)
    smoke = bool(smoke_boxes)
    bbox = _largest_bbox(seg_entries) or _largest_bbox(fire_boxes) or _largest_bbox(smoke_boxes)
    if bbox is None and fire_masks:
        bbox = _bbox_from_polygon(max(fire_masks, key=lambda m: cv2.contourArea(np.array([[p["x"] * im_w, p["y"] * im_h] for p in m], dtype=np.float32))))

    return {
        "fire": fire,
        "smoke": smoke,
        "bbox": bbox,
        "fire_masks": fire_masks,
        "fire_segment_area_pixels": round(fire_area_px, 1),
        "fire_boxes": fire_boxes,
        "smoke_boxes": smoke_boxes,
        "raw_detections": det_entries,
        "raw_segmentation_detections": seg_entries,
    }


def _confidence_and_explainability(y: dict) -> tuple[float, list[str]]:
    detection_confidences = [
        float(d.get("confidence", 0.0))
        for d in (
            (y.get("fire_boxes") or [])
            + (y.get("smoke_boxes") or [])
            + (y.get("raw_segmentation_detections") or [])
        )
        if d.get("confidence") is not None
    ]
    max_confidence = max(detection_confidences) if detection_confidences else 0.0
    explainability = []
    if y.get("fire"):
        explainability.append("Segmentation mask or fire-class detection activated")
    if y.get("smoke"):
        explainability.append("Smoke-class detector found a candidate region")
    if y.get("fire_segment_area_pixels", 0.0) > 0:
        explainability.append("Fire area estimated from mask pixels")
    if not explainability:
        explainability.append("No fire/smoke evidence crossed the confidence threshold")
    return max_confidence, explainability


def _frame_payload(t: float, y: dict, inference_ms: float) -> dict:
    bb = y.get("bbox")
    confidence, explainability = _confidence_and_explainability(y)
    return {
        "t": round(t, 3),
        "fire": bool(y.get("fire")),
        "smoke": bool(y.get("smoke")),
        "bbox": (
            {
                "x": round(bb["x"], 5),
                "y": round(bb["y"], 5),
                "w": round(bb["w"], 5),
                "h": round(bb["h"], 5),
            }
            if bb
            else None
        ),
        "fire_masks": y.get("fire_masks") or [],
        "fire_segment_area_pixels": y.get("fire_segment_area_pixels", 0.0),
        "smoke_boxes": y.get("smoke_boxes") or [],
        "fire_boxes": y.get("fire_boxes") or [],
        "confidence": round(confidence, 4),
        "inference_ms": round(inference_ms, 2),
        "segmentation_instances": len(y.get("fire_masks") or []),
        "explainability": explainability,
    }


def annotate_frame_bgr(frame_bgr: np.ndarray, y: dict, conf: float) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    cyan = (238, 211, 34)
    smoke_col = (248, 140, 129)

    for d in y.get("smoke_boxes") or []:
        b = d.get("bbox")
        if not b:
            continue
        x1 = int((b["x"] - b["w"] / 2) * w)
        y1 = int((b["y"] - b["h"] / 2) * h)
        x2 = int((b["x"] + b["w"] / 2) * w)
        y2 = int((b["y"] + b["h"] / 2) * h)
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), smoke_col, -1)
        out = cv2.addWeighted(overlay, 0.12, out, 0.88, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), smoke_col, 2, cv2.LINE_AA)

    fire_masks = y.get("fire_masks") or []
    if fire_masks:
        blend = out.copy()
        for poly in fire_masks:
            if not poly or len(poly) < 3:
                continue
            pts = np.array([[int(p["x"] * w), int(p["y"] * h)] for p in poly], dtype=np.int32)
            cv2.fillPoly(blend, [pts], cyan)
        out = cv2.addWeighted(blend, 0.2, out, 0.8, 0)
        for poly in fire_masks:
            if not poly or len(poly) < 3:
                continue
            pts = np.array([[int(p["x"] * w), int(p["y"] * h)] for p in poly], dtype=np.int32)
            cv2.polylines(out, [pts], True, cyan, 2, cv2.LINE_AA)
    elif y.get("bbox") and y.get("fire"):
        b = y["bbox"]
        x1 = int((b["x"] - b["w"] / 2) * w)
        y1 = int((b["y"] - b["h"] / 2) * h)
        x2 = int((b["x"] + b["w"] / 2) * w)
        y2 = int((b["y"] + b["h"] / 2) * h)
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), cyan, -1)
        out = cv2.addWeighted(overlay, 0.12, out, 0.88, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), cyan, 2, cv2.LINE_AA)

    parts = []
    if y.get("fire"):
        parts.append("Fire" + (" seg" if fire_masks else ""))
    if y.get("smoke"):
        parts.append("Smoke")
    text = " + ".join(parts) if parts else "No fire/smoke"
    text = f"{text}   conf={conf:.2f}   fire_area_px={y.get('fire_segment_area_pixels', 0):.0f}"
    cv2.rectangle(out, (8, 8), (min(w - 8, 660), 52), (24, 24, 26), -1)
    cv2.rectangle(out, (8, 8), (min(w - 8, 660), 52), (60, 60, 70), 1)
    cv2.putText(out, text, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 240, 250), 2, cv2.LINE_AA)
    return out


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "firewatch-yolo-inference",
        "det_repo": HF_MODEL_REPO,
        "det_file": HF_MODEL_FILENAME,
        "seg_repo": HF_SEG_MODEL_REPO,
        "seg_file": HF_SEG_MODEL_FILENAME,
        "det_loaded": _yolo_det is not None,
        "seg_loaded": _yolo_seg is not None,
    }


@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...), sample_interval_sec: float = 0.35, conf: float = 0.25):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}:
        raise HTTPException(status_code=400, detail="Unsupported video format.")

    sample_interval_sec = max(0.15, min(float(sample_interval_sec), 2.0))
    conf = max(0.05, min(float(conf), 0.95))

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file.")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        duration_sec = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        step = max(1, int(fps * sample_interval_sec))
        max_samples = int(os.getenv("MAX_VIDEO_SAMPLES", "180"))
        max_analyze_sec = float(os.getenv("MAX_ANALYZE_SECONDS", "120"))

        frames_out = []
        inference_times = []
        positive_frames = 0
        idx = 0
        while idx < frame_count and len(frames_out) < max_samples:
            if duration_sec > 0 and (idx / fps) > max_analyze_sec:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            t = idx / fps if fps > 0 else 0.0
            start = cv2.getTickCount()
            y = infer_frame_yolo(frame, conf=conf)
            inference_ms = ((cv2.getTickCount() - start) / cv2.getTickFrequency()) * 1000
            inference_times.append(inference_ms)
            if y.get("fire") or y.get("smoke"):
                positive_frames += 1
            frames_out.append(_frame_payload(t, y, inference_ms))
            idx += step

        cap.release()

        avg_inference_ms = sum(inference_times) / len(inference_times) if inference_times else 0.0
        confidence_values = [float(f.get("confidence", 0.0)) for f in frames_out if f.get("confidence", 0.0) > 0]
        mean_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        fire_areas = [float(f.get("fire_segment_area_pixels", 0.0)) for f in frames_out]

        return {
            "filename": file.filename,
            "video_width": width,
            "video_height": height,
            "duration_sec": round(min(duration_sec, max_analyze_sec), 2),
            "fps": round(fps, 2),
            "samples": len(frames_out),
            "positive_frames": positive_frames,
            "fire_frames": positive_frames,
            "conf": conf,
            "model_card": {
                "architecture": "Remote YOLO detector + YOLO segmentation",
                "detector_loaded": True,
                "segmentation_loaded": True,
                "confidence_threshold": conf,
                "sample_interval_sec": sample_interval_sec,
                "avg_inference_ms": round(avg_inference_ms, 2),
                "estimated_model_fps": round(1000 / avg_inference_ms, 2) if avg_inference_ms > 0 else 0,
                "mean_detection_confidence": round(mean_confidence, 4),
            },
            "explainability_summary": {
                "peak_fire_area_pixels": round(max(fire_areas) if fire_areas else 0, 1),
                "mean_fire_area_pixels": round(sum(fire_areas) / len(fire_areas), 1) if fire_areas else 0,
                "mask_positive_frames": sum(1 for f in frames_out if f.get("segmentation_instances", 0) > 0),
                "confidence_positive_frames": len(confidence_values),
            },
            "note": "Remote inference served by Hugging Face Spaces.",
            "frames": frames_out,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...), conf: float = 0.25):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    y = infer_frame_yolo(frame, conf=conf)
    annotated = annotate_frame_bgr(frame, y, conf)
    out_name = f"annotated_img_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
    out_path = OUTPUT_DIR / out_name
    cv2.imwrite(str(out_path), annotated)
    payload = _frame_payload(0, y, 0)
    payload.update(
        {
            "filename": file.filename,
            "image_width": frame.shape[1],
            "image_height": frame.shape[0],
            "conf": conf,
            "annotated_image_url": f"/outputs/{out_name}",
        }
    )
    return payload


@app.post("/analyze/video/export")
async def export_video(file: UploadFile = File(...), conf: float = 0.25, infer_stride: int = 1):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}:
        raise HTTPException(status_code=400, detail="Unsupported video format.")

    conf = max(0.05, min(float(conf), 0.95))
    infer_stride = max(1, min(int(infer_stride), 30))

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    out_name = f"FireWatch_annotated_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.mp4"
    out_path = OUTPUT_DIR / out_name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file.")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        max_out_frames = int(os.getenv("MAX_EXPORT_FRAMES", "2400"))

        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise HTTPException(status_code=500, detail="Could not create output video.")

        last_y = {"fire": False, "smoke": False, "bbox": None, "fire_masks": [], "fire_segment_area_pixels": 0.0, "fire_boxes": [], "smoke_boxes": []}
        frame_idx = 0
        written = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % infer_stride == 0:
                last_y = infer_frame_yolo(frame, conf=conf)
            writer.write(annotate_frame_bgr(frame, last_y, conf))
            frame_idx += 1
            written += 1
            if written >= max_out_frames:
                break

        cap.release()
        writer.release()
        return FileResponse(str(out_path), media_type="video/mp4", filename=out_name)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

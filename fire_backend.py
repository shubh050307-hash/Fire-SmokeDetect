"""
Fire Management System Backend
SQLite + FastAPI
- 3 zones with sprinkler options
- Detection image storage with overlays
- No confidence scores (UI adjustable)
- Clean, production-ready
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, selectinload
from sqlalchemy import select, update, delete, func, Column, Integer, String, Float, DateTime, ForeignKey, LargeBinary
from pydantic import BaseModel
from datetime import datetime
import os
import shutil
import time
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io
import json
import tempfile
from typing import Any, Optional

import cv2
import numpy as np
from real_rag_system import RealRAGSystem
from fire_agent import FireManagementAgent
import requests
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
ALERT_MODE = os.getenv("ALERT_MODE", "demo").strip().lower()
AUTO_ALERTS_ENABLED = os.getenv("AUTO_ALERTS_ENABLED", "false").strip().lower() == "true"
ALERT_CONFIRMATION_FRAMES = max(1, int(os.getenv("ALERT_CONFIRMATION_FRAMES", "3")))
ALERT_COOLDOWN_SECONDS = max(0, int(os.getenv("ALERT_COOLDOWN_SECONDS", "300")))
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local").strip().lower()
INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", "").strip().rstrip("/")
REMOTE_INFERENCE_TIMEOUT_SECONDS = max(30, int(os.getenv("REMOTE_INFERENCE_TIMEOUT_SECONDS", "300")))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174",
    ).split(",")
    if origin.strip()
]

PROJECT_ROOT = Path(__file__).resolve().parent
SEG_RESULTS_ZIP = PROJECT_ROOT / "fire_seg_final_results.zip"
SEG_WEIGHTS_PT = PROJECT_ROOT / "models" / "fire_seg" / "weights" / "best.pt"
DEFAULT_DETECT_PT = PROJECT_ROOT / "best.pt"
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "").strip()
HF_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "best.pt").strip() or "best.pt"
HF_SEG_MODEL_REPO = os.getenv("HF_SEG_MODEL_REPO", "").strip()
HF_SEG_MODEL_FILENAME = os.getenv("HF_SEG_MODEL_FILENAME", "best.pt").strip() or "best.pt"
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
HF_MODEL_CACHE_DIR = PROJECT_ROOT / "model_cache"
GOOGLE_TOKEN_FILE = os.getenv("GOOGLE_TOKEN_FILE", "token.json")

_yolo_det: Any = None
_yolo_seg: Any = None
_yolo_det_checked = False
_rag_system: Optional[Any] = None
_fire_agent: Optional[FireManagementAgent] = None
_last_alert_at_by_zone: dict[int, float] = {}

def generate_rag_answer(query: str, retrieved_docs: list, groq_key: str = None) -> str:
    """
    Use Groq LLM to synthesize an answer from retrieved documents
    This is the "GENERATE" part of RAG (Retrieval-Augmented Generation)
    """
    if not groq_key:
        groq_key = GROQ_API_KEY
    
    if not groq_key:
        # Fallback if no API key
        return "I found relevant fire-safety information but cannot synthesize answer (API key missing)."
    
    # Build document context
    doc_context = "\n\n".join([
        f"Document {i+1} ({d.get('title', 'Unknown')}, relevance: {d.get('similarity_score', 0):.1%}):\n{d.get('content', '')[:800]}"
        for i, d in enumerate(retrieved_docs[:3])
    ])
    
    # System prompt for answer synthesis
    system_msg = """You are a fire safety expert. Based on the retrieved documents provided, 
answer the user's fire safety question concisely and accurately. 
Focus on practical, actionable information. 
Be brief but comprehensive (2-3 paragraphs max).
If information is not in the documents, say so clearly."""
    
    user_msg = f"""Based on these fire-safety documents, answer: {query}

Documents:
{doc_context}

Provide a clear, practical answer."""
    
    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                "temperature": 0.5,
                "max_tokens": 800
            },
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("choices") and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
        
        error_detail = response.text
        try:
            error_detail = response.json().get("error", {}).get("message", response.text)
        except:
            pass
            
        return f"Error generating answer (HTTP {response.status_code}): {error_detail}"
    except Exception as e:
        return f"Could not synthesize answer: {str(e)}"


# ============= DATABASE SETUP =============
DATABASE_URL = f"sqlite+aiosqlite:///{PROJECT_ROOT / 'fire_system.db'}"
engine = create_async_engine(DATABASE_URL, connect_args={"check_same_thread": False})
AsyncSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)
Base = declarative_base()

# Create directories
IMAGES_DIR = PROJECT_ROOT / "detection_images"
IMAGES_DIR.mkdir(exist_ok=True)

# ============= DATABASE MODELS =============
class Zone(Base):
    __tablename__ = "zones"
    
    zone_id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    area_sqm = Column(Float)  # Square meters
    description = Column(String)
    location_address = Column(String)  # Detailed physical address
    
    def to_dict(self):
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "area_sqm": self.area_sqm,
            "description": self.description,
            "location_address": self.location_address
        }


class Detection(Base):
    __tablename__ = "detections"
    
    detection_id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    zone_id = Column(Integer, ForeignKey("zones.zone_id"))
    coordinates_x = Column(Float)  # Pixel X
    coordinates_y = Column(Float)  # Pixel Y
    bbox_w = Column(Float)  # Bounding box width
    bbox_h = Column(Float)  # Bounding box height
    image_filename = Column(String)  # Stored image path
    segment_area_pixels = Column(Float)  # Segmentation area
    raw_image_data = Column(LargeBinary)  # Original image (backup)
    
    def to_dict(self):
        return {
            "detection_id": self.detection_id,
            "timestamp": self.timestamp.isoformat(),
            "zone_id": self.zone_id,
            "coordinates": {"x": self.coordinates_x, "y": self.coordinates_y},
            "bbox": {"w": self.bbox_w, "h": self.bbox_h},
            "segment_area_pixels": self.segment_area_pixels,
            "image_filename": self.image_filename
        }


class Incident(Base):
    __tablename__ = "incidents"
    
    incident_id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    detection_id = Column(Integer, ForeignKey("detections.detection_id"))
    zone_id = Column(Integer, ForeignKey("zones.zone_id"))
    action_taken = Column(String)  # "Sprinkler activated", "Evacuation started"
    sprinkler_type = Column(String)  # "Sprinkler", "CO2", "Foam", "Water Mist"
    agent_reasoning = Column(String)  # Why this action?
    status = Column(String, default="active")  # active, resolved, manual_override
    
    def to_dict(self):
        return {
            "incident_id": self.incident_id,
            "timestamp": self.timestamp.isoformat(),
            "detection_id": self.detection_id,
            "zone_id": self.zone_id,
            "action_taken": self.action_taken,
            "sprinkler_type": self.sprinkler_type,
            "agent_reasoning": self.agent_reasoning,
            "status": self.status
        }


class SafetyProcedure(Base):
    __tablename__ = "safety_procedures"
    
    procedure_id = Column(Integer, primary_key=True)
    zone_id = Column(Integer, ForeignKey("zones.zone_id"))
    protocol_name = Column(String)
    evacuation_time_min = Column(Float)
    procedure_steps = Column(String)  # JSON string of steps
    suppression_type = Column(String)  # Primary suppression
    
    def to_dict(self):
        return {
            "procedure_id": self.procedure_id,
            "zone_id": self.zone_id,
            "protocol_name": self.protocol_name,
            "evacuation_time_min": self.evacuation_time_min,
            "procedure_steps": json.loads(self.procedure_steps),
            "suppression_type": self.suppression_type
        }


class Participant(Base):
    __tablename__ = "participants"
    
    participant_id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)
    role = Column(String, default="Stakeholder")
    is_active = Column(Integer, default=1)  # 1 for active, 0 for inactive
    
    def to_dict(self):
        return {
            "participant_id": self.participant_id,
            "name": self.name,
            "email": self.email,
            "role": self.role,
            "is_active": bool(self.is_active)
        }


# ============= PYDANTIC SCHEMAS =============
class ParticipantCreate(BaseModel):
    name: str
    email: str
    role: str = "Stakeholder"


class ParticipantUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class DetectionCreate(BaseModel):
    zone_id: int
    coordinates_x: float
    coordinates_y: float
    bbox_w: float
    bbox_h: float
    segment_area_pixels: float


class IncidentCreate(BaseModel):
    detection_id: int
    zone_id: int
    action_taken: str
    sprinkler_type: str  # "Sprinkler", "CO2", "Foam", "Water Mist"
    agent_reasoning: str


class ZoneCreate(BaseModel):
    name: str
    area_sqm: float
    description: str


class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = 3


class InstructorReportRequest(BaseModel):
    analysis: dict[str, Any]
    metrics: dict[str, Any]
    level: str
    insights: dict[str, Any]


# ============= FASTAPI APP =============
app = FastAPI(title="Fire Management System", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve detection images
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


# ============= CREATE TABLES (ASYNC) & CREDENTIAL CHECKS =============
@app.on_event("startup")
async def startup():
    # 1. Create DB Tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # 1.5. Seed Participants from Environment variable if table empty
    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(select(Participant))
            existing = result.scalars().first()
            if not existing:
                env_receivers = os.getenv("REMINDER_EMAIL_RECEIVERS", "")
                default_emails = [email.strip() for email in env_receivers.split(",") if email.strip()]
                for email in default_emails:
                    name = email.split("@")[0].replace(".", " ").title()
                    db.add(Participant(name=name, email=email, role="Stakeholder", is_active=1))
                await db.commit()
                print(f"✅ Pre-seeded {len(default_emails)} alert participants from env config.")
        except Exception as e:
            print(f"⚠️  Error pre-seeding participants: {e}")
    
    # 2. Check Gmail API (Alerts)
    if not os.path.exists(GOOGLE_TOKEN_FILE):
        print(f"⚠️  WARNING: '{GOOGLE_TOKEN_FILE}' not found. Gmail alerts will fail until OAuth is configured.")
    else:
        print(f"✅ Gmail API token detected at {GOOGLE_TOKEN_FILE}.")
        
    # 3. Check inference routing.
    if remote_inference_enabled():
        print(f"✅ Remote YOLO inference enabled: {INFERENCE_SERVICE_URL or 'INFERENCE_SERVICE_URL missing'}")
    else:
        print("ℹ️  Local YOLO inference mode enabled.")
        if DEFAULT_DETECT_PT.is_file():
            print(f"✅ YOLO detection weights found at {DEFAULT_DETECT_PT} (will load on first detection).")
        elif HF_MODEL_REPO:
            print(f"ℹ️  YOLO detection weights will be downloaded from '{HF_MODEL_REPO}' on first detection.")
        else:
            print(f"⚠️  WARNING: No YOLO detection weights found. Set HF_MODEL_REPO for hosted weights.")

        if SEG_WEIGHTS_PT.is_file():
            print(f"✅ YOLO segmentation weights found (will load on first detection).")
        elif HF_SEG_MODEL_REPO:
            print(f"ℹ️  YOLO segmentation weights will be downloaded from '{HF_SEG_MODEL_REPO}' on first detection.")
        else:
            print(f"ℹ️  YOLO segmentation weights not pre-loaded (will resolve from zip/HF on first use).")

    # 4. Eagerly initialize RAG System (lightweight)
    print("⏳ Initializing RAG system...")
    try:
        rag = get_rag_system()
        print(f"✅ RAG system initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")

    # 5. Eagerly initialize Fire Management Agent (lightweight)
    print("⏳ Initializing Fire Management Agent...")
    try:
        agent = get_fire_agent()
        print(f"✅ Fire Management Agent initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize Fire Management Agent: {e}")

    print("🚀 Startup complete — all systems checked.")


# ============= UTILITY FUNCTIONS =============
def create_beautiful_detection_image(image_data: bytes, zone_name: str, coords: tuple, bbox: tuple) -> str:
    """
    Create a beautiful detection image with overlays
    - Bounding box
    - Zone label
    - Timestamp
    - Flame icon (emoji or text)
    """
    try:
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Image dimensions
        img_w, img_h = img.size
        
        # Unpack coordinates and bbox
        x, y = coords
        w, h = bbox
        
        # Draw bounding box (bright red)
        x1, y1 = max(0, x - w/2), max(0, y - h/2)
        x2, y2 = min(img_w, x + w/2), min(img_h, y + h/2)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Draw semi-transparent overlay
        overlay = Image.new('RGBA', img.size, (255, 0, 0, 30))
        overlay_region = Image.new('RGBA', (int(x2-x1), int(y2-y1)), (255, 0, 0, 80))
        img_rgba = img.convert('RGBA')
        img_rgba.paste(overlay_region, (int(x1), int(y1)), overlay_region)
        img = img_rgba.convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Draw info box (top-left)
        box_width = 300
        box_height = 80
        draw.rectangle([10, 10, 10 + box_width, 10 + box_height], fill="black", outline="red", width=2)
        
        # Text
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        text_lines = [
            "🔥 FIRE DETECTED",
            f"Zone: {zone_name}",
            f"Time: {timestamp}"
        ]
        
        y_text = 20
        for line in text_lines:
            draw.text((20, y_text), line, fill="yellow")
            y_text += 20
        
        # Save image
        filename = f"detection_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = IMAGES_DIR / filename
        img.save(filepath, quality=95)
        
        return filename
    
    except Exception as e:
        print(f"Error creating detection image: {e}")
        return "error.png"


async def get_db():
    async with AsyncSessionLocal() as db:
        try:
            yield db
        finally:
            await db.close()


def detect_fire_bbox_bgr(frame_bgr: np.ndarray):
    """
    Heuristic fire/smoke-colored region in BGR frame.
    Returns (center_x, center_y, bbox_w, bbox_h) in pixels, or None.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    h, w = frame_bgr.shape[:2]
    blurred = cv2.GaussianBlur(frame_bgr, (21, 21), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 55, 55), (35, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 55, 55), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    min_area = max(500, (w * h) * 0.0015)
    if area < min_area:
        return None
    x, y, bw, bh = cv2.boundingRect(best)
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    return cx, cy, float(bw), float(bh)


def ensure_segmentation_weights() -> Path:
    """Resolve YOLO segmentation weights from local files, zip, or Hugging Face."""
    if SEG_WEIGHTS_PT.is_file():
        return SEG_WEIGHTS_PT
    if HF_SEG_MODEL_REPO:
        try:
            return Path(
                hf_hub_download(
                    repo_id=HF_SEG_MODEL_REPO,
                    filename=HF_SEG_MODEL_FILENAME,
                    token=HF_TOKEN,
                    cache_dir=str(HF_MODEL_CACHE_DIR),
                )
            )
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not download segmentation weights from Hugging Face repo "
                f"'{HF_SEG_MODEL_REPO}': {exc}"
            ) from exc
    if not SEG_RESULTS_ZIP.is_file():
        raise FileNotFoundError(
            f"Segmentation model not found. Place {SEG_RESULTS_ZIP.name} in {PROJECT_ROOT} "
            f"or extract weights to {SEG_WEIGHTS_PT} or set HF_SEG_MODEL_REPO."
        )
    SEG_WEIGHTS_PT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SEG_RESULTS_ZIP, "r") as zf:
        member = "weights/best.pt"
        if member not in zf.namelist():
            raise FileNotFoundError(f"{SEG_RESULTS_ZIP.name} is missing {member}")
        zf.extract(member, SEG_WEIGHTS_PT.parent.parent)
    return SEG_WEIGHTS_PT


def get_yolo_class():
    """Import Ultralytics only when local inference actually needs it."""
    from ultralytics import YOLO

    return YOLO


def get_yolo_seg():
    global _yolo_seg
    if _yolo_seg is None:
        print("⏳ Loading YOLO segmentation model (first use)...")
        wpath = ensure_segmentation_weights()
        _yolo_seg = get_yolo_class()(str(wpath))
        print("✅ YOLO segmentation model loaded.")
    return _yolo_seg


def get_yolo_det() -> Optional[Any]:
    """Optional detector from FIRE_DETECT_MODEL env or ./best.pt (not the seg weights file)."""
    global _yolo_det, _yolo_det_checked
    if _yolo_det_checked:
        return _yolo_det
    _yolo_det_checked = True
    env = os.environ.get("FIRE_DETECT_MODEL")
    path = Path(env).expanduser() if env else DEFAULT_DETECT_PT
    if not path.is_file():
        if HF_MODEL_REPO:
            try:
                path = Path(
                    hf_hub_download(
                        repo_id=HF_MODEL_REPO,
                        filename=HF_MODEL_FILENAME,
                        token=HF_TOKEN,
                        cache_dir=str(HF_MODEL_CACHE_DIR),
                    )
                )
            except Exception as exc:
                print(f"⚠️  Could not download YOLO weights from Hugging Face: {exc}")
                _yolo_det = None
                return None
        else:
            _yolo_det = None
            return None
    try:
        rpath = path.resolve()
        if rpath == SEG_WEIGHTS_PT.resolve():
            _yolo_det = None
            return None
    except OSError:
        pass
    print(f"⏳ Loading YOLO detection model from {path} (first use)...")
    _yolo_det = get_yolo_class()(str(path))
    print("✅ YOLO detection model loaded.")
    return _yolo_det


def get_rag_system() -> Any:
    global _rag_system
    if _rag_system is None:
        _rag_system = RealRAGSystem()
    return _rag_system


def get_fire_agent() -> FireManagementAgent:
    global _fire_agent
    if _fire_agent is None:
        _fire_agent = FireManagementAgent(rag_system=get_rag_system())
    return _fire_agent


def get_device_location(lat: float = None, lon: float = None) -> str:
    """Detect the physical address of the current device via GPS or IP Geolocation"""
    return resolve_device_location(lat=lat, lon=lon)["address"]


def resolve_device_location(lat: float = None, lon: float = None) -> dict:
    """Resolve coordinates and a readable address without leaking local variables to callers."""
    try:
        if lat is None or lon is None:
            geo_res = requests.get("http://ip-api.com/json/", timeout=3).json()
            if geo_res.get("status") == "success":
                lat = geo_res.get("lat")
                lon = geo_res.get("lon")
        
        if lat and lon:
            # 2. Reverse Geocode to get a readable street address (using OpenStreetMap Nominatim)
            headers = {"User-Agent": "FireWatchAI/1.0"}
            rev_res = requests.get(
                f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}", 
                headers=headers,
                timeout=3
            ).json()
            
            address = rev_res.get("display_name")
            if address:
                return {"address": address, "lat": lat, "lon": lon}
            return {"address": f"Lat: {lat}, Lon: {lon}", "lat": lat, "lon": lon}
            
    except Exception as e:
        print(f"Location detection failed: {e}")
    
    return {"address": "Unknown Live Location", "lat": lat, "lon": lon}


def should_dispatch_alert(zone_id: int) -> tuple[bool, str]:
    """Rate-limit outbound emergency notifications per zone."""
    if not AUTO_ALERTS_ENABLED:
        return False, "automatic alerts disabled"

    now = time.monotonic()
    last = _last_alert_at_by_zone.get(zone_id)
    if last is not None and now - last < ALERT_COOLDOWN_SECONDS:
        remaining = int(ALERT_COOLDOWN_SECONDS - (now - last))
        return False, f"cooldown active for {remaining}s"

    _last_alert_at_by_zone[zone_id] = now
    return True, "dispatch allowed"


def dispatch_confirmed_alert(
    background_tasks: BackgroundTasks,
    agent: FireManagementAgent,
    zone_name: str,
    address: str,
    lat: float = None,
    lon: float = None,
) -> None:
    """Send the configured alert type after detection confirmation."""
    reasoning = (
        f"Fire/smoke confirmed across {ALERT_CONFIRMATION_FRAMES} sampled frame(s). "
        "Location data attached when available."
    )

    if ALERT_MODE == "email":
        background_tasks.add_task(
            agent.send_emergency_email,
            zone_name=zone_name,
            address=address,
            severity="CRITICAL",
            action_taken="CONFIRMED ALERT DISPATCHED",
            reasoning=reasoning,
            lat=lat,
            lon=lon,
        )
    else:
        background_tasks.add_task(
            agent.record_demo_alert,
            channel="email",
            zone_name=zone_name,
            address=address,
            severity="CRITICAL",
            action_taken="CONFIRMED ALERT PREPARED",
            reasoning=reasoning,
            lat=lat,
            lon=lon,
        )


def remote_inference_enabled() -> bool:
    return INFERENCE_MODE == "remote"


def require_remote_inference_url() -> str:
    if not INFERENCE_SERVICE_URL:
        raise HTTPException(
            status_code=503,
            detail="Remote inference is enabled but INFERENCE_SERVICE_URL is not configured.",
        )
    return INFERENCE_SERVICE_URL


async def forward_upload_to_remote(
    endpoint: str,
    file: UploadFile,
    params: dict[str, Any],
) -> requests.Response:
    base_url = require_remote_inference_url()
    remote_url = f"{base_url}/{endpoint.lstrip('/')}"
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    print(f"↗️  Forwarding {file.filename or 'upload'} to remote inference: {remote_url}", flush=True)
    try:
        response = requests.post(
            remote_url,
            params=params,
            files={
                "file": (
                    file.filename or "upload",
                    raw,
                    file.content_type or "application/octet-stream",
                )
            },
            timeout=REMOTE_INFERENCE_TIMEOUT_SECONDS,
        )
    except requests.Timeout as exc:
        raise HTTPException(
            status_code=504,
            detail="Remote inference timed out. The Hugging Face Space may be waking up or the video is too long.",
        ) from exc
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Remote inference service unavailable: {exc}",
        ) from exc

    print(f"↙️  Remote inference response {response.status_code} from {endpoint}", flush=True)
    if response.status_code >= 400:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Remote inference failed: {detail}",
        )

    return response


def _boxes_from_result(result, im_w: int, im_h: int) -> list:
    out = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    for i in range(len(result.boxes)):
        xyxy = result.boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = map(float, xyxy)
        cx = ((x1 + x2) / 2) / im_w
        cy = ((y1 + y2) / 2) / im_h
        bw = (x2 - x1) / im_w
        bh = (y2 - y1) / im_h
        cls_id = int(result.boxes.cls[i]) if result.boxes.cls is not None else 0
        conf = float(result.boxes.conf[i]) if result.boxes.conf is not None else 0.0
        name = result.names.get(cls_id, str(cls_id)) if getattr(result, "names", None) else str(cls_id)
        out.append(
            {
                "bbox": {"x": cx, "y": cy, "w": bw, "h": bh},
                "class_id": cls_id,
                "confidence": round(conf, 4),
                "class_name": name,
            }
        )
    return out


def _masks_from_result(result, im_w: int, im_h: int) -> tuple[list, float]:
    polys: list = []
    area_px = 0.0
    if result.masks is None or result.masks.xy is None:
        return polys, area_px
    for xy in result.masks.xy:
        if xy is None or len(xy) < 3:
            continue
        poly = [{"x": float(px) / im_w, "y": float(py) / im_h} for px, py in xy]
        polys.append(poly)
        arr = np.array(xy, dtype=np.float32)
        area_px += float(cv2.contourArea(arr))
    return polys, area_px


def _largest_bbox(entries: list) -> Optional[dict]:
    if not entries:
        return None
    best = max(entries, key=lambda e: e["bbox"]["w"] * e["bbox"]["h"])
    return best["bbox"]


def _bbox_from_polygon(poly: list) -> dict:
    xs = [p["x"] for p in poly]
    ys = [p["y"] for p in poly]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    return {"x": cx, "y": cy, "w": w, "h": h}


def infer_frame_yolo(frame_bgr: np.ndarray, conf: float = 0.25) -> dict:
    """
    Run optional best.pt detector + segmentation model from zip.
    Returns normalized bbox (center), mask polygons, pixel area, fire flag.
    """
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
    det_model = get_yolo_det()
    seg_model = get_yolo_seg()

    det_entries: list = []
    if det_model is not None:
        dr = det_model(frame_bgr, conf=conf, verbose=False)[0]
        det_entries = _boxes_from_result(dr, im_w, im_h)

    sr = seg_model(frame_bgr, conf=conf, verbose=False)[0]
    seg_entries = _boxes_from_result(sr, im_w, im_h)
    fire_masks, fire_area_px = _masks_from_result(sr, im_w, im_h)

    fire_boxes = [d for d in det_entries if str(d.get("class_name", "")).lower() == "fire"]
    smoke_boxes = [d for d in det_entries if str(d.get("class_name", "")).lower() == "smoke"]

    fire = bool(fire_boxes) or bool(seg_entries) or bool(fire_masks)
    smoke = bool(smoke_boxes)
    bbox = _largest_bbox(seg_entries) or _largest_bbox(fire_boxes) or _largest_bbox(smoke_boxes)
    if bbox is None and fire_masks:
        def _poly_area_px(m):
            arr = np.array([[p["x"] * im_w, p["y"] * im_h] for p in m], dtype=np.float32)
            return cv2.contourArea(arr)

        bbox = _bbox_from_polygon(max(fire_masks, key=_poly_area_px))

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


def _dashed_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple, thickness: int = 2) -> None:
    dash, gap = 14, 10

    for x in range(x1, x2, dash + gap):
        xe = min(x + dash, x2)
        cv2.line(img, (xe, y1), (x, y1), color, thickness)
    for x in range(x1, x2, dash + gap):
        xe = min(x + dash, x2)
        cv2.line(img, (xe, y2), (x, y2), color, thickness)
    for y in range(y1, y2, dash + gap):
        ye = min(y + dash, y2)
        cv2.line(img, (x2, ye), (x2, y), color, thickness)
    for y in range(y1, y2, dash + gap):
        ye = min(y + dash, y2)
        cv2.line(img, (x1, ye), (x1, y), color, thickness)


def annotate_frame_bgr(frame_bgr: np.ndarray, y: dict, conf: float) -> np.ndarray:
    """Draw fire masks (cyan) and smoke boxes (indigo dashed) — matches frontend styling."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    cyan = (238, 211, 34)  # BGR ≈ #22d3ee
    smoke_col = (248, 140, 129)  # BGR ≈ #818cf8

    for d in y.get("smoke_boxes") or []:
        b = d.get("bbox")
        if not b:
            continue
        cx, cy, bw, bh = b["x"], b["y"], b["w"], b["h"]
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        x1, x2 = sorted([max(0, min(x1, w - 1)), max(0, min(x2, w - 1))])
        y1, y2 = sorted([max(0, min(y1, h - 1)), max(0, min(y2, h - 1))])
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), smoke_col, -1)
        out = cv2.addWeighted(overlay, 0.12, out, 0.88, 0)
        _dashed_rect(out, x1, y1, x2, y2, smoke_col, 2)

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
    else:
        fb = y.get("fire_boxes") or []
        box = None
        if fb and fb[0].get("bbox"):
            box = fb[0]["bbox"]
        elif y.get("bbox") and (y.get("fire") or fire_masks):
            box = y["bbox"]
        if box and y.get("fire"):
            cx, cy, bw, bh = box["x"], box["y"], box["w"], box["h"]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            overlay = out.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), cyan, -1)
            out = cv2.addWeighted(overlay, 0.12, out, 0.88, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), cyan, 2, cv2.LINE_AA)

    parts = []
    if y.get("fire"):
        parts.append("Fire" + (" seg" if fire_masks else ""))
    if y.get("smoke"):
        parts.append("Smoke")
    text = " · ".join(parts) if parts else "No fire/smoke"
    text = f"{text}   conf={conf:.2f}   fire_area_px={y.get('fire_segment_area_pixels', 0):.0f}"
    tw = max(420, min(w - 20, 18 * len(text) // 2))
    cv2.rectangle(out, (8, 8), (12 + tw, 52), (24, 24, 26), -1)
    cv2.rectangle(out, (8, 8), (12 + tw, 52), (60, 60, 70), 1)
    cv2.putText(out, text, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 240, 250), 2, cv2.LINE_AA)
    return out


def _report_pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "0.0%"


def _report_num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _report_text(value: Any, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def build_instructor_pdf(payload: InstructorReportRequest) -> bytes:
    """Create a polished PDF report for the Instructor Mode evidence board."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"PDF generation dependency missing: {exc}") from exc

    analysis = payload.analysis or {}
    metrics = payload.metrics or {}
    insights = payload.insights or {}
    model_card = analysis.get("model_card") or {}
    confusion = insights.get("confusion") or {}
    peak_frame = insights.get("peakFrame") or insights.get("peak_frame") or {}
    explanations = peak_frame.get("explainability") or ["No frame-level explanation was available."]

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=14 * mm,
        title="FireWatch AI Instructor Report",
    )

    styles = getSampleStyleSheet()
    cyan = colors.HexColor("#1fbfd0")
    cyan_light = colors.HexColor("#d9fbff")
    dark = colors.HexColor("#0c111b")
    panel = colors.HexColor("#121923")
    panel_alt = colors.HexColor("#17202b")
    muted = colors.HexColor("#70808c")
    line = colors.HexColor("#234653")
    danger = colors.HexColor("#d85858")
    green = colors.HexColor("#27c486")

    title = ParagraphStyle(
        "FireTitle",
        parent=styles["Title"],
        textColor=cyan_light,
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=28,
        spaceAfter=4,
    )
    subtitle = ParagraphStyle(
        "FireSubtitle",
        parent=styles["Normal"],
        textColor=colors.HexColor("#aebcc7"),
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        spaceAfter=8,
    )
    section = ParagraphStyle(
        "FireSection",
        parent=styles["Heading2"],
        textColor=cyan,
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        spaceBefore=8,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "FireBody",
        parent=styles["BodyText"],
        textColor=colors.HexColor("#dbe6ec"),
        fontName="Helvetica",
        fontSize=8.7,
        leading=12,
    )
    body_muted = ParagraphStyle(
        "FireMuted",
        parent=body,
        textColor=colors.HexColor("#96a6b2"),
        fontSize=8,
    )
    right = ParagraphStyle("Right", parent=body, alignment=TA_RIGHT)
    center = ParagraphStyle("Center", parent=body, alignment=TA_CENTER)

    def p(text: Any, style=body) -> Paragraph:
        safe = _report_text(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return Paragraph(safe, style)

    def small_card(label: str, value: str, detail: str = "") -> list:
        return [
            p(label.upper(), ParagraphStyle(f"{label}Label", parent=body_muted, textColor=cyan, fontName="Helvetica-Bold", fontSize=7, leading=9)),
            p(value, ParagraphStyle(f"{label}Value", parent=body, textColor=colors.white, fontName="Helvetica-Bold", fontSize=14, leading=18)),
            p(detail, body_muted),
        ]

    generated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    filename = _report_text(analysis.get("filename"), "Uploaded video")
    risk = int(_report_num(metrics.get("risk")))
    level = _report_text(payload.level, "low").upper()
    verdict = _report_text(insights.get("verdict"), "Continue monitoring")
    avg_conf = _report_num(insights.get("avgConfidence")) * 100
    avg_infer = _report_num(insights.get("avgInference"))
    fps_est = _report_text(model_card.get("estimated_model_fps"), "N/A")
    peak_area = _report_num(metrics.get("peak"))

    story = []
    header_data = [
        [
            [p("FIREWATCH AI", ParagraphStyle("Brand", parent=body, textColor=cyan, fontName="Helvetica-Bold", fontSize=8, leading=10)),
             Paragraph("Instructor Evidence Report", title),
             p(f"Generated: {generated}", subtitle)],
            [p("RISK LEVEL", ParagraphStyle("RiskLabel", parent=body_muted, alignment=TA_RIGHT, textColor=cyan, fontName="Helvetica-Bold", fontSize=7)),
             Paragraph(level, ParagraphStyle("RiskValue", parent=right, textColor=danger if level == "CRITICAL" else cyan_light, fontName="Helvetica-Bold", fontSize=18, leading=22)),
             p(f"{risk}/100 risk score", right)],
        ]
    ]
    header = Table(header_data, colWidths=[120 * mm, 45 * mm])
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), dark),
        ("BOX", (0, 0), (-1, -1), 0.75, line),
        ("INNERPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.extend([header, Spacer(1, 7)])

    cards = Table(
        [[
            small_card("Model Decision", verdict, f"{insights.get('positiveCount', 0)} / {insights.get('total', 0)} sampled frames flagged"),
            small_card("Inference Speed", f"{avg_infer:.0f} ms", f"{fps_est} estimated FPS"),
            small_card("Confidence", _report_pct(avg_conf), f"Threshold: {analysis.get('conf', model_card.get('confidence_threshold', 0.25))}"),
            small_card("Segmentation", str(insights.get("maskCount", 0)), "mask-positive frames"),
        ]],
        colWidths=[58 * mm, 35 * mm, 35 * mm, 37 * mm],
    )
    cards.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), panel),
        ("BOX", (0, 0), (-1, -1), 0.65, line),
        ("INNERGRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#1f3440")),
        ("INNERPADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.extend([cards, Spacer(1, 7)])

    story.append(Paragraph("Decision Summary", section))
    summary = Table(
        [
            [p("Video", body_muted), p(filename), p("Resolution", body_muted), p(f"{analysis.get('video_width', 0)} x {analysis.get('video_height', 0)}")],
            [p("Duration", body_muted), p(f"{analysis.get('duration_sec', 0)} sec"), p("Samples", body_muted), p(str(analysis.get("samples", 0)))],
            [p("First Detection", body_muted), p(f"{_report_num((insights.get('firstDetection') or {}).get('t')):.2f}s"), p("Peak Fire Area", body_muted), p(f"{peak_area:,.0f} px")],
        ],
        colWidths=[30 * mm, 52 * mm, 30 * mm, 53 * mm],
    )
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), panel),
        ("BOX", (0, 0), (-1, -1), 0.65, line),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#1d313b")),
        ("INNERPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([summary, Spacer(1, 4)])

    story.append(Paragraph("Deep Learning Evidence", section))
    evidence = Table(
        [
            [p("Architecture", body_muted), p(_report_text(model_card.get("architecture"), "YOLO detector + YOLO segmentation"))],
            [p("Detector Loaded", body_muted), p("Yes" if model_card.get("detector_loaded") else "Optional detector unavailable")],
            [p("Segmentation Loaded", body_muted), p("Yes" if model_card.get("segmentation_loaded", True) else "No")],
            [p("Mean Fire Area", body_muted), p(f"{_report_num((analysis.get('explainability_summary') or {}).get('mean_fire_area_pixels')):,.1f} px")],
        ],
        colWidths=[42 * mm, 123 * mm],
    )
    evidence.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), panel_alt),
        ("BOX", (0, 0), (-1, -1), 0.65, line),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#1d313b")),
        ("INNERPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([evidence, Spacer(1, 4)])

    story.append(Paragraph("Evaluation Snapshot", section))
    eval_table = Table(
        [
            [p("Precision", center), p("Recall", center), p("F1 Score", center), p("TP", center), p("FP", center), p("FN", center), p("TN", center)],
            [
                p(_report_pct(insights.get("estimatedPrecision")), center),
                p(_report_pct(insights.get("estimatedRecall")), center),
                p(_report_pct(insights.get("estimatedF1")), center),
                p(str(confusion.get("tp", 0)), center),
                p(str(confusion.get("fp", 0)), center),
                p(str(confusion.get("fn", 0)), center),
                p(str(confusion.get("tn", 0)), center),
            ],
        ],
        colWidths=[25 * mm, 25 * mm, 25 * mm, 22.5 * mm, 22.5 * mm, 22.5 * mm, 22.5 * mm],
    )
    eval_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#10252d")),
        ("BACKGROUND", (0, 1), (-1, 1), panel),
        ("TEXTCOLOR", (0, 0), (-1, 0), cyan),
        ("BOX", (0, 0), (-1, -1), 0.65, line),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#1d313b")),
        ("INNERPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([eval_table, p("These values summarize the analyzed sample. Use a labeled validation set for final academic precision, recall, F1, and confusion-matrix claims.", body_muted), Spacer(1, 4)])

    story.append(Paragraph("Explainable AI Trace", section))
    trace_rows = [[p(str(i + 1), center), p(item)] for i, item in enumerate(explanations)]
    trace = Table(trace_rows, colWidths=[12 * mm, 153 * mm])
    trace.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), panel),
        ("BOX", (0, 0), (-1, -1), 0.65, line),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#1d313b")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#10252d")),
        ("TEXTCOLOR", (0, 0), (0, -1), cyan),
        ("INNERPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([trace, Spacer(1, 4)])

    story.append(Paragraph("Response Pipeline", section))
    pipeline = Table(
        [
            [p("1", center), p("Raw Video"), p(f"Frame stream sampled at {model_card.get('sample_interval_sec', 0.2)}s intervals.")],
            [p("2", center), p("YOLO + Mask"), p("Bounding boxes, segmentation masks, confidence, and fire area are extracted.")],
            [p("3", center), p("Risk Engine"), p(f"Temporal fire/smoke evidence becomes a {risk}/100 response score.")],
        ],
        colWidths=[12 * mm, 34 * mm, 119 * mm],
    )
    pipeline.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), panel_alt),
        ("BOX", (0, 0), (-1, -1), 0.65, line),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#1d313b")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#10252d")),
        ("TEXTCOLOR", (0, 0), (0, -1), cyan),
        ("INNERPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([pipeline, Spacer(1, 6)])

    footer = Table([[p("FireWatch AI combines YOLO object detection, segmentation masks, temporal risk scoring, RAG-assisted guidance, and demo-safe response workflows.", body_muted)]], colWidths=[165 * mm])
    footer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#071018")),
        ("BOX", (0, 0), (-1, -1), 0.65, colors.HexColor("#173540")),
        ("INNERPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(footer)

    def page_canvas(canvas, pdf_doc):
        canvas.saveState()
        canvas.setFillColor(dark)
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        canvas.setStrokeColor(colors.HexColor("#102a33"))
        canvas.setLineWidth(0.7)
        canvas.line(16 * mm, 12 * mm, A4[0] - 16 * mm, 12 * mm)
        canvas.setFillColor(muted)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(16 * mm, 8 * mm, "FireWatch AI - Deep Learning Evidence Report")
        canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, f"Page {pdf_doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=page_canvas, onLaterPages=page_canvas)
    buffer.seek(0)
    return buffer.read()


# ============= PARTICIPANTS ENDPOINTS =============
@app.get("/api/participants")
async def get_participants(db: AsyncSession = Depends(get_db)):
    """Retrieve all email alert participants"""
    result = await db.execute(select(Participant).order_by(Participant.participant_id.asc()))
    participants = result.scalars().all()
    return [p.to_dict() for p in participants]


@app.post("/api/participants")
async def add_participant(participant: ParticipantCreate, db: AsyncSession = Depends(get_db)):
    """Add a new alert participant"""
    # Check if email already exists
    result = await db.execute(select(Participant).filter(Participant.email == participant.email))
    existing = result.scalars().first()
    if existing:
        raise HTTPException(status_code=400, detail="Participant with this email already exists.")
        
    db_participant = Participant(
        name=participant.name,
        email=participant.email,
        role=participant.role,
        is_active=1
    )
    db.add(db_participant)
    await db.commit()
    await db.refresh(db_participant)
    return db_participant.to_dict()


@app.patch("/api/participants/{participant_id}/status")
async def toggle_participant_status(participant_id: int, is_active: bool, db: AsyncSession = Depends(get_db)):
    """Toggle participant is_active status"""
    result = await db.execute(select(Participant).filter(Participant.participant_id == participant_id))
    participant = result.scalars().first()
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
        
    participant.is_active = 1 if is_active else 0
    await db.commit()
    return {"status": "Participant status updated", "participant_id": participant_id, "is_active": is_active}


@app.delete("/api/participants/{participant_id}")
async def delete_participant(participant_id: int, db: AsyncSession = Depends(get_db)):
    """Remove a participant from alerts"""
    result = await db.execute(select(Participant).filter(Participant.participant_id == participant_id))
    participant = result.scalars().first()
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
        
    await db.delete(participant)
    await db.commit()
    return {"status": "Participant deleted", "participant_id": participant_id}


# ============= INITIALIZATION ENDPOINTS =============
@app.post("/api/init/zones")
async def initialize_zones(db: AsyncSession = Depends(get_db)):
    """Initialize 3 zones (run once)"""
    zones_data = [
        {
            "name": "Lobby", 
            "area_sqm": 500, 
            "description": "Main entrance and reception area",
            "location_address": "Main Entrance, Ground Floor, 123 Innovation Drive, Tech City"
        },
        {
            "name": "Server Room", 
            "area_sqm": 200, 
            "description": "Data center with critical infrastructure",
            "location_address": "Restricted Access Wing, 2nd Floor, Building A, 123 Innovation Drive"
        },
        {
            "name": "Warehouse", 
            "area_sqm": 2000, 
            "description": "Storage and inventory area",
            "location_address": "Logistics Center, East Side, 125 Innovation Drive"
        }
    ]
    
    for zone_data in zones_data:
        result = await db.execute(select(Zone).filter(Zone.name == zone_data["name"]))
        existing = result.scalars().first()
        if not existing:
            db.add(Zone(**zone_data))
    
    await db.commit()
    return {"status": "Zones initialized", "count": 3}


@app.post("/api/init/procedures")
async def initialize_procedures(db: AsyncSession = Depends(get_db)):
    """Initialize safety procedures for each zone"""
    procedures = [
        {
            "zone_id": 1,
            "protocol_name": "Lobby Fire Response",
            "evacuation_time_min": 5,
            "procedure_steps": json.dumps([
                "1. Activate alarm system",
                "2. Clear all exits",
                "3. Direct people to assembly point A",
                "4. Activate sprinkler system"
            ]),
            "suppression_type": "Sprinkler"
        },
        {
            "zone_id": 2,
            "protocol_name": "Server Room Fire Response",
            "evacuation_time_min": 2,
            "procedure_steps": json.dumps([
                "1. Stop all operations",
                "2. Evacuate immediately",
                "3. Activate CO2 suppression (no manual override)",
                "4. Close fireproof doors"
            ]),
            "suppression_type": "CO2"
        },
        {
            "zone_id": 3,
            "protocol_name": "Warehouse Fire Response",
            "evacuation_time_min": 10,
            "procedure_steps": json.dumps([
                "1. Alert all personnel",
                "2. Move inventory away from fire",
                "3. Activate foam suppression system",
                "4. Establish perimeter"
            ]),
            "suppression_type": "Foam"
        }
    ]
    
    for proc in procedures:
        result = await db.execute(select(SafetyProcedure).filter(SafetyProcedure.zone_id == proc["zone_id"]))
        existing = result.scalars().first()
        if not existing:
            db.add(SafetyProcedure(**proc))
    
    await db.commit()
    return {"status": "Procedures initialized", "count": 3}


# ============= DETECTION ENDPOINTS =============
@app.post("/api/detections")
async def log_detection(
    background_tasks: BackgroundTasks,
    zone_id: int = Form(...),
    coordinates_x: float = Form(...),
    coordinates_y: float = Form(...),
    bbox_w: float = Form(...),
    bbox_h: float = Form(...),
    segment_area_pixels: float = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Log a detection with image
    Image will be beautified with overlays
    """
    try:
        # Read image
        image_data = await file.read()
        
        # Create beautiful image with overlays
        result = await db.execute(select(Zone).filter(Zone.zone_id == zone_id))
        zone = result.scalars().first()
        zone_name = zone.name if zone else f"Zone {zone_id}"
        
        filename = create_beautiful_detection_image(
            image_data,
            zone_name,
            coords=(coordinates_x, coordinates_y),
            bbox=(bbox_w, bbox_h)
        )
        
        # Store detection in DB
        db_detection = Detection(
            zone_id=zone_id,
            coordinates_x=coordinates_x,
            coordinates_y=coordinates_y,
            bbox_w=bbox_w,
            bbox_h=bbox_h,
            segment_area_pixels=segment_area_pixels,
            image_filename=filename,
            raw_image_data=image_data
        )
        
        db.add(db_detection)
        await db.commit()
        await db.refresh(db_detection)
        
        # AUTOMATIC EMERGENCY RESPONSE
        # Trigger the AI Agent reasoning in the background
        agent = get_fire_agent()
        
        # DYNAMIC LOCATION: Use live device location if possible
        live_location = resolve_device_location()
        live_address = live_location["address"]
        result = await db.execute(select(Zone).filter(Zone.zone_id == zone_id))
        zone = result.scalars().first()
        # Fallback to DB if live detection fails
        final_address = live_address if "Unknown" not in live_address else (zone.location_address if zone else "Unknown")
        
        agent_data = {
            "zone_id": zone_id,
            "address": final_address,
            "lat": live_location.get("lat") if "Unknown" not in live_address else None,
            "lon": live_location.get("lon") if "Unknown" not in live_address else None,
            "coordinates": {"x": coordinates_x, "y": coordinates_y},
            "segment_area_pixels": segment_area_pixels,
            "detection_id": db_detection.detection_id,
            "skip_email": not AUTO_ALERTS_ENABLED,
        }
        background_tasks.add_task(agent.reason, agent_data)
        
        return {
            "status": "Detection logged",
            "detection_id": db_detection.detection_id,
            "image_url": f"/images/{filename}",
            "zone": zone_name
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/detections")
async def get_detections(zone_id: int = None, db: AsyncSession = Depends(get_db)):
    """Get all detections, optionally filtered by zone"""
    stmt = select(Detection)
    if zone_id:
        stmt = stmt.filter(Detection.zone_id == zone_id)
    
    result = await db.execute(stmt.order_by(Detection.timestamp.desc()))
    detections = result.scalars().all()
    return [d.to_dict() for d in detections]


@app.get("/api/detections/{detection_id}")
async def get_detection(detection_id: int, db: AsyncSession = Depends(get_db)):
    """Get specific detection"""
    result = await db.execute(select(Detection).filter(Detection.detection_id == detection_id))
    detection = result.scalars().first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    return detection.to_dict()


# ============= INCIDENT ENDPOINTS =============
@app.post("/api/incidents")
async def log_incident(incident: IncidentCreate, db: AsyncSession = Depends(get_db)):
    """Log an incident (agent decision)"""
    db_incident = Incident(**incident.dict())
    db.add(db_incident)
    await db.commit()
    await db.refresh(db_incident)
    return {
        "status": "Incident logged",
        "incident_id": db_incident.incident_id,
        **db_incident.to_dict()
    }


@app.get("/api/incidents")
async def get_incidents(zone_id: int = None, db: AsyncSession = Depends(get_db)):
    """Get all incidents, optionally filtered by zone"""
    stmt = select(Incident)
    if zone_id:
        stmt = stmt.filter(Incident.zone_id == zone_id)
    
    result = await db.execute(stmt.order_by(Incident.timestamp.desc()))
    incidents = result.scalars().all()
    return [i.to_dict() for i in incidents]


@app.get("/api/incidents/{incident_id}")
async def get_incident(incident_id: int, db: AsyncSession = Depends(get_db)):
    """Get specific incident"""
    result = await db.execute(select(Incident).filter(Incident.incident_id == incident_id))
    incident = result.scalars().first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident.to_dict()


@app.patch("/api/incidents/{incident_id}/status")
async def update_incident_status(incident_id: int, status: str, db: AsyncSession = Depends(get_db)):
    """Update incident status (active, resolved, manual_override)"""
    result = await db.execute(select(Incident).filter(Incident.incident_id == incident_id))
    incident = result.scalars().first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    incident.status = status
    await db.commit()
    return {"status": "Incident updated", "incident_id": incident_id, "new_status": status}


# ============= ZONE ENDPOINTS =============
@app.get("/api/zones")
async def get_zones(db: AsyncSession = Depends(get_db)):
    """Get all zones"""
    result = await db.execute(select(Zone))
    zones = result.scalars().all()
    return [z.to_dict() for z in zones]


@app.get("/api/zones/{zone_id}")
async def get_zone(zone_id: int, db: AsyncSession = Depends(get_db)):
    """Get specific zone"""
    result = await db.execute(select(Zone).filter(Zone.zone_id == zone_id))
    zone = result.scalars().first()
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    return zone.to_dict()


# ============= PROCEDURE ENDPOINTS =============
@app.get("/api/procedures/{zone_id}")
async def get_procedure(zone_id: int, db: AsyncSession = Depends(get_db)):
    """Get safety procedure for a zone"""
    result = await db.execute(select(SafetyProcedure).filter(SafetyProcedure.zone_id == zone_id))
    procedure = result.scalars().first()
    if not procedure:
        raise HTTPException(status_code=404, detail="Procedure not found")
    return procedure.to_dict()


# ============= DASHBOARD ENDPOINTS =============
@app.get("/api/dashboard/summary")
async def get_dashboard_summary(db: AsyncSession = Depends(get_db)):
    """Get dashboard summary stats"""
    total_detections = await db.scalar(select(func.count(Detection.detection_id)))
    active_incidents = await db.scalar(select(func.count(Incident.incident_id)).filter(Incident.status == "active"))
    zones = await db.scalar(select(func.count(Zone.zone_id)))
    
    last_det_res = await db.execute(select(Detection).order_by(Detection.timestamp.desc()))
    last_det = last_det_res.scalars().first()
    
    return {
        "total_detections": total_detections,
        "active_incidents": active_incidents,
        "total_zones": zones,
        "last_detection": last_det.timestamp.isoformat() if last_det else None
    }


@app.get("/api/dashboard/incidents-by-zone")
async def get_incidents_by_zone(db: AsyncSession = Depends(get_db)):
    """Get incident counts by zone"""
    result_zones = await db.execute(select(Zone))
    zones = result_zones.scalars().all()
    result = []
    
    for zone in zones:
        count = await db.scalar(select(func.count(Incident.incident_id)).filter(Incident.zone_id == zone.zone_id))
        result.append({
            "zone_id": zone.zone_id,
            "zone_name": zone.name,
            "incident_count": count
        })
    
    return result


# ============= VIDEO ANALYSIS (sampled frames) =============
@app.post("/api/analyze/video")
async def analyze_uploaded_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sample_interval_sec: float = 0.35,
    conf: float = 0.25,
    zone_id: int = 1,
    lat: float = None,
    lon: float = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Sample frames from an uploaded video. Uses ./best.pt (or FIRE_DETECT_MODEL) plus
    weights from fire_seg_final_results.zip (YOLO segmentation) for masks and boxes.
    Bounding boxes are normalized (0–1); x,y are center. masks are lists of polygons in normalized coords.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Use mp4, webm, mov, avi, mkv, or m4v.",
        )

    sample_interval_sec = max(0.15, min(float(sample_interval_sec), 2.0))
    conf = max(0.05, min(float(conf), 0.95))

    if remote_inference_enabled():
        response = await forward_upload_to_remote(
            endpoint="/analyze/video",
            file=file,
            params={"sample_interval_sec": sample_interval_sec, "conf": conf},
        )
        return response.json()

    try:
        ensure_segmentation_weights()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

        if width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="Invalid video dimensions")

        duration_sec = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        max_analyze_sec = 180.0
        max_samples = 240
        step = max(1, int(fps * sample_interval_sec))

        frames_out = []
        idx = 0
        positive_frames = 0
        inference_times_ms = []

        while idx < frame_count and len(frames_out) < max_samples:
            if duration_sec > 0 and (idx / fps) > max_analyze_sec:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            t = idx / fps if fps > 0 else 0.0
            try:
                infer_start = time.perf_counter()
                y = infer_frame_yolo(frame, conf=conf)
                inference_ms = (time.perf_counter() - infer_start) * 1000
                inference_times_ms.append(inference_ms)
            except Exception as e:
                cap.release()
                raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

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
            if y["fire"]:
                explainability.append("Segmentation mask or fire-class detection activated")
            if y.get("smoke"):
                explainability.append("Smoke-class detector found a candidate region")
            if y.get("fire_segment_area_pixels", 0.0) > 0:
                explainability.append("Fire area estimated from mask pixels")
            if not explainability:
                explainability.append("No fire/smoke evidence crossed the confidence threshold")

            if y["fire"] or y.get("smoke"):
                positive_frames += 1
                
                if positive_frames == ALERT_CONFIRMATION_FRAMES:
                    live_location = resolve_device_location(lat=lat, lon=lon)
                    live_address = live_location["address"]
                    zone_res = await db.execute(select(Zone).filter(Zone.zone_id == zone_id))
                    zone = zone_res.scalars().first()
                    final_address = live_address if "Unknown" not in live_address else (zone.location_address if zone else "Unknown")
                    zone_name = zone.name if zone else f"Zone {zone_id}"
                    
                    agent = get_fire_agent()
                    can_dispatch, dispatch_reason = should_dispatch_alert(zone_id)
                    if can_dispatch:
                        dispatch_confirmed_alert(
                            background_tasks=background_tasks,
                            agent=agent,
                            zone_name=zone_name,
                            address=final_address,
                            lat=live_location.get("lat"),
                            lon=live_location.get("lon"),
                        )
                    else:
                        print(f"Alert not dispatched for zone {zone_id}: {dispatch_reason}")
                    
                    agent_data = {
                        "zone_id": zone_id,
                        "address": final_address,
                        "lat": live_location.get("lat"),
                        "lon": live_location.get("lon"),
                        "coordinates": y.get("bbox") or {"x": 0.5, "y": 0.5},
                        "segment_area_pixels": y.get("fire_segment_area_pixels", 0.0),
                        "is_confirmed_alert": True,
                        "skip_email": True,
                    }
                    background_tasks.add_task(agent.reason, agent_data)

                bb = y["bbox"]
                frame_payload = {
                    "t": round(t, 3),
                    "fire": bool(y["fire"]),
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
                    "confidence": round(max_confidence, 4),
                    "inference_ms": round(inference_ms, 2),
                    "segmentation_instances": len(y.get("fire_masks") or []),
                    "explainability": explainability,
                }
                frames_out.append(frame_payload)
            else:
                frames_out.append(
                    {
                        "t": round(t, 3),
                        "fire": False,
                        "smoke": bool(y.get("smoke")),
                        "bbox": None,
                        "fire_masks": [],
                        "fire_segment_area_pixels": 0.0,
                        "smoke_boxes": y.get("smoke_boxes") or [],
                        "fire_boxes": y.get("fire_boxes") or [],
                        "confidence": round(max_confidence, 4),
                        "inference_ms": round(inference_ms, 2),
                        "segmentation_instances": 0,
                        "explainability": explainability,
                    }
                )

            idx += step

        cap.release()
        
        # Cleanup tmp file
        try:
            os.remove(tmp_path)
        except:
            pass

        det_path_str = os.environ.get("FIRE_DETECT_MODEL") or str(DEFAULT_DETECT_PT)
        det_loaded = get_yolo_det() is not None
        avg_inference_ms = sum(inference_times_ms) / len(inference_times_ms) if inference_times_ms else 0.0
        analyzed_fps = 1000 / avg_inference_ms if avg_inference_ms > 0 else 0.0
        fire_areas = [float(f.get("fire_segment_area_pixels", 0.0)) for f in frames_out]
        peak_area = max(fire_areas) if fire_areas else 0.0
        mean_area = sum(fire_areas) / len(fire_areas) if fire_areas else 0.0
        confidence_values = [float(f.get("confidence", 0.0)) for f in frames_out if f.get("confidence", 0.0) > 0]
        mean_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        note_parts = [
            "YOLO segmentation from fire_seg_final_results.zip (extracted to models/fire_seg/weights/best.pt).",
        ]
        if det_loaded:
            note_parts.append(f"Detection head also loads: {det_path_str}.")
        else:
            note_parts.append(
                "Optional detector not loaded — add best.pt next to fire_backend.py or set FIRE_DETECT_MODEL."
            )

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
                "architecture": "YOLO detector + YOLO segmentation",
                "detector_loaded": det_loaded,
                "segmentation_loaded": True,
                "confidence_threshold": conf,
                "sample_interval_sec": sample_interval_sec,
                "avg_inference_ms": round(avg_inference_ms, 2),
                "estimated_model_fps": round(analyzed_fps, 2),
                "mean_detection_confidence": round(mean_confidence, 4),
            },
            "explainability_summary": {
                "peak_fire_area_pixels": round(peak_area, 1),
                "mean_fire_area_pixels": round(mean_area, 1),
                "mask_positive_frames": sum(1 for f in frames_out if f.get("segmentation_instances", 0) > 0),
                "confidence_positive_frames": len(confidence_values),
            },
            "note": " ".join(note_parts),
            "frames": frames_out,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/api/analyze/image")
async def analyze_uploaded_image(
    file: UploadFile = File(...),
    conf: float = 0.25,
):
    """Run YOLO on a single image; returns JSON + URL to an annotated PNG in /images/."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Use jpg, png, webp, or bmp.",
        )

    conf = max(0.05, min(float(conf), 0.95))

    if remote_inference_enabled():
        response = await forward_upload_to_remote(
            endpoint="/analyze/image",
            file=file,
            params={"conf": conf},
        )
        return response.json()

    try:
        ensure_segmentation_weights()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    im_h, im_w = frame.shape[:2]
    try:
        y = infer_frame_yolo(frame, conf=conf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    annotated = annotate_frame_bgr(frame, y, conf)
    out_name = f"annotated_img_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
    out_path = IMAGES_DIR / out_name
    cv2.imwrite(str(out_path), annotated)

    bb = y.get("bbox")
    return {
        "filename": file.filename,
        "image_width": im_w,
        "image_height": im_h,
        "conf": conf,
        "fire": y["fire"],
        "smoke": y["smoke"],
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
        "annotated_image_url": f"/images/{out_name}",
    }


@app.post("/api/analyze/video/export")
async def export_annotated_video(
    file: UploadFile = File(...),
    conf: float = 0.25,
    infer_stride: int = 1,
):
    """
    Render every frame (or every infer_stride-th frame for speed) with the same overlays as the UI.
    Returns an MP4 file download.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Use mp4, webm, mov, avi, mkv, or m4v.",
        )

    conf = max(0.05, min(float(conf), 0.95))
    infer_stride = max(1, min(int(infer_stride), 30))

    if remote_inference_enabled():
        response = await forward_upload_to_remote(
            endpoint="/analyze/video/export",
            file=file,
            params={"conf": conf, "infer_stride": infer_stride},
        )
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(file.filename).stem)[:80]
        return StreamingResponse(
            io.BytesIO(response.content),
            media_type=response.headers.get("content-type", "video/mp4"),
            headers={"Content-Disposition": f'attachment; filename="FireWatch_annotated_{safe}.mp4"'},
        )

    try:
        ensure_segmentation_weights()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    out_name = f"annotated_vid_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.mp4"
    out_path = IMAGES_DIR / out_name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        if width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="Invalid video dimensions")

        max_export_sec = 600.0
        max_out_frames = 8000
        duration_sec = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise HTTPException(status_code=500, detail="Could not create output video writer")

        last_y = {
            "fire": False,
            "smoke": False,
            "bbox": None,
            "fire_masks": [],
            "fire_segment_area_pixels": 0.0,
            "fire_boxes": [],
            "smoke_boxes": [],
        }

        frame_idx = 0
        written = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t = frame_idx / fps if fps > 0 else 0.0
            if duration_sec > max_export_sec and t > max_export_sec:
                break
            if frame_idx % infer_stride == 0:
                try:
                    last_y = infer_frame_yolo(frame, conf=conf)
                except Exception as e:
                    cap.release()
                    writer.release()
                    raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")
            annotated = annotate_frame_bgr(frame, last_y, conf)
            writer.write(annotated)
            written += 1
            frame_idx += 1
            if written >= max_out_frames:
                break

        cap.release()
        writer.release()

        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(file.filename).stem)[:80]
        download_name = f"FireWatch_annotated_{safe}.mp4"

        return FileResponse(
            path=str(out_path),
            media_type="video/mp4",
            filename=download_name,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/api/reports/instructor.pdf")
async def export_instructor_report_pdf(payload: InstructorReportRequest):
    """Generate a styled PDF report from the frontend Instructor Mode analysis state."""
    pdf_bytes = build_instructor_pdf(payload)
    filename = "FireWatch_Instructor_Report.pdf"
    raw_name = payload.analysis.get("filename") if isinstance(payload.analysis, dict) else None
    if raw_name:
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(str(raw_name)).stem)[:80]
        filename = f"FireWatch_Instructor_Report_{safe}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/rag/query")
def rag_query(payload: RAGQueryRequest):
    """Query the vector RAG system and return answer + sources."""
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    top_k = max(1, min(int(payload.top_k), 8))

    rag = get_rag_system()
    if not getattr(rag, "embeddings_model", None) or not getattr(rag, "vector_index", None):
        raise HTTPException(
            status_code=503,
            detail="RAG embeddings not initialized. Ensure sentence-transformers and faiss-cpu are installed.",
        )

    retrieved = rag.retrieve(query, top_k=top_k)
    if not retrieved:
        return {
            "query": query,
            "answer": "No relevant documents found in the fire-safety knowledge base.",
            "sources": [],
        }

    top_score = float(retrieved[0].get("similarity_score", 0.0))
    safety_keywords = {
        "fire", "smoke", "suppression", "sprinkler", "co2", "foam", "evacuation",
        "hazard", "alarm", "zone", "extinguisher", "incident", "safety", "nfpa",
        "response", "server room", "warehouse", "lobby",
    }
    q_lower = query.lower()
    has_safety_intent = any(k in q_lower for k in safety_keywords)
    if (not has_safety_intent) or top_score < 0.42:
        return {
            "query": query,
            "answer": (
                "I only answer fire-safety questions grounded in this project's RAG documents. "
                "Please ask about fire/smoke detection, suppression systems, evacuation, zones, or incident response."
            ),
            "sources": [],
        }

    # Generate synthesized answer using Groq LLM
    print(f"DEBUG: Generating RAG answer for query: {query}")
    answer = generate_rag_answer(query, retrieved)



    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "doc_id": d.get("doc_id"),
                "title": d.get("title"),
                "chunk_id": d.get("chunk_id"),
                "similarity_score": round(float(d.get("similarity_score", 0.0)), 4),
                "excerpt": str(d.get("content", ""))[:320],
            }
            for d in retrieved
        ],
    }


# ============= HEALTH CHECK =============
@app.get("/api/health")
def health_check():
    return {
        "status": "Backend is running",
        "db": "Connected",
        "alerts": {
            "mode": ALERT_MODE,
            "auto_enabled": AUTO_ALERTS_ENABLED,
            "confirmation_frames": ALERT_CONFIRMATION_FRAMES,
            "cooldown_seconds": ALERT_COOLDOWN_SECONDS,
        },
        "inference": {
            "mode": INFERENCE_MODE,
            "remote_enabled": remote_inference_enabled(),
            "service_url": INFERENCE_SERVICE_URL,
            "timeout_seconds": REMOTE_INFERENCE_TIMEOUT_SECONDS,
        },
        "cors_origins": CORS_ORIGINS,
    }


if __name__ == "__main__":
    import uvicorn
    print("🔥 Fire Management System Backend")
    print("📊 Starting on http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

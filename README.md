<div align="center">

# 🚨 FireWatch AI
**Advanced Fire and Smoke Detection & Monitoring Dashboard**

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://react.dev/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-00A6ED?style=for-the-badge)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](./LICENSE)

![System Demo](./demo.gif)

*FireWatch AI is a comprehensive, real-time fire and smoke monitoring system designed to provide rapid detection, tactical insights, and automated safety workflows.*
</div>

---

## ✨ Key Features

### 🧠 AI & Computer Vision
* **YOLO-based Video Analysis:** Real-time video frame sampling and fire/smoke detection using optional YOLOv8 model weights (with an intelligent heuristic fallback).
* **Groq-powered RAG Safety Assistant:** Optional Retrieval-Augmented Generation (RAG) for synthesizing immediate safety guidance and procedures on the fly.

### 💻 Tactical Dashboard (Frontend)
* **React + Vite:** Lightning-fast, modern web interface.
* **Data Visualization:** Animated overlays and timeline metrics for an intuitive view of detections and incident timelines.
* **Interactive Controls:** Response controls and a floating safety assistant to manage emergencies seamlessly.

### ⚙️ Robust Backend & Alerting
* **FastAPI Backend:** High-performance API driving the core system logic.
* **State Management:** SQLite database to efficiently manage monitoring zones, active detections, incident logs, and safety procedures.
* **Demo-Safe Alert Workflows:** Built-in Gmail alert modes with confirmation frames and cooldown guards to prevent spamming during testing or demos.

---

## 🏗️ Repository Architecture

```text
FireWatch AI
├── .github/workflows/ci.yml      # GitHub Actions checks
├── docs/
│   ├── ALERTING.md               # Demo/email alert modes and safety guardrails
│   └── DEPLOYMENT.md             # Vercel + Render + Hugging Face deployment guide
├── frontend/                     # React + Vite dashboard
│   ├── src/                      # Dashboard source
│   ├── public/                   # Static frontend assets
│   ├── package.json              # Frontend scripts and dependencies
│   └── package-lock.json         # Reproducible frontend installs
├── tests/                        # Manual integration/demo scripts
│   ├── manual_emergency.py       # Manual agent workflow check
│   ├── test_agent.py             # Manual agent/backend integration demo
│   └── test_backend.py           # Manual backend API demo

## Features

- Video upload and frame sampling for fire/smoke analysis.
- React + Vite dashboard with animated overlays, timeline metrics, response controls, and a floating safety assistant.
- FastAPI backend with SQLite-backed zones, detections, incidents, and safety procedures.
- Optional YOLO model weights for real detection, with a heuristic fallback when no local model is available.
- Optional Groq-powered RAG answer synthesis.
- Demo-safe Gmail alert modes with confirmation frames and cooldown guards.

## Repository Layout

```text
FireWatch AI
├── .github/workflows/ci.yml      # GitHub Actions checks
├── docs/
│   ├── ALERTING.md               # Demo/email alert modes and safety guardrails
│   └── DEPLOYMENT.md             # Vercel + Render + Hugging Face deployment guide
├── frontend/                     # React + Vite dashboard
│   ├── src/                      # Dashboard source
│   ├── public/                   # Static frontend assets
│   ├── package.json              # Frontend scripts and dependencies
│   └── package-lock.json         # Reproducible frontend installs
├── tests/                        # Manual integration/demo scripts
│   ├── manual_emergency.py       # Manual agent workflow check
│   ├── test_agent.py             # Manual agent/backend integration demo
│   └── test_backend.py           # Manual backend API demo
├── .env.example                  # Safe environment template
├── .gitattributes                # Text/binary and model-weight handling hints
├── .gitignore                    # Local secrets, models, caches, and runtime outputs
├── CONTRIBUTING.md               # Contribution and local check notes
├── SECURITY.md                   # Secret handling and safety notes
├── fire_agent.py                 # Agent tools, Gmail helpers, emergency response logic
├── fire_backend.py               # FastAPI API, detection pipeline, alert orchestration
├── real_rag_system.py            # FAISS/sentence-transformers RAG implementation
├── requirements.txt              # Python dependencies
├── package.json                  # Root convenience scripts
└── package-lock.json             # Root npm lockfile for wrapper scripts
```

Local runtime files such as `.env`, `credentials.json`, `token.json`, `fire_system.db`, `best.pt`, `models/*.pt`, and generated detection images are intentionally ignored by git.

The root `package.json` and `package-lock.json` should stay committed. They do not duplicate the frontend app; they provide repo-level commands like `npm run check`, while `frontend/package.json` owns the actual React dependencies.

Do not create `docs/credentials/` for secrets. Keep private local files in ignored folders such as `secrets/` or `local_assets/`; `docs/` is for public documentation.

## Prerequisites

- Python 3.10+
- Node.js 18+
- Optional: a YOLO model weight file such as `best.pt`
- Optional: a Groq API key for generated RAG responses
- Optional: Gmail API OAuth credentials for live email alerting

## Setup

1. Create and activate a Python environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Install frontend dependencies.

```bash
npm install --prefix frontend
```

3. Create local environment settings.

```bash
cp .env.example .env
```

For local development, update `.env` only as needed:

```env
FIRE_DETECT_MODEL=./best.pt
ALERT_MODE=demo
AUTO_ALERTS_ENABLED=false
ALERT_CONFIRMATION_FRAMES=3
ALERT_COOLDOWN_SECONDS=300
```

Keep `.env` private. Do not commit it.

Optional cleaner local layout:

```text
secrets/token.json
secrets/credentials.json
local_assets/best.pt
```

Then set:

```env
GOOGLE_TOKEN_FILE=secrets/token.json
FIRE_DETECT_MODEL=./local_assets/best.pt
```

4. Add a local model file if you have one.

```env
FIRE_DETECT_MODEL=./best.pt
```

Large model files should stay outside git. Use Git LFS, a release asset, or a model registry if you need to share them.

For deployment, the backend can download weights from Hugging Face Hub instead:

```env
HF_MODEL_REPO=your-username/firewatch-yolo
HF_MODEL_FILENAME=best.pt
HF_SEG_MODEL_REPO=your-username/firewatch-seg
HF_SEG_MODEL_FILENAME=best.pt
HF_TOKEN=your_read_token_for_private_repos
```

## Run Locally

Start the backend:

```bash
uvicorn fire_backend:app --reload --host 0.0.0.0 --port 8000
```

Start the frontend in another terminal:

```bash
npm run dev --prefix frontend
```

Then open the Vite URL, usually `http://localhost:5173`.

The frontend production build is generated in `frontend/dist/`. Frontend environment variables must use the `VITE_` prefix, and those values are public in the browser bundle.

Useful backend URLs:

- API health: `http://localhost:8000/api/health`
- API docs: `http://localhost:8000/docs`

## Deployment

Recommended split:

- Frontend: Vercel static Vite deployment.
- Backend: Render Python web service.

Vercel frontend settings:

- Root directory: `frontend`
- Framework preset: `Vite`
- Build command: `npm run build`
- Output directory: `dist`
- Environment variable: `VITE_BACKEND_URL=https://your-backend.onrender.com`

Render backend settings:

- Language: `Python 3`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn fire_backend:app --host 0.0.0.0 --port $PORT`
- Python version: set `PYTHON_VERSION=3.11.9` in Render

Backend environment variables:

```env
ALERT_MODE=demo
AUTO_ALERTS_ENABLED=false
CORS_ORIGINS=https://your-frontend.vercel.app
GROQ_API_KEY=your_key_here
HF_MODEL_REPO=your-username/firewatch-yolo
HF_MODEL_FILENAME=best.pt
```

For live Gmail alerts, add the Gmail variables from `.env.example` and keep OAuth files out of git.

Production notes:

- SQLite and generated detection images are local runtime files. Use persistent storage or a hosted database for anything beyond a demo.
- Model weights are intentionally not committed. Use Hugging Face Hub for `best.pt`, or add persistent storage if you prefer to keep weights on Render.
- The frontend does not need any change for model hosting. Only backend environment variables change.
- Vite exposes `VITE_*` values in browser code, so do not put secrets in frontend environment variables.

## Alerts

The default configuration does not contact real people. For live alerts, configure `.env` intentionally and read [docs/ALERTING.md](./docs/ALERTING.md).

Do not configure this project to contact real emergency services.

## Checks

```bash
npm run check
```

The scripts in `tests/` are manual integration/demo scripts. They expect the backend to be running and may create local database/image state, so they are not run automatically in CI.

## GitHub Hygiene

- Commit source, docs, lockfiles, and small static assets.
- Keep both root and frontend `package-lock.json` files committed for reproducible installs.
- Do not commit `.env`, OAuth tokens, credentials, local databases, generated captures, or model weights.
- Keep local-only files such as `.DS_Store`, `.venv/`, `__pycache__/`, `best.pt`, `fire_system.db`, `detection_images/*`, `model_cache/`, `secrets/`, `local_assets/`, and `local_backup/` out of git.
- Rotate any credential that was ever committed or shared publicly.
- Keep live alerting disabled unless you are deliberately testing with verified recipients.

  ## 📚 Documentation <a name="documentation"></a>

### 🤝 Contributing
Contributions are always welcome! Whether it is adding new themes. If you have a feature idea, feel free to open an Issue or submit a Pull Request.

<a name="contacts"></a>
## 🤝 Contacts

| Source | Link |
| :--- | :--- |
| **GitHub Profile** | [shubh050307-hash](https://github.com/shubh050307-hash) |
| **Project Repository** | [Firewatch -AI](https://github.com/shubh050307-hash/Fire-SmokeDetect/) |

### 🙌 Credits

<div align="center"><img src="https://placehold.co/100x3/9A1838/9A1838.png" width="10%" height="3px"><img src="https://placehold.co/100x3/B43E2C/B43E2C.png" width="10%" height="3px"><img src="https://placehold.co/100x3/C56821/C56821.png" width="10%" height="3px"><img src="https://placehold.co/100x3/CE921C/CE921C.png" width="10%" height="3px"><img src="https://placehold.co/100x3/D0B92B/D0B92B.png" width="10%" height="3px"><img src="https://placehold.co/100x3/81A543/81A543.png" width="10%" height="3px"><img src="https://placehold.co/100x3/258C5A/258C5A.png" width="10%" height="3px"><img src="https://placehold.co/100x3/166B8E/166B8E.png" width="10%" height="3px"><img src="https://placehold.co/100x3/154BB2/154BB2.png" width="10%" height="3px"><img src="https://placehold.co/100x3/0E2A80/0E2A80.png" width="10%" height="3px"></div>
<p align="center">
  <b>Built by SG's for the 🌏 </b>
  <br>
  Don't forget to ⭐ star the repo if you like this project!
</p>

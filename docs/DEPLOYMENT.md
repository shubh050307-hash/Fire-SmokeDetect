# Deployment Guide

This project should be deployed as two services:

- Frontend on Vercel.
- Backend on Render.
- YOLO weights on Hugging Face Hub.

## 1. Upload `best.pt` To Hugging Face

Create a Hugging Face account, then create a new model repository such as:

```text
your-username/firewatch-yolo
```

For a public demo model, a public repo is easiest. For private work, create a private repo and generate a Hugging Face read token.

Install the Hugging Face CLI locally:

```bash
python3 -m pip install -U huggingface_hub
hf auth login
```

Upload the model:

```bash
hf upload your-username/firewatch-yolo ./best.pt best.pt --repo-type model
```

The backend expects these Render environment variables:

```env
HF_MODEL_REPO=your-username/firewatch-yolo
HF_MODEL_FILENAME=best.pt
HF_SEG_MODEL_REPO=your-username/firewatch-seg
HF_SEG_MODEL_FILENAME=best.pt
HF_TOKEN=your_read_token_only_if_private
```

Do not commit `best.pt` to GitHub.

The committed backend supports these variables directly. If `FIRE_DETECT_MODEL` is missing on Render and `HF_MODEL_REPO` is set, the backend downloads `HF_MODEL_FILENAME` from Hugging Face Hub at startup or first model load. If the local segmentation weights or zip file are missing and `HF_SEG_MODEL_REPO` is set, the backend downloads `HF_SEG_MODEL_FILENAME` for the segmentation model too.

## 2. Deploy Backend On Render

Create a Render Web Service from the GitHub repo.

Use:

```text
Language: Python 3
Build Command: pip install -r requirements.txt
Start Command: uvicorn fire_backend:app --host 0.0.0.0 --port $PORT
```

Set:

```env
ALERT_MODE=demo
AUTO_ALERTS_ENABLED=false
CORS_ORIGINS=https://your-frontend.vercel.app
HF_MODEL_REPO=your-username/firewatch-yolo
HF_MODEL_FILENAME=best.pt
HF_SEG_MODEL_REPO=your-username/firewatch-seg
HF_SEG_MODEL_FILENAME=best.pt
GROQ_API_KEY=optional_for_rag_answers
```

If your Hugging Face repo is private, also set:

```env
HF_TOKEN=your_huggingface_read_token
```

After deploy, confirm:

```text
https://your-backend.onrender.com/api/health
```

## 3. Deploy Frontend On Vercel

Import the same GitHub repo into Vercel.

Use:

```text
Framework Preset: Vite
Root Directory: frontend
Build Command: npm run build
Output Directory: dist
```

Set:

```env
VITE_BACKEND_URL=https://your-backend.onrender.com
```

Redeploy the frontend after setting or changing `VITE_BACKEND_URL`.

## 4. Final CORS Update

Once Vercel gives you the final frontend URL, update Render:

```env
CORS_ORIGINS=https://your-frontend.vercel.app
```

Then redeploy the backend.

## Notes

- Render free instances may sleep, so the first request can be slow.
- Hugging Face model download can add cold-start time.
- SQLite and generated detection images are local runtime files. Use persistent storage or a hosted database for production.
- `VITE_*` frontend variables are public in the browser bundle. Never put secrets there.
- Keep `package.json` and `package-lock.json` committed at both the root and `frontend/` levels. The root files provide convenience scripts; the frontend files define the Vite app.
- Do not create `docs/credentials/`. For local-only private files, use ignored folders like `secrets/` and `local_assets/`.

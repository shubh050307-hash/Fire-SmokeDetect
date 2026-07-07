---
title: FireWatch YOLO Inference
emoji: 🔥
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
license: mit
short_description: YOLO inference API for FireWatch AI fire and smoke detection
---

# FireWatch YOLO Inference

Docker/FastAPI inference service for the FireWatch AI dashboard.

## Endpoints

- `GET /health`
- `POST /analyze/video`
- `POST /analyze/image`
- `POST /analyze/video/export`

## Required Space variables

Set these in **Settings -> Variables and secrets**:

```env
HF_MODEL_REPO=omerfarooq223/FireWatch-AI
HF_MODEL_FILENAME=detector-best.pt
HF_SEG_MODEL_REPO=omerfarooq223/FireWatch-AI
HF_SEG_MODEL_FILENAME=seg-best.pt
```

If the model repo is private, add `HF_TOKEN` as a secret.

## Render backend variables

After this Space builds, set these on Render:

```env
INFERENCE_MODE=remote
INFERENCE_SERVICE_URL=https://omerfarooq223-firewatch-yolo-inference.hf.space
```

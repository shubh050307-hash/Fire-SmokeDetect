# Contributing

Thanks for improving FireWatch AI.

## Local Setup

1. Create a Python environment and install `requirements.txt`.
2. Install the frontend with `npm install --prefix frontend`.
3. Copy `.env.example` to `.env` and keep alerting in demo mode unless you are deliberately testing live integrations.
4. For local model testing, keep `best.pt` in the project root or set `FIRE_DETECT_MODEL`.
5. For deployed model testing, set `HF_MODEL_REPO` and `HF_MODEL_FILENAME` instead of committing weights.
6. Keep model weights, databases, generated images, OAuth tokens, and credentials out of git.

## Checks Before Opening a Pull Request

```bash
npm run check
```

The `tests/` scripts are manual integration demos. Run them only after starting the backend and using local demo data.

Keep both root and frontend `package-lock.json` files committed. They make local checks and hosted builds reproducible.

## Pull Request Notes

- Describe user-facing behavior changes.
- Mention new environment variables or setup steps.
- Include screenshots or short clips for dashboard UI changes.
- Document any alerting behavior changes in `docs/ALERTING.md`.

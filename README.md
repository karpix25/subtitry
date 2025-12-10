# CPU Subtitle Cleaner

FastAPI service for removing subtitle overlays from trading videos using PaddleOCR for text detection and OpenCV inpainting. The service runs entirely on CPU and exposes HTTP endpoints for full-video cleaning plus preview utilities.

## Features
- PaddleOCR-based text detection with heuristic subtitle classification tuned for market UIs
- CPU-only Navier-Stokes inpainting via OpenCV
- Streaming frame pipeline without intermediate disk writes
- `/clean`, `/preview`, `/health` HTTP endpoints
- Optional webhook callbacks plus `/tasks/{id}` status polling for long jobs
- Docker image ready for EasyPanel deployments (port 8000)

## Project Layout
```
├── app/
│   ├── main.py
│   ├── video_processor.py
│   ├── text_detector.py
│   ├── classifier.py
│   ├── mask_builder.py
│   ├── inpainter.py
│   └── ffmpeg_utils.py
├── models/
│   └── subtitle_rules.json
├── requirements.txt
└── Dockerfile
```

## Local Development
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API
### `POST /clean`
Multipart upload (`file`) with optional form fields: `max_resolution`, `inpaint_radius`, `subtitle_intensity_threshold`, `callback_url`.

- Without `callback_url` the endpoint streams the processing and returns the cleaned video stats once finished.
- With `callback_url` the upload is accepted asynchronously: the response includes `task_id`, processing happens in the background, and the service POSTs the final payload (success or failure) to the provided URL. You can also poll `/tasks/{task_id}`.

### `POST /preview`
Returns single-frame before/after PNGs (base64) plus mask for debugging heuristics.

### `GET /health`
Simple readiness probe.

### `GET /tasks/{task_id}`
Returns the stored status/result for asynchronous jobs created via `callback_url`.

## Docker
Build and run:
```bash
docker build -t subtitle-cleaner .
docker run -p 8000:8000 subtitle-cleaner
```

EasyPanel automatically exposes port 8000.

## Testing Checklist
1. Trading chart with dynamic subtitles: subtitles removed, tooltips/UI intact.
2. Top-aligned subtitles with motion: cleaned without dents.
3. Subtitles over grid lines: Navier-Stokes restores lines.
4. Video without subtitles: frames unchanged (mask stays zero).

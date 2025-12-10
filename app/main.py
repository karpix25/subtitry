from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from .task_manager import TaskManager
from .video_processor import VideoProcessingOptions, VideoProcessor

app = FastAPI(title="Subtitle Cleaner", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
processor = VideoProcessor()
task_manager = TaskManager()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/clean")
async def clean_video(
    file: UploadFile = File(...),
    max_resolution: int = Form(1080),
    inpaint_radius: int = Form(4),
    subtitle_intensity_threshold: Optional[float] = Form(None),
    callback_url: Optional[str] = Form(None),
) -> JSONResponse:
    if file.content_type is None or not file.content_type.startswith("video"):
        raise HTTPException(status_code=400, detail="Expected video file upload")

    start_time = time.perf_counter()
    options = VideoProcessingOptions(
        max_resolution=max_resolution,
        inpaint_radius=inpaint_radius,
        subtitle_intensity_threshold=subtitle_intensity_threshold,
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "video").suffix) as tmp_in:
        shutil.copyfileobj(file.file, tmp_in)
        input_path = Path(tmp_in.name)

    if callback_url:
        task_id = task_manager.create_task(callback_url=callback_url)
        output_path = OUTPUT_DIR / f"cleaned_{task_id}.mp4"
        loop = asyncio.get_event_loop()
        try:
            future = loop.run_in_executor(
                None,
                lambda: _process_async_task(
                    task_id=task_id,
                    input_path=input_path,
                    output_path=output_path,
                    options=options,
                    callback_url=callback_url,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unable to schedule background task %s", task_id)
            task_manager.mark_failed(task_id, str(exc))
            input_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Failed to schedule processing") from exc
        future.add_done_callback(_log_future_exception)  # fire-and-forget
        payload = {"status": "accepted", "task_id": task_id}
        return JSONResponse(payload)

    try:
        output_path = OUTPUT_DIR / f"cleaned_{int(time.time() * 1000)}.mp4"
        stats = await asyncio.get_event_loop().run_in_executor(
            None, lambda: processor.process_video(input_path, output_path, options)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Video processing failed")
        raise HTTPException(status_code=500, detail="Processing failed") from exc
    finally:
        input_path.unlink(missing_ok=True)

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    payload = {
        "status": "ok",
        "video_url": str(output_path.resolve()),
        "time_ms": elapsed_ms,
        "stats": stats,
    }
    return JSONResponse(payload)


@app.post("/preview")
async def preview_frame(
    file: UploadFile = File(...),
    frame_number: int = Form(0),
    max_resolution: int = Form(720),
    inpaint_radius: int = Form(4),
):
    """Optional helper endpoint that returns a single before/after frame pair."""
    if frame_number < 0:
        raise HTTPException(status_code=400, detail="frame_number must be >= 0")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "video").suffix) as tmp_in:
        shutil.copyfileobj(file.file, tmp_in)
        input_path = Path(tmp_in.name)

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: processor.preview_frame(
                input_path,
                frame_number,
                VideoProcessingOptions(max_resolution=max_resolution, inpaint_radius=inpaint_radius),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Preview generation failed")
        raise HTTPException(status_code=500, detail="Preview failed") from exc
    finally:
        input_path.unlink(missing_ok=True)

    return JSONResponse(result)


@app.get("/tasks/{task_id}")
def get_task(task_id: str) -> JSONResponse:
    record = task_manager.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return JSONResponse(record)


def _process_async_task(
    *,
    task_id: str,
    input_path: Path,
    output_path: Path,
    options: VideoProcessingOptions,
    callback_url: Optional[str],
) -> None:
    start_time = time.perf_counter()
    task_manager.mark_processing(task_id)
    try:
        stats = processor.process_video(input_path, output_path, options)
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        payload = {
            "task_id": task_id,
            "status": "completed",
            "video_url": str(output_path.resolve()),
            "time_ms": elapsed_ms,
            "stats": stats,
        }
        task_manager.mark_completed(task_id, payload)
        if callback_url:
            _post_callback(callback_url, payload)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Background processing failed for task %s", task_id)
        error_message = str(exc)
        failure_payload = {
            "task_id": task_id,
            "status": "failed",
            "error": error_message,
        }
        task_manager.mark_failed(task_id, error_message)
        if callback_url:
            _post_callback(callback_url, failure_payload)
    finally:
        input_path.unlink(missing_ok=True)


def _post_callback(url: str, payload: dict) -> None:
    try:
        response = httpx.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.error("Callback to %s failed: %s", url, exc)


def _log_future_exception(future) -> None:
    try:
        future.result()
    except Exception as exc:  # noqa: BLE001
        logger.error("Background task raised: %s", exc)

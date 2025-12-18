from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from loguru import logger

from .task_manager import TaskManager
from .video_processor import VideoProcessingOptions, VideoProcessor

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background cleanup task
    logger.info("Starting Subtitle Cleaner v1.1 - Fixes Applied")
    task = asyncio.create_task(_cleanup_loop())
    yield
    # Cancel task on shutdown
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="Subtitle Cleaner", version="0.1.0", lifespan=lifespan)
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

# Mount output directory for static access
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/clean")
async def clean_video(
    file: Optional[UploadFile] = File(None),
    max_resolution: int = Form(1080),
    inpaint_radius: int = Form(4),
    subtitle_intensity_threshold: Optional[float] = Form(None),
    keyframe_interval: float = Form(0.5),
    bbox_padding: float = Form(0.1),
    language_hint: str = Form("auto"),
    video_url: Optional[str] = Form(None),
    callback_url: Optional[str] = Form(None),
    debug: bool = Form(True),
    dual_output: bool = Form(False),
    request: Request = None,
) -> JSONResponse:
    if file is None and video_url is None:
        raise HTTPException(status_code=400, detail="Either file upload or video_url is required")

    start_time = time.perf_counter()
    options = VideoProcessingOptions(
        max_resolution=max_resolution,
        inpaint_radius=inpaint_radius,
        subtitle_intensity_threshold=subtitle_intensity_threshold,
        keyframe_interval=keyframe_interval,
        bbox_padding=bbox_padding,
        language_hint=language_hint,
    )

    input_path = None
    if file:
        suffix = Path(file.filename or "video").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            shutil.copyfileobj(file.file, tmp_in)
            input_path = Path(tmp_in.name)
    elif video_url:
        suffix = ".mp4" # Basic assumption, could be improved
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            input_path = Path(tmp_in.name)
        
        # Download video
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(video_url, follow_redirects=True, timeout=60.0)
                resp.raise_for_status()
                with open(input_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                if input_path.exists():
                    input_path.unlink()
                raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")

    if not input_path:
            raise HTTPException(status_code=400, detail="No video input provided")

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
                    base_url=str(request.base_url) if request else "",
                    debug=debug,
                    dual_mode=dual_output,
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
            None, lambda: processor.process_video(input_path, output_path, options, debug=debug, dual_mode=dual_output)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Video processing failed")
        raise HTTPException(status_code=500, detail="Processing failed") from exc
    finally:
        input_path.unlink(missing_ok=True)

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    
    # Construct output URL for synchronous response
    clean_url = f"{request.base_url}output/{output_path.name}" if request else str(output_path.resolve())
    debug_url = None
    if stats.get("debug_output_path"):
        debug_path = Path(stats["debug_output_path"])
        debug_url = f"{request.base_url}output/{debug_path.name}" if request else str(debug_path.resolve())

    payload = {
        "status": "ok",
        "output_path": str(output_path.resolve()),
        "video_url": clean_url,
        "debug_video_url": debug_url,
        "processing_time_seconds": elapsed_ms / 1000,
        "time_ms": elapsed_ms,
        "total_frames": stats.get("frames", 0),
        "subtitle_frames": stats.get("subtitle_frames", 0),
        "keyframes_analyzed": stats.get("keyframes_analyzed", 0),
        "fps": stats.get("fps", 0),
        "duration": stats.get("duration", 0),
        "detected_region": stats.get("detected_region"),
    }
    return JSONResponse(payload)


@app.post("/preview")
async def preview_frame(
    file: UploadFile = File(...),
    frame_number: int = Form(0),
    max_resolution: int = Form(720),
    inpaint_radius: int = Form(4),
    bbox_padding: float = Form(0.1),
    language_hint: str = Form("auto"),
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
                VideoProcessingOptions(
                    max_resolution=max_resolution,
                    inpaint_radius=inpaint_radius,
                    bbox_padding=bbox_padding,
                    language_hint=language_hint,
                ),
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
    base_url: str = "",
    debug: bool = True,
    dual_mode: bool = False,
) -> None:
    start_time = time.perf_counter()
    logger.info(f"Task {task_id}: Processing started for {input_path.name}")
    task_manager.mark_processing(task_id)
    try:
        stats = processor.process_video(input_path, output_path, options, debug=debug, dual_mode=dual_mode)
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(f"Task {task_id}: Completed in {elapsed_ms}ms. Output: {output_path}")
        
        # Construct output URLs
        clean_url = f"{base_url}output/{output_path.name}" if base_url else str(output_path.resolve())
        debug_url = None
        
        if stats.get("debug_output_path"):
            debug_path = Path(stats["debug_output_path"])
            debug_url = f"{base_url}output/{debug_path.name}" if base_url else str(debug_path.resolve())

        payload = {
            "task_id": task_id,
            "status": "completed",
            "video_url": clean_url,
            "debug_video_url": debug_url,
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
        logger.info(f"Sending callback to {url} with status {payload.get('status')}")
        response = httpx.post(url, json=payload, timeout=30) # Increased timeout
        response.raise_for_status()
        logger.info(f"Callback successful: {response.status_code}")
    except Exception as exc:  # noqa: BLE001
        logger.error("Callback to %s failed: %s", url, exc)


def _log_future_exception(future) -> None:
    try:
        future.result()
    except Exception as exc:  # noqa: BLE001
        logger.error("Background task raised: %s", exc)


async def _cleanup_loop() -> None:
    """Periodically clean up old files in OUTPUT_DIR."""
    logger.info("Starting cleanup background task")
    while True:
        try:
            await asyncio.sleep(600)  # Check every 10 minutes
            now = time.time()
            cutoff = now - 3600  # 1 hour retention
            
            count = 0
            for file_path in OUTPUT_DIR.glob("*"):
                if file_path.is_file():
                    stat = file_path.stat()
                    # Check modified time
                    if stat.st_mtime < cutoff:
                        try:
                            file_path.unlink()
                            count += 1
                        except Exception as e:
                            logger.warning(f"Failed to delete {file_path}: {e}")
            
            if count > 0:
                logger.info(f"Cleaned up {count} old files")
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}")
            await asyncio.sleep(60)  # Wait a bit on error before retrying

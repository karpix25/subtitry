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
    dual_output: bool = Form(False), # New param
    request: Request = None,
) -> JSONResponse:
    # ... (omitted) ...
        # ...
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
                    dual_mode=dual_output, # Pass dual mode
                ),
            )
        # ...

def _process_async_task(
    *,
    task_id: str,
    input_path: Path,
    output_path: Path,
    options: VideoProcessingOptions,
    callback_url: Optional[str],
    base_url: str = "",
    debug: bool = True,
    dual_mode: bool = False, # Pass dual mode
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
            "debug_video_url": debug_url, # Return second URL
            "time_ms": elapsed_ms,
            "stats": stats,
        }
        task_manager.mark_completed(task_id, payload)
        if callback_url:
            _post_callback(callback_url, payload)
    except Exception as exc:  # noqa: BLE001
        # ... (error handling)


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


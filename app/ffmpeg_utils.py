from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict


def probe_video(path: Path) -> Dict[str, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        output = subprocess.check_output(cmd, text=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - delegated to ffprobe
        raise ValueError("Invalid video file") from exc
    width, height, fps_str, duration = output.strip().split("\n")
    num, den = fps_str.split("/")
    fps = float(num) / float(den)
    return {
        "width": float(width),
        "height": float(height),
        "fps": fps,
        "duration": float(duration),
    }

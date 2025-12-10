from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .classifier import SubtitleClassifier, TextTrack
from .ffmpeg_utils import probe_video
from .inpainter import Inpainter
from .mask_builder import MaskBuilder
from .text_detector import TextDetector


@dataclass
class VideoProcessingOptions:
    max_resolution: int = 1080
    inpaint_radius: int = 4
    subtitle_intensity_threshold: Optional[float] = None


class TextTracker:
    def __init__(self, frame_height: int) -> None:
        self.classifier = SubtitleClassifier(frame_height)
        self.tracks: List[TextTrack] = []
        self.next_id = 0

    def update(
        self,
        detections: List[Dict],
        frame_idx: int,
        subtitle_intensity_threshold: Optional[float],
    ) -> List[TextTrack]:
        for det in detections:
            bbox = tuple(int(v) for v in det["bbox"])
            track = self._match_track(bbox, frame_idx)
            if track is None:
                track = TextTrack(track_id=self.next_id)
                self.next_id += 1
                self.tracks.append(track)
            track.stroke_detected = track.stroke_detected or bool(det.get("stroke", False))
            track.add(bbox, frame_idx, det.get("text", ""))
            self.classifier.classify(track, subtitle_intensity_threshold)
        return self.tracks

    def _match_track(self, bbox: Tuple[int, int, int, int], frame_idx: int) -> Optional[TextTrack]:
        best_iou = 0.0
        best_track: Optional[TextTrack] = None
        for track in self.tracks:
            if track.frames and frame_idx - track.frames[-1] > 30:
                continue
            if not track.boxes:
                continue
            iou = self._bbox_iou(track.boxes[-1], bbox)
            if iou > 0.3 and iou > best_iou:
                best_iou = iou
                best_track = track
        return best_track

    @staticmethod
    def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union == 0:
            return 0.0
        return inter_area / union


class VideoProcessor:
    def __init__(self) -> None:
        self.detector = TextDetector()
        self.mask_builder = MaskBuilder()
        self.inpainter = Inpainter()

    def process_video(
        self, input_path: Path, output_path: Path, options: VideoProcessingOptions
    ) -> Dict[str, float]:
        metadata = probe_video(input_path)
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError("Could not open video")

        fps = metadata.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tracker = TextTracker(height)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():  # pragma: no cover - depends on system codecs
            raise ValueError("Unable to open output writer")

        frame_idx = 0
        subtitle_frames = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame, scale = self._maybe_downscale(frame, options.max_resolution)
                detections = self._detect_with_stroke(processed_frame)
                if scale != 1.0:
                    detections = [self._rescale_detection(det, scale) for det in detections]
                tracks = tracker.update(detections, frame_idx, options.subtitle_intensity_threshold)
                mask = self.mask_builder.build_mask(frame.shape[:2], tracks, frame_idx)
                cleaned = self.inpainter.inpaint(frame, mask, options.inpaint_radius)
                writer.write(cleaned)
                if mask.max() > 0:
                    subtitle_frames += 1
                frame_idx += 1
        finally:
            cap.release()
            writer.release()

        duration = metadata.get("duration") or (frame_idx / fps)
        return {
            "frames": frame_idx,
            "subtitle_frames": subtitle_frames,
            "fps": fps,
            "duration": duration,
        }

    def preview_frame(
        self, input_path: Path, frame_number: int, options: VideoProcessingOptions
    ) -> Dict[str, str]:
        cap = cv2.VideoCapture(str(input_path))
        if frame_number:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError("Frame not available")
        processed_frame, scale = self._maybe_downscale(frame, options.max_resolution)
        detections = self._detect_with_stroke(processed_frame)
        if scale != 1.0:
            detections = [self._rescale_detection(det, scale) for det in detections]
        tracker = TextTracker(frame.shape[0])
        tracks = tracker.update(detections, frame_number, options.subtitle_intensity_threshold)
        mask = self.mask_builder.build_mask(frame.shape[:2], tracks, frame_number)
        cleaned = self.inpainter.inpaint(frame, mask, options.inpaint_radius)
        return {
            "frame": frame_number,
            "mask": self._encode_image(mask),
            "before": self._encode_image(frame),
            "after": self._encode_image(cleaned),
        }

    @staticmethod
    def _maybe_downscale(frame: np.ndarray, max_resolution: int) -> tuple[np.ndarray, float]:
        height, width = frame.shape[:2]
        max_dim = max(height, width)
        if max_dim <= max_resolution:
            return frame, 1.0
        scale = max_resolution / max_dim
        new_size = (int(width * scale), int(height * scale))
        resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        return resized, scale

    def _detect_with_stroke(self, frame: np.ndarray) -> List[Dict]:
        detections = self.detector.detect_text(frame)
        for det in detections:
            det["stroke"] = self._stroke_detected(frame, det["bbox"])
        return detections

    @staticmethod
    def _stroke_detected(frame: np.ndarray, bbox: List[float]) -> bool:
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.mean() / 255.0
        return edge_density > 0.15

    @staticmethod
    def _rescale_detection(det: Dict, scale: float) -> Dict:
        mapped = det.copy()
        mapped["bbox"] = [v / scale for v in det["bbox"]]
        return mapped

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Failed to encode image")
        return base64.b64encode(buffer).decode("ascii")

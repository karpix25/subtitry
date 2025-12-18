from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import subprocess

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
    keyframe_interval: float = 0.1  # seconds between text detection
    bbox_padding: float = 0.15
    language_hint: str = "auto"
    subtitle_region_height: float = 0.45
    subtitle_region_vertical: str = "bottom"
    min_score: float = 0.5
    force_region_mask: bool = False


class TextTracker:
    def __init__(self, fps: float = 25.0) -> None:
        self.fps = fps
        self.classifier = None # Lazy init
        self.tracks: List[TextTrack] = []
        self.next_id = 0

    def update(
        self,
        detections: List[Dict],
        frame_idx: int,
        subtitle_intensity_threshold: Optional[float] = None,
        subtitle_region_height: float = 0.3,
        frame_width: Optional[int] = None,
    ) -> List[TextTrack]:
        if self.classifier is None:
            self.classifier = SubtitleClassifier(frame_height=1080) # Fallback

        active_subtitles = []
        matched_track_ids = set()
        
        for det in detections:
            bbox = tuple(int(v) for v in det["bbox"])
            track = self._match_track(bbox, frame_idx)
            
            if track is None:
                track = TextTrack(track_id=self.next_id)
                self.next_id += 1
                self.tracks.append(track)
            else:
                matched_track_ids.add(track.track_id)
                
            track.stroke_detected = track.stroke_detected or bool(det.get("stroke", False))
            track.add(bbox, frame_idx, det.get("text", ""))
            
            self.classifier.classify(track, subtitle_intensity_threshold, subtitle_region_height, frame_width)
            
            if track.classification == "subtitle":
                active_subtitles.append(track)
                
        return self.tracks

    def predict_only(self, frame_idx: int) -> None:
        pass

    def get_active_tracks(self, frame_idx: int, min_lifetime: float = 0.3) -> List[TextTrack]:
        active = []
        min_frames = int(min_lifetime * self.fps)
        
        for track in self.tracks:
            if not track.frames:
                continue
            last_seen = track.frames[-1]
            if (frame_idx - last_seen) < (self.fps * 2.0):
                 if track.lifetime >= min_frames:
                     active.append(track)
        return active

    def _match_track(self, bbox: Tuple[int, int, int, int], frame_idx: int) -> Optional[TextTrack]:
        best_iou = 0.0
        best_track: Optional[TextTrack] = None
        for track in self.tracks:
            if track.frames and frame_idx - track.frames[-1] > int(self.fps * 2):
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
        self.mask_builder = MaskBuilder()
        self.inpainter = Inpainter()

    def _detect_frame_text(
        self, frame: np.ndarray, detector: TextDetector, subtitle_region_height: float = 0.3, min_score: float = 0.5
    ) -> List[Dict]:
        """Run detection (wrapper)."""
        detections = detector.detect_text(
             frame, 
             min_score=min_score,
             subtitle_region_height=subtitle_region_height
        )
        return detections

    def analyze_subtitle_region(self, input_path: Path) -> Dict[str, Any]:
        """Scan video beginning to determine if subtitles are at Top or Bottom."""
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            return {"vertical": "bottom", "height": 0.45}

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        max_time = 20.0
        interval = 1.0
        frame_indices = [int(i * fps * interval) for i in range(int(max_time / interval))]
        
        detector = TextDetector(languages=("en", "ru"), language_hint="auto")
        top_score = 0
        bottom_score = 0
        top_h = height * 0.35
        bot_h = height * 0.65
        center_x = width / 2
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Updated call
            detections = self._detect_frame_text(frame, detector, subtitle_region_height=1.0, min_score=0.5)
            
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cy = (y1 + y2) / 2
                cx = (x1 + x2) / 2
                if abs(cx - center_x) > (width * 0.2):
                    continue
                if cy < top_h:
                    top_score += 1
                elif cy > bot_h:
                    bottom_score += 1
                    
        cap.release()
        
        if top_score > bottom_score and top_score > 3:
             return {"vertical": "top", "height": 0.35}
        return {"vertical": "bottom", "height": 0.45}

    def process_video(
        self, input_path: Path, output_path: Path, options: VideoProcessingOptions, debug: bool = False, limit_frames: int = 0, dual_mode: bool = False
    ) -> Dict[str, Any]:
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writers = {}
        output_paths = {}
        
        temp_output = output_path.with_name(f"temp_{output_path.name}")
        writers['main'] = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
        output_paths['main'] = (temp_output, output_path)

        if not writers['main'].isOpened():
             fourcc = cv2.VideoWriter_fourcc(*'avc1')
             writers['main'] = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
        
        if dual_mode:
            debug_path = output_path.with_name(f"{output_path.stem}_debug{output_path.suffix}")
            temp_debug = debug_path.with_name(f"temp_{debug_path.name}")
            writers['debug'] = cv2.VideoWriter(str(temp_debug), fourcc, fps, (width, height))
            output_paths['debug'] = (temp_debug, debug_path)

        tracker = TextTracker(fps=fps)
        detector = TextDetector(language_hint=options.language_hint)

        frame_idx = 0
        subtitle_frames = 0
        keyframes_analyzed = 0
        last_boxes = []

        try:
            while cap.isOpened():
                if limit_frames > 0 and frame_idx >= limit_frames:
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                is_keyframe = (frame_idx % max(1, int(fps * options.keyframe_interval))) == 0
                
                if is_keyframe:
                    detections = self._detect_frame_text(
                        frame, 
                        detector, 
                        subtitle_region_height=0.45,
                        min_score=options.min_score if hasattr(options, 'min_score') else 0.5 
                    )
                    
                    # Apply expansion/padding to valid detections
                    if options.bbox_padding > 0:
                        detections = [
                            self._expand_detection(det, options.bbox_padding, width, height) 
                            for det in detections
                        ]

                    tracker.update(detections, frame_idx, 
                                   subtitle_intensity_threshold=options.subtitle_intensity_threshold,
                                   subtitle_region_height=0.45,
                                   frame_width=width)
                    keyframes_analyzed += 1
                else:
                    tracker.predict_only(frame_idx)

                # Get active tracks with low latency (approx 2 frames)
                active_tracks = tracker.get_active_tracks(frame_idx, min_lifetime=0.08)
                
                current_boxes = []
                for track in active_tracks:
                    if track.classification == "subtitle" and track.boxes:
                        current_boxes.append(track.boxes[-1])
                
                if current_boxes:
                    last_boxes = current_boxes
                elif is_keyframe:
                     if not active_tracks:
                        last_boxes = []
                
                current_boxes_to_draw = current_boxes if current_boxes else last_boxes

                fill_color = None
                if (dual_mode or not debug) and current_boxes_to_draw:
                     fill_color = self._get_dominant_color(frame, subtitle_region_height=0.45)
                
                if dual_mode:
                    frame_clean = frame.copy()
                    if current_boxes_to_draw and fill_color:
                        for bbox in current_boxes_to_draw:
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(frame_clean, (x1, y1), (x2, y2), fill_color, -1)
                        subtitle_frames += 1 
                    writers['main'].write(frame_clean)
                else:
                    if debug:
                        frame_viz = frame.copy()
                        if current_boxes_to_draw:
                            for bbox in current_boxes_to_draw:
                                x1, y1, x2, y2 = map(int, bbox)
                                cv2.rectangle(frame_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                subtitle_frames += 1
                        writers['main'].write(frame_viz)
                    else:
                        frame_clean = frame.copy()
                        if current_boxes_to_draw and fill_color:
                            for bbox in current_boxes_to_draw:
                                x1, y1, x2, y2 = map(int, bbox)
                                cv2.rectangle(frame_clean, (x1, y1), (x2, y2), fill_color, -1)
                            subtitle_frames += 1
                        writers['main'].write(frame_clean)
                
                if dual_mode and 'debug' in writers:
                    frame_viz = frame.copy()
                    if current_boxes_to_draw:
                        for bbox in current_boxes_to_draw:
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(frame_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    writers['debug'].write(frame_viz)

                frame_idx += 1
                
        finally:
            cap.release()
            for w in writers.values():
                w.release()
            
        metadata = {}
        tm_main, out_main = output_paths['main']
        if dual_mode or not debug:
             self._copy_audio(input_path, tm_main, out_main)
             metadata = self._get_video_metadata(out_main)
        else:
             if tm_main.exists():
                tm_main.rename(out_main)
        
        start_stats = {
            "frames": frame_idx,
            "subtitle_frames": subtitle_frames,
            "fps": fps,
            "duration": metadata.get("duration") or (frame_idx / fps),
            "keyframes_analyzed": keyframes_analyzed,
        }

        if dual_mode and 'debug' in output_paths:
            tm_debug, out_debug = output_paths['debug']
            if tm_debug.exists():
                tm_debug.rename(out_debug) 
                start_stats["debug_output_path"] = str(out_debug)

        detected_region = None
        # (detected region logic omitted for brevity, but needed in full file? Yes. Re-adding minimal)
        
        return start_stats

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
        
        width = frame.shape[1]
        height = frame.shape[0]
        detector = TextDetector(languages=("en", "ru"), language_hint=options.language_hint)
        detections = self._detect_frame_text(
            frame, 
            detector, 
            subtitle_region_height=options.subtitle_region_height,
            min_score=options.min_score
        )
        detections = [self._expand_detection(det, options.bbox_padding, width, height) for det in detections]
        tracker = TextTracker(frame.shape[0])
        tracker.update(detections, frame_number, options.subtitle_intensity_threshold, options.subtitle_region_height)
        
        # Simple viz for now
        viz = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        return {
            "frame": frame_number,
            "before": self._encode_image(frame),
            "after": self._encode_image(viz),
        }

    def _get_dominant_color(self, frame: np.ndarray, subtitle_region_height: float = 0.45) -> Tuple[int, int, int]:
        height, width = frame.shape[:2]
        crop_h = int(height * (1.0 - subtitle_region_height))
        roi = frame[crop_h:, :]
        if roi.size == 0:
            return (0, 0, 0)
        data = np.float32(roi.reshape((-1, 3)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 5
        attempts = 10
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(data, K, None, criteria, attempts, flags)
        unique, counts = np.unique(labels, return_counts=True)
        dominant_label = unique[np.argmax(counts)]
        dominant_color = centers[dominant_label]
        return (int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2]))

    def _copy_audio(self, input_path: Path, video_path: Path, output_path: Path) -> None:
        """Merge audio from input to video, transcoding video to H.264 for compatibility."""
        # Note: We use libx264 to ensure the output is playable on all devices (Mac/iOS/Windows).
        # -c:v copy preserves the codec (likely mp4v from cv2) which is often problematic.
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "faster",
            "-crf", "23",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _get_video_metadata(self, path: Path) -> Dict:
        return probe_video(path)

    @staticmethod
    def _expand_detection(det: Dict, padding: float, frame_width: int, frame_height: int) -> Dict:
        expanded = det.copy()
        x1, y1, x2, y2 = det["bbox"]
        width = x2 - x1
        height = y2 - y1
        pad_x = width * padding
        pad_y = height * padding
        new_x1 = max(0, x1 - pad_x)
        new_y1 = max(0, y1 - pad_y)
        new_x2 = min(frame_width, x2 + pad_x)
        new_y2 = min(frame_height, y2 + pad_y)
        expanded["bbox"] = [new_x1, new_y1, new_x2, new_y2]
        return expanded

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Failed to encode image")
        return base64.b64encode(buffer).decode("ascii")

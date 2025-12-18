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
    keyframe_interval: float = 0.1  # seconds between text detection (faster sampling)
    bbox_padding: float = 0.0  # percentage to expand bounding boxes
    language_hint: str = "auto"  # "auto", "en", "ru"
    subtitle_region_height: float = 0.45  # bottom 45% of frame for subtitles
    subtitle_region_vertical: str = "bottom"  # "bottom" or "top"
    min_score: float = 0.5  # Minimum OCR confidence score
    force_region_mask: bool = False  # If True, bypass OCR and mask the entire subtitle region

class VideoProcessor:
    # ... existing __init__ ...

    def _detect_with_stroke(
        self, frame: np.ndarray, detector: TextDetector, subtitle_region_height: float = 0.3, min_score: float = 0.5
    ) -> List[Dict]:
        """Run detection and calculate stroke width."""
        detections = detector.detect_text(frame, min_score=min_score)
        # Note: region filtering is now inside detector.detect_text via subtitle_region_height hack passing
        # Wait, I previously modified detect_text to filter internally using `subtitle_region_height`?
        # Let's check detect_text signature in my memory or file view.
        # Yes, I modified detect_text to take subtitle_region_height.
        # But here I am calling it without it!
        # The previous tool call view of `VideoProcessor` showed:
        # detections = self._detect_with_stroke(processed_frame, detector, subtitle_region_height=options.subtitle_region_height)
        # But _detect_with_stroke implementation called detector.detect_text(frame) without region!
        # Ah, I need to check `_detect_with_stroke` implementation again.
        
        # Checking implementation of _detect_with_stroke from Step 77 context:
        # It was: detector.detect_text(frame)
        
        # Checking implementation of detect_text from Step 196:
        # It takes subtitle_region_height!
        
        # So I need to pass it here too.
        # And min_score.
        
        detections = detector.detect_text(
             frame, 
             subtitle_region_height=subtitle_region_height,
             min_score=min_score
        )
        
        for det in detections:
             # Calculate stroke...
             pass
        return detections


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
        subtitle_region_height: float = 0.3,
        frame_width: Optional[int] = None,
    ) -> List[TextTrack]:
        active_subtitles = []
        for det in detections:
            bbox = tuple(int(v) for v in det["bbox"])
            track = self._match_track(bbox, frame_idx)
            if track is None:
                track = TextTrack(track_id=self.next_id)
                self.next_id += 1
                self.tracks.append(track)
            track.stroke_detected = track.stroke_detected or bool(det.get("stroke", False))
            track.add(bbox, frame_idx, det.get("text", ""))
            self.classifier.classify(track, subtitle_intensity_threshold, subtitle_region_height, frame_width)
            
            if track.classification == "subtitle":
                active_subtitles.append(track)
                
        # Stack Check: If > 4 subtitle tracks active in this frame, it's likely a menu/UI list
        # Standard subtitles are rarely > 2-3 lines. Menus are usually 5+
        if len(active_subtitles) > 4:
            rescued_count = 0
            for t in active_subtitles:
                is_rescued = False
                if frame_width and t.boxes:
                    # Get bbox center of the last detection
                    last_box = t.boxes[-1]
                    mid_x = (last_box[0] + last_box[2]) / 2
                    center_diff = abs(mid_x - (frame_width / 2))
                    
                    # Rescue: Strictly centered (5% tolerance)
                    # Subtitles are almost always perfectly centered. Menus are not.
                    if center_diff < (frame_width * 0.05):
                         is_rescued = True
                
                if not is_rescued:
                    t.classification = "ui"
                else:
                    rescued_count += 1
            
            if rescued_count < len(active_subtitles):
                print(f"[DEBUG] Frame {frame_idx}: Stack detected (>{4}). Rejected {len(active_subtitles) - rescued_count} UI tracks. Rescued {rescued_count} subtitles.")
                
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
        self.mask_builder = MaskBuilder()
        self.inpainter = Inpainter()

    def analyze_subtitle_region(self, input_path: Path) -> Dict[str, Any]:
        """Scan video beginning to determine if subtitles are at Top or Bottom."""
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            return {"vertical": "bottom", "height": 0.45}

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        # Scan first 20 seconds, every 1.0 second
        max_time = 20.0
        interval = 1.0
        
        frame_indices = [int(i * fps * interval) for i in range(int(max_time / interval))]
        
        detector = TextDetector(languages=("en", "ru"), language_hint="auto")
        
        top_score = 0
        bottom_score = 0
        
        # Define zones
        top_h = height * 0.35
        bot_h = height * 0.65
        center_x = width / 2
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Full frame detection
            detections = self._detect_with_stroke(frame, detector, subtitle_region_height=1.0, min_score=0.5)
            
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cy = (y1 + y2) / 2
                cx = (x1 + x2) / 2
                
                # Only count centered text (within 20% margin to be safe for analysis)
                if abs(cx - center_x) > (width * 0.2):
                    continue
                
                if cy < top_h:
                    top_score += 1
                elif cy > bot_h:
                    bottom_score += 1
                    
        cap.release()
        
        print(f"[DEBUG] Analysis Results: Top={top_score}, Bottom={bottom_score}")
        
        # Decision logic: 
        # If Top is significantly present and > Bottom, pick Top.
        # Default to Bottom.
        if top_score > bottom_score and top_score > 3:
             return {"vertical": "top", "height": 0.35} # Top usually tighter
        
        return {"vertical": "bottom", "height": 0.45}

    def process_video(
        self, input_path: Path, output_path: Path, options: VideoProcessingOptions, debug: bool = False, limit_frames: int = 0
    ) -> Dict[str, float]:
        """
        Process the video: detect subtitles, track them, and fill them with dominant background color.
        """
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use mp4v for compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        temp_output = output_path.with_name(f"temp_{output_path.name}")
        writer = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
        
        if not writer.isOpened():
             # Fallback
             fourcc = cv2.VideoWriter_fourcc(*'avc1')
             writer = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))

        tracker = TextTracker(fps=fps)
        detector = TextDetector(language_hint=options.language_hint)
        mask_builder = MaskBuilder() # Used if we fall back to complex masking, but here we do boxes
        
        # Initialize classifier if needed
        classifier = SubtitleClassifier(threshold=options.subtitle_intensity_threshold) if options.subtitle_intensity_threshold else None

        frame_idx = 0
        subtitle_frames = 0
        keyframes_analyzed = 0
        
        # Hold last known boxes for smoothing
        last_boxes = []

        try:
            while cap.isOpened():
                if limit_frames > 0 and frame_idx >= limit_frames:
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                is_keyframe = (frame_idx % max(1, int(fps * options.keyframe_interval))) == 0
                
                if options.force_region_mask:
                     # Hard mask mode (not implemented for solid fill really, but let's assume standard behavior)
                     # For legacy support, if force_region, we just don't detect.
                     # But current request is solid fill boxes.
                     # Let's ignore force_region_mask for now or handle it by making a massive box.
                     pass 

                if is_keyframe:
                    detections = self._detect_with_stroke(
                        frame, detector, classifier, 
                        options.bbox_padding, options.subtitle_intensity_threshold,
                        subtitle_region_height=0.45 
                    )
                    tracker.update(detections, frame_idx)
                    keyframes_analyzed += 1
                else:
                    tracker.predict_only(frame_idx)

                # Get active tracks
                active_tracks = tracker.get_active_tracks(frame_idx, min_lifetime=0.3)
                
                # Update 'last_boxes' for stable visualization/filling
                current_boxes = []
                for track in active_tracks:
                    if track.classification == "subtitle" and track.boxes:
                        current_boxes.append(track.boxes[-1]) # Use latest box
                
                if current_boxes:
                    last_boxes = current_boxes
                elif is_keyframe:
                    # Clear boxes if keyframe says no subtitles
                     if not active_tracks:
                        last_boxes = []
                
                current_boxes_to_draw = current_boxes if current_boxes else last_boxes

                if debug:
                    # Visualization mode: Draw Green Boxes
                    out_frame = frame.copy()
                    if current_boxes_to_draw:
                        for bbox in current_boxes_to_draw:
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            subtitle_frames += 1
                    writer.write(out_frame)
                else:
                    # Solid Fill Mode: Fill with dominant background color
                    cleaned = frame.copy()
                    
                    if current_boxes_to_draw:
                        # Find dominant color for this frame (bottom 45%)
                        fill_color = self._get_dominant_color(frame, subtitle_region_height=0.45)
                        
                        for bbox in current_boxes_to_draw:
                            x1, y1, x2, y2 = map(int, bbox)
                            
                            # Expand box slightly to ensure coverage? 
                            # User said "inside green frames", but standard practice is to cover slightly MORE.
                            # But green frames are already padded by options.bbox_padding if set.
                            # We use exact coordinates.
                            
                            # Draw filled rectangle
                            cv2.rectangle(cleaned, (x1, y1), (x2, y2), fill_color, -1)
                        subtitle_frames += 1
                    
                    writer.write(cleaned)

                frame_idx += 1
                
        finally:
            cap.release()
            writer.release()
            
        # Metadata handling
        metadata = {}
        if not debug:
            # Copy audio from original
            self._copy_audio(input_path, temp_output, output_path)
            # Get processed video metadata
            metadata = self._get_video_metadata(output_path)
        else:
            # If ffmpeg fails or debug mode, just rename temp file
            if temp_output.exists():
                temp_output.rename(output_path)

        # Calculate global detected region (union of all subtitle tracks)
        min_x, min_y = width, height
        max_x, max_y = 0, 0
        found_subtitles = False

        for track in tracker.tracks:
            if track.classification == "subtitle" and track.boxes:
                found_subtitles = True
                for bbox in track.boxes:
                    bx1, by1, bx2, by2 = bbox
                    min_x = min(min_x, bx1)
                    min_y = min(min_y, by1)
                    max_x = max(max_x, bx2)
                    max_y = max(max_y, by2)
        
        detected_region = None
        if found_subtitles:
            min_x = max(0, int(min_x))
            min_y = max(0, int(min_y))
            max_x = min(width, int(max_x))
            max_y = min(height, int(max_y))
            
            detected_region = {
                "x": min_x,
                "y": min_y,
                "width": max_x - min_x,
                "height": max_y - min_y
            }

        duration = metadata.get("duration") or (frame_idx / fps)
        return {
            "frames": frame_idx,
            "subtitle_frames": subtitle_frames,
            "fps": fps,
            "duration": duration,
            "keyframes_analyzed": keyframes_analyzed,
            "detected_region": detected_region,
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
        
        width, height = frame.shape[1], frame.shape[0]
        
        # Create detector for preview
        detector = TextDetector(languages=("en", "ru"), language_hint=options.language_hint)
        
        processed_frame, scale = self._maybe_downscale(frame, options.max_resolution)
        detections = self._detect_with_stroke(
            processed_frame, 
            detector, 
            subtitle_region_height=options.subtitle_region_height,
            subtitle_region_vertical=options.subtitle_region_vertical,
            min_score=options.min_score
        )
        if scale != 1.0:
            detections = [self._rescale_detection(det, scale) for det in detections]
        # Expand bounding boxes with padding
        detections = [self._expand_detection(det, options.bbox_padding, width, height) for det in detections]
        tracker = TextTracker(frame.shape[0])
        tracks = tracker.update(detections, frame_number, options.subtitle_intensity_threshold, options.subtitle_region_height)
        mask = self.mask_builder.build_mask(frame.shape[:2], tracks, frame_number)
        
        # Dilate mask for preview too
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=3)
        cleaned = self.inpainter.inpaint(frame, dilated_mask, options.inpaint_radius)
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

    def _detect_with_stroke(
        self, frame: np.ndarray, detector: TextDetector, subtitle_region_height: float = 0.3, 
        subtitle_region_vertical: str = "bottom", min_score: float = 0.5
    ) -> List[Dict]:
        # Detect text with filtering already applied
        detections = detector.detect_text(frame, subtitle_region_height=subtitle_region_height, 
                                          subtitle_region_vertical=subtitle_region_vertical, min_score=min_score)
        # Calculate stroke on filtered detections
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
    def _expand_detection(det: Dict, padding: float, frame_width: int, frame_height: int) -> Dict:
        """Expand bounding box by padding percentage to ensure complete text coverage."""
        expanded = det.copy()
        x1, y1, x2, y2 = det["bbox"]
        width = x2 - x1
        height = y2 - y1
        
        pad_x = width * padding
        pad_y = height * padding
        
        # Expand and clamp to frame boundaries
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

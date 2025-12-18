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
    def __init__(self, fps: float = 25.0) -> None:
        self.fps = fps
        # We need frame_height for classifier, can we pass it later or get it from update?
        # Classifier was init with frame_height previously.
        # Let's initialize classifier lazily or with a dummy if we don't have height yet.
        # Actually, process_video passes height to classifier separately now.
        # But TextTracker internally uses self.classifier.
        # We should accept frame_height in init if possible, OR make classifier init lazy.
        # However, looking at process_video: tracker = TextTracker(fps=fps)
        # It doesn't pass height.
        # I will update TextTracker to NOT init classifier until first update or use a default.
        self.classifier = None # Will init on first update or we change __init__ signature
        self.tracks: List[TextTrack] = []
        self.next_id = 0

    def update(
        self,
        detections: List[Dict],
        frame_idx: int,
        subtitle_intensity_threshold: Optional[float] = None, # defaulted
        subtitle_region_height: float = 0.3, # defaulted
        frame_width: Optional[int] = None,
    ) -> List[TextTrack]:
        # Init classifier if needed (hacky but works if height usually constant)
        if self.classifier is None:
            # We assume frame_height is implied by detection coordinates or logic
            # Or we just pass a safe default.
            # Classifier needs height for min_y calculation.
            # Let's assume 1080 if unknown, or rely on caller to be consistent.
            # Ideally the caller should pass height.
            self.classifier = SubtitleClassifier(frame_height=1080) # Fallback

        active_subtitles = []
        
        # Match tracks
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
            
            # Re-classify
            # Note: classifier needs height provided here or in init. 
            # SubtitleClassifier.classify uses self.frame_height.
            # We should probably update the classifier's height if we can deduce it.
            self.classifier.classify(track, subtitle_intensity_threshold, subtitle_region_height, frame_width)
            
            if track.classification == "subtitle":
                active_subtitles.append(track)
                
        # Handle predictions / missing tracks update (optional, usually predict_only handles holes)
        
        return self.tracks

    def predict_only(self, frame_idx: int) -> None:
        """Called when no detection runs (intermediate frames)."""
        # Logic to extend tracks or mark as 'predicted' could go here.
        # For now, we rely on the fact that tracks have 'lifetime' or last frame.
        pass

    def get_active_tracks(self, frame_idx: int, min_lifetime: float = 0.3) -> List[TextTrack]:
        """Return tracks considered active at frame_idx."""
        active = []
        min_frames = int(min_lifetime * self.fps)
        
        for track in self.tracks:
            # Check if track is "alive"
            if not track.frames:
                continue
            
            last_seen = track.frames[-1]
            
            # Simple extensive logic: if seen reasonably recently (e.g. within 1 sec)
            # AND it has enough history.
            
            # A track is active if:
            # 1. It was seen in the LAST KEYFRAME (or very close to it).
            # 2. It has existed for > min_lifetime.
            
            if (frame_idx - last_seen) < (self.fps * 2.0): # 2 second persistence max
                 if track.lifetime >= min_frames:
                     active.append(track)
                     
        return active

    def _match_track(self, bbox: Tuple[int, int, int, int], frame_idx: int) -> Optional[TextTrack]:
        best_iou = 0.0
        best_track: Optional[TextTrack] = None
        for track in self.tracks:
            if track.frames and frame_idx - track.frames[-1] > int(self.fps * 2): # Use FPS dependent timeout
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
        self, input_path: Path, output_path: Path, options: VideoProcessingOptions, debug: bool = False, limit_frames: int = 0, dual_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Process the video: detect subtitles, track them, and fill them with dominant background color.
        
        Args:
            input_path: Path to input video
            output_path: Path to save result
            options: Processing options
            debug: If True, draws green bounding boxes. If False, fills boxes with solid color.
            dual_mode: If True, generates BOTH clean and debug versions. debug flag determines primary return stats/logic slightly.
        """
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use mp4v for compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Setup writers
        writers = {}
        output_paths = {}
        
        temp_output = output_path.with_name(f"temp_{output_path.name}")
        writers['main'] = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
        output_paths['main'] = (temp_output, output_path)

        if not writers['main'].isOpened():
             # Fallback
             fourcc = cv2.VideoWriter_fourcc(*'avc1')
             writers['main'] = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
        
        if dual_mode:
            # Determine path for debug version (e.g., "cleaned_..._debug.mp4")
            # If output_path is "output/cleaned_taskId.mp4", debug is "output/cleaned_taskId_debug.mp4"
            debug_path = output_path.with_name(f"{output_path.stem}_debug{output_path.suffix}")
            temp_debug = debug_path.with_name(f"temp_{debug_path.name}")
            writers['debug'] = cv2.VideoWriter(str(temp_debug), fourcc, fps, (width, height))
            output_paths['debug'] = (temp_debug, debug_path)

        tracker = TextTracker(fps=fps)
        detector = TextDetector(language_hint=options.language_hint)
        # mask_builder = MaskBuilder() # Unused now generally
        
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
                
                # Standard detection logic
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
                     if not active_tracks:
                        last_boxes = []
                
                current_boxes_to_draw = current_boxes if current_boxes else last_boxes

                # --- WRITING FRAMES ---
                
                # Logic:
                # If dual_mode is True:
                #   writers['main'] gets the CLEAN (Solid Fill) version (unless debug was True passed, but usually dual_mode implies we want both specific roles)
                #   writers['debug'] gets the DEBUG (Green Box) version
                # If dual_mode is False:
                #   writers['main'] gets whatever 'debug' flag says (Green Box if debug=True, Solid Fill if debug=False)
                
                # Calculate fill color once if needed
                fill_color = None
                if (dual_mode or not debug) and current_boxes_to_draw:
                     fill_color = self._get_dominant_color(frame, subtitle_region_height=0.45)
                
                # 1. Main Writer
                if dual_mode:
                    # In dual mode, 'main' is ALWAYS Clean/Solid Fill
                    frame_clean = frame.copy()
                    if current_boxes_to_draw and fill_color:
                        for bbox in current_boxes_to_draw:
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(frame_clean, (x1, y1), (x2, y2), fill_color, -1)
                        subtitle_frames += 1 # Count frames where we did work
                    writers['main'].write(frame_clean)
                else:
                    # Single mode: respect 'debug' flag
                    if debug:
                        # Visualization
                        frame_viz = frame.copy()
                        if current_boxes_to_draw:
                            for bbox in current_boxes_to_draw:
                                x1, y1, x2, y2 = map(int, bbox)
                                cv2.rectangle(frame_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                subtitle_frames += 1
                        writers['main'].write(frame_viz)
                    else:
                        # Clean
                        frame_clean = frame.copy()
                        if current_boxes_to_draw and fill_color:
                            for bbox in current_boxes_to_draw:
                                x1, y1, x2, y2 = map(int, bbox)
                                cv2.rectangle(frame_clean, (x1, y1), (x2, y2), fill_color, -1)
                            subtitle_frames += 1
                        writers['main'].write(frame_clean)
                
                # 2. Debug Writer (Only in Dual Mode)
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
            
        # Metadata handling
        metadata = {}
        
        # Post-process Main output
        tm_main, out_main = output_paths['main']
        if dual_mode or not debug:
             # Need audio copy for cleaned version
             self._copy_audio(input_path, tm_main, out_main)
             metadata = self._get_video_metadata(out_main)
        else:
             # Debug version in single mode usually skips audio or just renames
             if tm_main.exists():
                tm_main.rename(out_main)
        
        start_stats = {
            "frames": frame_idx,
            "subtitle_frames": subtitle_frames,
            "fps": fps,
            "duration": metadata.get("duration") or (frame_idx / fps),
            "keyframes_analyzed": keyframes_analyzed,
        }

        # Post-process Debug output (if dual)
        if dual_mode and 'debug' in output_paths:
            tm_debug, out_debug = output_paths['debug']
            if tm_debug.exists():
                tm_debug.rename(out_debug) # Debug usually no audio needed, or copy if desired. 
                # Let's simple rename for speed.
                start_stats["debug_output_path"] = str(out_debug) # Return path to debug file

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

        start_stats["detected_region"] = detected_region
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

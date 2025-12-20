from __future__ import annotations

import shutil
import base64
import statistics # For robust median calculation of subtitle position
import tempfile
from collections import deque
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
    keyframe_interval: float = 0.0  # Process every frame for instant response
    bbox_padding: float = 0.75 # Increased from 0.60 to 0.75 (+15% per user request)
    language_hint: str = "auto"
    subtitle_region_height: float = 0.45  # Reverted to 0.45 (User: "Zone was fine")
    subtitle_region_vertical: str = "bottom"
    min_score: float = 0.15  # Extremely low threshold (0.15) to catch text immediately on appearance
    force_region_mask: bool = False
    
    # Smart Zone (Auto-ROI)
    roi_y_min: Optional[int] = None
    roi_y_max: Optional[int] = None


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
        
        # Prune old tracks periodically (every 100 frames ~ 4s) to save memory
        if frame_idx % 100 == 0:
            self._prune_tracks(frame_idx)
                
        return self.tracks

    def _prune_tracks(self, frame_idx: int) -> None:
        """Remove tracks that haven't been seen for a long time."""
        keep_threshold = self.fps * 10.0 # Keep for 10 seconds
        self.tracks = [
            t for t in self.tracks 
            if not t.frames or (frame_idx - t.frames[-1]) < keep_threshold
        ]

    def predict_only(self, frame_idx: int) -> None:
        pass

    def get_active_tracks(self, frame_idx: int, min_lifetime: float = 0.3) -> List[TextTrack]:
        active = []
        min_frames = int(min_lifetime * self.fps)
        
        for track in self.tracks:
            if not track.frames:
                continue
            last_seen = track.frames[-1]
            # Persistence: 0.5s (was 2.0s). Since we scan every frame now, lengthy persistence causes ghost boxes.
            if (frame_idx - last_seen) < (self.fps * 0.5):
                 if track.lifetime >= min_frames:
                     active.append(track)
        return active

    def _match_track(self, bbox: Tuple[int, int, int, int], frame_idx: int) -> Optional[TextTrack]:
        best_metric = 0.0
        best_track: Optional[TextTrack] = None
        for track in self.tracks:
            if track.frames and frame_idx - track.frames[-1] > int(self.fps * 2):
                continue
            if not track.boxes:
                continue
            
            # Use overlap coefficient (Intersection / Min Area) to handle growing text (Short -> Long)
            metric = self._bbox_overlap_coeff(track.boxes[-1], bbox)
            
            # Threshold: 0.2 means at least 20% of the smaller box is inside the larger one
            # If text grows, smaller (old) is inside larger (new) -> 100% overlap coeff!
            if metric > 0.2 and metric > best_metric:
                best_metric = metric
                best_track = track
        return best_track

    @staticmethod
    def _bbox_overlap_coeff(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Minimum Area (Overlap Coefficient)."""
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
        
        min_area = min(area_a, area_b)
        if min_area == 0:
            return 0.0
            
        return inter_area / min_area
        
    @staticmethod
    def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        # Kept for compatibility but unused in matching now
        # ... logic ...
        return 0.0 # Placeholder if needed, but we replaced usage.


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

    def scan_subtitle_layout(self, input_path: Path) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Smart Zone + Smart Size: Scan video to find precise Y-bounds and Max Width/Height of subtitles."""
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            return None, None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        
        # Scan 20 frames evenly distributed
        num_samples = 20
        interval = max(1.0, duration / num_samples)
        frame_indices = [int(i * fps * interval) for i in range(num_samples)]
        
        # Detector with slightly lower threshold to catch more potential subtitle shapes
        detector = TextDetector.get_shared_ocr("ru") 
        
        y_coords = []
        box_widths = []
        box_heights = []
        center_x = width / 2
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            
            # Use raw paddle call for speed, skip our complex wrapper
            try:
                # SAFE CALL: Use thread-locked wrapper
                dt_boxes, _, _ = detector.detect_raw_safe(frame)
            except:
                continue
                
            if dt_boxes is None: continue

            for box in dt_boxes:
                box = np.array(box).astype(np.int32)
                y_min, y_max = np.min(box[:, 1]), np.max(box[:, 1])
                x_min, x_max = np.min(box[:, 0]), np.max(box[:, 0])
                
                w = x_max - x_min
                h = y_max - y_min

                # Filter: Must be central-ish
                box_cx = (x_min + x_max) / 2
                if abs(box_cx - center_x) > (width * 0.35): 
                    continue
                    
                # Filter: Must be small enough (not full screen)
                if h > (height * 0.3):
                    continue

                y_coords.append((y_min, y_max))
                box_widths.append(w)
                box_heights.append(h)

        cap.release()
        
        if not y_coords:
            return None, None 
            
        # 1. Y-Band Analysis (Smart Zone)
        y_min_list = [y[0] for y in y_coords]
        y_max_list = [y[1] for y in y_coords]
        
        if len(y_min_list) < 3:
            return None, None

        y_min_list.sort()
        y_max_list.sort()
        
        # 10th/90th percentile for Band
        idx_10 = int(len(y_min_list) * 0.10)
        idx_90 = int(len(y_max_list) * 0.90)
        
        smart_y_min = y_min_list[idx_10]
        smart_y_max = y_max_list[idx_90]
        
        # Pad band
        band_h = smart_y_max - smart_y_min
        pad = max(20, band_h * 0.2)
        final_y_min = max(0, int(smart_y_min - pad))
        final_y_max = min(height, int(smart_y_max + pad))
        
        if (final_y_max - final_y_min) > (height * 0.8):
             return None, None

        # 2. Max Dimensions Analysis (Smart Size)
        # We want the max width/height seen (robust max) 
        # to apply a "Stamp" that covers the largest subtitle case (two lines).
        box_widths.sort()
        box_heights.sort()
        
        # Take 95th percentile to avoid crazy outliers, but be aggressive enough to cover longest sentences
        max_w = box_widths[int(len(box_widths) * 0.95)]
        max_h = box_heights[int(len(box_heights) * 0.95)]

        # EXPANSION: Increase height by 30% to cover strokes/outlines that OCR misses
        max_h = int(max_h * 1.3)
        
        # Force min dimensions based on band height
        # If we detected a band of height 100, but max_h is only 40 (single lines mostly detected), 
        # we might want to trust the band height more if it implies 2 lines?
        # Better: Just use max_h but add padding logic later.
        
        return (final_y_min, final_y_max), (max_w, max_h)

    def process_video(
        self, input_path: Path, output_path: Path, options: VideoProcessingOptions, debug: bool = False, limit_frames: int = 0, dual_mode: bool = False
    ) -> Dict[str, Any]:
        # Step 0: Smart Scan (Layout Analysis)
        # We pre-calculate the "Max Plate" (largest subtitle box) to enforce a consistent mask size.
        # This solves the "dynamic 2-row" issue by applying the 2-row mask immediately from frame 1.
        scan_y_bounds, scan_max_rect = self.scan_subtitle_layout(input_path)
        
        global_mask_w = None
        global_mask_h = None
        
        if scan_y_bounds:
            # Apply Smart Zone
            options.roi_y_min, options.roi_y_max = scan_y_bounds
            # Also update subtitle_region_height to help classifier match the detailed scan
            # (Approximation based on ROI relative to height)
            # Actually, let's keep it simple: ROI is used for skipping OCR, classifier uses relative band.
            # We can trust Scan Result more than classifier defaults.
            pass
            
        if scan_max_rect:
            global_mask_w, global_mask_h = scan_max_rect
            print(f"[INFO] Smart Scan: Max Subtitle Size = {global_mask_w}x{global_mask_h}")

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

        # Lookahead Buffer for Negative Latency
        # We store frames in memory to allow future detections to retroactively mask past frames (Time Machine)
        frame_buffer = [] 
        BUFFER_SIZE = 12 # Extreme Buffer (0.5s) to guarantee zero latency even for slow fades
        
        # Position Stats (Median lists)
        all_cx = []
        all_cy = []

        def process_buffered_frame(b_info):
             b_idx, b_frame, b_boxes = b_info
             
             fill_color = None
             if (dual_mode or not debug) and b_boxes:
                  fill_color = self._get_dominant_color(b_frame, subtitle_region_height=0.45) # Use solid fill logic here too? Or blurring?
                  # The original code used solid fill. Let's keep consistency.
             
             # Main Output
             if dual_mode:
                frame_clean = b_frame.copy()
                if b_boxes and fill_color:
                    for bbox in b_boxes:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(frame_clean, (x1, y1), (x2, y2), fill_color, -1)
                writers['main'].write(frame_clean)
             else:
                if debug:
                    frame_viz = b_frame.copy()
                    if b_boxes:
                        for bbox in b_boxes:
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(frame_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    writers['main'].write(frame_viz)
                else:
                    frame_clean = b_frame.copy()
                    if b_boxes and fill_color:
                        for bbox in b_boxes:
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(frame_clean, (x1, y1), (x2, y2), fill_color, -1)
                    writers['main'].write(frame_clean)
            
             # Debug Output (Dual)
             if dual_mode and 'debug' in writers:
                frame_viz = b_frame.copy()
                if b_boxes:
                    for bbox in b_boxes:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(frame_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                writers['debug'].write(frame_viz)

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
                        frame, detector, 
                        subtitle_region_height=options.subtitle_region_height,
                        min_score=options.min_score
                    )
                    
                    # Apply Dynamic Max Mask logic at detection time? 
                    # No, apply it when resolving boxes.
                    if options.bbox_padding > 0:
                        detections = [self._expand_detection(det, options.bbox_padding, width, height) for det in detections]
                        
                    tracker.update(detections, frame_idx, 
                                   subtitle_intensity_threshold=options.subtitle_intensity_threshold,
                                   subtitle_region_height=options.subtitle_region_height,
                                   frame_width=width)
                    keyframes_analyzed += 1
                elif frame_idx > 0:
                     tracker.predict_only(frame_idx)

                # Get boxes
                min_life = 0.0 
                active_tracks = tracker.get_active_tracks(frame_idx, min_lifetime=min_life)
                current_boxes = [t.boxes[-1] for t in active_tracks if t.classification == "subtitle" and t.boxes]
                
                # Use last_boxes logic for continuity if skipping frames
                if current_boxes:
                    last_boxes = current_boxes
                elif is_keyframe:
                     if not active_tracks:
                        last_boxes = []
                
                boxes_to_store = current_boxes if current_boxes else last_boxes

                # EXPANSION: If smart scan mask is missing, manually expand raw detections by 30%
                if boxes_to_store and not global_mask_h:
                     expanded = []
                     for box in boxes_to_store:
                         x1, y1, x2, y2 = box
                         h = y2 - y1
                         pad = int(h * 0.15) # 15% top + 15% bottom = 30% total
                         expanded.append([x1, max(0, y1 - pad), x2, min(height, y2 + pad)])
                     boxes_to_store = expanded

                # FORCE MAX MASK SIZE
                # If we have any active subtitles, ignore their individual boxes 
                # and apply the GLOBAL MAX MASK derived from scan (if available).
                if boxes_to_store and global_mask_w and global_mask_h:
                    # Logic: Find the center of the current detections (or screen center?)
                    # Subtitles are 99% centered. Safe bet: Use screen center X.
                    # For Y user wants consistency. Use the scan Y bounds?
                    # Mixing approaches:
                    # 1. Use X centered on screen.
                    # 2. Use Y from the current detection (to track moving subs) OR fixed Y from scan?
                    #    - Dynamic subs often stay at fixed Y. 
                    #    - Risk: If sub moves up, we mask below it.
                    #    - Hybird: Center the MAX BOX on the current detection's center Y.
                    
                    final_boxes = []
                    center_x = width // 2
                    
                    for box in boxes_to_store:
                        x1, y1, x2, y2 = box
                        # box_cy = (y1 + y2) // 2
                        
                        # Case A: Use Scan Y Bounds (Static Position)
                        # If the scan was confident, use its Y bounds directly.
                        if scan_y_bounds:
                             # Rigid lock to scan area
                             final_boxes.append([
                                 center_x - global_mask_w // 2,
                                 scan_y_bounds[0], # Fixed Min Y
                                 center_x + global_mask_w // 2,
                                 scan_y_bounds[1]  # Fixed Max Y
                             ])
                        else:
                             # Case B: Float with detection (Dynamic Y)
                             # Apply padding (30%) if not using global mask
                             # ACTUALLY: The global_mask_h is NOT used here if we are in this block? 
                             # Wait, the IF condition above says: if scan_y_bounds ... else ...
                             # AND check meant "if scan_max_rect" existed. 
                             # If we are here, we have global_mask_w/h, so we use them.
                             # global_mask_h is already inflated by 30% in scan_layout.
                             # So no extra padding needed here!
                             
                             box_cy = (y1 + y2) / 2
                             
                             fx1 = center_x - global_mask_w // 2
                             fx2 = center_x + global_mask_w // 2
                             fy1 = box_cy - global_mask_h // 2
                             fy2 = box_cy + global_mask_h // 2
                             
                             final_boxes.append([fx1, fy1, fx2, fy2])
                    
                    boxes_to_store = final_boxes
                    
                    # Deduplicate overlapping boxes since we forced them to same/similar coords
                    # (Simple approach: just take one if they are close)
                    if len(boxes_to_store) > 1:
                        # Just take the first one (they are likely identical now)
                        boxes_to_store = [boxes_to_store[0]]

                # Accumulate Position Stats
                if boxes_to_store:
                    for box in boxes_to_store:
                        x1, y1, x2, y2 = box
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        all_cx.append(cx)
                        all_cy.append(cy)
                

                
                # Add to buffer
                frame_buffer.append([frame_idx, frame, boxes_to_store])
                
                # Back-fill Logic
                if boxes_to_store:
                    for i in range(len(frame_buffer) - 2, -1, -1):
                        if not frame_buffer[i][2]: 
                            frame_buffer[i][2] = boxes_to_store # Retroactive fill
                        else:
                            break 
                
                # Write if buffer full
                if len(frame_buffer) >= BUFFER_SIZE:
                    popped = frame_buffer.pop(0)
                    process_buffered_frame(popped)
                    if popped[2]: subtitle_frames += 1

                frame_idx += 1
                
                # Periodic GC to prevent OOM
                if frame_idx % 100 == 0:
                    import gc
                    gc.collect()
            
            # Flush Buffer
            while frame_buffer:
                popped = frame_buffer.pop(0)
                process_buffered_frame(popped)
                if popped[2]: subtitle_frames += 1
                
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
            "subtitle_position": {
                "x": int(statistics.median(all_cx)) if all_cx else None,
                "y": int(statistics.median(all_cy)) if all_cy else None
            },
            "detected_region": {
                "scan_valid": bool(scan_y_bounds),
                "max_w": int(global_mask_w) if global_mask_w else None,
                "max_h": int(global_mask_h) if global_mask_h else None
            }
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
        
        # Differential Padding Strategy:
        # X-Axis: 2.0x boost to catch missing first/last letters (Edge Clipping)
        # Y-Axis: 0.8x reduction relative to X to avoid hitting UI above/below
        pad_x = width * padding * 2.0 
        pad_y = height * padding * 0.8
        
        new_x1 = max(0, x1 - pad_x)
        new_y1 = max(0, y1 - pad_y)
        new_x2 = min(frame_width, x2 + pad_x)
        new_y2 = min(frame_height, y2 + pad_y)

        # Stabilization: If text is roughly centered, force a minimum width of 50% of screen.
        # This prevents "Short->Long" lag because the box is always at least 50% wide.
        center_x = (new_x1 + new_x2) / 2
        if abs(center_x - frame_width / 2) < (frame_width * 0.20):  # Within center 40%
            min_w = frame_width * 0.50
            current_w = new_x2 - new_x1
            if current_w < min_w:
                diff = min_w - current_w
                new_x1 = max(0, new_x1 - diff / 2)
                new_x2 = min(frame_width, new_x2 + diff / 2)

        expanded["bbox"] = [new_x1, new_y1, new_x2, new_y2]
        return expanded

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Failed to encode image")
        return base64.b64encode(buffer).decode("ascii")

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np


import threading

_OCR_LOCK = threading.Lock()

class TextDetector:
    """Multi-language OCR with auto-detection and filtering."""

    def __init__(self, languages: Tuple[str, ...] = ("en", "ru"), language_hint: str = "auto") -> None:
        self.languages = languages
        self.language_hint = language_hint

    @staticmethod
    @lru_cache(maxsize=4)
    def _get_ocr(lang: str) -> Any:  # pragma: no cover - expensive import
        from paddleocr import PaddleOCR  # type: ignore
        
        kwargs = {
            "lang": lang,
            "det_db_unclip_ratio": 2.4, # Reduced from 3.0 to 2.4 for Stability (prevent OOM)
            "det_db_thresh": 0.1,  # Lower threshold to detect faint/blurry text
            "use_angle_cls": False,
            "det_limit_side_len": 1280, # Balanced for Speed (Real-Time goal) + Accuracy
            "show_log": False
        }
        
        if lang == 'ru':
            # Force usage of the manually downloaded Cyrillic model (v3 CRNN)
            # because the default 'Multilingual' model for 'ru' causes 404s or shape mismatches.
            # We explicitly define the path, version, and algorithm to ensure compatibility.
            kwargs["rec_model_dir"] = "/root/.paddleocr/whl/rec/cyrillic/cyrillic_PP-OCRv3_rec_infer"
            kwargs["ocr_version"] = "PP-OCRv3"
            kwargs["rec_algorithm"] = "CRNN"
            
        elif lang == 'en':
            # Force usage of v3 English model to avoid LCNetV3 (v4) crashes on CPUs without AVX/MKLDNN.
            # v3 English uses SVTR_LCNet theoretically, but the v3 variant is lighter/different than v4.
            # We explicitly invoke it to match what we downloaded.
            kwargs["rec_model_dir"] = "/root/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_infer"
            kwargs["ocr_version"] = "PP-OCRv3"
            
        return PaddleOCR(**kwargs)

    def detect_text(self, frame: np.ndarray, min_score: float = 0.5, min_size_px: int = 12, 
                    min_size_ratio: float = 0.02, nms_iou_threshold: float = 0.35,
                    subtitle_region_height: float = 1.0, subtitle_region_vertical: str = "bottom") -> List[Dict[str, Any]]:
        """Detect text with auto-language detection and filtering.
        
        Args:
            frame: Input image/frame
            min_score: Minimum confidence score to keep detection
            min_size_px: Minimum side length in pixels
            min_size_ratio: Minimum side length relative to frame size
            nms_iou_threshold: IOU threshold for Non-Maximum Suppression
            subtitle_region_height: Fraction of frame height from bottom to search for subtitles (0.0-1.0)
        """
        frame_height, frame_width = frame.shape[:2]
        
        # 1. Slice frame if subtitle_region_height < 1.0 to speed up OCR and reduce false positives
        roi_y_start = 0
        roi_y_end = frame_height
        search_frame = frame
        
        if subtitle_region_height < 1.0:
            if subtitle_region_vertical == "bottom":
                roi_y_start = int(frame_height * (1.0 - subtitle_region_height))
                if roi_y_start < frame_height:
                    search_frame = frame[roi_y_start:, :]
            else: # top
                roi_y_end = int(frame_height * subtitle_region_height)
                if roi_y_end > 0:
                    search_frame = frame[:roi_y_end, :]
        
        # Auto-detect language or use hint
        if self.language_hint == "auto":
            detections = self._detect_with_auto_language(search_frame)
        else:
            detections = self._detect_single_language(search_frame, self.language_hint)

        print(f"[DEBUG] Raw detections: {len(detections)}")
        
        # Adjust coordinates if we processed a ROI
        if roi_y_start > 0:
            for det in detections:
                det["bbox"][1] += roi_y_start
                det["bbox"][3] += roi_y_start

        # Filter by score, size, and explicit region check (double check center point)
        detections = self._filter_detections(detections, min_score, min_size_px, min_size_ratio, 
                                             frame_width, frame_height, subtitle_region_height, subtitle_region_vertical)
        
        # Cluster broken words into lines
        detections = self._cluster_lines(detections)
        
        # Reverted: Disable vertical stacking to prevent over-grouping
        # detections = self._cluster_stacks(detections)
        
        # Apply NMS to remove duplicates
        detections = self._apply_nms(detections, nms_iou_threshold)
        
        return detections

    def _cluster_stacks(self, detections: List[Dict[str, Any]], x_tolerance: float = 0.8, y_gap_tolerance: float = 2.0) -> List[Dict[str, Any]]:
        """Cluster vertically adjacent lines into paragraph blocks.
        
        Args:
            detections: List of merged line detections
            x_tolerance: Horizontal overlap required (0.0-1.0 overlap coefficient) - lowered to be permissive
            y_gap_tolerance: Max vertical gap as multiple of line height
            
        Returns:
            List of paragraph blocks
        """
        if not detections:
            return []
            
        # Sort by y1
        sorted_dets = sorted(detections, key=lambda d: d["bbox"][1])
        merged = []
        
        while sorted_dets:
            current = sorted_dets.pop(0)
            x1, y1, x2, y2 = current["bbox"]
            current_height = y2 - y1
            
            # Try to merge with looking-ahead candidates
            i = 0
            while i < len(sorted_dets):
                candidate = sorted_dets[i]
                cx1, cy1, cx2, cy2 = candidate["bbox"]
                candidate_height = cy2 - cy1
                
                # Check 1: Vertical proximity (candidate top close to current bottom)
                # Gap between current bottom (y2) and candidate top (cy1)
                gap_y = dir_gap = cy1 - y2
                
                # Allow slight negative gap (overlap) or positive gap within tolerance
                avg_h = (current_height + candidate_height) / 2
                allowed_gap = avg_h * y_gap_tolerance
                
                if gap_y < allowed_gap:
                    # Check 2: Horizontal Overlap
                    # Do they share X-space?
                    # Overlap X
                    ox = max(0, min(x2, cx2) - max(x1, cx1))
                    # Union X width
                    ux = max(x2, cx2) - min(x1, cx1)
                    
                    # If they overlap significantly relative to the smaller width
                    min_w = min(x2-x1, cx2-cx1)
                    
                    if ox > 0: # Simple overlap check is usually enough for subtitles
                         # Merge!
                        nx1 = min(x1, cx1)
                        ny1 = min(y1, cy1)
                        nx2 = max(x2, cx2)
                        ny2 = max(y2, cy2)
                        
                        # Update current
                        x1, y1, x2, y2 = nx1, ny1, nx2, ny2
                        current["bbox"] = [x1, y1, x2, y2]
                        current["text"] += "\n" + candidate["text"]
                        current["score"] = max(current["score"], candidate["score"])
                        
                        # Remove candidate
                        sorted_dets.pop(i)
                        continue # Stay at i=0 to check next candidates against new merged block
                
                i += 1
            
            merged.append(current)
            
        return merged

    def _cluster_lines(self, detections: List[Dict[str, Any]], y_tolerance: float = 0.5) -> List[Dict[str, Any]]:
        """Cluster horizontally adjacent text boxes into lines.
        
        Args:
            detections: List of detection dicts (bbox=[x1,y1,x2,y2], score, text)
            y_tolerance: Vertical tolerance as fraction of box height to consider same line
            
        Returns:
            List of merged detections
        """
        if not detections:
            return []
            
        # Sort by y1 then x1
        sorted_dets = sorted(detections, key=lambda d: (d["bbox"][1], d["bbox"][0]))
        merged = []
        
        while sorted_dets:
            current = sorted_dets.pop(0)
            x1, y1, x2, y2 = current["bbox"]
            current_height = y2 - y1
            
            # Try to merge with remaining detections
            i = 0
            while i < len(sorted_dets):
                candidate = sorted_dets[i]
                cx1, cy1, cx2, cy2 = candidate["bbox"]
                candidate_height = cy2 - cy1
                
                # Check vertical alignment (center points should be close)
                center_y = (y1 + y2) / 2
                cand_center_y = (cy1 + cy2) / 2
                avg_height = (current_height + candidate_height) / 2
                
                if abs(center_y - cand_center_y) < (avg_height * y_tolerance):
                    # Check horizontal proximity (allow small gap or overlap)
                    # Gap threshold: 50.0 * height (Infinite bridge: merge EVERYTHING on the same line)
                    # This ensures that "THE   I    TANT" becomes one solid bar, hiding missing letters.
                    gap = max(0, cx1 - x2) 
                    overlap = max(0, min(x2, cx2) - max(x1, cx1))
                    
                    if gap < (avg_height * 50.0) or overlap > 0:
                        # Merge
                        x1 = min(x1, cx1)
                        y1 = min(y1, cy1)
                        x2 = max(x2, cx2)
                        y2 = max(y2, cy2)
                        
                        # Merge text and update score
                        current["text"] += " " + candidate.get("text", "")
                        # Weighted score could be better, but simple average is fine for now
                        current["score"] = (current["score"] + candidate["score"]) / 2
                        current["bbox"] = [x1, y1, x2, y2]
                        current_height = y2 - y1
                        
                        sorted_dets.pop(i)
                        continue
                
                i += 1
            
            merged.append(current)
            
        return merged

    def _detect_with_auto_language(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Try multiple languages and pick the best one."""
        best_detections = []
        best_score = 0.0
        best_count = -1  # Start at -1 to distinguish from key loop
        
        for lang in self.languages:
            detections = self._detect_single_language(frame, lang)
            # We want to check all languages, even if one returns empty, 
            # but usually empty means bad language match unless frame is empty.
            
            count = len(detections)
            avg_score = sum(d["score"] for d in detections) / count if count > 0 else 0.0
            
            # Pick language with most detections, or highest avg score if tied
            if count > best_count or (count == best_count and avg_score > best_score):
                best_detections = detections
                best_count = count
                best_score = avg_score
        
        return best_detections

    def _detect_single_language(self, frame: np.ndarray, lang: str) -> List[Dict[str, Any]]:
        """Detect text in a single language."""
        ocr = self._get_ocr(lang)
        
        # Serialize access to shared PaddleOCR instance to prevent threading issues
        with _OCR_LOCK:
            results = ocr.ocr(frame)
            
        detections: List[Dict[str, Any]] = []
        
        # Handle None or empty results
        if not results:
            return detections
        
        if isinstance(results, dict):
            # Handle new dictionary-style output (PP-Structure/PaddleX format)
            # Keys: rec_texts, rec_scores, dt_polys
            texts = results.get("rec_texts", [])
            scores = results.get("rec_scores", [])
            polys = results.get("dt_polys", [])
            
            if not texts or not polys or len(texts) != len(polys):
                # Try specific box format if polys is missing
                # Fallback to empty if not parseable
                return detections
                
            for i, text in enumerate(texts):
                try:
                    score = float(scores[i]) if i < len(scores) else 0.0
                    poly = polys[i]
                    
                    # Poly is usually [ [x1, y1], [x2, y2], ... ] numpy array
                    if isinstance(poly, np.ndarray):
                        xs = poly[:, 0]
                        ys = poly[:, 1]
                        x1, x2 = float(xs.min()), float(xs.max())
                        y1, y2 = float(ys.min()), float(ys.max())
                    else:
                        continue
                        
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "text": str(text),
                        "score": score,
                    })
                except Exception as e:
                    print(f"[WARN] Failed to parse dict item {i}: {e}")
                    continue
            return detections
            
        # Handle list-based output (Legacy/standard OCR)
        # Handle empty list or None in first element
        if not results:
             print("[DEBUG] Empty results list")
             return detections
        if not results[0]:
             print("[DEBUG] Empty first result")
             return detections
        
        if isinstance(results[0], dict):
             res_dict = results[0]
             # Reuse logic
             texts = res_dict.get("rec_texts", [])
             scores = res_dict.get("rec_scores", [])
             polys = res_dict.get("dt_polys", [])
             
             for i, text in enumerate(texts):
                try:
                    score = float(scores[i]) if i < len(scores) else 0.0
                    poly = polys[i]
                    if isinstance(poly, np.ndarray):
                        xs = poly[:, 0]
                        ys = poly[:, 1]
                        x1, x2 = float(xs.min()), float(xs.max())
                        y1, y2 = float(ys.min()), float(ys.max())
                        
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "text": str(text),
                            "score": score,
                        })
                    else:
                        print(f"[DEBUG] Poly {i} is not ndarray: {type(poly)}")
                except Exception as e:
                    print(f"[DEBUG] Exception parsing item {i}: {e}")
                    continue
             return detections

        # Standard list of lists format
        for item in results[0]:
            try:
                # Skip non-list/tuple items
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                
                bbox, text_info = item
                
                # Validate bbox
                if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                    continue
                
                # Validate text_info
                if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
                    continue
                
                text, score = text_info
                x_coords = [pt[0] for pt in bbox]
                y_coords = [pt[1] for pt in bbox]
                
                detections.append({
                    "bbox": [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
                    "text": str(text),
                    "score": float(score),
                })
            except (ValueError, TypeError, IndexError, AttributeError):
                # Skip malformed detections
                continue
        
        return detections

    def _filter_detections(self, detections: List[Dict[str, Any]], min_score: float, 
                          min_size_px: int, min_size_ratio: float, 
                          frame_width: int, frame_height: int,
                          subtitle_region_height: float = 1.0,
                          subtitle_region_vertical: str = "bottom") -> List[Dict[str, Any]]:
        """Filter detections by score, size, and region."""
        filtered = []
        min_y = int(frame_height * (1.0 - subtitle_region_height))
        max_y = int(frame_height * subtitle_region_height)
        
        for det in detections:
            # Score filter
            if det["score"] < min_score:
                # print(f"[DEBUG] Dropped low score: {det['score']:.2f}")
                continue
            
            x1, y1, x2, y2 = det["bbox"]
            width = x2 - x1
            height = y2 - y1
            center_y = (y1 + y2) / 2
            
            # Pixel size filter
            if width < min_size_px or height < min_size_px:
                # print(f"[DEBUG] Dropped small: {width}x{height}")
                continue
            
            # Ratio filter
            width_ratio = width / frame_width
            height_ratio = height / frame_height
            if width_ratio < min_size_ratio or height_ratio < min_size_ratio:
                # print(f"[DEBUG] Dropped ratio: {width_ratio:.3f}")
                continue
                
            # Region filter: Center of box must be within subtitle region
            if subtitle_region_height < 1.0:
                if subtitle_region_vertical == "bottom":
                    if center_y < min_y:
                        # print(f"[DEBUG] Dropped region (bottom): y={center_y:.1f} < min_y={min_y:.1f}")
                        continue
                else: # top
                    if center_y > max_y:
                        # print(f"[DEBUG] Dropped region (top): y={center_y:.1f} > max_y={max_y:.1f}")
                        continue
            
            filtered.append(det)
        
        return filtered

    def _apply_nms(self, detections: List[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if not detections:
            return []
        
        # Sort by score descending
        detections = sorted(detections, key=lambda x: x["score"], reverse=True)
        keep = []
        
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping boxes
            detections = [
                det for det in detections 
                if self._bbox_iou(best["bbox"], det["bbox"]) < iou_threshold
            ]
        
        return keep

    @staticmethod
    def _bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - inter_area
        
        return inter_area / union if union > 0 else 0.0


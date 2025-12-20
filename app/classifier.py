from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TextTrack:
    track_id: int
    boxes: List[tuple[int, int, int, int]] = field(default_factory=list)
    frames: List[int] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    classification: Optional[str] = None
    stroke_detected: bool = False
    merged_bbox: Optional[tuple[int, int, int, int]] = None

    def add(self, bbox: tuple[int, int, int, int], frame_idx: int, text: str) -> None:
        self.boxes.append(bbox)
        self.frames.append(frame_idx)
        self.texts.append(text)
        
        # Update merged_bbox
        if self.merged_bbox is None:
            self.merged_bbox = bbox
        else:
            x1 = min(self.merged_bbox[0], bbox[0])
            y1 = min(self.merged_bbox[1], bbox[1])
            x2 = max(self.merged_bbox[2], bbox[2])
            y2 = max(self.merged_bbox[3], bbox[3])
            self.merged_bbox = (x1, y1, x2, y2)

    @property
    def lifetime(self) -> int:
        if not self.frames:
            return 0
        return self.frames[-1] - self.frames[0] + 1

    @property
    def avg_y(self) -> float:
        if not self.boxes:
            return 0.0
        return sum((b[1] + b[3]) / 2 for b in self.boxes) / len(self.boxes)

    @property
    def avg_height(self) -> float:
        if not self.boxes:
            return 0.0
        return sum((b[3] - b[1]) for b in self.boxes) / len(self.boxes)

    @property
    def text_len(self) -> float:
        if not self.texts:
            return 0.0
        return sum(len(t) for t in self.texts) / len(self.texts)


class SubtitleClassifier:
    """Rule-based classifier tuned for trading UIs."""

    def __init__(self, frame_height: int) -> None:
        self.frame_height = frame_height

    def classify(self, track: TextTrack, subtitle_intensity_threshold: Optional[float] = None, subtitle_region_height: float = 0.3, frame_width: Optional[int] = None) -> str:
        # Define subtitle band at bottom of screen based on region height
        min_y = self.frame_height * (1.0 - subtitle_region_height)
        
        # Check if track is within the subtitle band
        # We check if center point (avg_y) is in the lower region
        in_vertical_band = track.avg_y >= min_y
        
        score_gate = True
        if subtitle_intensity_threshold is not None:
            score_gate = self._score_gate(track, subtitle_intensity_threshold)

        # Log detailed reason for debug
        problems = []
        if track.lifetime >= 3000:
            problems.append(f"lifetime({track.lifetime})>=3000")
        if not in_vertical_band:
            problems.append(f"y({track.avg_y:.1f})<{min_y:.1f}")
        # Changed from <= 2 to < 1 to allow short words like "I", "A", "NO", "YC".
        # We rely on centering and vertical position to filter noise.
        if track.text_len < 1:
            problems.append(f"len({track.text_len:.1f})<1")
        if not (track.stroke_detected or score_gate):
            problems.append("no_stroke")
            
        # Centering check (if frame_width provided)
        if frame_width:
             # Calculate track center X
             if track.merged_bbox:
                 mid_x = (track.merged_bbox[0] + track.merged_bbox[2]) / 2
             elif track.boxes:
                 # fallback to last box
                 b = track.boxes[-1]
                 mid_x = (b[0] + b[2]) / 2
             else:
                 mid_x = 0
             
             center_diff = abs(mid_x - (frame_width / 2))
             # Threshold: Allow 10% deviation from center (STRICT).
             # Was 25%, but that allowed side UIs. Subtitles are usually <5% off.
             if center_diff > (frame_width * 0.10):
                  problems.append(f"off_center({center_diff:.0f})")

        if not problems:
            track.classification = "subtitle"
        else:
            track.classification = "ui"
            # Optional: print for debug, or could attach to track for visualization
            print(f"[DEBUG] Track {track.track_id} rejected: {', '.join(problems)} Text: '{track.texts[-1] if track.texts else ''}'")
            
        return track.classification

    @staticmethod
    def _score_gate(track: TextTrack, threshold: float) -> bool:
        """Fallback gate when explicit stroke detection is not available."""
        if not track.texts:
            return False
        avg_density = sum(len(text.replace(" ", "")) for text in track.texts) / len(track.texts)
        return avg_density >= threshold

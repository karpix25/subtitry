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

    def add(self, bbox: tuple[int, int, int, int], frame_idx: int, text: str) -> None:
        self.boxes.append(bbox)
        self.frames.append(frame_idx)
        self.texts.append(text)

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

    def classify(self, track: TextTrack, subtitle_intensity_threshold: Optional[float] = None) -> str:
        band = 0.3 * self.frame_height
        in_vertical_band = track.avg_y < band or track.avg_y > (self.frame_height - band)
        score_gate = True
        if subtitle_intensity_threshold is not None:
            score_gate = self._score_gate(track, subtitle_intensity_threshold)

        if (
            track.lifetime < 120
            and in_vertical_band
            and track.text_len > 8
            and (track.stroke_detected or score_gate)
        ):
            track.classification = "subtitle"
        else:
            track.classification = "ui"
        return track.classification

    @staticmethod
    def _score_gate(track: TextTrack, threshold: float) -> bool:
        """Fallback gate when explicit stroke detection is not available."""
        if not track.texts:
            return False
        avg_density = sum(len(text.replace(" ", "")) for text in track.texts) / len(track.texts)
        return avg_density >= threshold

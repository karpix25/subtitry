from __future__ import annotations

import numpy as np

from .classifier import TextTrack


class MaskBuilder:
    def build_mask(self, frame_shape: tuple[int, int], tracks: list[TextTrack], frame_idx: int) -> np.ndarray:
        mask = np.zeros(frame_shape, dtype=np.uint8)
        for track in tracks:
            if track.classification != "subtitle":
                continue
            if frame_idx not in track.frames:
                continue
            bbox = track.boxes[track.frames.index(frame_idx)]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            mask[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)] = 255
        return mask

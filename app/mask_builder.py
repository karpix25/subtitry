from __future__ import annotations

import numpy as np

from .classifier import TextTrack


class MaskBuilder:
    def build_mask(self, frame_shape: tuple[int, int], tracks: list[TextTrack], frame_idx: int) -> np.ndarray:
        mask = np.zeros(frame_shape, dtype=np.uint8)
        for track in tracks:
            if track.classification != "subtitle":
                continue
            if not track.frames:
                continue
            
            # Check recency: persistence for 30 frames (~1 sec)
            last_seen = track.frames[-1]
            if (frame_idx - last_seen) > 30:
                continue

            # Strict Inpainting: use the EXACT box for the current frame if available.
            # Only use specific box to match user's "tight" requirement.
            bbox = None
            if frame_idx in track.frames:
                 bbox = track.boxes[track.frames.index(frame_idx)]
            else:
                 # Persistence: use the last known box
                 bbox = track.boxes[-1]
                
            x1, y1, x2, y2 = [int(v) for v in bbox]
            mask[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)] = 255
        return mask

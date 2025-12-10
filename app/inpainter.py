from __future__ import annotations

import cv2
import numpy as np


class Inpainter:
    def inpaint(self, frame: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
        if mask.max() == 0:
            return frame
        radius = max(1, min(radius, 12))
        return cv2.inpaint(frame, mask, radius, cv2.INPAINT_NS)

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

import numpy as np


class TextDetector:
    """Thin wrapper around PaddleOCR for CPU-only inference."""

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_ocr(lang: str) -> Any:  # pragma: no cover - expensive import
        from paddleocr import PaddleOCR  # type: ignore

        return PaddleOCR(use_gpu=False, lang=lang)

    def detect_text(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        ocr = self._get_ocr(self.lang)
        results = ocr.ocr(frame, cls=False)
        detections: List[Dict[str, Any]] = []
        for block in results:
            for bbox, (text, score) in block:
                x_coords = [pt[0] for pt in bbox]
                y_coords = [pt[1] for pt in bbox]
                detections.append(
                    {
                        "bbox": [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
                        "text": text,
                        "score": float(score),
                    }
                )
        return detections

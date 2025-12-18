import os
import tarfile
import requests
from pathlib import Path

# PaddleOCR default cache structure
BASE_DIR = Path("/root/.paddleocr/whl")

MODELS = {
    "det": {
        "en": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
        "ml": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar",
    },
    "rec": {
        "en": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
        "ru": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_rec_infer.tar",
    },
    "cls": {
        "ch": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"
    }
}

# ... (omitted) ...

def main():
    # 1. Detection (English/Common)
    download_and_extract(MODELS["det"]["en"], "det", "en")
    # 1.1 Detection (Multilingual for 'ru' support)
    download_and_extract(MODELS["det"]["ml"], "det", "ml")
    
    # 2. Recognition (English)
    download_and_extract(MODELS["rec"]["en"], "rec", "en")
    
    # 3. Recognition (Russian/Multilingual)
    # PaddleOCR(lang='ru') typically maps to 'multilingual' structure in v3
    download_and_extract(MODELS["rec"]["ru"], "rec", "multilingual")
    
    # 4. Classification
    # CLS model is usually language-agnostic in path (or uses flat structure under cls)
    # Error expected: .../whl/cls/ch_ppocr_mobile_v2.0_cls_infer/...
    download_and_extract(MODELS["cls"]["ch"], "cls", "")

if __name__ == "__main__":
    main()

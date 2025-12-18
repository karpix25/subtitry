import os
import tarfile
try:
    import requests
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import requests: {e}")
    exit(1)
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

# Mapping for 'ru' lang in directory structure
# When lang='ru', Paddle looks in 'rec/multilingual/Multilingual_PP-OCRv3_rec_infer' 
# OR 'rec/cyrillic/...' depending on version.
# PP-OCRv3 usually uses 'Multilingual' for non-Latin.
# We will download to standard locations.

def download_and_extract(url, category, subdir):
    filename = url.split("/")[-1]
    model_name = filename.replace(".tar", "")
    
    # PaddleOCR Structure: BASE / category / subdir / model_name / filename
    # Example: .../whl/det/en/en_PP-OCRv3_det_infer/en_PP-OCRv3_det_infer.tar
    
    parent_dir = BASE_DIR / category / subdir
    model_dir = parent_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    tar_path = model_dir / filename
    
    print(f"Downloading {url} to {tar_path}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(tar_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
                    
        print(f"Extracting {tar_path}...")
        # Extract to parent_dir because tar usually contains the model_name directory
        # e.g. tar contains "en_PP-OCRv3_det_infer/..." and we want it in ".../det/en/en_PP-OCRv3_det_infer/"
        # extracting to parent_dir (".../det/en") will create/merge into ".../det/en/en_PP-OCRv3_det_infer/"
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=parent_dir)
            
        print("Done.")
    except Exception as e:
        print(f"Error processing {url}: {e}")
        # Don't exit, try next model? No, strict.
        raise e

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

import os
import tarfile
import urllib.request
import urllib.error
import sys
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

def download_and_extract(url, category, subdir):
    print(f"[START] Processing {url}")
    try:
        filename = url.split("/")[-1]
        model_name = filename.replace(".tar", "")
        
        # PaddleOCR Structure: BASE / category / subdir / model_name / filename
        
        parent_dir = BASE_DIR / category
        if subdir:
            parent_dir = parent_dir / subdir
            
        model_dir = parent_dir / model_name
        
        print(f"[INFO] Creating directory: {model_dir}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        tar_path = model_dir / filename
        
        print(f"[DOWNLOAD] Downloading to {tar_path}...")
        
        # Use urllib to avoid external dependencies
        try:
            with urllib.request.urlopen(url) as response:
                content_length = response.getheader('Content-Length')
                print(f"[DOWNLOAD] Content-Length: {content_length}")
                
                with open(tar_path, 'wb') as f:
                    downloaded = 0
                    block_size = 8192
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded += len(buffer)
                        f.write(buffer)
                        
            print(f"[DOWNLOAD] Completed. Size: {tar_path.stat().st_size} bytes")
            
        except urllib.error.URLError as e:
            print(f"[ERROR] Failed to download {url}: {e}")
            raise e

        print(f"[EXTRACT] Extracting {tar_path} to {parent_dir}...")
        try:
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=parent_dir)
            print("[EXTRACT] Done.")
        except Exception as e:
            print(f"[ERROR] Failed to extract tar: {e}")
            raise e
            
    except Exception as e:
        print(f"[CRITICAL] Error in download_and_extract: {e}")
        # Re-raise to ensure script exits with non-zero code
        raise e

def main():
    print("--- Starting PaddleOCR Model Download ---")
    try:
        # 1. Detection (English/Common) - v3
        download_and_extract(MODELS["det"]["en"], "det", "en")
        
        # 1.1 Detection (Multilingual for 'ru' support) - v3
        download_and_extract(MODELS["det"]["ml"], "det", "ml")
        
        # 2. Recognition (English) - v4
        download_and_extract(MODELS["rec"]["en"], "rec", "en")
        
        # 3. Recognition (Russian/Multilingual) - v3
        download_and_extract(MODELS["rec"]["ru"], "rec", "multilingual")
        
        # 4. Classification - v2 (No subdir)
        download_and_extract(MODELS["cls"]["ch"], "cls", "")
        
        print("--- All Downloads Completed Successfully ---")
        
    except Exception as e:
        print(f"--- FAILURE: {e} ---")
        sys.exit(1)

if __name__ == "__main__":
    main()

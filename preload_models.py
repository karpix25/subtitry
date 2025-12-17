from paddleocr import PaddleOCR
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preload():
    logger.info("Pre-loading PaddleOCR models...")
    # Initialize for English (force download)
    PaddleOCR(lang='en', show_log=False)
    logger.info("English model loaded.")
    
    # Initialize for Russian (force download)
    PaddleOCR(lang='ru', show_log=False)
    logger.info("Russian model loaded.")
    
    logger.info("All models pre-loaded successfully.")

if __name__ == "__main__":
    preload()

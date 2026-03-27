import gc
import cv2
import torch
import easyocr

# ================= OCR READER =================

_reader = None
_ocr_call_count = 0

# reinitialise reader every 500 OCR calls to flush EasyOCR's internal cache
# this is the main fix for the stair-stepping CPU RAM growth
OCR_REINIT_INTERVAL = 500

def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=True)
    return _reader

def _reinit_reader():
    global _reader
    _reader = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _reader = easyocr.Reader(['en'], gpu=True)
    print("[OCR] Reader reinitialised to flush memory cache")

# ================= PREPROCESSING =================

def preprocess_for_led(img):
    """
    Enhance LED-style images for OCR
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

    # Only upscale if crop is genuinely small — avoids processing 4x pixels every frame
    h, w = enhanced.shape[:2]
    if h < 32 or w < 64:
        enhanced = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return enhanced

# ================= OCR =================

def run_ocr_on_image(img_or_path):
    """
    Accepts either:
      - a numpy array (cv2 image)
      - a file path string
    Returns recognized text
    """
    global _ocr_call_count

    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path)
        if img is None:
            return {"text": ""}
    else:
        img = img_or_path

    pre = preprocess_for_led(img)

    # reinitialise reader every 500 calls to free EasyOCR internal cache
    _ocr_call_count += 1
    if _ocr_call_count % OCR_REINIT_INTERVAL == 0:
        _reinit_reader()

    results = _get_reader().readtext(pre)

    texts = [text for _, text, score in results if score > 0.3]
    final_text = " ".join(texts).strip()

    return {"text": final_text}
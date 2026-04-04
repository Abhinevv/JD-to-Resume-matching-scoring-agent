"""
utils/file_extractor.py
-----------------------
Handles text extraction from PDF files (via pdfplumber) and
image files (via pytesseract OCR).
"""

import os
import io
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file using pdfplumber.
    Handles multi-page PDFs — all pages are concatenated.
    Returns empty string on failure.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber not installed. Run: pip install pdfplumber")
        return ""

    text_parts = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                else:
                    logger.debug(f"Page {page_num} returned no text (may be image-based).")
    except Exception as e:
        logger.error(f"Failed to extract PDF '{file_path}': {e}")
        return ""

    return "\n".join(text_parts)


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """
    Same as extract_text_from_pdf but accepts raw bytes (for FastAPI uploads).
    """
    try:
        import pdfplumber
    except ImportError:
        return ""

    text_parts = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        logger.error(f"Failed to extract PDF from bytes: {e}")
        return ""

    return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# Image OCR extraction
# ---------------------------------------------------------------------------

def extract_text_from_image(file_path: str) -> str:
    """
    Extract text from an image file using pytesseract (Tesseract OCR).
    Supports PNG, JPEG, TIFF, BMP, etc.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.error("pytesseract / Pillow not installed.")
        return ""

    try:
        img = Image.open(file_path)
        # Optional: convert to greyscale for better OCR accuracy
        img = img.convert("L")
        text = pytesseract.image_to_string(img, lang="eng")
        return text
    except Exception as e:
        logger.error(f"OCR failed for '{file_path}': {e}")
        return ""


def extract_text_from_image_bytes(file_bytes: bytes, filename: str = "") -> str:
    """
    OCR on raw image bytes (for FastAPI uploads).
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""

    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("L")
        return pytesseract.image_to_string(img, lang="eng")
    except Exception as e:
        logger.error(f"OCR bytes failed ({filename}): {e}")
        return ""


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def extract_text(file_path: str) -> str:
    """
    Auto-detect file type and extract text accordingly.
    Supported: .pdf, .png, .jpg, .jpeg, .tiff, .bmp, .gif, .webp, .txt
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}:
        return extract_text_from_image(file_path)
    elif ext in {".txt", ".md"}:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Could not read text file '{file_path}': {e}")
            return ""
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return ""


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """
    Dispatcher for uploaded bytes (FastAPI UploadFile scenario).
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf_bytes(file_bytes)
    elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}:
        return extract_text_from_image_bytes(file_bytes, filename)
    elif ext in {".txt", ".md"}:
        try:
            return file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return ""


def save_uploaded_file(file_bytes: bytes, filename: str, save_dir: str = "data/raw_resumes") -> str:
    """
    Persist an uploaded file to disk and return its absolute path.
    Creates the directory if it doesn't exist.
    """
    os.makedirs(save_dir, exist_ok=True)
    dest = os.path.join(save_dir, filename)
    with open(dest, "wb") as f:
        f.write(file_bytes)
    logger.info(f"Saved uploaded file to: {dest}")
    return dest

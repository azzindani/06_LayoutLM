"""
OCR module - lazy imports.
"""

__all__ = ['BaseOCREngine', 'OCRResult', 'EasyOCREngine', 'create_ocr_engine']


def __getattr__(name):
    if name in ('BaseOCREngine', 'OCRResult'):
        from core.ocr.base import BaseOCREngine, OCRResult
        return BaseOCREngine if name == 'BaseOCREngine' else OCRResult
    if name in ('EasyOCREngine', 'create_ocr_engine'):
        from core.ocr.easyocr_engine import EasyOCREngine, create_ocr_engine
        return EasyOCREngine if name == 'EasyOCREngine' else create_ocr_engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

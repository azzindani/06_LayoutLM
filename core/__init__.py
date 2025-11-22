"""
Core module - lazy imports to avoid heavy dependencies during CI.
"""

__all__ = [
    'DocumentProcessor',
    'get_processor',
    'create_ocr_engine',
    'load_model',
    'run_inference',
]


def __getattr__(name):
    if name == 'DocumentProcessor':
        from core.pipeline.document_processor import DocumentProcessor
        return DocumentProcessor
    if name == 'get_processor':
        from core.pipeline.document_processor import get_processor
        return get_processor
    if name == 'create_ocr_engine':
        from core.ocr.easyocr_engine import create_ocr_engine
        return create_ocr_engine
    if name == 'load_model':
        from core.model.loader import load_model
        return load_model
    if name == 'run_inference':
        from core.model.inference import run_inference
        return run_inference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

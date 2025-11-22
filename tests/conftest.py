"""
Pytest configuration and shared fixtures.
"""

import pytest
from PIL import Image


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = Image.new('RGB', (800, 600), color='white')
    return img


@pytest.fixture
def sample_image_with_text():
    """Create a test image with text."""
    from PIL import ImageDraw

    img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Name:", fill='black')
    draw.text((150, 50), "John Doe", fill='black')
    draw.text((50, 100), "Date:", fill='black')
    draw.text((150, 100), "2024-01-15", fill='black')
    return img


@pytest.fixture
def sample_ocr_results():
    """Create sample OCR results."""
    from core.ocr.base import OCRResult

    return [
        OCRResult(text="Name:", bbox=(50, 50, 120, 70), confidence=0.95),
        OCRResult(text="John Doe", bbox=(150, 50, 280, 70), confidence=0.92),
        OCRResult(text="Date:", bbox=(50, 100, 100, 120), confidence=0.93),
        OCRResult(text="2024-01-15", bbox=(150, 100, 280, 120), confidence=0.91),
    ]


@pytest.fixture
def sample_inference_result():
    """Create sample inference result."""
    return {
        "predictions": [0, 3, 5, 0, 3, 5],
        "confidence_scores": [0.9, 0.95, 0.92, 0.88, 0.93, 0.91],
        "word_ids": [None, 0, 1, None, 2, 3],
        "words": ["Name:", "John Doe", "Date:", "2024-01-15"],
        "boxes": [
            (50, 50, 120, 70),
            (150, 50, 280, 70),
            (50, 100, 100, 120),
            (150, 100, 280, 120)
        ],
        "normalized_boxes": [
            [83, 125, 200, 175],
            [250, 125, 466, 175],
            [83, 250, 166, 300],
            [250, 250, 466, 300]
        ],
        "image_size": (600, 400)
    }


@pytest.fixture
def sample_processing_result():
    """Create sample processing result."""
    return {
        "status": "success",
        "processing_time_ms": 1234.56,
        "results": [
            {
                "page": 1,
                "entities": [
                    {
                        "text": "Name:",
                        "label": "QUESTION",
                        "confidence": 0.95,
                        "bbox": {"x1": 50, "y1": 50, "x2": 120, "y2": 70}
                    },
                    {
                        "text": "John Doe",
                        "label": "ANSWER",
                        "confidence": 0.92,
                        "bbox": {"x1": 150, "y1": 50, "x2": 280, "y2": 70}
                    }
                ]
            }
        ],
        "metadata": {
            "model_version": "layoutlmv3-funsd-v1",
            "ocr_engine": "easyocr",
            "image_size": [600, 400]
        }
    }

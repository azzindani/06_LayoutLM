#!/usr/bin/env python
"""
Test OCR engine functionality.

Run with: python -m core.ocr.test_ocr
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_ocr_result():
    """Test OCRResult dataclass."""
    from core.ocr.base import OCRResult

    result = OCRResult(
        text="Hello World",
        bbox=(10, 20, 100, 50),
        confidence=0.95
    )

    assert result.text == "Hello World"
    assert result.bbox == (10, 20, 100, 50)
    assert result.confidence == 0.95
    print("  ✓ OCRResult dataclass works correctly")


def test_easyocr_engine():
    """Test EasyOCR engine initialization and extraction."""
    try:
        from PIL import Image, ImageDraw
        from core.ocr.easyocr_engine import EasyOCREngine

        # Create test image with text
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "Test Document", fill='black')
        draw.text((20, 50), "Hello World", fill='black')

        # Initialize engine
        engine = EasyOCREngine(languages=['en'], gpu=False)
        engine.initialize()
        print("  ✓ EasyOCR engine initialized")

        # Extract text
        results = engine.extract_text(img)
        print(f"  ✓ Extracted {len(results)} text regions")

        for r in results:
            print(f"    - '{r.text}' at {r.bbox} (conf: {r.confidence:.2f})")

        engine.shutdown()
        print("  ✓ EasyOCR engine shutdown")

    except ImportError as e:
        print(f"  ⚠ Skipped (missing dependency): {e}")


def test_ocr_factory():
    """Test OCR engine factory."""
    try:
        from core.ocr.easyocr_engine import create_ocr_engine

        engine = create_ocr_engine("easyocr", languages=['en'], gpu=False)
        assert engine.name == "easyocr"
        print("  ✓ OCR factory creates correct engine")

    except ImportError as e:
        print(f"  ⚠ Skipped (missing dependency): {e}")


def main():
    print("=" * 60)
    print("OCR ENGINE TEST")
    print("=" * 60)
    print()

    print("1. Testing OCRResult dataclass...")
    test_ocr_result()
    print()

    print("2. Testing EasyOCR engine...")
    test_easyocr_engine()
    print()

    print("3. Testing OCR factory...")
    test_ocr_factory()
    print()

    print("=" * 60)
    print("OCR ENGINE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

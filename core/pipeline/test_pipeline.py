#!/usr/bin/env python
"""
Test document processing pipeline.

Run with: python -m core.pipeline.test_pipeline
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_image_utils():
    """Test image utilities."""
    try:
        from PIL import Image
        from core.pipeline.image_utils import (
            load_image, validate_image, resize_image, preprocess_image
        )

        # Create test image
        img = Image.new('RGB', (800, 600), color='white')

        # Test validation
        validate_image(img)
        print("  ✓ Image validation works")

        # Test resize
        large_img = Image.new('RGB', (3000, 2000), color='white')
        resized = resize_image(large_img, max_size=1000)
        assert resized.size[0] <= 1000 and resized.size[1] <= 1000
        print(f"  ✓ Image resize: {large_img.size} -> {resized.size}")

        # Test preprocess
        processed = preprocess_image(img)
        assert processed.mode == 'RGB'
        print("  ✓ Image preprocess works")

    except ImportError as e:
        print(f"  ⚠ Skipped (missing dependency): {e}")


def test_export():
    """Test export utilities."""
    from core.pipeline.export import export_json, export_csv, export_xml

    test_result = {
        "status": "success",
        "processing_time_ms": 1234.56,
        "results": [{
            "page": 1,
            "entities": [{
                "text": "Name:",
                "label": "QUESTION",
                "confidence": 0.95,
                "bbox": {"x1": 10, "y1": 20, "x2": 80, "y2": 40}
            }]
        }],
        "metadata": {
            "model_version": "v1",
            "ocr_engine": "easyocr"
        }
    }

    # Test JSON
    json_out = export_json(test_result)
    assert "success" in json_out
    print(f"  ✓ JSON export: {len(json_out)} chars")

    # Test CSV
    csv_out = export_csv(test_result)
    assert "page" in csv_out
    print(f"  ✓ CSV export: {len(csv_out)} chars")

    # Test XML
    xml_out = export_xml(test_result)
    assert "<document>" in xml_out
    print(f"  ✓ XML export: {len(xml_out)} chars")


def test_document_processor():
    """Test full document processor pipeline."""
    try:
        from PIL import Image, ImageDraw
        from core.pipeline.document_processor import DocumentProcessor

        # Create test image with text
        img = Image.new('RGB', (600, 400), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "Name:", fill='black')
        draw.text((150, 50), "John Doe", fill='black')
        draw.text((50, 100), "Date:", fill='black')
        draw.text((150, 100), "2024-01-15", fill='black')

        # Initialize processor
        processor = DocumentProcessor()
        print("  Initializing processor (loading models)...")
        processor.initialize()
        print(f"  ✓ Processor initialized on {processor._device.device_name}")

        # Process image
        print("  Processing test image...")
        start = time.time()
        result = processor.process_image(img)
        elapsed = (time.time() - start) * 1000

        print(f"  ✓ Processing complete in {elapsed:.2f}ms")
        print(f"    Status: {result['status']}")
        print(f"    Entities found: {len(result['results'][0]['entities'])}")

        for entity in result['results'][0]['entities']:
            print(f"      - {entity['text']}: {entity['label']} ({entity['confidence']:.2f})")

        # Test batch processing
        print("  Testing batch processing...")
        results = processor.process_batch([img, img])
        print(f"  ✓ Batch processed {len(results)} images")

        processor.shutdown()
        print("  ✓ Processor shutdown")

    except ImportError as e:
        print(f"  ⚠ Skipped (missing dependency): {e}")


def main():
    print("=" * 60)
    print("PIPELINE TEST")
    print("=" * 60)
    print()

    print("1. Testing image utilities...")
    test_image_utils()
    print()

    print("2. Testing export utilities...")
    test_export()
    print()

    print("3. Testing document processor...")
    test_document_processor()
    print()

    print("=" * 60)
    print("PIPELINE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

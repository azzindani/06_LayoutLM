#!/usr/bin/env python
"""
Test model loading and inference.

Run with: python -m core.model.test_model
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_label_mapping():
    """Test label ID to name mapping."""
    from core.model.loader import get_label_mapping

    labels = get_label_mapping()

    assert 0 in labels
    assert labels[0] == "O"
    assert labels[3] == "B-QUESTION"
    assert labels[5] == "B-ANSWER"

    print(f"  ✓ Label mapping: {len(labels)} labels")
    for id, name in labels.items():
        print(f"    {id}: {name}")


def test_model_loading():
    """Test model and processor loading."""
    try:
        from core.model.loader import load_model, clear_model_cache

        print("  Loading model (this may take a while on first run)...")
        model, processor = load_model(
            "nielsr/layoutlmv3-finetuned-funsd",
            device="cpu"
        )

        print(f"  ✓ Model loaded: {type(model).__name__}")
        print(f"  ✓ Processor loaded: {type(processor).__name__}")

        # Test caching
        model2, processor2 = load_model(
            "nielsr/layoutlmv3-finetuned-funsd",
            device="cpu"
        )
        assert model is model2, "Model caching failed"
        print("  ✓ Model caching works")

        clear_model_cache()
        print("  ✓ Model cache cleared")

    except ImportError as e:
        print(f"  ⚠ Skipped (missing dependency): {e}")


def test_inference():
    """Test model inference."""
    try:
        from PIL import Image, ImageDraw
        from core.ocr.base import OCRResult
        from core.model.inference import run_inference

        # Create test image
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "Name:", fill='black')
        draw.text((150, 50), "John Doe", fill='black')

        # Mock OCR results
        ocr_results = [
            OCRResult(text="Name:", bbox=(50, 50, 120, 70), confidence=0.95),
            OCRResult(text="John Doe", bbox=(150, 50, 280, 70), confidence=0.92),
        ]

        print("  Running inference...")
        result = run_inference(img, ocr_results, device="cpu")

        print(f"  ✓ Predictions: {len(result['predictions'])} tokens")
        print(f"  ✓ Words: {result['words']}")
        print(f"  ✓ Image size: {result['image_size']}")

    except ImportError as e:
        print(f"  ⚠ Skipped (missing dependency): {e}")


def test_postprocess():
    """Test post-processing utilities."""
    from core.model.postprocess import (
        Entity, unnormalize_bbox, format_output, process_predictions
    )

    # Test unnormalize_bbox
    bbox = unnormalize_bbox([500, 500, 1000, 1000], 800, 600)
    assert bbox == {"x1": 400, "y1": 300, "x2": 800, "y2": 600}
    print("  ✓ unnormalize_bbox works")

    # Test Entity
    entity = Entity(
        text="Test",
        label="QUESTION",
        confidence=0.95,
        bbox={"x1": 10, "y1": 20, "x2": 100, "y2": 50}
    )
    assert entity.text == "Test"
    print("  ✓ Entity dataclass works")

    # Test format_output
    output = format_output([entity], (800, 600), 1234.5)
    assert output["status"] == "success"
    assert len(output["results"][0]["entities"]) == 1
    print("  ✓ format_output works")


def main():
    print("=" * 60)
    print("MODEL TEST")
    print("=" * 60)
    print()

    print("1. Testing label mapping...")
    test_label_mapping()
    print()

    print("2. Testing post-processing...")
    test_postprocess()
    print()

    print("3. Testing model loading...")
    test_model_loading()
    print()

    print("4. Testing inference...")
    test_inference()
    print()

    print("=" * 60)
    print("MODEL TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

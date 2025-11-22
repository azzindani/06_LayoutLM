#!/usr/bin/env python
"""
Complete system integration test.

Run with: python -m core.test_complete_system
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config():
    """Test configuration system."""
    from config import get_config, reset_config

    reset_config()
    config = get_config()

    assert config.model.name == "nielsr/layoutlmv3-finetuned-funsd"
    assert config.hardware.device == "auto"
    assert config.api.port == 8000

    print("  ✓ Configuration loaded correctly")
    print(f"    Model: {config.model.name}")
    print(f"    Device: {config.hardware.device}")
    print(f"    API Port: {config.api.port}")
    print(f"    UI Port: {config.ui.server_port}")


def test_logging():
    """Test logging infrastructure."""
    from infrastructure.logger_utils import setup_logging, get_logger

    setup_logging(level="INFO", format_type="standard")
    logger = get_logger("test")
    logger.info("Test log message")
    print("  ✓ Logging system works")


def test_hardware_detection():
    """Test hardware detection."""
    try:
        from infrastructure.hardware_detection import detect_device

        device = detect_device("auto")
        print(f"  ✓ Hardware detected: {device.device_name}")
        print(f"    Type: {device.device_type}")
        print(f"    PyTorch device: {device.torch_device}")

    except ImportError as e:
        print(f"  ⚠ Skipped (missing dependency): {e}")


def test_exceptions():
    """Test custom exceptions."""
    from infrastructure.exceptions import (
        LayoutLMServiceError,
        InvalidImageError,
        ModelLoadError
    )

    try:
        raise InvalidImageError("Test error")
    except LayoutLMServiceError:
        pass  # Expected

    print("  ✓ Exception hierarchy works correctly")


def test_end_to_end():
    """Test complete end-to-end processing."""
    try:
        from PIL import Image, ImageDraw
        from core.pipeline.document_processor import DocumentProcessor
        from core.pipeline.export import export_json, export_csv

        print("  Creating test document...")
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)

        # Add form-like content
        draw.text((50, 50), "REGISTRATION FORM", fill='black')
        draw.text((50, 100), "First Name:", fill='black')
        draw.text((200, 100), "John", fill='black')
        draw.text((50, 140), "Last Name:", fill='black')
        draw.text((200, 140), "Doe", fill='black')
        draw.text((50, 180), "Email:", fill='black')
        draw.text((200, 180), "john.doe@example.com", fill='black')
        draw.text((50, 220), "Date:", fill='black')
        draw.text((200, 220), "2024-01-15", fill='black')

        # Process
        print("  Initializing processor...")
        processor = DocumentProcessor()
        processor.initialize()

        print("  Processing document...")
        start = time.time()
        result = processor.process_image(img)
        elapsed = (time.time() - start) * 1000

        # Verify result
        assert result["status"] == "success"
        entities = result["results"][0]["entities"]

        print(f"  ✓ Document processed in {elapsed:.2f}ms")
        print(f"    Found {len(entities)} entities:")

        for entity in entities:
            print(f"      [{entity['label']}] {entity['text']} (conf: {entity['confidence']:.2f})")

        # Test export
        json_output = export_json(result)
        csv_output = export_csv(result)

        print(f"  ✓ Exported to JSON ({len(json_output)} chars)")
        print(f"  ✓ Exported to CSV ({len(csv_output)} chars)")

        processor.shutdown()
        print("  ✓ Processor shutdown complete")

    except ImportError as e:
        print(f"  ⚠ Skipped (missing dependency): {e}")


def main():
    print("=" * 60)
    print("COMPLETE SYSTEM TEST")
    print("=" * 60)
    print()

    print("1. Testing configuration...")
    test_config()
    print()

    print("2. Testing logging...")
    test_logging()
    print()

    print("3. Testing hardware detection...")
    test_hardware_detection()
    print()

    print("4. Testing exceptions...")
    test_exceptions()
    print()

    print("5. Testing end-to-end processing...")
    test_end_to_end()
    print()

    print("=" * 60)
    print("COMPLETE SYSTEM TEST FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()

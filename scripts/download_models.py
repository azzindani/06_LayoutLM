#!/usr/bin/env python
"""
Download and cache models for offline use.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_models(
    model_name: str = "nielsr/layoutlmv3-finetuned-funsd",
    cache_dir: str = "./models"
):
    """
    Download and cache LayoutLMv3 model and processor.

    Args:
        model_name: HuggingFace model name
        cache_dir: Directory to cache models
    """
    from transformers import (
        LayoutLMv3ForTokenClassification,
        LayoutLMv3Processor
    )

    print(f"Downloading model: {model_name}")
    print(f"Cache directory: {cache_dir}")

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Download processor
    print("\nDownloading processor...")
    processor = LayoutLMv3Processor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        apply_ocr=False
    )
    print("  Processor downloaded")

    # Download model
    print("\nDownloading model...")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    print("  Model downloaded")

    print("\n" + "=" * 50)
    print("Download complete!")
    print(f"Models cached in: {cache_dir}")
    print("=" * 50)


def download_easyocr(languages: list = None):
    """
    Download EasyOCR models.

    Args:
        languages: List of language codes (default: ['en'])
    """
    import easyocr

    languages = languages or ["en"]

    print(f"\nDownloading EasyOCR models for: {languages}")

    # Initialize reader (triggers download)
    reader = easyocr.Reader(languages, gpu=False)

    print("  EasyOCR models downloaded")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and cache models for offline use"
    )
    parser.add_argument(
        "--model",
        default="nielsr/layoutlmv3-finetuned-funsd",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--cache-dir",
        default="./models",
        help="Cache directory"
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip EasyOCR model download"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Model Download Script")
    print("=" * 50)

    # Download LayoutLMv3
    download_models(args.model, args.cache_dir)

    # Download EasyOCR
    if not args.skip_ocr:
        download_easyocr()

    print("\nAll models downloaded successfully!")

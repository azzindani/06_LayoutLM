"""
Model loading and caching utilities.
"""

from typing import Tuple, Optional
import os

from infrastructure.logger_utils import get_logger
from infrastructure.exceptions import ModelLoadError

logger = get_logger(__name__)

# Global model cache
_model_cache = {}
_processor_cache = {}


def load_model(
    model_name: str,
    cache_dir: str = "./models",
    device: str = "auto"
) -> Tuple:
    """
    Load LayoutLMv3 model and processor.

    Args:
        model_name: HuggingFace model name or path
        cache_dir: Directory to cache model files
        device: Device to load model on ('auto', 'cpu', 'cuda')

    Returns:
        Tuple of (model, processor)
    """
    global _model_cache, _processor_cache

    cache_key = f"{model_name}_{device}"

    # Return cached model if available
    if cache_key in _model_cache:
        logger.debug(f"Using cached model: {model_name}")
        return _model_cache[cache_key], _processor_cache[cache_key]

    try:
        from transformers import (
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Processor
        )
        import torch

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading model: {model_name} on {device}")

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Load processor
        processor = LayoutLMv3Processor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            apply_ocr=False  # We use our own OCR
        )

        # Load model
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        # Move model to device
        model = model.to(device)
        model.eval()

        # Cache for future use
        _model_cache[cache_key] = model
        _processor_cache[cache_key] = processor

        logger.info(f"Model loaded successfully on {device}")
        return model, processor

    except Exception as e:
        raise ModelLoadError(f"Failed to load model {model_name}: {e}")


def get_model_config(model_name: str, cache_dir: str = "./models"):
    """
    Get model configuration without loading the full model.

    Args:
        model_name: HuggingFace model name or path
        cache_dir: Directory to cache model files

    Returns:
        Model configuration
    """
    try:
        from transformers import LayoutLMv3Config

        config = LayoutLMv3Config.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        return config
    except Exception as e:
        raise ModelLoadError(f"Failed to load model config: {e}")


def clear_model_cache() -> None:
    """Clear all cached models (useful for memory management)."""
    global _model_cache, _processor_cache

    import gc
    import torch

    _model_cache.clear()
    _processor_cache.clear()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Model cache cleared")


def get_label_mapping(model_name: str = "nielsr/layoutlmv3-finetuned-funsd"):
    """
    Get the label ID to name mapping for the model.

    Returns:
        Dict mapping label IDs to label names
    """
    # FUNSD dataset labels
    return {
        0: "O",
        1: "B-HEADER",
        2: "I-HEADER",
        3: "B-QUESTION",
        4: "I-QUESTION",
        5: "B-ANSWER",
        6: "I-ANSWER",
    }


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL LOADER TEST")
    print("=" * 60)

    # Test label mapping
    labels = get_label_mapping()
    print(f"  Label mapping: {labels}")

    # Test model loading (will download if needed)
    try:
        model, processor = load_model(
            "nielsr/layoutlmv3-finetuned-funsd",
            device="cpu"
        )
        print(f"  Model type: {type(model).__name__}")
        print(f"  Processor type: {type(processor).__name__}")

        # Test cache
        model2, processor2 = load_model(
            "nielsr/layoutlmv3-finetuned-funsd",
            device="cpu"
        )
        print(f"  Cache working: {model is model2}")

        clear_model_cache()
        print("  Cache cleared successfully")

    except Exception as e:
        print(f"  Model loading skipped: {e}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

"""
Custom exceptions for the LayoutLM Document Processing Service.
"""


class LayoutLMServiceError(Exception):
    """Base exception for all service errors."""
    pass


class ConfigurationError(LayoutLMServiceError):
    """Error in configuration settings."""
    pass


class ImageProcessingError(LayoutLMServiceError):
    """Error during image processing."""
    pass


class InvalidImageError(ImageProcessingError):
    """Invalid or corrupted image file."""
    pass


class ImageTooLargeError(ImageProcessingError):
    """Image exceeds maximum dimensions or file size."""
    pass


class UnsupportedFormatError(ImageProcessingError):
    """Unsupported image format."""
    pass


class OCRError(LayoutLMServiceError):
    """Error during OCR processing."""
    pass


class OCREngineError(OCRError):
    """Error initializing or running OCR engine."""
    pass


class ModelError(LayoutLMServiceError):
    """Error with the LayoutLM model."""
    pass


class ModelLoadError(ModelError):
    """Error loading the model."""
    pass


class InferenceError(ModelError):
    """Error during model inference."""
    pass


class PipelineError(LayoutLMServiceError):
    """Error in the processing pipeline."""
    pass


class ExportError(LayoutLMServiceError):
    """Error during result export."""
    pass


class ValidationError(LayoutLMServiceError):
    """Input validation error."""
    pass


if __name__ == "__main__":
    print("=" * 60)
    print("EXCEPTIONS TEST")
    print("=" * 60)

    # Test exception hierarchy
    try:
        raise InvalidImageError("Test invalid image")
    except ImageProcessingError as e:
        print(f"  Caught ImageProcessingError: {e}")

    try:
        raise ModelLoadError("Test model load error")
    except LayoutLMServiceError as e:
        print(f"  Caught LayoutLMServiceError: {e}")

    print("  All exceptions properly inherit from base")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

"""
Image processing utilities.
"""

import io
from typing import Tuple, Optional, List
from PIL import Image

from infrastructure.logger_utils import get_logger
from infrastructure.exceptions import (
    InvalidImageError,
    ImageTooLargeError,
    UnsupportedFormatError
)

logger = get_logger(__name__)

# Supported formats
SUPPORTED_FORMATS = {'PNG', 'JPEG', 'JPG', 'TIFF', 'BMP', 'GIF'}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_DIMENSION = 10000
MIN_DIMENSION = 100


def load_image(source) -> Image.Image:
    """
    Load an image from various sources.

    Args:
        source: File path, bytes, or file-like object

    Returns:
        PIL Image in RGB mode

    Raises:
        InvalidImageError: If image cannot be loaded
        UnsupportedFormatError: If format not supported
    """
    try:
        if isinstance(source, str):
            # File path
            image = Image.open(source)
        elif isinstance(source, bytes):
            # Bytes
            image = Image.open(io.BytesIO(source))
        elif hasattr(source, 'read'):
            # File-like object
            image = Image.open(source)
        elif isinstance(source, Image.Image):
            # Already a PIL Image
            image = source
        else:
            raise InvalidImageError(f"Unsupported source type: {type(source)}")

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image

    except Exception as e:
        if isinstance(e, (InvalidImageError, UnsupportedFormatError)):
            raise
        raise InvalidImageError(f"Failed to load image: {e}")


def validate_image(image: Image.Image) -> None:
    """
    Validate image dimensions and format.

    Args:
        image: PIL Image to validate

    Raises:
        ImageTooLargeError: If image exceeds size limits
        UnsupportedFormatError: If format not supported
    """
    width, height = image.size

    # Check dimensions
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ImageTooLargeError(
            f"Image dimensions ({width}x{height}) exceed maximum ({MAX_DIMENSION}x{MAX_DIMENSION})"
        )

    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise InvalidImageError(
            f"Image dimensions ({width}x{height}) below minimum ({MIN_DIMENSION}x{MIN_DIMENSION})"
        )

    # Check format
    if image.format and image.format.upper() not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Unsupported format: {image.format}. Supported: {SUPPORTED_FORMATS}"
        )


def resize_image(
    image: Image.Image,
    max_size: int = 2000,
    maintain_aspect: bool = True
) -> Image.Image:
    """
    Resize image if it exceeds maximum size.

    Args:
        image: PIL Image to resize
        max_size: Maximum dimension (width or height)
        maintain_aspect: Whether to maintain aspect ratio

    Returns:
        Resized image
    """
    width, height = image.size

    if width <= max_size and height <= max_size:
        return image

    if maintain_aspect:
        ratio = min(max_size / width, max_size / height)
        new_size = (int(width * ratio), int(height * ratio))
    else:
        new_size = (max_size, max_size)

    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    logger.debug(f"Resized image from {image.size} to {resized.size}")
    return resized


def preprocess_image(
    source,
    max_size: int = 2000,
    validate: bool = True
) -> Image.Image:
    """
    Load, validate, and preprocess an image.

    Args:
        source: Image source (path, bytes, or PIL Image)
        max_size: Maximum dimension after resizing
        validate: Whether to validate the image

    Returns:
        Preprocessed PIL Image
    """
    # Load image
    image = load_image(source)

    # Validate
    if validate:
        validate_image(image)

    # Resize if necessary
    image = resize_image(image, max_size)

    return image


def load_pdf_pages(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """
    Load PDF pages as images.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for rendering

    Returns:
        List of PIL Images (one per page)
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        images = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        doc.close()
        logger.info(f"Loaded {len(images)} pages from PDF")
        return images

    except ImportError:
        raise ImportError("PyMuPDF required for PDF support. Install with: pip install PyMuPDF")
    except Exception as e:
        raise InvalidImageError(f"Failed to load PDF: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("IMAGE UTILS TEST")
    print("=" * 60)

    # Create test image
    test_img = Image.new('RGB', (800, 600), color='white')

    # Test validation
    validate_image(test_img)
    print("  Validation passed")

    # Test resize
    large_img = Image.new('RGB', (3000, 2000), color='white')
    resized = resize_image(large_img, max_size=1000)
    print(f"  Resize: {large_img.size} -> {resized.size}")

    # Test preprocess
    processed = preprocess_image(test_img)
    print(f"  Preprocess: {processed.size}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

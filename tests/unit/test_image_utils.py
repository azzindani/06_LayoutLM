"""
Unit tests for image utilities.
"""

import pytest
from PIL import Image

from core.pipeline.image_utils import (
    load_image,
    validate_image,
    resize_image,
    preprocess_image,
    MAX_DIMENSION,
    MIN_DIMENSION
)
from infrastructure.exceptions import (
    InvalidImageError,
    ImageTooLargeError
)


@pytest.mark.unit
class TestImageUtils:
    """Tests for image utilities."""

    def test_load_image_pil(self, sample_image):
        """Loads PIL Image correctly."""
        result = load_image(sample_image)
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'

    def test_load_image_converts_mode(self):
        """Converts non-RGB images to RGB."""
        # Create grayscale image
        gray_img = Image.new('L', (100, 100))
        result = load_image(gray_img)

        assert result.mode == 'RGB'

    def test_validate_image_valid(self, sample_image):
        """Validates correct image without error."""
        # Should not raise
        validate_image(sample_image)

    def test_validate_image_too_large(self):
        """Raises error for oversized image."""
        large_img = Image.new('RGB', (MAX_DIMENSION + 1, 100))

        with pytest.raises(ImageTooLargeError):
            validate_image(large_img)

    def test_validate_image_too_small(self):
        """Raises error for undersized image."""
        small_img = Image.new('RGB', (MIN_DIMENSION - 1, MIN_DIMENSION - 1))

        with pytest.raises(InvalidImageError):
            validate_image(small_img)

    def test_resize_image_maintains_aspect(self):
        """Resize maintains aspect ratio."""
        img = Image.new('RGB', (2000, 1000))
        resized = resize_image(img, max_size=1000)

        assert resized.size == (1000, 500)

    def test_resize_image_no_upscale(self):
        """Resize doesn't upscale small images."""
        img = Image.new('RGB', (500, 300))
        resized = resize_image(img, max_size=1000)

        assert resized.size == (500, 300)

    def test_preprocess_image_full_pipeline(self, sample_image):
        """Preprocess handles full pipeline."""
        result = preprocess_image(sample_image)

        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'

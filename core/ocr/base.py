"""
Abstract base class for OCR engines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image


@dataclass
class OCRResult:
    """Result from OCR processing."""
    text: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float


class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the OCR engine.

        Returns:
            True if initialization successful
        """
        pass

    @abstractmethod
    def extract_text(self, image: Image.Image) -> List[OCRResult]:
        """
        Extract text and bounding boxes from an image.

        Args:
            image: PIL Image to process

        Returns:
            List of OCRResult with text, bounding boxes, and confidence
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the engine name."""
        pass

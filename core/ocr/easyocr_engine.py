"""
EasyOCR engine implementation.
"""

from typing import List
import numpy as np
from PIL import Image

from core.ocr.base import BaseOCREngine, OCRResult
from infrastructure.logger_utils import get_logger
from infrastructure.exceptions import OCREngineError

logger = get_logger(__name__)


class EasyOCREngine(BaseOCREngine):
    """OCR engine using EasyOCR."""

    def __init__(self, languages: List[str] = None, gpu: bool = True):
        """
        Initialize EasyOCR engine.

        Args:
            languages: List of language codes (default: ['en'])
            gpu: Whether to use GPU acceleration
        """
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._reader = None

    def initialize(self) -> bool:
        """Initialize the EasyOCR reader."""
        try:
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
            logger.info(f"EasyOCR initialized with languages: {self.languages}")
            return True
        except Exception as e:
            raise OCREngineError(f"Failed to initialize EasyOCR: {e}")

    def extract_text(self, image: Image.Image) -> List[OCRResult]:
        """
        Extract text from image using EasyOCR.

        Args:
            image: PIL Image to process

        Returns:
            List of OCRResult
        """
        if self._reader is None:
            self.initialize()

        # Convert PIL Image to numpy array
        image_array = np.array(image)

        # Run OCR
        results = self._reader.readtext(image_array)

        ocr_results = []
        for bbox, text, confidence in results:
            # Convert polygon bbox to rectangle (x1, y1, x2, y2)
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]

            x1 = int(min(x_coords))
            y1 = int(min(y_coords))
            x2 = int(max(x_coords))
            y2 = int(max(y_coords))

            ocr_results.append(OCRResult(
                text=text,
                bbox=(x1, y1, x2, y2),
                confidence=float(confidence)
            ))

        logger.debug(f"EasyOCR extracted {len(ocr_results)} text regions")
        return ocr_results

    def shutdown(self) -> None:
        """Clean up EasyOCR resources."""
        self._reader = None
        logger.info("EasyOCR shutdown complete")

    @property
    def name(self) -> str:
        return "easyocr"


# Factory function
def create_ocr_engine(engine_type: str = "easyocr", **kwargs) -> BaseOCREngine:
    """
    Create an OCR engine instance.

    Args:
        engine_type: Type of OCR engine ('easyocr')
        **kwargs: Additional arguments for the engine

    Returns:
        OCR engine instance
    """
    engines = {
        "easyocr": EasyOCREngine,
    }

    if engine_type not in engines:
        raise ValueError(f"Unknown OCR engine: {engine_type}")

    return engines[engine_type](**kwargs)


if __name__ == "__main__":
    print("=" * 60)
    print("EASYOCR ENGINE TEST")
    print("=" * 60)

    # Create test image
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Hello World", fill='black')
    draw.text((10, 50), "Test OCR", fill='black')

    # Test OCR
    engine = EasyOCREngine(gpu=False)
    engine.initialize()
    results = engine.extract_text(img)

    print(f"  Found {len(results)} text regions")
    for r in results:
        print(f"    - '{r.text}' at {r.bbox} (conf: {r.confidence:.2f})")

    engine.shutdown()

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

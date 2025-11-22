"""
Main document processing pipeline.
"""

import time
from typing import Dict, Any, List, Optional
from PIL import Image

from config import get_config
from core.ocr.easyocr_engine import create_ocr_engine
from core.model.inference import run_inference
from core.model.postprocess import (
    process_predictions,
    aggregate_entities,
    format_output
)
from core.pipeline.image_utils import preprocess_image, load_pdf_pages
from infrastructure.logger_utils import get_logger
from infrastructure.hardware_detection import detect_device
from infrastructure.exceptions import PipelineError

logger = get_logger(__name__)


class DocumentProcessor:
    """Main pipeline for document processing."""

    def __init__(self, config=None):
        """
        Initialize the document processor.

        Args:
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()
        self._ocr_engine = None
        self._device = None

    def initialize(self) -> None:
        """Initialize OCR engine and detect hardware."""
        # Detect device
        self._device = detect_device(self.config.hardware.device)
        logger.info(f"Using device: {self._device.device_name}")

        # Initialize OCR engine
        use_gpu = self._device.device_type == "cuda"
        self._ocr_engine = create_ocr_engine(
            self.config.ocr.engine,
            languages=self.config.ocr.languages,
            gpu=use_gpu
        )
        self._ocr_engine.initialize()
        logger.info("Document processor initialized")

    def process_image(self, image_source) -> Dict[str, Any]:
        """
        Process a single image.

        Args:
            image_source: Image path, bytes, or PIL Image

        Returns:
            Processing results
        """
        start_time = time.time()

        try:
            # Ensure initialized
            if self._ocr_engine is None:
                self.initialize()

            # Preprocess image
            image = preprocess_image(image_source)
            logger.debug(f"Image preprocessed: {image.size}")

            # Run OCR
            ocr_results = self._ocr_engine.extract_text(image)
            logger.debug(f"OCR extracted {len(ocr_results)} text regions")

            # Handle empty OCR results
            if not ocr_results:
                processing_time = (time.time() - start_time) * 1000
                return format_output(
                    entities=[],
                    image_size=image.size,
                    processing_time_ms=processing_time,
                    ocr_engine=self.config.ocr.engine
                )

            # Run inference
            inference_result = run_inference(
                image,
                ocr_results,
                model_name=self.config.model.name,
                device=self._device.torch_device,
                cache_dir=self.config.model.cache_dir
            )

            # Post-process
            entities = process_predictions(
                inference_result,
                confidence_threshold=self.config.model.confidence_threshold
            )

            # Aggregate consecutive entities
            entities = aggregate_entities(entities)

            # Format output
            processing_time = (time.time() - start_time) * 1000
            result = format_output(
                entities=entities,
                image_size=image.size,
                processing_time_ms=processing_time,
                ocr_engine=self.config.ocr.engine
            )

            logger.info(f"Processed image in {processing_time:.2f}ms, found {len(entities)} entities")
            return result

        except Exception as e:
            raise PipelineError(f"Pipeline failed: {e}")

    def process_pdf(self, pdf_path: str, dpi: int = 200) -> Dict[str, Any]:
        """
        Process a PDF document.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for rendering

        Returns:
            Processing results for all pages
        """
        start_time = time.time()

        try:
            # Load PDF pages
            pages = load_pdf_pages(pdf_path, dpi)
            logger.info(f"Processing PDF with {len(pages)} pages")

            all_results = []
            for i, page_image in enumerate(pages):
                page_result = self.process_image(page_image)

                # Update page number
                if page_result["results"]:
                    page_result["results"][0]["page"] = i + 1

                all_results.append(page_result)

            # Combine results
            processing_time = (time.time() - start_time) * 1000
            combined = {
                "status": "success",
                "processing_time_ms": round(processing_time, 2),
                "results": [r["results"][0] for r in all_results if r["results"]],
                "metadata": {
                    "model_version": "layoutlmv3-funsd-v1",
                    "ocr_engine": self.config.ocr.engine,
                    "total_pages": len(pages)
                }
            }

            return combined

        except Exception as e:
            raise PipelineError(f"PDF processing failed: {e}")

    def process_batch(self, image_sources: List) -> List[Dict[str, Any]]:
        """
        Process multiple images.

        Args:
            image_sources: List of image sources

        Returns:
            List of processing results
        """
        results = []
        for source in image_sources:
            try:
                result = self.process_image(source)
                results.append(result)
            except Exception as e:
                results.append({
                    "status": "error",
                    "error": str(e)
                })
        return results

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._ocr_engine:
            self._ocr_engine.shutdown()
        logger.info("Document processor shutdown complete")


# Singleton instance
_processor: Optional[DocumentProcessor] = None


def get_processor() -> DocumentProcessor:
    """Get the global document processor instance."""
    global _processor
    if _processor is None:
        _processor = DocumentProcessor()
    return _processor


if __name__ == "__main__":
    print("=" * 60)
    print("DOCUMENT PROCESSOR TEST")
    print("=" * 60)

    # Create test image with text
    from PIL import Image, ImageDraw

    img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Name:", fill='black')
    draw.text((150, 50), "John Doe", fill='black')
    draw.text((50, 100), "Date:", fill='black')
    draw.text((150, 100), "2024-01-15", fill='black')

    # Save test image
    test_path = "/tmp/test_document.png"
    img.save(test_path)

    try:
        processor = DocumentProcessor()
        processor.initialize()

        result = processor.process_image(test_path)
        print(f"  Status: {result['status']}")
        print(f"  Processing time: {result['processing_time_ms']:.2f}ms")
        print(f"  Entities found: {len(result['results'][0]['entities'])}")

        for entity in result['results'][0]['entities']:
            print(f"    - {entity['text']}: {entity['label']} ({entity['confidence']:.2f})")

        processor.shutdown()

    except Exception as e:
        print(f"  Test skipped: {e}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

"""
Integration tests for the document processing pipeline.
"""

import pytest
from PIL import Image, ImageDraw

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.mark.integration
class TestPipeline:
    """Integration tests for document processor."""

    def test_pipeline_end_to_end(self, sample_image_with_text):
        """Full pipeline processes sample and returns results."""
        from core.pipeline.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        processor.initialize()

        result = processor.process_image(sample_image_with_text)

        assert result["status"] == "success"
        assert "processing_time_ms" in result
        assert "results" in result
        assert "metadata" in result

        processor.shutdown()

    def test_pipeline_empty_image(self):
        """Pipeline handles image with no text."""
        from core.pipeline.document_processor import DocumentProcessor

        # Create blank image
        blank = Image.new('RGB', (400, 300), color='white')

        processor = DocumentProcessor()
        processor.initialize()

        result = processor.process_image(blank)

        assert result["status"] == "success"
        # May have empty entities
        assert "results" in result

        processor.shutdown()

    def test_pipeline_batch(self, sample_image_with_text):
        """Pipeline handles batch of images."""
        from core.pipeline.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        processor.initialize()

        # Create batch
        images = [sample_image_with_text, sample_image_with_text]
        results = processor.process_batch(images)

        assert len(results) == 2
        for result in results:
            assert result["status"] == "success"

        processor.shutdown()


@pytest.mark.integration
class TestOCREngine:
    """Integration tests for OCR engine."""

    def test_easyocr_extracts_text(self, sample_image_with_text):
        """EasyOCR extracts text from sample image."""
        easyocr = pytest.importorskip("easyocr")

        from core.ocr.easyocr_engine import EasyOCREngine

        engine = EasyOCREngine(gpu=False)
        engine.initialize()

        results = engine.extract_text(sample_image_with_text)

        assert len(results) > 0
        texts = [r.text for r in results]

        # Should find some text
        assert any(len(t) > 0 for t in texts)

        engine.shutdown()

    def test_easyocr_returns_bboxes(self, sample_image_with_text):
        """EasyOCR returns valid bounding boxes."""
        easyocr = pytest.importorskip("easyocr")

        from core.ocr.easyocr_engine import EasyOCREngine

        engine = EasyOCREngine(gpu=False)
        engine.initialize()

        results = engine.extract_text(sample_image_with_text)

        for result in results:
            x1, y1, x2, y2 = result.bbox
            assert x1 >= 0
            assert y1 >= 0
            assert x2 > x1
            assert y2 > y1

        engine.shutdown()


@pytest.mark.integration
class TestModelInference:
    """Integration tests for model inference."""

    def test_model_loads(self):
        """LayoutLMv3 model loads successfully."""
        from core.model.loader import load_model

        model, processor = load_model(
            "nielsr/layoutlmv3-finetuned-funsd",
            device="cpu"
        )

        assert model is not None
        assert processor is not None

    def test_inference_returns_predictions(self, sample_image_with_text, sample_ocr_results):
        """Model returns predictions for input."""
        from core.model.inference import run_inference

        result = run_inference(
            sample_image_with_text,
            sample_ocr_results,
            device="cpu"
        )

        assert "predictions" in result
        assert "confidence_scores" in result
        assert "words" in result
        assert len(result["predictions"]) > 0

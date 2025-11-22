"""
Unit tests for post-processing utilities.
"""

import pytest

from core.model.postprocess import (
    Entity,
    process_predictions,
    aggregate_entities,
    format_output,
    unnormalize_bbox
)


@pytest.mark.unit
class TestPostprocess:
    """Tests for post-processing utilities."""

    def test_unnormalize_bbox(self):
        """Converts 0-1000 scale to pixel coordinates."""
        bbox = [100, 200, 300, 400]
        result = unnormalize_bbox(bbox, 1000, 1000)

        assert result == {"x1": 100, "y1": 200, "x2": 300, "y2": 400}

    def test_unnormalize_bbox_scaling(self):
        """Scales bbox correctly for different image sizes."""
        bbox = [500, 500, 1000, 1000]
        result = unnormalize_bbox(bbox, 800, 600)

        assert result["x1"] == 400
        assert result["y1"] == 300
        assert result["x2"] == 800
        assert result["y2"] == 600

    def test_process_predictions(self, sample_inference_result):
        """Processes predictions into entities."""
        entities = process_predictions(sample_inference_result)

        assert len(entities) > 0
        for entity in entities:
            assert hasattr(entity, 'text')
            assert hasattr(entity, 'label')
            assert hasattr(entity, 'confidence')
            assert hasattr(entity, 'bbox')

    def test_process_predictions_filters_low_confidence(self, sample_inference_result):
        """Filters entities below confidence threshold."""
        entities = process_predictions(sample_inference_result, confidence_threshold=0.99)

        # Should filter out all entities with conf < 0.99
        assert len(entities) == 0

    def test_aggregate_entities_merges_adjacent(self):
        """Aggregates adjacent entities with same label."""
        entities = [
            Entity(text="John", label="ANSWER", confidence=0.9,
                   bbox={"x1": 100, "y1": 50, "x2": 140, "y2": 70}),
            Entity(text="Doe", label="ANSWER", confidence=0.85,
                   bbox={"x1": 145, "y1": 50, "x2": 180, "y2": 70}),
        ]

        aggregated = aggregate_entities(entities)

        assert len(aggregated) == 1
        assert aggregated[0].text == "John Doe"
        assert aggregated[0].label == "ANSWER"

    def test_aggregate_entities_keeps_different_labels(self):
        """Keeps entities with different labels separate."""
        entities = [
            Entity(text="Name:", label="QUESTION", confidence=0.9,
                   bbox={"x1": 50, "y1": 50, "x2": 100, "y2": 70}),
            Entity(text="John", label="ANSWER", confidence=0.85,
                   bbox={"x1": 110, "y1": 50, "x2": 150, "y2": 70}),
        ]

        aggregated = aggregate_entities(entities)

        assert len(aggregated) == 2

    def test_format_output_structure(self):
        """Format output has correct structure."""
        entities = [
            Entity(text="Test", label="QUESTION", confidence=0.9,
                   bbox={"x1": 10, "y1": 20, "x2": 100, "y2": 40})
        ]

        result = format_output(
            entities=entities,
            image_size=(800, 600),
            processing_time_ms=1234.5
        )

        assert result["status"] == "success"
        assert result["processing_time_ms"] == 1234.5
        assert len(result["results"]) == 1
        assert result["results"][0]["page"] == 1
        assert len(result["results"][0]["entities"]) == 1
        assert "metadata" in result

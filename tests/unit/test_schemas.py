"""
Unit tests for API schemas.
"""

import pytest
from pydantic import ValidationError

from api.schemas.request import ProcessRequest, BatchProcessRequest
from api.schemas.response import (
    BoundingBox,
    Entity,
    PageResult,
    ProcessResponse,
    HealthResponse
)


@pytest.mark.unit
class TestRequestSchemas:
    """Tests for request schemas."""

    def test_process_request_defaults(self):
        """ProcessRequest has correct defaults."""
        request = ProcessRequest()
        assert request.confidence_threshold == 0.5

    def test_process_request_valid_threshold(self):
        """ProcessRequest accepts valid threshold."""
        request = ProcessRequest(confidence_threshold=0.8)
        assert request.confidence_threshold == 0.8

    def test_process_request_invalid_threshold(self):
        """ProcessRequest rejects invalid threshold."""
        with pytest.raises(ValidationError):
            ProcessRequest(confidence_threshold=1.5)

        with pytest.raises(ValidationError):
            ProcessRequest(confidence_threshold=-0.1)


@pytest.mark.unit
class TestResponseSchemas:
    """Tests for response schemas."""

    def test_bounding_box(self):
        """BoundingBox schema works correctly."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=50)
        assert bbox.x1 == 10
        assert bbox.y2 == 50

    def test_entity(self):
        """Entity schema works correctly."""
        entity = Entity(
            text="Test",
            label="QUESTION",
            confidence=0.95,
            bbox=BoundingBox(x1=10, y1=20, x2=100, y2=50)
        )
        assert entity.text == "Test"
        assert entity.label == "QUESTION"

    def test_page_result(self):
        """PageResult schema works correctly."""
        result = PageResult(
            page=1,
            entities=[
                Entity(
                    text="Test",
                    label="QUESTION",
                    confidence=0.95,
                    bbox=BoundingBox(x1=10, y1=20, x2=100, y2=50)
                )
            ]
        )
        assert result.page == 1
        assert len(result.entities) == 1

    def test_health_response(self):
        """HealthResponse schema works correctly."""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            device="CPU",
            model_loaded=True
        )
        assert response.status == "healthy"
        assert response.model_loaded is True

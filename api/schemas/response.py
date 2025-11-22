"""
Response schemas for API endpoints.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int


class Entity(BaseModel):
    """Extracted entity from document."""
    text: str
    label: str
    confidence: float
    bbox: BoundingBox


class PageResult(BaseModel):
    """Results for a single page."""
    page: int
    entities: List[Entity]


class Metadata(BaseModel):
    """Processing metadata."""
    model_version: str
    ocr_engine: str
    image_size: List[int]


class ProcessResponse(BaseModel):
    """Response for document processing."""
    status: str
    processing_time_ms: float
    results: List[PageResult]
    metadata: Metadata


class ErrorResponse(BaseModel):
    """Error response."""
    status: str = "error"
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    device: str
    model_loaded: bool

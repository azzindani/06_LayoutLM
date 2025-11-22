"""
Request schemas for API endpoints.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class ProcessRequest(BaseModel):
    """Request for single image processing."""
    confidence_threshold: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for entities"
    )


class BatchProcessRequest(BaseModel):
    """Request for batch image processing."""
    confidence_threshold: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for entities"
    )


class ExportRequest(BaseModel):
    """Request for exporting results."""
    format: str = Field(
        default="json",
        description="Export format: json, csv, xml"
    )

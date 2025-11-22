"""
Export endpoints.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from core.pipeline.export import export_result
from infrastructure.exceptions import ExportError

router = APIRouter()


@router.post("/export/{format}")
async def export_results(format: str, result: dict):
    """
    Export processing results to specified format.

    Args:
        format: Output format (json, csv, xml)
        result: Processing result to export

    Returns:
        Exported content in requested format
    """
    try:
        content = export_result(result, format)

        # Set content type based on format
        content_types = {
            "json": "application/json",
            "csv": "text/csv",
            "xml": "application/xml"
        }

        return PlainTextResponse(
            content=content,
            media_type=content_types.get(format, "text/plain")
        )

    except ExportError as e:
        raise HTTPException(status_code=400, detail=str(e))

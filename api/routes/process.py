"""
Document processing endpoints.
"""

from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse

from api.schemas.response import ProcessResponse, ErrorResponse
from core.pipeline.document_processor import get_processor
from infrastructure.logger_utils import get_logger
from infrastructure.exceptions import LayoutLMServiceError

router = APIRouter()
logger = get_logger(__name__)


@router.post("/process", response_model=ProcessResponse)
async def process_document(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(default=0.5)
):
    """
    Process a single document image.

    Args:
        file: Uploaded image file
        confidence_threshold: Minimum confidence for entities

    Returns:
        Processing results with extracted entities
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Expected image."
            )

        # Read file content
        content = await file.read()

        # Process document
        processor = get_processor()
        result = processor.process_image(content)

        return JSONResponse(content=result)

    except LayoutLMServiceError as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.post("/batch")
async def process_batch(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = Form(default=0.5)
):
    """
    Process multiple document images.

    Args:
        files: List of uploaded image files
        confidence_threshold: Minimum confidence for entities

    Returns:
        List of processing results
    """
    try:
        processor = get_processor()
        results = []

        for file in files:
            try:
                content = await file.read()
                result = processor.process_image(content)
                result["filename"] = file.filename
                results.append(result)
            except Exception as e:
                results.append({
                    "status": "error",
                    "filename": file.filename,
                    "error": str(e)
                })

        return JSONResponse(content={"results": results})

    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(default=0.5),
    dpi: int = Form(default=200)
):
    """
    Process a PDF document.

    Args:
        file: Uploaded PDF file
        confidence_threshold: Minimum confidence for entities
        dpi: Resolution for rendering PDF pages

    Returns:
        Processing results for all pages
    """
    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Expected PDF file"
            )

        # Save to temporary file
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Process PDF
            processor = get_processor()
            result = processor.process_pdf(tmp_path, dpi=dpi)
            return JSONResponse(content=result)
        finally:
            # Clean up
            os.unlink(tmp_path)

    except LayoutLMServiceError as e:
        logger.error(f"PDF processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

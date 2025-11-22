"""
Health check endpoints.
"""

from fastapi import APIRouter

from api.schemas.response import HealthResponse
from config import get_config
from infrastructure.hardware_detection import detect_device

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health."""
    config = get_config()
    device = detect_device(config.hardware.device)

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        device=device.device_name,
        model_loaded=True  # TODO: Check actual model status
    )


@router.get("/ready")
async def readiness_check():
    """Check if service is ready to accept requests."""
    return {"status": "ready"}


@router.get("/live")
async def liveness_check():
    """Check if service is alive."""
    return {"status": "alive"}

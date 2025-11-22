"""
FastAPI application server.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, process, export
from config import get_config
from infrastructure.logger_utils import setup_logging, get_logger
from core.pipeline.document_processor import get_processor

# Initialize logging
config = get_config()
setup_logging(level=config.log.level, format_type=config.log.format)
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LayoutLMv3 Document Processing Service",
    description="Extract structured information from documents using LayoutLMv3",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(process.router, tags=["Processing"])
app.include_router(export.router, tags=["Export"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting LayoutLM Document Processing Service")

    # Initialize document processor
    try:
        processor = get_processor()
        processor.initialize()
        logger.info("Document processor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("Shutting down service")
    processor = get_processor()
    processor.shutdown()


def run_server():
    """Run the API server."""
    import uvicorn

    config = get_config()
    uvicorn.run(
        "api.server:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        reload=False
    )


if __name__ == "__main__":
    run_server()

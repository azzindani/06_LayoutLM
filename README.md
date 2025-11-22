# LayoutLMv3 Document Processing System

Production-ready document understanding system using LayoutLMv3 for extracting structured information from forms and documents.

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        GRADIO WEB UI                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Upload  │  │  Preview │  │  Results │  │  Export Options  │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API LAYER (FastAPI)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  /health │  │ /process │  │  /batch  │  │  /export/{fmt}   │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PROCESSING PIPELINE                          │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │  Image   │ -> │   OCR    │ -> │LayoutLM  │ -> │   Post   │   │
│  │  Loader  │    │ (EasyOCR)│    │Inference │    │Processing│   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘   │
│                                                                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INFRASTRUCTURE LAYER                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Model   │  │  Config  │  │  Logger  │  │  Hardware Detect │ │
│  │  Manager │  │  Manager │  │  Utils   │  │  (CPU/GPU)       │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Input Image/PDF
       │
       ▼
┌─────────────────┐
│  Preprocessing  │ → Resize, normalize, validate format
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    EasyOCR      │ → Extract text + bounding boxes
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│LayoutLMv3Proc.  │ → Tokenize text, encode layout + visual features
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│LayoutLMv3Model  │ → Token classification (NER)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Post-processing │ → Aggregate tokens, map labels, format output
└────────┬────────┘
         │
         ▼
Structured JSON Output
```

### Component Relationships

| Component | Depends On | Provides |
|-----------|------------|----------|
| Gradio UI | API Layer | User interface |
| API Layer | Pipeline, Config | REST endpoints |
| Pipeline | OCR, Model, PostProc | Document processing |
| OCR Engine | Hardware Config | Text + bounding boxes |
| Model Manager | Hardware Config | LayoutLMv3 inference |
| Config Manager | None | Settings for all components |

---

## Features

### MVP (Minimum Viable Product)

- [ ] Single image document processing
- [ ] Field extraction (Questions, Answers, Headers)
- [ ] JSON output with bounding boxes and labels
- [ ] Gradio web interface for upload and visualization
- [ ] Basic error handling and input validation
- [ ] Configuration management via environment variables
- [ ] Health check endpoint

### Phase 1: Production Ready

- [ ] Batch document processing
- [ ] PDF support (multi-page)
- [ ] Export formats (JSON, CSV, XML)
- [ ] Logging and monitoring
- [ ] Docker containerization
- [ ] Unit and integration tests
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] API documentation (OpenAPI/Swagger)

### Phase 2: Enhanced Features

- [ ] Confidence score thresholds
- [ ] Custom model fine-tuning interface
- [ ] Multiple OCR engine support (Tesseract, PaddleOCR)
- [ ] Async processing with job queue
- [ ] Result caching (Redis)
- [ ] User authentication
- [ ] Processing history and analytics

### Phase 3: Advanced

- [ ] Multi-language OCR support
- [ ] Custom label/entity types
- [ ] Model versioning and A/B testing
- [ ] Kubernetes deployment configs
- [ ] Auto-scaling based on load
- [ ] Webhook callbacks for async results

---

## Directory Structure

```
layoutlm_service/
├── config.py                    # Central configuration
├── main.py                      # Application entry point
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Multi-service orchestration
├── .env.example                 # Environment template
├── README.md                    # This file
├── WORKFLOW.md                  # Development methodology
│
├── core/                        # Business logic
│   ├── __init__.py
│   ├── ocr/                     # OCR engines
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract OCR interface
│   │   ├── easyocr_engine.py    # EasyOCR implementation
│   │   └── tesseract_engine.py  # Tesseract implementation (Phase 2)
│   │
│   ├── model/                   # LayoutLM model management
│   │   ├── __init__.py
│   │   ├── loader.py            # Model loading and caching
│   │   ├── inference.py         # Inference logic
│   │   └── postprocess.py       # Output formatting
│   │
│   └── pipeline/                # Processing orchestration
│       ├── __init__.py
│       ├── document_processor.py # Main pipeline
│       ├── image_utils.py       # Image preprocessing
│       └── export.py            # Output format converters
│
├── api/                         # REST API
│   ├── __init__.py
│   ├── server.py                # FastAPI application
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py            # Health check endpoints
│   │   ├── process.py           # Document processing endpoints
│   │   └── export.py            # Export endpoints
│   └── schemas/                 # Pydantic models
│       ├── __init__.py
│       ├── request.py           # Input schemas
│       └── response.py          # Output schemas
│
├── ui/                          # User interfaces
│   ├── __init__.py
│   └── gradio_app.py            # Gradio web interface
│
├── infrastructure/              # Shared utilities
│   ├── __init__.py
│   ├── hardware_detection.py    # CPU/GPU detection
│   ├── logger_utils.py          # Logging configuration
│   └── exceptions.py            # Custom exceptions
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── unit/                    # Unit tests (no GPU)
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_image_utils.py
│   │   ├── test_postprocess.py
│   │   ├── test_export.py
│   │   └── test_schemas.py
│   └── integration/             # Integration tests (requires GPU)
│       ├── __init__.py
│       ├── test_ocr_engine.py
│       ├── test_model_inference.py
│       ├── test_pipeline.py
│       └── test_api.py
│
├── models/                      # Model artifacts (gitignored)
│   └── .gitkeep
│
├── data/                        # Sample data for testing
│   ├── samples/
│   │   └── sample_form.png
│   └── expected/
│       └── sample_form.json
│
└── scripts/                     # Utility scripts
    ├── download_models.py       # Pre-download models
    ├── benchmark.py             # Performance testing
    └── convert_notebook.py      # Convert original notebook
```

---

## Expectations

### Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Single image latency | < 2s (GPU), < 10s (CPU) | 300 DPI, A4 size |
| Batch throughput | 30+ docs/min (GPU) | Concurrent processing |
| Model load time | < 30s | First request cold start |
| Memory usage | < 4GB GPU VRAM | LayoutLMv3 base |
| API response time | < 100ms | Health check |

### Input Requirements

- **Supported formats**: PNG, JPG, JPEG, TIFF, PDF
- **Recommended resolution**: 150-300 DPI
- **Max file size**: 10MB per image, 50MB per PDF
- **Max pages**: 50 pages per PDF
- **Image dimensions**: Min 100x100, Max 10000x10000 pixels

### Output Format

```json
{
  "status": "success",
  "processing_time_ms": 1234,
  "results": [
    {
      "page": 1,
      "entities": [
        {
          "text": "Name:",
          "label": "QUESTION",
          "confidence": 0.95,
          "bbox": {
            "x1": 100, "y1": 200,
            "x2": 200, "y2": 230
          }
        },
        {
          "text": "John Doe",
          "label": "ANSWER",
          "confidence": 0.92,
          "bbox": {
            "x1": 210, "y1": 200,
            "x2": 350, "y2": 230
          }
        }
      ]
    }
  ],
  "metadata": {
    "model_version": "layoutlmv3-funsd-v1",
    "ocr_engine": "easyocr",
    "image_size": [1000, 1400]
  }
}
```

### Entity Labels

| Label | Description | Color (UI) |
|-------|-------------|------------|
| HEADER | Document headers/titles | Orange |
| QUESTION | Form field labels/keys | Blue |
| ANSWER | Form field values | Green |
| OTHER | Unclassified text | Gray |

---

## Roadmap

### Week 1: Foundation

#### Day 1-2: Project Setup
- [ ] Initialize directory structure
- [ ] Create config.py with environment variable support
- [ ] Set up logging infrastructure
- [ ] Create requirements.txt with pinned versions
- [ ] Add hardware detection module

#### Day 3-4: Core Pipeline
- [ ] Port OCR logic from notebook to core/ocr/
- [ ] Implement model loader with caching
- [ ] Create inference module
- [ ] Build post-processing for output formatting
- [ ] Create document processor pipeline class

#### Day 5: Basic API
- [ ] Set up FastAPI application
- [ ] Implement /health endpoint
- [ ] Implement /process endpoint for single image
- [ ] Add request/response schemas

### Week 2: UI and Testing

#### Day 1-2: Gradio Interface
- [ ] Create Gradio app with image upload
- [ ] Add visualization of results with bounding boxes
- [ ] Implement JSON output display
- [ ] Add download button for results

#### Day 3-4: Unit Tests
- [ ] Test configuration loading
- [ ] Test image preprocessing
- [ ] Test post-processing logic
- [ ] Test export formats
- [ ] Test API schemas

#### Day 5: Integration Tests
- [ ] Test OCR engine
- [ ] Test model inference
- [ ] Test full pipeline
- [ ] Test API endpoints

### Week 3: Production Hardening

#### Day 1-2: Docker and CI/CD
- [ ] Create Dockerfile
- [ ] Create docker-compose.yml
- [ ] Set up GitHub Actions workflow
- [ ] Add linting (black, flake8)

#### Day 3-4: Enhanced Features
- [ ] Add batch processing endpoint
- [ ] Implement PDF support
- [ ] Add export formats (CSV, XML)
- [ ] Implement error handling improvements

#### Day 5: Documentation and Deploy
- [ ] Complete API documentation
- [ ] Write deployment guide
- [ ] Create .env.example with all options
- [ ] Performance benchmarking

---

## Test Strategy

### Test Categories

```python
# Unit tests - fast, no external dependencies
@pytest.mark.unit
def test_normalize_bbox():
    pass

# Integration tests - requires models/GPU
@pytest.mark.integration
def test_full_pipeline():
    pass

# Slow tests - performance benchmarks
@pytest.mark.slow
def test_batch_processing_throughput():
    pass
```

### Test Coverage by Component

#### Configuration (tests/unit/test_config.py)
```python
def test_config_loads_from_env():
    """Config reads environment variables correctly"""

def test_config_defaults():
    """Config has sensible defaults when env vars missing"""

def test_config_validation():
    """Config validates values (e.g., confidence threshold 0-1)"""
```

#### Image Utils (tests/unit/test_image_utils.py)
```python
def test_load_image_png():
    """Loads PNG images correctly"""

def test_load_image_jpg():
    """Loads JPG images correctly"""

def test_resize_image():
    """Resizes while maintaining aspect ratio"""

def test_normalize_image():
    """Normalizes pixel values to expected range"""

def test_invalid_image_raises():
    """Raises exception for invalid/corrupt images"""
```

#### Post-processing (tests/unit/test_postprocess.py)
```python
def test_unnormalize_bbox():
    """Converts 0-1000 scale to pixel coordinates"""

def test_aggregate_tokens():
    """Merges sub-word tokens into words"""

def test_map_labels():
    """Converts label IDs to human-readable strings"""

def test_filter_by_confidence():
    """Filters results below threshold"""
```

#### Export (tests/unit/test_export.py)
```python
def test_export_json():
    """Exports results to valid JSON"""

def test_export_csv():
    """Exports results to CSV with correct columns"""

def test_export_xml():
    """Exports results to valid XML"""
```

#### OCR Engine (tests/integration/test_ocr_engine.py)
```python
@pytest.mark.integration
def test_easyocr_extracts_text():
    """EasyOCR extracts text from sample image"""

@pytest.mark.integration
def test_easyocr_returns_bboxes():
    """EasyOCR returns valid bounding boxes"""

@pytest.mark.integration
def test_easyocr_handles_empty_image():
    """EasyOCR handles images with no text gracefully"""
```

#### Model Inference (tests/integration/test_model_inference.py)
```python
@pytest.mark.integration
def test_model_loads():
    """LayoutLMv3 model loads successfully"""

@pytest.mark.integration
def test_inference_returns_logits():
    """Model returns logits of expected shape"""

@pytest.mark.integration
def test_inference_with_sample():
    """Model produces expected labels for sample document"""
```

#### Pipeline (tests/integration/test_pipeline.py)
```python
@pytest.mark.integration
def test_pipeline_end_to_end():
    """Full pipeline processes sample and returns results"""

@pytest.mark.integration
def test_pipeline_batch():
    """Pipeline handles batch of images"""

@pytest.mark.integration
def test_pipeline_pdf():
    """Pipeline processes PDF document"""
```

#### API (tests/integration/test_api.py)
```python
@pytest.mark.integration
def test_health_endpoint():
    """GET /health returns 200"""

@pytest.mark.integration
def test_process_endpoint():
    """POST /process returns expected results"""

@pytest.mark.integration
def test_process_invalid_image():
    """POST /process with invalid image returns 400"""

@pytest.mark.integration
def test_batch_endpoint():
    """POST /batch processes multiple images"""
```

### Running Tests

```bash
# All unit tests (CI environment)
pytest tests/unit -v -m unit

# All integration tests (requires GPU)
pytest tests/integration -v -m integration

# Specific component
pytest tests/unit/test_postprocess.py -v

# With coverage
pytest --cov=layoutlm_service --cov-report=html

# Run module directly (uses __main__ block)
python -m core.pipeline.document_processor
python -m core.ocr.easyocr_engine
```

---

## Environment Variables

```bash
# Model Configuration
MODEL_NAME=nielsr/layoutlmv3-finetuned-funsd
MODEL_CACHE_DIR=./models
CONFIDENCE_THRESHOLD=0.5

# OCR Configuration
OCR_ENGINE=easyocr
OCR_LANGUAGES=en

# Hardware
DEVICE=auto  # auto, cpu, cuda, cuda:0
MAX_BATCH_SIZE=8

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# UI
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=false

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

---

## Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Download models (optional, downloads on first use)
python scripts/download_models.py

# Run Gradio UI
python ui/gradio_app.py

# Run API server
python api/server.py

# Run both (using main.py)
python main.py --mode all
```

### Docker

```bash
# Build image
docker build -t layoutlm-service .

# Run container
docker run -p 7860:7860 -p 8000:8000 --gpus all layoutlm-service

# Docker Compose (with all services)
docker-compose up
```

### Production Checklist

- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Environment variables documented
- [ ] Health check endpoint working
- [ ] Logging configured
- [ ] Error handling complete
- [ ] Docker image builds successfully
- [ ] API documentation generated
- [ ] Performance benchmarks meet targets
- [ ] Security review (no secrets in code)

---

## Development Commands

```bash
# Testing
pytest -m unit -v                    # Unit tests only
pytest -m integration -v             # Integration tests
pytest --cov=. --cov-report=html     # With coverage

# Linting
black .                              # Format code
flake8 .                             # Check style
isort .                              # Sort imports

# Git
git status
git add -A && git commit -m "message"
git push -u origin <branch-name>

# Development
python ui/gradio_app.py              # Run Gradio
python api/server.py                 # Run API
python main.py --help                # Show options
```

---

## Contributing

1. Create feature branch: `claude/<feature>-<session-id>`
2. Follow directory structure and naming conventions
3. Add tests for new functionality
4. Run `pytest -m unit` before committing
5. Use lazy imports for heavy dependencies
6. Update this README for new features

---

## License

[Add your license here]

---

## References

- [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
- [Hugging Face Model](https://huggingface.co/nielsr/layoutlmv3-finetuned-funsd)
- [FUNSD Dataset](https://guillaumejaume.github.io/FUNSD/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Gradio Documentation](https://gradio.app/docs/)

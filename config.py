"""
Central configuration management for LayoutLM Document Processing Service.
All settings are loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """Model-related configuration."""
    name: str = "nielsr/layoutlmv3-finetuned-funsd"
    cache_dir: str = "./models"
    confidence_threshold: float = 0.5


@dataclass
class OCRConfig:
    """OCR engine configuration."""
    engine: str = "easyocr"
    languages: List[str] = field(default_factory=lambda: ["en"])


@dataclass
class HardwareConfig:
    """Hardware and resource configuration."""
    device: str = "auto"  # auto, cpu, cuda, cuda:0
    max_batch_size: int = 8


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


@dataclass
class UIConfig:
    """Gradio UI configuration."""
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    share: bool = False


@dataclass
class LogConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"


@dataclass
class Config:
    """Main configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    api: APIConfig = field(default_factory=APIConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    log: LogConfig = field(default_factory=LogConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            model=ModelConfig(
                name=os.getenv("MODEL_NAME", "nielsr/layoutlmv3-finetuned-funsd"),
                cache_dir=os.getenv("MODEL_CACHE_DIR", "./models"),
                confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),
            ),
            ocr=OCRConfig(
                engine=os.getenv("OCR_ENGINE", "easyocr"),
                languages=os.getenv("OCR_LANGUAGES", "en").split(","),
            ),
            hardware=HardwareConfig(
                device=os.getenv("DEVICE", "auto"),
                max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "8")),
            ),
            api=APIConfig(
                host=os.getenv("API_HOST", "0.0.0.0"),
                port=int(os.getenv("API_PORT", "8000")),
                workers=int(os.getenv("API_WORKERS", "1")),
            ),
            ui=UIConfig(
                server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
                server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
                share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
            ),
            log=LogConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
                format=os.getenv("LOG_FORMAT", "json"),
            ),
        )


# Global configuration instance
_config: Config = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None


if __name__ == "__main__":
    print("=" * 60)
    print("CONFIG TEST")
    print("=" * 60)

    config = get_config()
    print(f"  Model: {config.model.name}")
    print(f"  Device: {config.hardware.device}")
    print(f"  API Port: {config.api.port}")
    print(f"  UI Port: {config.ui.server_port}")
    print(f"  Log Level: {config.log.level}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

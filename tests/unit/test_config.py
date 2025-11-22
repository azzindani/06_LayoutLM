"""
Unit tests for configuration management.
"""

import os
import pytest

from config import Config, get_config, reset_config


@pytest.mark.unit
class TestConfig:
    """Tests for Config class."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def test_config_defaults(self):
        """Config has sensible defaults."""
        config = Config()

        assert config.model.name == "nielsr/layoutlmv3-finetuned-funsd"
        assert config.model.confidence_threshold == 0.5
        assert config.hardware.device == "auto"
        assert config.api.port == 8000
        assert config.ui.server_port == 7860

    def test_config_from_env(self):
        """Config reads from environment variables."""
        os.environ["MODEL_NAME"] = "test-model"
        os.environ["CONFIDENCE_THRESHOLD"] = "0.8"
        os.environ["API_PORT"] = "9000"

        try:
            config = Config.from_env()

            assert config.model.name == "test-model"
            assert config.model.confidence_threshold == 0.8
            assert config.api.port == 9000
        finally:
            # Clean up
            del os.environ["MODEL_NAME"]
            del os.environ["CONFIDENCE_THRESHOLD"]
            del os.environ["API_PORT"]

    def test_get_config_singleton(self):
        """get_config returns singleton instance."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_reset_config(self):
        """reset_config clears singleton."""
        config1 = get_config()
        reset_config()
        config2 = get_config()

        assert config1 is not config2

    def test_config_validation(self):
        """Config validates values correctly."""
        config = Config()

        # Confidence threshold should be 0-1
        assert 0 <= config.model.confidence_threshold <= 1

        # Ports should be positive
        assert config.api.port > 0
        assert config.ui.server_port > 0

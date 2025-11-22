"""
Infrastructure module - lazy imports.
"""

__all__ = ['get_logger', 'setup_logging', 'detect_device', 'LayoutLMServiceError']


def __getattr__(name):
    if name in ('get_logger', 'setup_logging'):
        from infrastructure.logger_utils import get_logger, setup_logging
        return get_logger if name == 'get_logger' else setup_logging
    if name in ('detect_device', 'DeviceInfo'):
        from infrastructure.hardware_detection import detect_device, DeviceInfo
        return detect_device if name == 'detect_device' else DeviceInfo
    if name == 'LayoutLMServiceError':
        from infrastructure.exceptions import LayoutLMServiceError
        return LayoutLMServiceError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

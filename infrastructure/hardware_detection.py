"""
Hardware detection and device management.
"""

import os
from dataclasses import dataclass
from typing import Optional

from infrastructure.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DeviceInfo:
    """Information about available compute devices."""
    device_type: str  # 'cpu' or 'cuda'
    device_id: Optional[int]  # GPU index if cuda
    device_name: str  # Human-readable name
    total_memory_gb: Optional[float]  # GPU memory if available

    @property
    def torch_device(self) -> str:
        """Get the PyTorch device string."""
        if self.device_type == "cpu":
            return "cpu"
        return f"cuda:{self.device_id}" if self.device_id is not None else "cuda"


def detect_device(preferred: str = "auto") -> DeviceInfo:
    """
    Detect the best available compute device.

    Args:
        preferred: Preferred device ('auto', 'cpu', 'cuda', 'cuda:0', etc.)

    Returns:
        DeviceInfo with detected device information
    """
    # Lazy import torch to avoid import errors in CI
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False
        logger.warning("PyTorch not available, using CPU")

    # Handle explicit CPU request
    if preferred == "cpu":
        return DeviceInfo(
            device_type="cpu",
            device_id=None,
            device_name="CPU",
            total_memory_gb=None
        )

    # Handle explicit CUDA device
    if preferred.startswith("cuda"):
        if not cuda_available:
            logger.warning(f"Requested {preferred} but CUDA not available, falling back to CPU")
            return DeviceInfo(
                device_type="cpu",
                device_id=None,
                device_name="CPU",
                total_memory_gb=None
            )

        # Parse device ID
        device_id = 0
        if ":" in preferred:
            device_id = int(preferred.split(":")[1])

        import torch
        device_name = torch.cuda.get_device_name(device_id)
        total_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)

        return DeviceInfo(
            device_type="cuda",
            device_id=device_id,
            device_name=device_name,
            total_memory_gb=round(total_memory, 2)
        )

    # Auto-detect best device
    if preferred == "auto":
        if cuda_available:
            import torch
            device_id = 0
            device_name = torch.cuda.get_device_name(device_id)
            total_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)

            logger.info(f"Auto-detected GPU: {device_name} ({total_memory:.2f} GB)")

            return DeviceInfo(
                device_type="cuda",
                device_id=device_id,
                device_name=device_name,
                total_memory_gb=round(total_memory, 2)
            )
        else:
            logger.info("No GPU available, using CPU")
            return DeviceInfo(
                device_type="cpu",
                device_id=None,
                device_name="CPU",
                total_memory_gb=None
            )

    # Default to CPU
    return DeviceInfo(
        device_type="cpu",
        device_id=None,
        device_name="CPU",
        total_memory_gb=None
    )


def get_optimal_batch_size(device_info: DeviceInfo, base_size: int = 8) -> int:
    """
    Calculate optimal batch size based on device memory.

    Args:
        device_info: Device information
        base_size: Base batch size for standard GPU

    Returns:
        Recommended batch size
    """
    if device_info.device_type == "cpu":
        return max(1, base_size // 4)

    if device_info.total_memory_gb is None:
        return base_size

    # Scale batch size based on GPU memory
    # Assuming 4GB as baseline for batch_size=8
    memory_factor = device_info.total_memory_gb / 4.0
    return max(1, int(base_size * memory_factor))


if __name__ == "__main__":
    print("=" * 60)
    print("HARDWARE DETECTION TEST")
    print("=" * 60)

    # Test auto detection
    device = detect_device("auto")
    print(f"  Device: {device.device_name}")
    print(f"  Type: {device.device_type}")
    print(f"  PyTorch device: {device.torch_device}")
    if device.total_memory_gb:
        print(f"  Memory: {device.total_memory_gb} GB")

    # Test batch size calculation
    batch_size = get_optimal_batch_size(device)
    print(f"  Optimal batch size: {batch_size}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

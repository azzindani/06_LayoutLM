#!/usr/bin/env python
"""
Performance benchmarking for document processing.
"""

import os
import sys
import time
import statistics
from typing import List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw


def create_test_image(width: int = 800, height: int = 600) -> Image.Image:
    """Create a test image with sample text."""
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Add sample text
    draw.text((50, 50), "Name:", fill='black')
    draw.text((150, 50), "John Doe", fill='black')
    draw.text((50, 100), "Date:", fill='black')
    draw.text((150, 100), "2024-01-15", fill='black')
    draw.text((50, 150), "Address:", fill='black')
    draw.text((150, 150), "123 Main Street", fill='black')
    draw.text((50, 200), "City:", fill='black')
    draw.text((150, 200), "New York", fill='black')

    return img


def benchmark_single(processor, image: Image.Image, runs: int = 10) -> dict:
    """
    Benchmark single image processing.

    Args:
        processor: DocumentProcessor instance
        image: Test image
        runs: Number of runs

    Returns:
        Benchmark results
    """
    times = []

    # Warmup
    processor.process_image(image)

    # Benchmark
    for _ in range(runs):
        start = time.time()
        processor.process_image(image)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    return {
        "runs": runs,
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0
    }


def benchmark_batch(processor, images: List[Image.Image], runs: int = 5) -> dict:
    """
    Benchmark batch processing.

    Args:
        processor: DocumentProcessor instance
        images: List of test images
        runs: Number of runs

    Returns:
        Benchmark results
    """
    times = []

    # Warmup
    processor.process_batch(images)

    # Benchmark
    for _ in range(runs):
        start = time.time()
        processor.process_batch(images)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    batch_size = len(images)
    mean_time = statistics.mean(times)

    return {
        "batch_size": batch_size,
        "runs": runs,
        "total_mean_ms": mean_time,
        "per_image_ms": mean_time / batch_size,
        "throughput_per_min": (60000 / mean_time) * batch_size
    }


def run_benchmarks(device: str = "auto"):
    """Run all benchmarks."""
    from core.pipeline.document_processor import DocumentProcessor
    from config import get_config

    print("=" * 60)
    print("LayoutLMv3 Performance Benchmark")
    print("=" * 60)

    # Initialize processor
    config = get_config()
    config.hardware.device = device

    processor = DocumentProcessor(config)
    processor.initialize()

    print(f"\nDevice: {processor._device.device_name}")
    print(f"Model: {config.model.name}")

    # Create test images
    test_image = create_test_image()
    batch_images = [create_test_image() for _ in range(8)]

    # Single image benchmark
    print("\n" + "-" * 60)
    print("Single Image Processing")
    print("-" * 60)

    single_results = benchmark_single(processor, test_image, runs=10)
    print(f"  Runs: {single_results['runs']}")
    print(f"  Mean: {single_results['mean_ms']:.2f} ms")
    print(f"  Median: {single_results['median_ms']:.2f} ms")
    print(f"  Min: {single_results['min_ms']:.2f} ms")
    print(f"  Max: {single_results['max_ms']:.2f} ms")
    print(f"  Std Dev: {single_results['std_ms']:.2f} ms")

    # Batch benchmark
    print("\n" + "-" * 60)
    print("Batch Processing")
    print("-" * 60)

    batch_results = benchmark_batch(processor, batch_images, runs=5)
    print(f"  Batch Size: {batch_results['batch_size']}")
    print(f"  Total Time: {batch_results['total_mean_ms']:.2f} ms")
    print(f"  Per Image: {batch_results['per_image_ms']:.2f} ms")
    print(f"  Throughput: {batch_results['throughput_per_min']:.1f} images/min")

    # Cleanup
    processor.shutdown()

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)

    return {
        "single": single_results,
        "batch": batch_results
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run performance benchmarks"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto, cpu, cuda)"
    )

    args = parser.parse_args()
    run_benchmarks(args.device)

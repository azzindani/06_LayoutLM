#!/usr/bin/env python
"""
Example usage of the LayoutLMv3 Document Processing Service.

Run with: python -m core.example_usage
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    code = '''
from PIL import Image
from core.pipeline.document_processor import DocumentProcessor

# Initialize processor
processor = DocumentProcessor()
processor.initialize()

# Process an image
result = processor.process_image("document.png")

# Print results
for entity in result["results"][0]["entities"]:
    print(f"{entity['label']}: {entity['text']}")

# Cleanup
processor.shutdown()
'''
    print(code)


def example_custom_config():
    """Custom configuration example."""
    print("=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)

    code = '''
import os

# Set environment variables before importing
os.environ["MODEL_NAME"] = "nielsr/layoutlmv3-finetuned-funsd"
os.environ["DEVICE"] = "cuda:0"
os.environ["CONFIDENCE_THRESHOLD"] = "0.7"

from core.pipeline.document_processor import DocumentProcessor

processor = DocumentProcessor()
processor.initialize()

# Process with custom threshold
result = processor.process_image("document.png")
'''
    print(code)


def example_batch_processing():
    """Batch processing example."""
    print("=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)

    code = '''
from core.pipeline.document_processor import DocumentProcessor

processor = DocumentProcessor()
processor.initialize()

# Process multiple images
images = ["doc1.png", "doc2.png", "doc3.png"]
results = processor.process_batch(images)

for i, result in enumerate(results):
    print(f"Document {i+1}: {len(result['results'][0]['entities'])} entities")

processor.shutdown()
'''
    print(code)


def example_export_formats():
    """Export formats example."""
    print("=" * 60)
    print("Example 4: Export to Different Formats")
    print("=" * 60)

    code = '''
from core.pipeline.document_processor import DocumentProcessor
from core.pipeline.export import export_json, export_csv, export_xml

processor = DocumentProcessor()
processor.initialize()
result = processor.process_image("document.png")

# Export to JSON
with open("output.json", "w") as f:
    f.write(export_json(result))

# Export to CSV
with open("output.csv", "w") as f:
    f.write(export_csv(result))

# Export to XML
with open("output.xml", "w") as f:
    f.write(export_xml(result))
'''
    print(code)


def example_api_usage():
    """API usage example."""
    print("=" * 60)
    print("Example 5: Using the REST API")
    print("=" * 60)

    code = '''
import requests

# Start the API server: python main.py --mode api

# Process a document
with open("document.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/process",
        files={"file": f},
        data={"confidence_threshold": 0.5}
    )

result = response.json()
print(f"Found {len(result['results'][0]['entities'])} entities")

# Health check
response = requests.get("http://localhost:8000/health")
print(f"Status: {response.json()['status']}")
'''
    print(code)


def example_gradio_customization():
    """Gradio customization example."""
    print("=" * 60)
    print("Example 6: Customizing Gradio UI")
    print("=" * 60)

    code = '''
import gradio as gr
from core.pipeline.document_processor import get_processor

processor = get_processor()
processor.initialize()

def process(image, threshold):
    processor.config.model.confidence_threshold = threshold
    result = processor.process_image(image)
    return result

# Create custom interface
demo = gr.Interface(
    fn=process,
    inputs=[
        gr.Image(type="pil"),
        gr.Slider(0, 1, value=0.5, label="Confidence")
    ],
    outputs=gr.JSON()
)

demo.launch()
'''
    print(code)


def run_live_demo():
    """Run a live demo if dependencies are available."""
    print("=" * 60)
    print("Live Demo")
    print("=" * 60)

    try:
        from PIL import Image, ImageDraw
        from core.pipeline.document_processor import DocumentProcessor
        from core.pipeline.export import export_json

        # Create sample document
        img = Image.new('RGB', (600, 400), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "Name:", fill='black')
        draw.text((150, 50), "Alice Smith", fill='black')
        draw.text((50, 100), "Email:", fill='black')
        draw.text((150, 100), "alice@example.com", fill='black')

        # Process
        processor = DocumentProcessor()
        processor.initialize()
        result = processor.process_image(img)

        print("\nProcessing Result:")
        print("-" * 40)
        for entity in result["results"][0]["entities"]:
            print(f"  [{entity['label']:10}] {entity['text']}")

        print(f"\nProcessing time: {result['processing_time_ms']:.2f}ms")
        processor.shutdown()

    except ImportError as e:
        print(f"\nCannot run live demo: {e}")
        print("Install dependencies with: pip install -r requirements.txt")


def main():
    print("\n" + "=" * 60)
    print("LAYOUTLMV3 DOCUMENT PROCESSING - EXAMPLES")
    print("=" * 60 + "\n")

    example_basic_usage()
    print()

    example_custom_config()
    print()

    example_batch_processing()
    print()

    example_export_formats()
    print()

    example_api_usage()
    print()

    example_gradio_customization()
    print()

    run_live_demo()

    print("\n" + "=" * 60)
    print("For more information, see README.md")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

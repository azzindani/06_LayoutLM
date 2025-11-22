"""
Gradio web interface for document processing.
"""

import json
from PIL import Image, ImageDraw

import gradio as gr

from config import get_config
from core.pipeline.document_processor import get_processor
from core.pipeline.export import export_json, export_csv
from infrastructure.logger_utils import setup_logging, get_logger

# Initialize
config = get_config()
setup_logging(level=config.log.level, format_type="standard")
logger = get_logger(__name__)

# Color mapping for labels
LABEL_COLORS = {
    "HEADER": (255, 165, 0),    # Orange
    "QUESTION": (0, 0, 255),    # Blue
    "ANSWER": (0, 255, 0),      # Green
    "OTHER": (128, 128, 128),   # Gray
}


def draw_entities(image: Image.Image, entities: list) -> Image.Image:
    """
    Draw bounding boxes and labels on image.

    Args:
        image: PIL Image
        entities: List of entity dictionaries

    Returns:
        Annotated image
    """
    # Make a copy to draw on
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for entity in entities:
        bbox = entity.get("bbox", {})
        label = entity.get("label", "OTHER")
        text = entity.get("text", "")
        confidence = entity.get("confidence", 0)

        # Get coordinates
        x1 = bbox.get("x1", 0)
        y1 = bbox.get("y1", 0)
        x2 = bbox.get("x2", 0)
        y2 = bbox.get("y2", 0)

        # Get color for label
        color = LABEL_COLORS.get(label, LABEL_COLORS["OTHER"])

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label
        label_text = f"{label}: {confidence:.2f}"
        draw.text((x1, y1 - 15), label_text, fill=color)

    return annotated


def process_document(
    image,
    confidence_threshold: float = 0.5
):
    """
    Process uploaded document image.

    Args:
        image: Uploaded image
        confidence_threshold: Minimum confidence for entities

    Returns:
        Tuple of (annotated_image, json_results, csv_results)
    """
    if image is None:
        return None, "No image uploaded", ""

    try:
        # Get processor
        processor = get_processor()

        # Update confidence threshold in config
        processor.config.model.confidence_threshold = confidence_threshold

        # Process image
        if not hasattr(processor, '_ocr_engine') or processor._ocr_engine is None:
            processor.initialize()

        result = processor.process_image(image)

        # Get entities
        entities = []
        if result.get("results"):
            entities = result["results"][0].get("entities", [])

        # Draw annotations
        pil_image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        annotated = draw_entities(pil_image, entities)

        # Format outputs
        json_output = export_json(result)
        csv_output = export_csv(result)

        # Summary
        summary = f"Found {len(entities)} entities in {result.get('processing_time_ms', 0):.2f}ms"

        return annotated, json_output, csv_output, summary

    except Exception as e:
        logger.error(f"Processing error: {e}")
        return None, f"Error: {str(e)}", "", f"Error: {str(e)}"


def create_demo():
    """Create the Gradio interface."""
    with gr.Blocks(title="LayoutLMv3 Document Processor") as demo:
        gr.Markdown("""
        # LayoutLMv3 Document Processing

        Upload a document image to extract structured information (questions, answers, headers).
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Input
                input_image = gr.Image(
                    label="Upload Document",
                    type="pil"
                )

                confidence_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Confidence Threshold"
                )

                process_btn = gr.Button("Process Document", variant="primary")

            with gr.Column(scale=1):
                # Output image
                output_image = gr.Image(
                    label="Annotated Document",
                    type="pil"
                )

                status_text = gr.Textbox(
                    label="Status",
                    interactive=False
                )

        with gr.Row():
            with gr.Column():
                # JSON output
                json_output = gr.Code(
                    label="JSON Results",
                    language="json",
                    lines=15
                )

            with gr.Column():
                # CSV output
                csv_output = gr.Code(
                    label="CSV Results",
                    language="plaintext",
                    lines=15
                )

        # Download buttons
        with gr.Row():
            json_download = gr.File(label="Download JSON")
            csv_download = gr.File(label="Download CSV")

        # Process button action
        process_btn.click(
            fn=process_document,
            inputs=[input_image, confidence_slider],
            outputs=[output_image, json_output, csv_output, status_text]
        )

        # Examples
        gr.Markdown("### Color Legend")
        gr.Markdown("""
        - 🟠 **Orange**: Headers
        - 🔵 **Blue**: Questions/Labels
        - 🟢 **Green**: Answers/Values
        - ⚪ **Gray**: Other
        """)

    return demo


def run_ui():
    """Run the Gradio interface."""
    config = get_config()

    # Initialize processor
    processor = get_processor()
    try:
        processor.initialize()
        logger.info("Processor initialized for Gradio UI")
    except Exception as e:
        logger.warning(f"Processor initialization deferred: {e}")

    # Create and launch demo
    demo = create_demo()
    demo.launch(
        server_name=config.ui.server_name,
        server_port=config.ui.server_port,
        share=config.ui.share
    )


if __name__ == "__main__":
    run_ui()

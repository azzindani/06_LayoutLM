"""
Gradio web interface for document processing.
Styled to match the original Jupyter notebook visualization.
"""

import json
from PIL import Image, ImageDraw, ImageFont

import gradio as gr

from config import get_config
from core.pipeline.document_processor import get_processor
from core.pipeline.export import export_json, export_csv
from infrastructure.logger_utils import setup_logging, get_logger

# Initialize
config = get_config()
setup_logging(level=config.log.level, format_type="standard")
logger = get_logger(__name__)

# Color mapping matching the notebook style
# label2color = {'question': 'blue', 'answer': 'green', 'header': 'orange', 'other': 'violet'}
LABEL_COLORS = {
    "HEADER": "orange",
    "QUESTION": "blue",
    "ANSWER": "green",
    "OTHER": "violet",
}


def draw_entities(image: Image.Image, entities: list) -> Image.Image:
    """
    Draw bounding boxes and labels on image.
    Matches the visualization style from the original notebook.

    Args:
        image: PIL Image
        entities: List of entity dictionaries

    Returns:
        Annotated image
    """
    # Make a copy to draw on
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Try to load default font
    try:
        font = ImageFont.load_default()
    except:
        font = None

    # Track drawn boxes to avoid duplicates (like in notebook)
    drawn_boxes = set()

    for entity in entities:
        bbox = entity.get("bbox", {})
        label = entity.get("label", "OTHER").lower()
        text = entity.get("text", "")
        confidence = entity.get("confidence", 0)

        # Get coordinates
        x1 = bbox.get("x1", 0)
        y1 = bbox.get("y1", 0)
        x2 = bbox.get("x2", 0)
        y2 = bbox.get("y2", 0)

        # Convert to tuple for set comparison
        box_tuple = (x1, y1, x2, y2)

        # Skip if already drawn
        if box_tuple in drawn_boxes:
            continue

        # Get color for label (matching notebook style)
        color = LABEL_COLORS.get(label.upper(), "violet")

        # Draw rectangle with width=2 (like notebook)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label text (like notebook: label at top-left with offset)
        label_text = label
        if font:
            draw.text((x1 + 10, y1 - 10), text=label_text, fill=color, font=font)
        else:
            draw.text((x1 + 10, y1 - 10), text=label_text, fill=color)

        # Add to drawn boxes
        drawn_boxes.add(box_tuple)

    return annotated


def process_document(image, confidence_threshold: float = 0.5):
    """
    Process uploaded document image.

    Args:
        image: Uploaded image
        confidence_threshold: Minimum confidence for entities

    Returns:
        Tuple of (annotated_image, json_results, status)
    """
    if image is None:
        return None, "{}", "No image uploaded"

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

        # Summary
        summary = f"Found {len(entities)} entities in {result.get('processing_time_ms', 0):.2f}ms"

        return annotated, json_output, summary

    except Exception as e:
        logger.error(f"Processing error: {e}")
        return None, "{}", f"Error: {str(e)}"


def create_demo():
    """Create the Gradio interface with notebook-like theme."""

    # Simple, clean interface matching notebook style
    with gr.Blocks(
        title="LayoutLMv3 Document Processor",
        theme=gr.themes.Default()
    ) as demo:

        gr.Markdown("""
        # LayoutLMv3 Document Processing

        Upload a document image to extract structured information.

        **Color Legend:**
        - <span style="color:orange">■</span> **Header** - Document titles/headers
        - <span style="color:blue">■</span> **Question** - Form field labels
        - <span style="color:green">■</span> **Answer** - Form field values
        - <span style="color:violet">■</span> **Other** - Other text
        """)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Input Document",
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

            with gr.Column():
                output_image = gr.Image(
                    label="Annotated Result",
                    type="pil"
                )

                status_text = gr.Textbox(
                    label="Status",
                    interactive=False
                )

        with gr.Row():
            json_output = gr.Code(
                label="JSON Output",
                language="json",
                lines=20
            )

        # Process action
        process_btn.click(
            fn=process_document,
            inputs=[input_image, confidence_slider],
            outputs=[output_image, json_output, status_text]
        )

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

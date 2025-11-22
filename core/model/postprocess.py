"""
Post-processing utilities for model outputs.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from core.model.loader import get_label_mapping
from infrastructure.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Entity:
    """Extracted entity from document."""
    text: str
    label: str
    confidence: float
    bbox: Dict[str, int]  # x1, y1, x2, y2


def process_predictions(
    inference_result: Dict[str, Any],
    confidence_threshold: float = 0.5
) -> List[Entity]:
    """
    Process model predictions into structured entities.

    Args:
        inference_result: Output from run_inference
        confidence_threshold: Minimum confidence to include entity

    Returns:
        List of Entity objects
    """
    predictions = inference_result["predictions"]
    confidence_scores = inference_result["confidence_scores"]
    word_ids = inference_result["word_ids"]
    words = inference_result["words"]
    boxes = inference_result["boxes"]

    label_map = get_label_mapping()
    entities = []

    # Track processed words to avoid duplicates
    processed_words = set()

    for idx, (pred, conf) in enumerate(zip(predictions, confidence_scores)):
        word_id = word_ids[idx] if idx < len(word_ids) else None

        # Skip special tokens and already processed words
        if word_id is None or word_id in processed_words:
            continue

        processed_words.add(word_id)

        # Get label name
        label_name = label_map.get(pred, "O")

        # Skip 'O' (Other) labels and low confidence
        if label_name == "O" or conf < confidence_threshold:
            continue

        # Get word and bbox
        if word_id < len(words):
            text = words[word_id]
            bbox = boxes[word_id]

            # Convert to simplified label (remove B-/I- prefix)
            simple_label = label_name.split("-")[-1] if "-" in label_name else label_name

            entities.append(Entity(
                text=text,
                label=simple_label,
                confidence=float(conf),
                bbox={
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3]
                }
            ))

    logger.debug(f"Processed {len(entities)} entities")
    return entities


def aggregate_entities(entities: List[Entity]) -> List[Entity]:
    """
    Aggregate consecutive entities with the same label.

    Args:
        entities: List of entities to aggregate

    Returns:
        Aggregated entities
    """
    if not entities:
        return []

    aggregated = []
    current = None

    for entity in entities:
        if current is None:
            current = entity
        elif entity.label == current.label:
            # Check if adjacent (simple proximity check)
            if abs(entity.bbox["x1"] - current.bbox["x2"]) < 50:
                # Merge entities
                current = Entity(
                    text=f"{current.text} {entity.text}",
                    label=current.label,
                    confidence=min(current.confidence, entity.confidence),
                    bbox={
                        "x1": min(current.bbox["x1"], entity.bbox["x1"]),
                        "y1": min(current.bbox["y1"], entity.bbox["y1"]),
                        "x2": max(current.bbox["x2"], entity.bbox["x2"]),
                        "y2": max(current.bbox["y2"], entity.bbox["y2"])
                    }
                )
            else:
                aggregated.append(current)
                current = entity
        else:
            aggregated.append(current)
            current = entity

    if current:
        aggregated.append(current)

    return aggregated


def format_output(
    entities: List[Entity],
    image_size: tuple,
    processing_time_ms: float,
    model_version: str = "layoutlmv3-funsd-v1",
    ocr_engine: str = "easyocr"
) -> Dict[str, Any]:
    """
    Format entities into the standard output format.

    Args:
        entities: List of extracted entities
        image_size: (width, height) of processed image
        processing_time_ms: Processing time in milliseconds
        model_version: Version of the model used
        ocr_engine: OCR engine used

    Returns:
        Formatted output dictionary
    """
    return {
        "status": "success",
        "processing_time_ms": round(processing_time_ms, 2),
        "results": [
            {
                "page": 1,
                "entities": [
                    {
                        "text": e.text,
                        "label": e.label,
                        "confidence": round(e.confidence, 3),
                        "bbox": e.bbox
                    }
                    for e in entities
                ]
            }
        ],
        "metadata": {
            "model_version": model_version,
            "ocr_engine": ocr_engine,
            "image_size": list(image_size)
        }
    }


def unnormalize_bbox(
    bbox: List[int],
    image_width: int,
    image_height: int
) -> Dict[str, int]:
    """
    Convert normalized bbox (0-1000 scale) to pixel coordinates.

    Args:
        bbox: Normalized bounding box [x1, y1, x2, y2]
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Bounding box in pixel coordinates
    """
    return {
        "x1": int(bbox[0] * image_width / 1000),
        "y1": int(bbox[1] * image_height / 1000),
        "x2": int(bbox[2] * image_width / 1000),
        "y2": int(bbox[3] * image_height / 1000)
    }


if __name__ == "__main__":
    print("=" * 60)
    print("POSTPROCESS TEST")
    print("=" * 60)

    # Test unnormalize_bbox
    bbox = [100, 200, 300, 250]
    result = unnormalize_bbox(bbox, 1000, 1000)
    print(f"  Unnormalize bbox: {bbox} -> {result}")

    # Test entity creation
    entity = Entity(
        text="Name:",
        label="QUESTION",
        confidence=0.95,
        bbox={"x1": 10, "y1": 20, "x2": 80, "y2": 40}
    )
    print(f"  Entity: {entity.text} ({entity.label})")

    # Test format_output
    output = format_output(
        entities=[entity],
        image_size=(800, 600),
        processing_time_ms=1234.5
    )
    print(f"  Output status: {output['status']}")
    print(f"  Entities count: {len(output['results'][0]['entities'])}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

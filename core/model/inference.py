"""
Model inference logic.
"""

from typing import List, Dict, Any
from PIL import Image

from core.ocr.base import OCRResult
from core.model.loader import load_model, get_label_mapping
from infrastructure.logger_utils import get_logger
from infrastructure.exceptions import InferenceError

logger = get_logger(__name__)


def run_inference(
    image: Image.Image,
    ocr_results: List[OCRResult],
    model_name: str = "nielsr/layoutlmv3-finetuned-funsd",
    device: str = "auto",
    cache_dir: str = "./models"
) -> Dict[str, Any]:
    """
    Run LayoutLMv3 inference on an image with OCR results.

    Args:
        image: PIL Image
        ocr_results: List of OCR results with text and bounding boxes
        model_name: HuggingFace model name
        device: Device to run inference on
        cache_dir: Model cache directory

    Returns:
        Dictionary with predictions and metadata
    """
    try:
        import torch

        # Load model and processor
        model, processor = load_model(model_name, cache_dir, device)

        # Prepare inputs
        words = [r.text for r in ocr_results]
        boxes = [list(r.bbox) for r in ocr_results]

        # Normalize bounding boxes to 0-1000 scale
        width, height = image.size
        normalized_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            normalized_boxes.append([
                int(1000 * x1 / width),
                int(1000 * y1 / height),
                int(1000 * x2 / width),
                int(1000 * y2 / height)
            ])

        # Encode inputs
        encoding = processor(
            image,
            words,
            boxes=normalized_boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        )

        # Move to device
        device_str = next(model.parameters()).device
        encoding = {k: v.to(device_str) for k, v in encoding.items()}

        # Run inference
        with torch.no_grad():
            outputs = model(**encoding)

        # Get predictions
        logits = outputs.logits
        predictions = logits.argmax(-1).squeeze().tolist()

        # Handle single prediction case
        if isinstance(predictions, int):
            predictions = [predictions]

        # Get confidence scores
        probabilities = torch.softmax(logits, dim=-1)
        confidence_scores = probabilities.max(-1).values.squeeze().tolist()

        if isinstance(confidence_scores, float):
            confidence_scores = [confidence_scores]

        # Get token-to-word mapping
        word_ids = encoding.word_ids()

        return {
            "predictions": predictions,
            "confidence_scores": confidence_scores,
            "word_ids": word_ids,
            "words": words,
            "boxes": boxes,
            "normalized_boxes": normalized_boxes,
            "image_size": (width, height)
        }

    except Exception as e:
        raise InferenceError(f"Inference failed: {e}")


def batch_inference(
    images: List[Image.Image],
    ocr_results_list: List[List[OCRResult]],
    model_name: str = "nielsr/layoutlmv3-finetuned-funsd",
    device: str = "auto",
    cache_dir: str = "./models"
) -> List[Dict[str, Any]]:
    """
    Run batch inference on multiple images.

    Args:
        images: List of PIL Images
        ocr_results_list: List of OCR results for each image
        model_name: HuggingFace model name
        device: Device to run inference on
        cache_dir: Model cache directory

    Returns:
        List of inference results
    """
    results = []
    for image, ocr_results in zip(images, ocr_results_list):
        result = run_inference(
            image, ocr_results, model_name, device, cache_dir
        )
        results.append(result)
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("INFERENCE TEST")
    print("=" * 60)

    # Create test data
    from PIL import Image, ImageDraw

    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Name:", fill='black')
    draw.text((100, 10), "John Doe", fill='black')

    # Mock OCR results
    ocr_results = [
        OCRResult(text="Name:", bbox=(10, 10, 80, 30), confidence=0.95),
        OCRResult(text="John Doe", bbox=(100, 10, 200, 30), confidence=0.92),
    ]

    try:
        result = run_inference(img, ocr_results, device="cpu")
        print(f"  Predictions: {result['predictions'][:10]}...")
        print(f"  Words: {result['words']}")
        print(f"  Image size: {result['image_size']}")
    except Exception as e:
        print(f"  Inference test skipped: {e}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

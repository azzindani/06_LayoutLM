"""
Export utilities for different output formats.
"""

import json
import csv
import io
from typing import Dict, Any, List
from xml.etree import ElementTree as ET
from xml.dom import minidom

from infrastructure.logger_utils import get_logger
from infrastructure.exceptions import ExportError

logger = get_logger(__name__)


def export_json(result: Dict[str, Any], pretty: bool = True) -> str:
    """
    Export result to JSON string.

    Args:
        result: Processing result dictionary
        pretty: Whether to format with indentation

    Returns:
        JSON string
    """
    try:
        if pretty:
            return json.dumps(result, indent=2)
        return json.dumps(result)
    except Exception as e:
        raise ExportError(f"JSON export failed: {e}")


def export_csv(result: Dict[str, Any]) -> str:
    """
    Export result to CSV string.

    Args:
        result: Processing result dictionary

    Returns:
        CSV string
    """
    try:
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            "page", "text", "label", "confidence",
            "x1", "y1", "x2", "y2"
        ])

        # Write entities
        for page_result in result.get("results", []):
            page_num = page_result.get("page", 1)
            for entity in page_result.get("entities", []):
                bbox = entity.get("bbox", {})
                writer.writerow([
                    page_num,
                    entity.get("text", ""),
                    entity.get("label", ""),
                    entity.get("confidence", 0),
                    bbox.get("x1", 0),
                    bbox.get("y1", 0),
                    bbox.get("x2", 0),
                    bbox.get("y2", 0)
                ])

        return output.getvalue()
    except Exception as e:
        raise ExportError(f"CSV export failed: {e}")


def export_xml(result: Dict[str, Any]) -> str:
    """
    Export result to XML string.

    Args:
        result: Processing result dictionary

    Returns:
        XML string
    """
    try:
        root = ET.Element("document")

        # Metadata
        metadata = ET.SubElement(root, "metadata")
        for key, value in result.get("metadata", {}).items():
            elem = ET.SubElement(metadata, key)
            elem.text = str(value)

        # Processing info
        ET.SubElement(root, "status").text = result.get("status", "")
        ET.SubElement(root, "processing_time_ms").text = str(
            result.get("processing_time_ms", 0)
        )

        # Results
        results_elem = ET.SubElement(root, "results")
        for page_result in result.get("results", []):
            page_elem = ET.SubElement(results_elem, "page")
            page_elem.set("number", str(page_result.get("page", 1)))

            for entity in page_result.get("entities", []):
                entity_elem = ET.SubElement(page_elem, "entity")
                ET.SubElement(entity_elem, "text").text = entity.get("text", "")
                ET.SubElement(entity_elem, "label").text = entity.get("label", "")
                ET.SubElement(entity_elem, "confidence").text = str(
                    entity.get("confidence", 0)
                )

                bbox = entity.get("bbox", {})
                bbox_elem = ET.SubElement(entity_elem, "bbox")
                for key in ["x1", "y1", "x2", "y2"]:
                    bbox_elem.set(key, str(bbox.get(key, 0)))

        # Pretty print
        xml_string = ET.tostring(root, encoding="unicode")
        dom = minidom.parseString(xml_string)
        return dom.toprettyxml(indent="  ")

    except Exception as e:
        raise ExportError(f"XML export failed: {e}")


def export_result(
    result: Dict[str, Any],
    format: str = "json"
) -> str:
    """
    Export result to specified format.

    Args:
        result: Processing result dictionary
        format: Output format ('json', 'csv', 'xml')

    Returns:
        Formatted string
    """
    exporters = {
        "json": export_json,
        "csv": export_csv,
        "xml": export_xml
    }

    if format not in exporters:
        raise ExportError(f"Unknown format: {format}. Supported: {list(exporters.keys())}")

    return exporters[format](result)


if __name__ == "__main__":
    print("=" * 60)
    print("EXPORT TEST")
    print("=" * 60)

    # Test data
    test_result = {
        "status": "success",
        "processing_time_ms": 1234.56,
        "results": [
            {
                "page": 1,
                "entities": [
                    {
                        "text": "Name:",
                        "label": "QUESTION",
                        "confidence": 0.95,
                        "bbox": {"x1": 10, "y1": 20, "x2": 80, "y2": 40}
                    },
                    {
                        "text": "John Doe",
                        "label": "ANSWER",
                        "confidence": 0.92,
                        "bbox": {"x1": 90, "y1": 20, "x2": 200, "y2": 40}
                    }
                ]
            }
        ],
        "metadata": {
            "model_version": "layoutlmv3-funsd-v1",
            "ocr_engine": "easyocr"
        }
    }

    # Test JSON
    json_output = export_json(test_result)
    print(f"  JSON export: {len(json_output)} chars")

    # Test CSV
    csv_output = export_csv(test_result)
    print(f"  CSV export: {len(csv_output)} chars")
    print(f"    Header: {csv_output.split(chr(10))[0]}")

    # Test XML
    xml_output = export_xml(test_result)
    print(f"  XML export: {len(xml_output)} chars")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

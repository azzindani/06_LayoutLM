"""
Unit tests for export utilities.
"""

import json
import pytest

from core.pipeline.export import (
    export_json,
    export_csv,
    export_xml,
    export_result
)
from infrastructure.exceptions import ExportError


@pytest.mark.unit
class TestExport:
    """Tests for export utilities."""

    def test_export_json(self, sample_processing_result):
        """Exports to valid JSON."""
        output = export_json(sample_processing_result)

        # Should be valid JSON
        parsed = json.loads(output)
        assert parsed["status"] == "success"

    def test_export_json_pretty(self, sample_processing_result):
        """Exports pretty-formatted JSON."""
        output = export_json(sample_processing_result, pretty=True)

        # Pretty JSON has newlines
        assert "\n" in output

    def test_export_csv(self, sample_processing_result):
        """Exports to CSV with correct columns."""
        output = export_csv(sample_processing_result)

        lines = output.strip().split("\n")

        # Check header
        header = lines[0]
        assert "page" in header
        assert "text" in header
        assert "label" in header
        assert "confidence" in header

        # Check data rows
        assert len(lines) >= 2  # Header + at least 1 data row

    def test_export_xml(self, sample_processing_result):
        """Exports to valid XML."""
        output = export_xml(sample_processing_result)

        # Should contain XML structure
        assert "<?xml" in output
        assert "<document>" in output
        assert "<status>success</status>" in output
        assert "<entity>" in output

    def test_export_result_json(self, sample_processing_result):
        """export_result handles JSON format."""
        output = export_result(sample_processing_result, "json")
        assert json.loads(output)

    def test_export_result_csv(self, sample_processing_result):
        """export_result handles CSV format."""
        output = export_result(sample_processing_result, "csv")
        assert "page" in output

    def test_export_result_xml(self, sample_processing_result):
        """export_result handles XML format."""
        output = export_result(sample_processing_result, "xml")
        assert "<document>" in output

    def test_export_result_unknown_format(self, sample_processing_result):
        """export_result raises error for unknown format."""
        with pytest.raises(ExportError):
            export_result(sample_processing_result, "unknown")

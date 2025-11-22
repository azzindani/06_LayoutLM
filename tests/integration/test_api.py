"""
Integration tests for API endpoints.
"""

import io
import pytest
from PIL import Image

httpx = pytest.importorskip("httpx")


@pytest.mark.integration
class TestAPIEndpoints:
    """Integration tests for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from api.server import app

        return TestClient(app)

    def test_health_endpoint(self, client):
        """GET /health returns 200."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_ready_endpoint(self, client):
        """GET /ready returns 200."""
        response = client.get("/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_live_endpoint(self, client):
        """GET /live returns 200."""
        response = client.get("/live")

        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_process_endpoint(self, client, sample_image_with_text):
        """POST /process returns results."""
        # Convert image to bytes
        img_bytes = io.BytesIO()
        sample_image_with_text.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        response = client.post(
            "/process",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"confidence_threshold": "0.5"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "results" in data

    def test_process_invalid_file(self, client):
        """POST /process with invalid file returns 400."""
        response = client.post(
            "/process",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )

        assert response.status_code == 400

    def test_batch_endpoint(self, client, sample_image_with_text):
        """POST /batch processes multiple images."""
        # Convert image to bytes
        img_bytes1 = io.BytesIO()
        sample_image_with_text.save(img_bytes1, format='PNG')
        img_bytes1.seek(0)

        img_bytes2 = io.BytesIO()
        sample_image_with_text.save(img_bytes2, format='PNG')
        img_bytes2.seek(0)

        response = client.post(
            "/batch",
            files=[
                ("files", ("test1.png", img_bytes1, "image/png")),
                ("files", ("test2.png", img_bytes2, "image/png"))
            ]
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2

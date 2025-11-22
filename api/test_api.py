#!/usr/bin/env python
"""
Test API endpoints.

Run with: python -m api.test_api
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_schemas():
    """Test API schemas."""
    try:
        from api.schemas.request import ProcessRequest
        from api.schemas.response import (
            HealthResponse, BoundingBox, Entity, PageResult
        )

        # Test request
        req = ProcessRequest(confidence_threshold=0.7)
        assert req.confidence_threshold == 0.7
        print("  ✓ ProcessRequest schema works")

        # Test response schemas
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=50)
        assert bbox.x1 == 10
        print("  ✓ BoundingBox schema works")

        entity = Entity(
            text="Test",
            label="QUESTION",
            confidence=0.95,
            bbox=bbox
        )
        assert entity.text == "Test"
        print("  ✓ Entity schema works")

        health = HealthResponse(
            status="healthy",
            version="1.0.0",
            device="CPU",
            model_loaded=True
        )
        assert health.status == "healthy"
        print("  ✓ HealthResponse schema works")

    except ImportError as e:
        print(f"  ⚠ Skipped (missing pydantic): {e}")


def test_routes():
    """Test API route imports."""
    try:
        from api.routes import health, process, export

        assert hasattr(health, 'router')
        assert hasattr(process, 'router')
        assert hasattr(export, 'router')

        print("  ✓ All route modules load correctly")

    except ImportError as e:
        print(f"  ⚠ Skipped (missing dependency): {e}")


def test_server_creation():
    """Test FastAPI app creation."""
    try:
        from api.server import app

        assert app.title == "LayoutLMv3 Document Processing Service"
        print("  ✓ FastAPI app created successfully")
        print(f"    Title: {app.title}")
        print(f"    Version: {app.version}")

    except ImportError as e:
        print(f"  ⚠ Skipped (missing dependency): {e}")


def test_endpoints_with_client():
    """Test endpoints using TestClient."""
    try:
        from fastapi.testclient import TestClient
        from api.server import app

        client = TestClient(app)

        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        print("  ✓ GET /health returns 200")

        # Test ready endpoint
        response = client.get("/ready")
        assert response.status_code == 200
        print("  ✓ GET /ready returns 200")

        # Test live endpoint
        response = client.get("/live")
        assert response.status_code == 200
        print("  ✓ GET /live returns 200")

        print("\n  Endpoint responses:")
        print(f"    /health: {client.get('/health').json()}")
        print(f"    /ready: {client.get('/ready').json()}")
        print(f"    /live: {client.get('/live').json()}")

    except ImportError as e:
        print(f"  ⚠ Skipped (missing dependency): {e}")


def main():
    print("=" * 60)
    print("API TEST")
    print("=" * 60)
    print()

    print("1. Testing schemas...")
    test_schemas()
    print()

    print("2. Testing routes...")
    test_routes()
    print()

    print("3. Testing server creation...")
    test_server_creation()
    print()

    print("4. Testing endpoints with client...")
    test_endpoints_with_client()
    print()

    print("=" * 60)
    print("API TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

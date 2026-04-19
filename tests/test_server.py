"""
FastAPI server integration tests.
Requires:  pip install httpx fastapi
"""
from __future__ import annotations

import json
import pytest

try:
    from fastapi.testclient import TestClient
    from server import app
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi/httpx not installed")


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


SQUARE_PAYLOAD = {
    "points": ["A","B","C","D"],
    "constraints": [{"type":"square","points":["A","B","C","D"]}],
    "side_length": 1.0,
}

PYRAMID_PAYLOAD = {
    "points": ["A","B","C","D","S","J"],
    "constraints": [
        {"type":"square",               "points": ["A","B","C","D"]},
        {"type":"pyramid",              "points": ["S","A","B","C","D"]},
        {"type":"equilateral_triangle", "points": ["S","A","B"]},
        {"type":"right_angle",          "points": ["S","A","D"]},
        {"type":"midpoint",             "point":  "J","segment":["S","D"]},
    ],
    "side_length": 1.0,
}


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestSolveEndpoint:
    def test_square_returns_200(self, client):
        resp = client.post("/solve", json=SQUARE_PAYLOAD)
        assert resp.status_code == 200

    def test_square_has_points(self, client):
        resp = client.post("/solve", json=SQUARE_PAYLOAD)
        body = resp.json()
        assert set(body["points"].keys()) == {"A","B","C","D"}

    def test_square_has_edges(self, client):
        resp = client.post("/solve", json=SQUARE_PAYLOAD)
        body = resp.json()
        assert len(body["edges"]) == 4

    def test_square_has_faces(self, client):
        resp = client.post("/solve", json=SQUARE_PAYLOAD)
        body = resp.json()
        assert len(body["faces"]) == 1

    def test_pyramid_all_points(self, client):
        resp = client.post("/solve", json=PYRAMID_PAYLOAD)
        assert resp.status_code == 200
        body = resp.json()
        assert set(body["points"].keys()) == {"A","B","C","D","S","J"}

    def test_pyramid_apex_above_base(self, client):
        resp = client.post("/solve", json=PYRAMID_PAYLOAD)
        S = resp.json()["points"]["S"]
        assert S["z"] > 0.5

    def test_invalid_payload_returns_422(self, client):
        resp = client.post("/solve", json={"bad": "payload"})
        assert resp.status_code == 422

    def test_point3d_has_xyz_fields(self, client):
        resp = client.post("/solve", json=SQUARE_PAYLOAD)
        pt_A = resp.json()["points"]["A"]
        assert set(pt_A.keys()) == {"x","y","z"}

    def test_violations_field_present(self, client):
        resp = client.post("/solve", json=SQUARE_PAYLOAD)
        assert "violations" in resp.json()

    def test_normalize_flag(self, client):
        payload = {**SQUARE_PAYLOAD, "normalize": True}
        resp = client.post("/solve", json=payload)
        pts = resp.json()["points"]
        # Centroid should be near zero
        cx = sum(pts[n]["x"] for n in "ABCD") / 4
        cy = sum(pts[n]["y"] for n in "ABCD") / 4
        assert abs(cx) < 1e-6 and abs(cy) < 1e-6

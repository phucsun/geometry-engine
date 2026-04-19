"""Shared pytest fixtures."""
from __future__ import annotations
import json
import pytest
from geometry_engine import GeometryEngine


@pytest.fixture
def engine() -> GeometryEngine:
    return GeometryEngine()


def make_input(**kwargs) -> str:
    """Build a minimal GeometryInput JSON string."""
    return json.dumps(kwargs)

from __future__ import annotations

import math

import numpy as np

from geometry_engine import GeometryEngine
from geometry_engine.models import Constraint, GeometryInput


EPS = 1e-5


def _plane_angle(points1: list[np.ndarray], points2: list[np.ndarray]) -> float:
    n1 = np.cross(points1[1] - points1[0], points1[2] - points1[0])
    n2 = np.cross(points2[1] - points2[0], points2[2] - points2[0])
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    return math.degrees(math.acos(np.clip(abs(float(np.dot(n1, n2))), -1.0, 1.0)))


def test_triangular_pyramid_equal_side_face_angle_resolves_without_violations():
    engine = GeometryEngine()
    result = engine.solve(
        GeometryInput(
            points=["S", "A", "B", "C"],
            constraints=[
                Constraint(type="right_triangle", points=["A", "B", "C"], length=1.0, width=math.sqrt(3.0)),
                Constraint(type="equal_side_face_angle", points=["S", "A", "B", "C"], degrees=60.0),
                Constraint(type="apex", points=["S", "A", "B", "C"]),
            ],
            side_length=1.0,
            validate_constraints=True,
        )
    )

    assert result.unresolved_points == []
    assert result.violations == []

    pts = {name: np.array([pt.x, pt.y, pt.z]) for name, pt in result.points.items()}
    assert np.allclose(pts["A"], np.array([0.0, 0.0, 0.0]), atol=EPS)
    assert np.allclose(pts["B"], np.array([1.0, 0.0, 0.0]), atol=EPS)
    assert np.allclose(pts["C"], np.array([0.0, math.sqrt(3.0), 0.0]), atol=EPS)

    base_abc = [pts["A"], pts["B"], pts["C"]]
    for side in (
        [pts["S"], pts["A"], pts["B"]],
        [pts["S"], pts["A"], pts["C"]],
        [pts["S"], pts["B"], pts["C"]],
    ):
        assert abs(_plane_angle(side, base_abc) - 60.0) < 1e-4

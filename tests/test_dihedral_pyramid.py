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


def test_square_pyramid_dihedral_angle_resolves_without_violations():
    engine = GeometryEngine()
    result = engine.solve(
        GeometryInput(
            points=["S", "A", "B", "C", "D", "M", "N"],
            constraints=[
                Constraint(type="square", points=["A", "B", "C", "D"]),
                Constraint(
                    type="perpendicular_to_plane",
                    point="S",
                    from_point="A",
                    points=["A", "B", "C", "D"],
                ),
                Constraint(
                    type="dihedral_angle",
                    point="S",
                    from_point="A",
                    segment=["B", "D"],
                    points=["A", "B", "C", "D"],
                    degrees=60.0,
                ),
                Constraint(type="apex", points=["S", "A", "B", "C", "D"]),
                Constraint(type="midpoint", point="M", segment=["S", "B"]),
                Constraint(type="midpoint", point="N", segment=["S", "C"]),
            ],
            side_length=1.0,
            validate_constraints=True,
        )
    )

    assert result.unresolved_points == []
    assert result.violations == []

    pts = {name: np.array([pt.x, pt.y, pt.z]) for name, pt in result.points.items()}
    assert abs(pts["S"][0]) < EPS
    assert abs(pts["S"][1]) < EPS
    assert abs(pts["S"][2] - math.sqrt(1.5)) < EPS

    plane_sbd = [pts["S"], pts["B"], pts["D"]]
    plane_abcd = [pts["A"], pts["B"], pts["D"]]
    assert abs(_plane_angle(plane_sbd, plane_abcd) - 60.0) < 1e-4

    expected_m = (pts["S"] + pts["B"]) / 2.0
    expected_n = (pts["S"] + pts["C"]) / 2.0
    assert np.allclose(pts["M"], expected_m, atol=EPS)
    assert np.allclose(pts["N"], expected_n, atol=EPS)

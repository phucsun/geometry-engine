"""Repair rules for pyramid-style spatial geometry problems.

Supported cases include dihedral angle statements, side faces making the same
angle with the base, and common LLM mistakes that map plane angles to point
angles.
"""
from __future__ import annotations

import re
from typing import Any

from .common import (
    _choose_side_and_base_plane,
    _find_perpendicular_foot,
    _has_explicit_perpendicular_symbol,
    _normalize_math_text,
)


def _repair_equal_side_face_angle_constraints(
    constraints: list[dict[str, Any]],
    problem_text: str,
) -> None:
    normalized_text = _normalize_math_text(problem_text)
    match = re.search(
        r"các mặt bên\s*\(?([A-Z]{3,8})\)?\s*,\s*\(?([A-Z]{3,8})\)?\s*,\s*\(?([A-Z]{3,8})\)?\s*cùng tạo với mặt đáy góc\s*([0-9]+(?:\.[0-9]+)?)",
        normalized_text,
        re.IGNORECASE,
    )
    if not match:
        return

    plane1, plane2, plane3, degrees_str = match.groups()
    common = set(plane1) & set(plane2) & set(plane3)
    if len(common) != 1:
        return
    apex = next(iter(common))
    base = [name for name in plane1 if name != apex]
    if len(base) != 2:
        base = [name for name in plane2 if name != apex]
    if len(base) != 2:
        return

    triangle_points = [base[0], base[1], next(name for name in plane3 if name not in {apex, *base})]
    side_face_point_sets = {
        frozenset([apex, triangle_points[0], triangle_points[1]]),
        frozenset([apex, triangle_points[0], triangle_points[2]]),
        frozenset([apex, triangle_points[1], triangle_points[2]]),
    }
    target_degrees = float(degrees_str)

    def _is_wrong_plane_angle_mapping(constraint: dict[str, Any]) -> bool:
        if constraint.get("type") != "angle":
            return False

        raw_degrees = constraint.get("degrees")
        try:
            deg = float(raw_degrees)
        except (TypeError, ValueError):
            return False
        if abs(deg - target_degrees) > 1e-9:
            return False

        points = constraint.get("points")
        if not isinstance(points, list):
            return False

        # Wrong pattern 1: malformed angle with 4 points.
        if len(points) > 3:
            return True

        # Wrong pattern 2: LLM maps plane-plane angle into 3-point angles
        # on side faces, e.g. angle(SAB), angle(SAC), angle(SBC).
        if len(points) == 3 and frozenset(points) in side_face_point_sets:
            return True

        return False

    constraints[:] = [
        constraint
        for constraint in constraints
        if not (
            isinstance(constraint, dict)
            and _is_wrong_plane_angle_mapping(constraint)
        )
    ]

    # If the problem only states equal side-face dihedral angles to the base,
    # a direct "S ⟂ base" constraint is usually hallucinated by LLM and conflicts
    # with the intended geometry (e.g. projection is only said to lie inside base).
    # Keep explicit perpendicular constraints only when text explicitly contains it.
    if not _has_explicit_perpendicular_symbol(problem_text):
        constraints[:] = [
            constraint
            for constraint in constraints
            if not (
                isinstance(constraint, dict)
                and constraint.get("type") == "perpendicular_to_plane"
                and constraint.get("point") == apex
                and isinstance(constraint.get("points"), list)
                and len(constraint["points"]) >= 3
                and set(triangle_points).issubset(set(constraint["points"]))
            )
        ]

    if not any(
        isinstance(constraint, dict)
        and constraint.get("type") == "equal_side_face_angle"
        and constraint.get("points") == [apex, *triangle_points]
        for constraint in constraints
    ):
        constraints.append(
            {
                "type": "equal_side_face_angle",
                "points": [apex, *triangle_points],
                "degrees": target_degrees,
            }
        )




def _repair_dihedral_constraints(
    constraints: list[dict[str, Any]],
    problem_text: str,
) -> None:
    normalized_text = _normalize_math_text(problem_text)
    match = re.search(
        r"góc giữa hai mặt phẳng\s*\(?([A-Z]{3,8})\)?\s*,\s*\(?([A-Z]{3,8})\)?\s*bằng\s*([0-9]+(?:\.[0-9]+)?)",
        normalized_text,
        re.IGNORECASE,
    )
    if not match:
        return

    plane1, plane2, degrees_str = match.groups()
    side_plane, base_plane = _choose_side_and_base_plane(plane1, plane2)
    if not side_plane or not base_plane:
        return

    edge = sorted(set(side_plane) & set(base_plane))
    apex = next((name for name in side_plane if name not in edge), None)
    foot = _find_perpendicular_foot(constraints, apex)
    if apex is None or foot is None or len(edge) != 2:
        return

    constraints[:] = [
        constraint
        for constraint in constraints
        if not (
            isinstance(constraint, dict)
            and constraint.get("type") == "angle"
            and constraint.get("points") == list(side_plane)
            and float(constraint.get("degrees", 0.0)) == float(degrees_str)
        )
    ]

    constraints.append(
        {
            "type": "dihedral_angle",
            "point": apex,
            "from_point": foot,
            "segment": edge,
            "points": list(base_plane),
            "degrees": float(degrees_str),
        }
    )



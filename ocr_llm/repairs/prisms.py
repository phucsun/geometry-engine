"""Repair rules for triangular prism problem families.

Supported cases include right triangular prisms and oblique triangular prisms
with symbolic lengths such as a√3, lateral-edge angles, and points placed by
ratios on a base edge.
"""
from __future__ import annotations

import re
from typing import Any

from .common import _coerce_numeric_value, _normalize_math_text


def _repair_right_triangular_prism_constraints(
    constraints: list[dict[str, Any]],
    payload: dict[str, Any],
    problem_text: str,
) -> None:
    normalized_text = _normalize_math_text(problem_text)
    if "lăng trụ đứng" not in normalized_text.lower():
        return

    prism_match = re.search(
        r"lăng trụ đứng\s*([A-Z]{3})\s*\.\s*([A-Z]'[A-Z]'[A-Z]')",
        normalized_text,
        re.IGNORECASE,
    )
    if not prism_match:
        return

    base = list(prism_match.group(1).upper())
    top = re.findall(r"[A-Z]'", prism_match.group(2).upper())
    if len(base) != 3 or len(top) != 3:
        return

    top_by_base = {name[0]: name for name in top}
    prism_points = [*base, *(top_by_base.get(name) for name in base)]
    if any(name is None for name in prism_points):
        return

    segment_lengths = _extract_segment_lengths(normalized_text)
    height = segment_lengths.get(frozenset({base[0], top_by_base[base[0]]}))
    if height is None:
        return

    right_vertex = _extract_right_triangle_vertex(normalized_text)
    if right_vertex is None or right_vertex not in base:
        return

    other_vertices = [name for name in base if name != right_vertex]
    right_triangle = {
        "type": "right_triangle",
        "points": [right_vertex, *other_vertices],
    }
    leg1 = segment_lengths.get(frozenset({right_vertex, other_vertices[0]}))
    leg2 = segment_lengths.get(frozenset({right_vertex, other_vertices[1]}))
    if leg1 is not None:
        right_triangle["length"] = leg1
    if leg2 is not None:
        right_triangle["width"] = leg2

    right_prism = {
        "type": "right_prism",
        "points": prism_points,
        "height": height,
    }

    prism_point_set = set(prism_points)
    constraints[:] = [
        constraint
        for constraint in constraints
        if not (
            isinstance(constraint, dict)
            and (
                constraint.get("type") in {"rectangle", "prism", "perpendicular_to_plane"}
                and prism_point_set.intersection(set(constraint.get("points") or []))
            )
        )
    ]

    constraints[:] = [
        constraint
        for constraint in constraints
        if not (
            isinstance(constraint, dict)
            and constraint.get("type") == "right_triangle"
            and set(constraint.get("points") or []) == set(base)
        )
    ]

    constraints.insert(0, right_prism)
    constraints.insert(0, right_triangle)

    points = payload.get("points")
    if isinstance(points, list):
        for name in prism_points:
            if name not in points:
                points.append(name)




def _repair_oblique_triangular_prism_constraints(
    constraints: list[dict[str, Any]],
    payload: dict[str, Any],
    problem_text: str,
) -> None:
    normalized_text = _normalize_math_text(problem_text)
    prism_match = re.search(
        r"lăng trụ\s*([A-Z]{3})\s*\.\s*([A-Z]'[A-Z]'[A-Z]')",
        normalized_text,
        re.IGNORECASE,
    )
    if not prism_match:
        return

    base = list(prism_match.group(1).upper())
    top = re.findall(r"[A-Z]'", prism_match.group(2).upper())
    if len(base) != 3 or len(top) != 3:
        return

    h_point = _extract_ratio_point_on_segment(normalized_text)
    if not h_point:
        return
    point_name, segment, ratio = h_point

    side_angle = _extract_lateral_edge_base_angle(normalized_text)
    if side_angle is None:
        return

    base_angle = _extract_named_base_angle(normalized_text, base)
    if base_angle is None:
        return
    angle_vertex, angle_degrees = base_angle

    length_constraints = _extract_symbolic_edge_lengths(normalized_text)
    edge1 = frozenset({angle_vertex, base[0]})
    edge2 = frozenset({angle_vertex, base[1]})
    edge3 = frozenset({angle_vertex, base[2]})
    known_edges = {
        frozenset(edge): length
        for edge, length in length_constraints.items()
        if angle_vertex in edge
    }
    if len(known_edges) < 2:
        return

    # The specialized solver expects the base order [A,B,C] with the angle at C,
    # and length/width are the two sides from the angle vertex to the first two points.
    other_vertices = [name for name in base if name != angle_vertex]
    if len(other_vertices) != 2:
        return
    length = known_edges.get(frozenset({angle_vertex, other_vertices[0]}))
    width = known_edges.get(frozenset({angle_vertex, other_vertices[1]}))
    if length is None or width is None:
        return

    ordered_base = [other_vertices[0], other_vertices[1], angle_vertex]
    top_by_base = {name[0]: name for name in top}
    ordered_top = [top_by_base.get(name) for name in ordered_base]
    if any(name is None for name in ordered_top):
        return

    prism_points = [*ordered_base, *ordered_top]
    prism_constraint = {
        "type": "oblique_prism",
        "points": prism_points,
        "point": point_name,
        "segment": segment,
        "length": length,
        "width": width,
        "height": side_angle,
        "ratio": ratio,
        "degrees": angle_degrees,
    }

    prism_point_set = set(prism_points)
    base_set = set(ordered_base)
    constraints[:] = [
        constraint
        for constraint in constraints
        if not (
            isinstance(constraint, dict)
            and (
                constraint.get("type") in {"rectangle", "right_triangle"}
                and base_set.intersection(set(constraint.get("points") or []))
            )
            or (
                isinstance(constraint, dict)
                and constraint.get("type") in {"perpendicular_to_plane", "angle", "right_angle"}
                and prism_point_set.intersection(set(constraint.get("points") or []))
            )
        )
    ]
    constraints.insert(0, prism_constraint)

    points = payload.get("points")
    if isinstance(points, list):
        for name in [*prism_points, point_name]:
            if name not in points:
                points.append(name)




def _extract_segment_lengths(problem_text: str) -> dict[frozenset[str], float]:
    result: dict[frozenset[str], float] = {}
    for raw_segment, raw_value in re.findall(
        r"([A-Z']+)\s*=\s*([0-9]*(?:\.[0-9]+)?\s*\*?\s*(?:a(?:\s*(?:√|sqrt)\s*[0-9]+)?|[0-9]+(?:\.[0-9]+)?))",
        problem_text,
        re.IGNORECASE,
    ):
        points = re.findall(r"[A-Z]'?", raw_segment.upper())
        if len(points) != 2:
            continue
        value = _coerce_numeric_value(raw_value)
        if isinstance(value, (int, float)):
            result[frozenset(points)] = float(value)
    return result




def _extract_symbolic_edge_lengths(problem_text: str) -> dict[frozenset[str], float]:
    return _extract_segment_lengths(problem_text)




def _extract_right_triangle_vertex(problem_text: str) -> str | None:
    match = re.search(r"tam giác vuông tại\s*([A-Z])", problem_text, re.IGNORECASE)
    return match.group(1).upper() if match else None




def _extract_named_base_angle(problem_text: str, base: list[str]) -> tuple[str, float] | None:
    base_set = set(base)
    for raw_points, raw_degrees in re.findall(
        r"([A-Z]{3})\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        problem_text,
        re.IGNORECASE,
    ):
        points = raw_points.upper()
        if set(points) == base_set:
            return points[1], float(raw_degrees)
    return None




def _extract_lateral_edge_base_angle(problem_text: str) -> float | None:
    match = re.search(
        r"cạnh bên\s+hợp với mặt phẳng đáy góc\s*([0-9]+(?:\.[0-9]+)?)",
        problem_text,
        re.IGNORECASE,
    )
    return float(match.group(1)) if match else None




def _extract_ratio_point_on_segment(problem_text: str) -> tuple[str, list[str], float] | None:
    match = re.search(
        r"điểm\s+([A-Z])\s+trên cạnh\s+([A-Z])([A-Z])\s+sao cho\s+\1([A-Z])\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Z])\1",
        problem_text,
        re.IGNORECASE,
    )
    if not match:
        return None
    point, s1, s2, end1, coefficient, end2 = match.groups()
    point = point.upper()
    s1 = s1.upper()
    s2 = s2.upper()
    end1 = end1.upper()
    end2 = end2.upper()
    if {end1, end2} != {s1, s2}:
        return None

    k = float(coefficient)
    if end1 == s2 and end2 == s1:
        # H on BC, HC = k * HB => BH / BC = 1 / (k + 1)
        ratio = 1.0 / (k + 1.0)
    else:
        # HB = k * HC => BH / BC = k / (k + 1)
        ratio = k / (k + 1.0)
    return point, [s1, s2], ratio



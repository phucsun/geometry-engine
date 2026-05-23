"""Common repair orchestration and schema-level cleanup for LLM payloads.

These rules are intentionally conservative: they fix field placement, numeric
normalisation, derived point syntax, and dispatch problem-family repairs without
changing the public GeometryInput schema.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any


def _repair_geometry_payload(
    payload: dict[str, Any] | str,
    problem_text: str,
) -> dict[str, Any] | str:
    """Repair common LLM field-mapping mistakes before Pydantic validation."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return payload
    if not isinstance(payload, dict):
        return payload

    constraints = payload.get("constraints")
    if not isinstance(constraints, list):
        return payload

    _repair_centroid_constraints(constraints)
    _repair_intersection_constraints(constraints, problem_text)
    _repair_perpendicular_constraints(constraints, problem_text)
    from .pyramids import _repair_dihedral_constraints, _repair_equal_side_face_angle_constraints
    from .prisms import _repair_oblique_triangular_prism_constraints, _repair_right_triangular_prism_constraints

    _repair_dihedral_constraints(constraints, problem_text)
    _normalize_numeric_fields(payload)
    _repair_right_triangle_vertex(constraints, problem_text)
    _repair_right_triangle_lengths(constraints)
    _repair_right_triangular_prism_constraints(constraints, payload, problem_text)
    _repair_equal_side_face_angle_constraints(constraints, problem_text)
    _repair_oblique_triangular_prism_constraints(constraints, payload, problem_text)

    pyramid = _extract_pyramid(problem_text)
    if pyramid:
        apex, base = pyramid
        if _mentions_parallelogram_base(problem_text) and not _has_constraint(
            constraints, "parallelogram", base
        ):
            constraints.insert(0, {"type": "parallelogram", "points": base})
        if not _has_any_constraint(constraints, {"apex", "pyramid", "regular_pyramid"}):
            insert_at = 1 if _has_constraint(constraints, "parallelogram", base) else 0
            constraints.insert(insert_at, {"type": "apex", "points": [apex, *base]})

    payload["constraints"] = _sort_constraints(constraints)
    return payload




def _normalize_numeric_fields(payload: dict[str, Any]) -> None:
    for field in ("side_length",):
        if field in payload:
            payload[field] = _coerce_numeric_value(payload[field])

    constraints = payload.get("constraints")
    if not isinstance(constraints, list):
        return
    for constraint in constraints:
        if not isinstance(constraint, dict):
            continue
        for field in ("length", "width", "height", "ratio", "degrees"):
            if field in constraint:
                constraint[field] = _coerce_numeric_value(constraint[field])




def _coerce_numeric_value(value: Any) -> Any:
    if isinstance(value, (int, float)) or value is None:
        return value
    if not isinstance(value, str):
        return value

    cleaned = value.strip().lower().replace(" ", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("{", "").replace("}", "")

    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", cleaned):
        return float(cleaned)
    if cleaned == "a":
        return 1.0

    match = re.fullmatch(r"a(?:√|sqrt)([+-]?\d+(?:\.\d+)?)", cleaned)
    if match:
        return math.sqrt(float(match.group(1)))

    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)a(?:√|sqrt)([+-]?\d+(?:\.\d+)?)", cleaned)
    if match:
        return float(match.group(1)) * math.sqrt(float(match.group(2)))

    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)a", cleaned)
    if match:
        return float(match.group(1))

    match = re.fullmatch(r"a/([+-]?\d+(?:\.\d+)?)", cleaned)
    if match:
        denominator = float(match.group(1))
        if abs(denominator) > 1e-12:
            return 1.0 / denominator

    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)/a", cleaned)
    if match:
        return float(match.group(1))

    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\*?a/([+-]?\d+(?:\.\d+)?)", cleaned)
    if match:
        denominator = float(match.group(2))
        if abs(denominator) > 1e-12:
            return float(match.group(1)) / denominator

    return value




def _repair_right_triangle_lengths(constraints: list[dict[str, Any]]) -> None:
    right_triangle = next(
        (
            constraint
            for constraint in constraints
            if isinstance(constraint, dict) and constraint.get("type") == "right_triangle"
        ),
        None,
    )
    if right_triangle is None:
        return

    pts = right_triangle.get("points") or []
    if len(pts) != 3:
        return
    right_vertex, p_name, q_name = pts

    edge_lengths: dict[frozenset[str], float] = {}
    remaining_constraints: list[dict[str, Any]] = []
    for constraint in constraints:
        if constraint is right_triangle:
            continue
        if (
            isinstance(constraint, dict)
            and constraint.get("type") == "edge_length"
            and isinstance(constraint.get("segment"), list)
            and len(constraint["segment"]) == 2
            and isinstance(constraint.get("length"), (int, float))
        ):
            edge_lengths[frozenset(constraint["segment"])] = float(constraint["length"])
            continue
        remaining_constraints.append(constraint)

    leg1 = edge_lengths.get(frozenset({right_vertex, p_name}))
    leg2 = edge_lengths.get(frozenset({right_vertex, q_name}))
    hyp = edge_lengths.get(frozenset({p_name, q_name}))

    if leg1 is not None:
        right_triangle["length"] = leg1
    if leg2 is not None:
        right_triangle["width"] = leg2
    elif leg1 is not None and hyp is not None and hyp > leg1:
        right_triangle["width"] = float((hyp ** 2 - leg1 ** 2) ** 0.5)
    elif leg2 is None and leg1 is None and hyp is not None:
        remaining_constraints.extend(
            [
                {"type": "edge_length", "segment": [right_vertex, p_name], "length": hyp},
                {"type": "edge_length", "segment": [p_name, q_name], "length": hyp},
            ]
        )

    constraints[:] = remaining_constraints
    constraints.append(right_triangle)




def _repair_right_triangle_vertex(
    constraints: list[dict[str, Any]],
    problem_text: str,
) -> None:
    match = re.search(r"tam giác vuông tại\s*([A-Z])", _normalize_math_text(problem_text), re.IGNORECASE)
    if not match:
        return

    right_vertex = match.group(1).upper()
    for constraint in constraints:
        if not isinstance(constraint, dict) or constraint.get("type") != "right_triangle":
            continue
        points = constraint.get("points")
        if not isinstance(points, list) or len(points) != 3 or right_vertex not in points:
            continue
        if points[0] == right_vertex:
            return
        others = [name for name in points if name != right_vertex]
        constraint["points"] = [right_vertex, *others]
        return




def _repair_centroid_constraints(constraints: list[dict[str, Any]]) -> None:
    for constraint in constraints:
        if not isinstance(constraint, dict):
            continue
        if (
            constraint.get("type") == "centroid"
            and not constraint.get("points")
            and isinstance(constraint.get("segment"), list)
        ):
            constraint["points"] = constraint["segment"]
            constraint.pop("segment", None)




def _repair_intersection_constraints(
    constraints: list[dict[str, Any]],
    problem_text: str,
) -> None:
    for constraint in constraints:
        if not isinstance(constraint, dict):
            continue
        if constraint.get("type") == "intersection" and not constraint.get("points"):
            plane_points = _extract_plane_points(problem_text)
            if plane_points:
                constraint["points"] = plane_points




def _repair_perpendicular_constraints(
    constraints: list[dict[str, Any]],
    problem_text: str,
) -> None:
    match = re.search(r"([A-Z])([A-Z])\s*(?:\\perp|⊥)\s*\(?\$?([A-Z]{3,8})\$?\)?", problem_text)
    if not match:
        return
    p1, p2, plane = match.groups()
    plane_points = list(plane)

    if p1 in plane_points and p2 not in plane_points:
        foot, apex = p1, p2
    elif p2 in plane_points and p1 not in plane_points:
        foot, apex = p2, p1
    else:
        return

    repaired = False
    for constraint in constraints:
        if not isinstance(constraint, dict):
            continue
        if constraint.get("type") != "perpendicular_to_plane":
            continue
        if constraint.get("point") in (None, apex):
            constraint["point"] = apex
        if not constraint.get("from_point"):
            constraint["from_point"] = foot
        if not constraint.get("points"):
            constraint["points"] = plane_points
        repaired = True

    if not repaired:
        constraints.append(
            {
                "type": "perpendicular_to_plane",
                "point": apex,
                "from_point": foot,
                "points": plane_points,
            }
        )




def _extract_pyramid(problem_text: str) -> tuple[str, list[str]] | None:
    match = re.search(r"hình\s+chóp\s+([A-Z])\s*\.\s*([A-Z]{3,8})", problem_text)
    if not match:
        return None
    return match.group(1), list(match.group(2))




def _extract_plane_points(problem_text: str) -> list[str] | None:
    matches = re.findall(r"mặt\s+phẳng\s*\(?\$?([A-Z]{3,8})\$?\)?", problem_text)
    if not matches:
        matches = re.findall(r"\(\s*([A-Z]{3,8})\s*\)", problem_text)
    if not matches:
        return None
    return list(matches[-1])




def _normalize_math_text(problem_text: str) -> str:
    text = problem_text
    text = text.replace("$", "")
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\(", "(").replace("\\)", ")")
    text = re.sub(r"\\(?:circ|degree|degrees?)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\^0", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()




def _has_explicit_perpendicular_symbol(problem_text: str) -> bool:
    # Explicit symbolic forms: SA ⟂ (ABCD) or SA \perp (ABCD)
    if re.search(r"([A-Z])([A-Z])\s*(?:\\perp|⊥)\s*\(?\$?[A-Z]{3,8}\$?\)?", problem_text):
        return True
    # Explicit textual form: SA vuông góc (ABCD) / SA vuông góc với mặt phẳng (ABCD)
    normalized = _normalize_math_text(problem_text).lower()
    if re.search(r"[a-z]\s*[a-z]\s+vuông\s+góc(?:\s+với\s+mặt\s+phẳng)?\s*\(?[a-z]{3,8}\)?", normalized):
        return True
    return False




def _mentions_parallelogram_base(problem_text: str) -> bool:
    return "hình bình hành" in problem_text.lower()




def _choose_side_and_base_plane(plane1: str, plane2: str) -> tuple[str | None, str | None]:
    if len(plane1) == 3 and len(plane2) >= 3:
        return plane1, plane2
    if len(plane2) == 3 and len(plane1) >= 3:
        return plane2, plane1
    return None, None




def _find_perpendicular_foot(
    constraints: list[dict[str, Any]],
    apex: str | None,
) -> str | None:
    if apex is None:
        return None
    for constraint in constraints:
        if (
            isinstance(constraint, dict)
            and constraint.get("type") == "perpendicular_to_plane"
            and constraint.get("point") == apex
            and constraint.get("from_point")
        ):
            return str(constraint["from_point"])
    return None




def _has_constraint(
    constraints: list[Any],
    constraint_type: str,
    points: list[str],
) -> bool:
    return any(
        isinstance(constraint, dict)
        and constraint.get("type") == constraint_type
        and constraint.get("points") == points
        for constraint in constraints
    )




def _has_any_constraint(constraints: list[Any], constraint_types: set[str]) -> bool:
    return any(
        isinstance(constraint, dict) and constraint.get("type") in constraint_types
        for constraint in constraints
    )




def _sort_constraints(constraints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {
        "square": 0,
        "rectangle": 0,
        "parallelogram": 0,
        "rhombus": 0,
        "trapezoid": 0,
        "right_triangle": 0,
        "oblique_prism": 0,
        "right_prism": 1,
        "perpendicular_to_plane": 1,
        "dihedral_angle": 2,
        "equal_side_face_angle": 2,
        "apex": 3,
        "regular_pyramid": 3,
        "pyramid": 3,
        "midpoint": 4,
        "centroid": 4,
        "intersection": 4,
    }
    return sorted(constraints, key=lambda constraint: priority.get(constraint.get("type", ""), 10))


# Dùng GeometryInput để validate lại JSON từ LLM, đảm bảo đúng schema và kiểu dữ liệu

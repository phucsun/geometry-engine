"""Constraint handler registry for GeometryEngine.

The registry maps constraint type strings from GeometryInput to grouped handler
mixins. Internal backend-only constraints are kept here too so engine.py remains
an orchestrator rather than a long list of handler implementations.
"""
from __future__ import annotations

import logging
from typing import Callable

from .models import Constraint

logger = logging.getLogger(__name__)

_HANDLER_NAMES: dict[str, str] = {
    # 2-D base anchors.
    "square": "_handle_square",
    "rectangle": "_handle_rectangle",
    "parallelogram": "_handle_parallelogram",
    "rhombus": "_handle_rhombus",
    "trapezoid": "_handle_trapezoid",
    "equilateral_triangle": "_handle_equilateral_triangle",
    "isosceles_triangle": "_handle_isosceles_triangle",
    "right_triangle": "_handle_right_triangle",
    "regular_hexagon": "_handle_regular_hexagon",
    "regular_polygon": "_handle_regular_polygon",
    # 3-D solids and pyramids/prisms.
    "regular_tetrahedron": "_handle_regular_tetrahedron",
    "cube": "_handle_cube",
    "rectangular_prism": "_handle_rectangular_prism",
    "prism": "_handle_prism",
    "oblique_prism": "_handle_oblique_prism",
    "regular_octahedron": "_handle_regular_octahedron",
    "right_prism": "_handle_right_prism",
    "apex": "_handle_apex",
    "regular_pyramid": "_handle_apex",
    "pyramid": "_handle_apex",
    "truncated_pyramid": "_handle_truncated_pyramid",
    # Derived construction points.
    "midpoint": "_handle_midpoint",
    "ratio_point": "_handle_ratio_point",
    "centroid": "_handle_centroid",
    "circumcenter": "_handle_circumcenter",
    "orthocenter": "_handle_orthocenter",
    "incenter": "_handle_incenter",
    "equidistant": "_handle_equidistant",
    "angle_bisector": "_handle_angle_bisector",
    "median": "_handle_median",
    "foot_perpendicular": "_handle_foot_perpendicular",
    "foot_on_plane": "_handle_foot_on_plane",
    "perpendicular_to_plane": "_handle_perpendicular_to_plane",
    "symmetric": "_handle_symmetric",
    "intersection": "_handle_intersection",
    # Backend-only THPT rules.
    "equal_side_face_angle": "_handle_equal_side_face_angle",
    "dihedral_angle": "_handle_dihedral_angle",
    # Filtering and disambiguation.
    "right_angle": "_handle_right_angle",
    "angle": "_handle_angle",
    "distance": "_handle_distance",
    "edge_length": "_handle_edge_length",
    "on_line": "_handle_on_line",
    "collinear": "_handle_collinear",
}

_PASSTHROUGH_TYPES = {"parallel", "perpendicular", "coplanar"}


def get_handler(engine: object, ctype: str) -> Callable[[Constraint], bool]:
    """Return the bound handler for a GeometryInput constraint type."""
    if ctype in _PASSTHROUGH_TYPES:
        return lambda _c: True

    method_name = _HANDLER_NAMES.get(ctype)
    if method_name is None:
        logger.warning("Unknown constraint '%s' — skipped.", ctype)
        return lambda _c: True
    return getattr(engine, method_name)

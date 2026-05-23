"""Problem-type catalog used to keep prompts and repairs focused.

The detector is local and regex-based. It does not solve the problem; it only
selects the smallest relevant prompt rule set for OCR/LLM extraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re


class ProblemType(StrEnum):
    SQUARE_PYRAMID = "square_pyramid"
    DIHEDRAL_PYRAMID = "dihedral_pyramid"
    EQUAL_SIDE_FACE_ANGLE_PYRAMID = "equal_side_face_angle_pyramid"
    RIGHT_TRIANGULAR_PRISM = "right_triangular_prism"
    OBLIQUE_TRIANGULAR_PRISM = "oblique_triangular_prism"
    GENERIC_SHAPES = "generic_shapes"


@dataclass(frozen=True)
class ProblemTypeSpec:
    type: ProblemType
    description: str
    allowed_constraints: tuple[str, ...]


PROBLEM_TYPE_SPECS: dict[ProblemType, ProblemTypeSpec] = {
    ProblemType.SQUARE_PYRAMID: ProblemTypeSpec(
        ProblemType.SQUARE_PYRAMID,
        "Pyramids with square/rectangle/parallelogram bases and side-face constraints.",
        ("square", "rectangle", "parallelogram", "apex", "midpoint", "equilateral_triangle", "right_angle"),
    ),
    ProblemType.DIHEDRAL_PYRAMID: ProblemTypeSpec(
        ProblemType.DIHEDRAL_PYRAMID,
        "Pyramids with an angle between two planes.",
        ("square", "rectangle", "parallelogram", "apex", "perpendicular_to_plane", "midpoint"),
    ),
    ProblemType.EQUAL_SIDE_FACE_ANGLE_PYRAMID: ProblemTypeSpec(
        ProblemType.EQUAL_SIDE_FACE_ANGLE_PYRAMID,
        "Triangular pyramids whose side faces make equal angles with the base.",
        ("right_triangle", "edge_length", "apex"),
    ),
    ProblemType.RIGHT_TRIANGULAR_PRISM: ProblemTypeSpec(
        ProblemType.RIGHT_TRIANGULAR_PRISM,
        "Right triangular prism drawings such as ABC.A'B'C'.",
        ("right_triangle", "right_prism", "midpoint", "edge_length"),
    ),
    ProblemType.OBLIQUE_TRIANGULAR_PRISM: ProblemTypeSpec(
        ProblemType.OBLIQUE_TRIANGULAR_PRISM,
        "Oblique triangular prism drawings with lateral edge angle and projection constraints.",
        ("prism", "edge_length", "angle", "ratio_point"),
    ),
    ProblemType.GENERIC_SHAPES: ProblemTypeSpec(
        ProblemType.GENERIC_SHAPES,
        "Generic shapes without a specialized THPT repair rule.",
        ("square", "rectangle", "parallelogram", "right_triangle", "midpoint", "centroid", "intersection", "apex"),
    ),
}


def detect_problem_type(problem_text: str) -> ProblemType:
    """Classify OCR text into the nearest supported drawing family."""
    text = problem_text.lower()
    if "lăng trụ đứng" in text and re.search(r"[a-z]{3}\s*\.\s*[a-z]'[a-z]'[a-z]'", text):
        return ProblemType.RIGHT_TRIANGULAR_PRISM
    if "lăng trụ" in text and "cạnh bên hợp" in text:
        return ProblemType.OBLIQUE_TRIANGULAR_PRISM
    if "các mặt bên" in text and "cùng tạo với mặt đáy" in text:
        return ProblemType.EQUAL_SIDE_FACE_ANGLE_PYRAMID
    if "góc giữa hai mặt phẳng" in text:
        return ProblemType.DIHEDRAL_PYRAMID
    if "hình chóp" in text:
        return ProblemType.SQUARE_PYRAMID
    return ProblemType.GENERIC_SHAPES

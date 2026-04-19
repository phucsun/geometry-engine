"""
Pydantic models for GeometryEngine input and output.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Input ─────────────────────────────────────────────────────────────────────

class Constraint(BaseModel):
    """A single geometric constraint produced by an LLM from a math problem."""

    type: str
    """Constraint kind: 'square', 'equilateral_triangle', 'midpoint', …"""

    # ── Point lists ──────────────────────────────────────────────────────────
    points: Optional[list[str]] = None
    """Ordered point names for polygon/shape constraints.
    Also used as reference points for centroid, symmetric, perpendicular_to_plane,
    foot_on_plane, intersection."""

    point: Optional[str] = None
    """Single result point name (midpoint, ratio_point, centroid, foot_on_plane…)."""

    segment: Optional[list[str]] = None
    """Two-point segment [P, Q] — used by midpoint, foot_perpendicular, intersection."""

    from_point: Optional[str] = None
    """Source point — used by foot_perpendicular, foot_on_plane, symmetric,
    perpendicular_to_plane."""

    # ── Numeric parameters ───────────────────────────────────────────────────
    length: Optional[float] = None
    """Primary length / x-dimension."""

    width: Optional[float] = None
    """Secondary / y-dimension (rectangle, prism)."""

    height: Optional[float] = None
    """Height / z-dimension (rectangular_prism, prism, truncated_pyramid)."""

    ratio: Optional[float] = None
    """Ratio in [0, 1] — for ratio_point: G = A + ratio*(B-A);
    for truncated_pyramid: top_side / base_side."""

    degrees: Optional[float] = None
    """Angle in degrees — used by 'angle' constraint (e.g., 60.0)."""


class GeometryInput(BaseModel):
    """Full input to the GeometryEngine."""

    points: list[str]
    """All point names that must appear in the output."""

    constraints: list[Constraint]
    """Ordered list of constraints defining the geometry."""

    side_length: float = Field(default=1.0, gt=0, description="Default unit side length.")

    normalize: bool = Field(
        default=False,
        description="Center the model at origin and scale to unit sphere.",
    )

    validate_constraints: bool = Field(
        default=True,
        description="Verify that solved coordinates satisfy every constraint.",
    )


# ── Output ────────────────────────────────────────────────────────────────────

class Point3D(BaseModel):
    """3-D Cartesian coordinate."""
    x: float
    y: float
    z: float


class Edge(BaseModel):
    """Line segment between two named points (Unity: LineRenderer)."""
    p1: str
    p2: str


class Face(BaseModel):
    """Ordered polygon face (Unity: Mesh). Vertices in CCW order."""
    vertices: list[str]


class GeometryOutput(BaseModel):
    """Complete output from the GeometryEngine."""

    points: dict[str, Point3D]
    """Resolved 3-D coordinates for every requested point."""

    edges: list[Edge] = Field(default_factory=list)
    """All structural edges (de-duplicated) — for LineRenderer."""

    faces: list[Face] = Field(default_factory=list)
    """All polygon faces — for Mesh generation."""

    unresolved_points: list[str] = Field(default_factory=list)
    """Points that could not be placed."""

    violations: list[str] = Field(default_factory=list)
    """Human-readable descriptions of unsatisfied constraints."""

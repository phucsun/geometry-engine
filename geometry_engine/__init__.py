"""GeometryEngine — constraint-based 3-D coordinate solver."""

from .engine import GeometryEngine, SolverError
from .models import (
    Constraint,
    Edge,
    Face,
    GeometryInput,
    GeometryOutput,
    Point3D,
)
from .topology import TopologyBuilder
from .validator import ConstraintValidator
from .normalizer import Normalizer

__all__ = [
    "GeometryEngine",
    "SolverError",
    "GeometryInput",
    "GeometryOutput",
    "Constraint",
    "Point3D",
    "Edge",
    "Face",
    "TopologyBuilder",
    "ConstraintValidator",
    "Normalizer",
]

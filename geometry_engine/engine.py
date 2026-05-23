"""
GeometryEngine — constraint propagation solver.

Algorithm overview
------------------
1. Fixed-point loop: each pass tries every pending constraint handler.
   A handler returns True (made progress) or False (prerequisites missing).
2. Multi-right-angle detector: when the loop stalls, scan for patterns like
       right_angle [S, A, B]  +  right_angle [S, A, D]
   where S is unknown and A, B, D are placed.  This encodes "SA ⊥ plane(ABD)",
   the most frequent construction in Vietnamese HS geometry.
   → places S on ±normal from A, generates two candidates.
3. Candidate disambiguation: right_angle / angle / distance filters narrow
   multiple candidates; the last is broken by the z-priority heuristic.
4. Post-processing: topology (edges/faces), constraint validation, normalisation.

Supported constraint types
--------------------------
Shape anchors (place from scratch):
  square, rectangle, rhombus, trapezoid, equilateral_triangle,
  isosceles_triangle, right_triangle, regular_tetrahedron, cube,
  rectangular_prism, prism, regular_hexagon, regular_octahedron

Derived points (require prerequisites):
  midpoint, ratio_point, centroid, foot_perpendicular, foot_on_plane,
  perpendicular_to_plane, symmetric, intersection, apex/regular_pyramid/pyramid,
  truncated_pyramid

Filtering / disambiguation:
  right_angle, angle, distance, edge_length, on_line, parallel, perpendicular
"""
from __future__ import annotations

import logging
import numpy as np

from .errors import SolverError
from .handlers import (
    BaseShapeHandlers,
    ConstraintHandlers,
    DerivedPointHandlers,
    SolidShapeHandlers,
    SpecialRuleHandlers,
)
from .models import Constraint, GeometryInput, GeometryOutput, Point3D
from .topology import TopologyBuilder
from .validator import ConstraintValidator
from .normalizer import Normalizer
from .registry import get_handler
from .utils import normalize

logger = logging.getLogger(__name__)


class GeometryEngine(
    BaseShapeHandlers,
    SolidShapeHandlers,
    DerivedPointHandlers,
    SpecialRuleHandlers,
    ConstraintHandlers,
):

    def __init__(self) -> None:
        self.coords: dict[str, np.ndarray] = {}
        self._candidates: dict[str, list[np.ndarray]] = {}
        self._side_length: float = 1.0

    # ── Public API ────────────────────────────────────────────────────────────

    def solve(self, input_data: GeometryInput) -> GeometryOutput:
        self.coords = {}
        self._candidates = {}
        self._side_length = input_data.side_length

        pending = list(input_data.constraints)
        max_iter = len(pending) * 4 + 30

        for _ in range(max_iter):
            if not pending:
                break
            progress, pending = self._one_pass(pending)
            if not progress:
                # Try the perpendicular-system solver first
                if self._try_perpendicular_system(input_data.constraints):
                    continue
                if self._try_equilateral_right_angle_system(input_data.constraints):
                    continue
                if not self._commit_one_candidate():
                    break

        if pending:
            logger.warning("Unresolved constraints: %s", [c.type for c in pending])

        self._commit_all_candidates()

        # Topology
        builder = TopologyBuilder()
        for c in input_data.constraints:
            builder.process(c)
        edges, faces = builder.build()

        # Validation
        violations: list[str] = []
        if input_data.validate_constraints:
            violations = ConstraintValidator(self.coords).validate(input_data.constraints)
            for v in violations:
                logger.warning("Violation: %s", v)

        unresolved = [p for p in input_data.points if p not in self.coords]
        result_points: dict[str, Point3D] = {
            name: Point3D(
                x=round(float(self.coords[name][0]), 10),
                y=round(float(self.coords[name][1]), 10),
                z=round(float(self.coords[name][2]), 10),
            )
            for name in input_data.points
            if name in self.coords
        }

        output = GeometryOutput(
            points=result_points,
            edges=edges,
            faces=faces,
            unresolved_points=unresolved,
            violations=violations,
        )
        if input_data.normalize:
            output = Normalizer().normalize(output)
        return output

    def solve_json(self, json_str: str) -> dict:
        data = GeometryInput.model_validate_json(json_str)
        return self.solve(data).model_dump()

    # ── Constraint propagation ────────────────────────────────────────────────

    def _one_pass(
        self, pending: list[Constraint]
    ) -> tuple[bool, list[Constraint]]:
        progress = False
        still: list[Constraint] = []
        for c in pending:
            try:
                if self._get_handler(c.type)(c):
                    progress = True
                else:
                    still.append(c)
            except SolverError as exc:
                logger.error("SolverError '%s': %s", c.type, exc)
                still.append(c)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error '%s': %s", c.type, exc)
                still.append(c)
        return progress, still

    def _get_handler(self, ctype: str):
        return get_handler(self, ctype)

    # ── Candidate management ─────────────────────────────────────────────────

    def _commit_one_candidate(self) -> bool:
        for name, candidates in list(self._candidates.items()):
            if name not in self.coords:
                self.coords[name] = self._best_candidate(candidates)
                del self._candidates[name]
                return True
        return False

    def _commit_all_candidates(self) -> None:
        for name, candidates in list(self._candidates.items()):
            if name not in self.coords:
                self.coords[name] = self._best_candidate(candidates)
        self._candidates.clear()

    @staticmethod
    def _best_candidate(candidates: list[np.ndarray]) -> np.ndarray:
        """Prefer highest z, then y, then x → apex above base."""
        return max(
            candidates,
            key=lambda c: (round(c[2], 8), round(c[1], 8), round(c[0], 8)),
        )

    # ═══════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _planar_perp(direction: np.ndarray) -> np.ndarray:
        """Unit vector ⊥ direction that stays in XY plane when possible."""
        d = normalize(direction)
        cross = np.cross(d, np.array([0., 0., 1.]))
        if float(np.linalg.norm(cross)) > 1e-8:
            return normalize(cross)
        return np.array([0., 1., 0.])

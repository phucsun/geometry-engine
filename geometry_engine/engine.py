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
from collections import defaultdict
from typing import Callable

import numpy as np

from .models import Constraint, GeometryInput, GeometryOutput, Point3D
from .topology import TopologyBuilder
from .validator import ConstraintValidator
from .normalizer import Normalizer
from .utils import (
    are_perpendicular,
    centroid,
    cosine_of_angle,
    dist,
    equilateral_apex_candidates,
    intersect_line_plane,
    intersect_two_lines,
    midpoint,
    normalize,
    perpendicular_pair,
    plane_from_points,
    polygon_normal,
    project_point_onto_line,
    project_point_onto_plane,
    ratio_point,
    reflect_over_line,
    reflect_over_plane,
    reflect_over_point,
)

logger = logging.getLogger(__name__)


class SolverError(Exception):
    """Raised when a constraint is malformed or fundamentally unsolvable."""


class GeometryEngine:

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

    def _get_handler(self, ctype: str) -> Callable[[Constraint], bool]:
        m: dict[str, Callable[[Constraint], bool]] = {
            # ── Shape anchors ─────────────────────────────────────────────
            "square":                self._handle_square,
            "rectangle":             self._handle_rectangle,
            "rhombus":               self._handle_rhombus,
            "trapezoid":             self._handle_trapezoid,
            "equilateral_triangle":  self._handle_equilateral_triangle,
            "isosceles_triangle":    self._handle_isosceles_triangle,
            "right_triangle":        self._handle_right_triangle,
            "regular_tetrahedron":   self._handle_regular_tetrahedron,
            "cube":                  self._handle_cube,
            "rectangular_prism":     self._handle_rectangular_prism,
            "prism":                 self._handle_prism,
            "regular_hexagon":       self._handle_regular_hexagon,
            "regular_octahedron":    self._handle_regular_octahedron,
            # ── Derived points ────────────────────────────────────────────
            "midpoint":              self._handle_midpoint,
            "ratio_point":           self._handle_ratio_point,
            "centroid":              self._handle_centroid,
            "foot_perpendicular":    self._handle_foot_perpendicular,
            "foot_on_plane":         self._handle_foot_on_plane,
            "perpendicular_to_plane":self._handle_perpendicular_to_plane,
            "symmetric":             self._handle_symmetric,
            "intersection":          self._handle_intersection,
            "apex":                  self._handle_apex,
            "regular_pyramid":       self._handle_apex,
            "pyramid":               self._handle_apex,
            "truncated_pyramid":     self._handle_truncated_pyramid,
            # ── Filters / disambiguation ──────────────────────────────────
            "right_angle":           self._handle_right_angle,
            "angle":                 self._handle_angle,
            "distance":              self._handle_distance,
            "edge_length":           self._handle_edge_length,
            "on_line":               self._handle_on_line,
            "parallel":              lambda _c: True,
            "perpendicular":         lambda _c: True,
        }
        if ctype not in m:
            logger.warning("Unknown constraint '%s' — skipped.", ctype)
            return lambda _c: True
        return m[ctype]

    # ═══════════════════════════════════════════════════════════════════════
    # SHAPE ANCHORS
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_square(self, c: Constraint) -> bool:
        pts = c.points or []
        if len(pts) != 4:
            raise SolverError(f"'square' needs 4 points, got {len(pts)}")
        A, B, C, D = pts
        if all(p in self.coords for p in pts):
            return True
        if not any(p in self.coords for p in pts):
            s = self._side_length
            self.coords[A] = np.array([0., 0., 0.])
            self.coords[B] = np.array([s,  0., 0.])
            self.coords[C] = np.array([s,  s,  0.])
            self.coords[D] = np.array([0., s,  0.])
            return True
        if A in self.coords and B in self.coords and C not in self.coords and D not in self.coords:
            ab = self.coords[B] - self.coords[A]
            s = float(np.linalg.norm(ab))
            perp = self._planar_perp(ab) * s
            self.coords[C] = self.coords[B] + perp
            self.coords[D] = self.coords[A] + perp
            return True
        return False

    def _handle_rectangle(self, c: Constraint) -> bool:
        pts = c.points or []
        if len(pts) != 4:
            raise SolverError(f"'rectangle' needs 4 points, got {len(pts)}")
        A, B, C, D = pts
        if all(p in self.coords for p in pts):
            return True
        if not any(p in self.coords for p in pts):
            length = c.length or self._side_length
            width  = c.width  or self._side_length
            self.coords[A] = np.array([0.,     0.,    0.])
            self.coords[B] = np.array([length, 0.,    0.])
            self.coords[C] = np.array([length, width, 0.])
            self.coords[D] = np.array([0.,     width, 0.])
            return True
        if A in self.coords and B in self.coords and C not in self.coords and D not in self.coords:
            ab = self.coords[B] - self.coords[A]
            w = c.width or self._side_length
            perp = self._planar_perp(ab) * w
            self.coords[C] = self.coords[B] + perp
            self.coords[D] = self.coords[A] + perp
            return True
        return False

    def _handle_rhombus(self, c: Constraint) -> bool:
        pts = c.points or []
        if len(pts) != 4:
            raise SolverError(f"'rhombus' needs 4 points, got {len(pts)}")
        A, B, C, D = pts
        if all(p in self.coords for p in pts):
            return True
        if not any(p in self.coords for p in pts):
            s = self._side_length
            self.coords[A] = np.array([0.,          0.,                    0.])
            self.coords[B] = np.array([s,           0.,                    0.])
            self.coords[D] = np.array([s / 2.,  s * np.sqrt(3.) / 2., 0.])
            self.coords[C] = self.coords[B] + self.coords[D] - self.coords[A]
            return True
        return False

    def _handle_trapezoid(self, c: Constraint) -> bool:
        pts = c.points or []
        if len(pts) != 4:
            raise SolverError(f"'trapezoid' needs 4 points, got {len(pts)}")
        A, B, C, D = pts
        if all(p in self.coords for p in pts):
            return True
        if not any(p in self.coords for p in pts):
            ab = c.length or self._side_length
            dc = c.width  or (ab / 2.)
            h  = c.height or self._side_length
            off = (ab - dc) / 2.
            self.coords[A] = np.array([0.,        0., 0.])
            self.coords[B] = np.array([ab,        0., 0.])
            self.coords[C] = np.array([ab - off,  h,  0.])
            self.coords[D] = np.array([off,        h,  0.])
            return True
        return False

    def _handle_equilateral_triangle(self, c: Constraint) -> bool:
        pts = c.points or []
        if len(pts) != 3:
            raise SolverError(f"'equilateral_triangle' needs 3 points, got {len(pts)}")
        known = [p for p in pts if p in self.coords]
        unres = [p for p in pts if p not in self.coords and p not in self._candidates]
        if len(unres) == 0:
            return True
        if len(known) == 0 and len(unres) == 3:
            P, Q, R = pts
            s = self._side_length
            self.coords[P] = np.array([0.,   0., 0.])
            self.coords[Q] = np.array([s,    0., 0.])
            self.coords[R] = np.array([s/2., s * np.sqrt(3.)/2., 0.])
            return True
        if len(known) == 2 and len(unres) == 1:
            P_pos = self.coords[known[0]]
            Q_pos = self.coords[known[1]]
            self._candidates[unres[0]] = equilateral_apex_candidates(P_pos, Q_pos)
            return True
        return False

    def _handle_isosceles_triangle(self, c: Constraint) -> bool:
        pts = c.points or []
        if len(pts) != 3:
            return False
        apex_n, b1_n, b2_n = pts
        if apex_n in self.coords:
            return True
        if b1_n not in self.coords or b2_n not in self.coords:
            return False
        if apex_n in self._candidates:
            return True
        B1, B2 = self.coords[b1_n], self.coords[b2_n]
        M = midpoint(B1, B2)
        h = c.length if c.length is not None else dist(B1, B2)
        u = normalize(B2 - B1)
        v, w = perpendicular_pair(u)
        self._candidates[apex_n] = [M+h*w, M-h*w, M+h*v, M-h*v]
        return True

    def _handle_right_triangle(self, c: Constraint) -> bool:
        pts = c.points or []
        if len(pts) != 3:
            return False
        rv, P, Q = pts
        return self._handle_right_angle(Constraint(type="right_angle", points=[P, rv, Q]))

    def _handle_regular_tetrahedron(self, c: Constraint) -> bool:
        pts = c.points or []
        if len(pts) != 4:
            return False
        A, B, C, D = pts
        if all(p in self.coords for p in pts):
            return True
        if any(p in self.coords for p in pts):
            return False
        s = self._side_length
        h_tri  = s * np.sqrt(3.) / 2.
        ctr    = np.array([s/2., h_tri/3., 0.])
        apex_h = s * np.sqrt(2./3.)
        self.coords[A] = np.array([0.,   0.,    0.])
        self.coords[B] = np.array([s,    0.,    0.])
        self.coords[C] = np.array([s/2., h_tri, 0.])
        self.coords[D] = ctr + np.array([0., 0., apex_h])
        return True

    def _handle_cube(self, c: Constraint) -> bool:
        pts = c.points or []
        if len(pts) != 8:
            return False
        if any(p in self.coords for p in pts):
            return True
        s = self._side_length
        A, B, C, D, E, F, G, H = pts
        self.coords[A] = np.array([0., 0., 0.])
        self.coords[B] = np.array([s,  0., 0.])
        self.coords[C] = np.array([s,  s,  0.])
        self.coords[D] = np.array([0., s,  0.])
        self.coords[E] = np.array([0., 0., s ])
        self.coords[F] = np.array([s,  0., s ])
        self.coords[G] = np.array([s,  s,  s ])
        self.coords[H] = np.array([0., s,  s ])
        return True

    def _handle_rectangular_prism(self, c: Constraint) -> bool:
        """Hình hộp chữ nhật với 3 chiều độc lập: length × width × height."""
        pts = c.points or []
        if len(pts) != 8:
            return False
        if any(p in self.coords for p in pts):
            return True
        length = c.length or self._side_length
        width  = c.width  or self._side_length
        height = c.height or self._side_length
        A, B, C, D, Ap, Bp, Cp, Dp = pts
        self.coords[A]  = np.array([0.,     0.,    0.     ])
        self.coords[B]  = np.array([length, 0.,    0.     ])
        self.coords[C]  = np.array([length, width, 0.     ])
        self.coords[D]  = np.array([0.,     width, 0.     ])
        self.coords[Ap] = np.array([0.,     0.,    height ])
        self.coords[Bp] = np.array([length, 0.,    height ])
        self.coords[Cp] = np.array([length, width, height ])
        self.coords[Dp] = np.array([0.,     width, height ])
        return True

    def _handle_prism(self, c: Constraint) -> bool:
        """Lăng trụ tam giác đều: đáy ABC trong mặt phẳng XY, đỉnh DEF phía trên."""
        pts = c.points or []
        if len(pts) != 6:
            return False
        if any(p in self.coords for p in pts):
            return True
        A, B, C, D, E, F = pts
        s = self._side_length
        h_tri   = s * np.sqrt(3.) / 2.
        prism_h = c.height or s
        self.coords[A] = np.array([0.,   0.,     0.      ])
        self.coords[B] = np.array([s,    0.,     0.      ])
        self.coords[C] = np.array([s/2., h_tri,  0.      ])
        self.coords[D] = np.array([0.,   0.,     prism_h ])
        self.coords[E] = np.array([s,    0.,     prism_h ])
        self.coords[F] = np.array([s/2., h_tri,  prism_h ])
        return True

    def _handle_regular_hexagon(self, c: Constraint) -> bool:
        """Lục giác đều trong mặt phẳng XY."""
        pts = c.points or []
        if len(pts) != 6:
            raise SolverError("'regular_hexagon' needs 6 points")
        if any(p in self.coords for p in pts):
            return True
        s = self._side_length
        for i, name in enumerate(pts):
            angle = np.radians(60. * i)
            self.coords[name] = np.array([s * np.cos(angle), s * np.sin(angle), 0.])
        return True

    def _handle_regular_octahedron(self, c: Constraint) -> bool:
        """
        Bát diện đều: 6 đỉnh theo sơ đồ (+/-1, 0, 0), (0, +/-1, 0), (0, 0, +/-1).
        points = [Top, Bottom, E1, E2, E3, E4].
        """
        pts = c.points or []
        if len(pts) != 6:
            raise SolverError("'regular_octahedron' needs 6 points")
        if any(p in self.coords for p in pts):
            return True
        s = self._side_length / np.sqrt(2.)     # so that all edges = side_length
        T, B, E1, E2, E3, E4 = pts
        self.coords[T]  = np.array([ 0.,  0.,  s ])
        self.coords[B]  = np.array([ 0.,  0., -s ])
        self.coords[E1] = np.array([ s,   0.,  0.])
        self.coords[E2] = np.array([ 0.,  s,   0.])
        self.coords[E3] = np.array([-s,   0.,  0.])
        self.coords[E4] = np.array([ 0., -s,   0.])
        return True

    # ═══════════════════════════════════════════════════════════════════════
    # DERIVED POINTS
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_midpoint(self, c: Constraint) -> bool:
        J = c.point
        seg = c.segment or []
        if not J or len(seg) != 2:
            raise SolverError("'midpoint' needs 'point' and 'segment=[P,Q]'")
        P, Q = seg
        if J in self.coords:
            return True
        if P not in self.coords or Q not in self.coords:
            return False
        self.coords[J] = midpoint(self.coords[P], self.coords[Q])
        return True

    def _handle_ratio_point(self, c: Constraint) -> bool:
        """G = A + ratio*(B-A)."""
        G = c.point
        seg = c.segment or []
        t = c.ratio
        if not G or len(seg) != 2 or t is None:
            raise SolverError("'ratio_point' needs 'point', 'segment=[A,B]', 'ratio'")
        A, B = seg
        if G in self.coords:
            return True
        if A not in self.coords or B not in self.coords:
            return False
        self.coords[G] = ratio_point(self.coords[A], self.coords[B], t)
        return True

    def _handle_centroid(self, c: Constraint) -> bool:
        """
        Trọng tâm.  point=G, points=[A,B,C,...].
        G = mean(A, B, C, …).
        """
        G = c.point
        ref = c.points or []
        if not G or not ref:
            raise SolverError("'centroid' needs 'point' and 'points'")
        if G in self.coords:
            return True
        if not all(p in self.coords for p in ref):
            return False
        self.coords[G] = centroid([self.coords[p] for p in ref])
        return True

    def _handle_foot_perpendicular(self, c: Constraint) -> bool:
        """Chân đường vuông góc từ from_point xuống đường thẳng segment=[A,B]."""
        H = c.point
        S = c.from_point
        seg = c.segment or []
        if not H or not S or len(seg) != 2:
            raise SolverError(
                "'foot_perpendicular' needs 'point', 'from_point', 'segment=[A,B]'"
            )
        A, B = seg
        if H in self.coords:
            return True
        for n in [S, A, B]:
            if n not in self.coords:
                return False
        self.coords[H] = project_point_onto_line(
            self.coords[S], self.coords[A], self.coords[B]
        )
        return True

    def _handle_foot_on_plane(self, c: Constraint) -> bool:
        """
        Hình chiếu điểm from_point lên mặt phẳng xác định bởi points.
        point=H, from_point=S, points=[A,B,C,D,...].
        """
        H = c.point
        S = c.from_point
        ref = c.points or []
        if not H or not S or len(ref) < 3:
            raise SolverError(
                "'foot_on_plane' needs 'point', 'from_point', 'points'(≥3)"
            )
        if H in self.coords:
            return True
        if S not in self.coords or not all(p in self.coords for p in ref):
            return False
        plane_pt, normal = plane_from_points([self.coords[p] for p in ref])
        self.coords[H] = project_point_onto_plane(self.coords[S], plane_pt, normal)
        return True

    def _handle_perpendicular_to_plane(self, c: Constraint) -> bool:
        """
        SA ⊥ mặt phẳng (ABCD…).
        point=S, from_point=A (foot of perpendicular), points=[A,B,C,D,...], length=h.

        S = A + h * normal(plane).
        """
        S = c.point
        foot_name = c.from_point
        ref = c.points or []
        if not S or not foot_name or len(ref) < 3:
            raise SolverError(
                "'perpendicular_to_plane' needs 'point', 'from_point', 'points'(≥3)"
            )
        if S in self.coords:
            return True
        if foot_name not in self.coords or not all(p in self.coords for p in ref):
            return False
        _, normal = plane_from_points([self.coords[p] for p in ref])
        h = c.length if c.length is not None else self._side_length
        foot = self.coords[foot_name]
        # Offer two candidates (up / down); z-priority heuristic will choose up
        self._candidates[S] = [foot + h * normal, foot - h * normal]
        return True

    def _handle_symmetric(self, c: Constraint) -> bool:
        """
        Điểm đối xứng.  point=P', from_point=P, points=reference.

        len(points)==1  → đối xứng qua điểm M:       P' = 2M - P
        len(points)==2  → đối xứng qua đường AB:      P' = reflect_over_line
        len(points)>=3  → đối xứng qua mặt phẳng ABC: P' = reflect_over_plane
        """
        Pp = c.point
        P  = c.from_point
        ref = c.points or []
        if not Pp or not P or not ref:
            raise SolverError(
                "'symmetric' needs 'point', 'from_point', 'points'(reference)"
            )
        if Pp in self.coords:
            return True
        if P not in self.coords or not all(p in self.coords for p in ref):
            return False

        P_pos = self.coords[P]

        if len(ref) == 1:
            self.coords[Pp] = reflect_over_point(P_pos, self.coords[ref[0]])
        elif len(ref) == 2:
            self.coords[Pp] = reflect_over_line(
                P_pos, self.coords[ref[0]], self.coords[ref[1]]
            )
        else:
            plane_pt, normal = plane_from_points([self.coords[p] for p in ref])
            self.coords[Pp] = reflect_over_plane(P_pos, plane_pt, normal)
        return True

    def _handle_intersection(self, c: Constraint) -> bool:
        """
        Giao điểm.  point=I.

        segment=[A,B], points=[C,D]       → giao 2 đường thẳng AB và CD
        segment=[A,B], points=[C,D,E,...] → giao đường AB với mặt phẳng CDE…
        """
        I = c.point
        seg = c.segment or []
        ref = c.points or []
        if not I or len(seg) != 2 or len(ref) < 2:
            raise SolverError(
                "'intersection' needs 'point', 'segment=[A,B]', 'points'(≥2)"
            )
        if I in self.coords:
            return True
        all_names = list(seg) + list(ref)
        if not all(p in self.coords for p in all_names):
            return False

        A, B = self.coords[seg[0]], self.coords[seg[1]]

        if len(ref) == 2:
            # Line–line intersection
            C, D = self.coords[ref[0]], self.coords[ref[1]]
            pt = intersect_two_lines(A, B, C, D)
            if pt is None:
                logger.warning("Intersection of lines %s and %s not found.", seg, ref)
                return True  # skip gracefully
            self.coords[I] = pt
        else:
            # Line–plane intersection
            plane_pos, normal = plane_from_points([self.coords[p] for p in ref])
            direction = B - A
            pt = intersect_line_plane(A, direction, plane_pos, normal)
            if pt is None:
                logger.warning("Line %s parallel to plane %s.", seg, ref)
                return True
            self.coords[I] = pt
        return True

    def _handle_apex(self, c: Constraint) -> bool:
        """Đỉnh chóp points[0] đặt trên đường thẳng vuông góc qua tâm đáy."""
        pts = c.points or []
        if len(pts) < 2:
            return False
        apex = pts[0]
        base = pts[1:]
        if apex in self.coords:
            return True
        if not all(p in self.coords for p in base):
            return False
        base_positions = [self.coords[p] for p in base]
        center = centroid(base_positions)
        normal = polygon_normal(base_positions)
        h = c.length if c.length is not None else self._side_length
        self.coords[apex] = center + h * normal
        return True

    def _handle_truncated_pyramid(self, c: Constraint) -> bool:
        """
        Hình chóp cụt.
        points = [base1, base2, …, baseN, top1, top2, …, topN] (2N points).
        ratio = top_side / base_side (default 0.5).
        height = vertical distance between bases (default side_length).
        """
        pts = c.points or []
        if len(pts) < 4 or len(pts) % 2 != 0:
            return False
        n = len(pts) // 2
        base_names = pts[:n]
        top_names  = pts[n:]
        if all(p in self.coords for p in pts):
            return True
        if not all(p in self.coords for p in base_names):
            return False
        base_positions = [self.coords[p] for p in base_names]
        center = centroid(base_positions)
        normal = polygon_normal(base_positions)
        h = c.height if c.height is not None else self._side_length
        r = c.ratio  if c.ratio  is not None else 0.5
        top_center = center + h * normal
        for i, name in enumerate(top_names):
            if name not in self.coords:
                self.coords[name] = top_center + r * (base_positions[i] - center)
        return True

    # ═══════════════════════════════════════════════════════════════════════
    # FILTERING / DISAMBIGUATION
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_right_angle(self, c: Constraint) -> bool:
        """
        Góc vuông tại đỉnh giữa. points = [arm1, vertex, arm2].
        Dùng để lọc candidates của arm1 hoặc arm2.
        """
        pts = c.points or []
        if len(pts) != 3:
            raise SolverError(f"'right_angle' needs 3 points, got {len(pts)}")
        arm1, vertex, arm2 = pts
        if all(p in self.coords for p in pts):
            return True
        if vertex not in self.coords:
            return False
        V = self.coords[vertex]
        for unknown, known in [(arm1, arm2), (arm2, arm1)]:
            if unknown not in self._candidates:
                continue
            if known not in self.coords:
                continue
            vk = self.coords[known] - V
            filtered = [
                cand for cand in self._candidates[unknown]
                if are_perpendicular(cand - V, vk)
            ]
            if filtered:
                if len(filtered) == 1:
                    self.coords[unknown] = filtered[0]
                    del self._candidates[unknown]
                else:
                    self._candidates[unknown] = filtered
                return True
        return False

    def _handle_angle(self, c: Constraint) -> bool:
        """
        Góc cụ thể (degrees) tại đỉnh giữa. points=[arm1, vertex, arm2].
        Lọc candidates của arm1 hoặc arm2 bằng cosine.
        """
        pts = c.points or []
        deg = c.degrees
        if len(pts) != 3 or deg is None:
            return False
        arm1, vertex, arm2 = pts
        if all(p in self.coords for p in pts):
            return True
        if vertex not in self.coords:
            return False
        V = self.coords[vertex]
        cos_target = np.cos(np.radians(deg))
        tol = 1e-3
        for unknown, known in [(arm1, arm2), (arm2, arm1)]:
            if unknown not in self._candidates:
                continue
            if known not in self.coords:
                continue
            vk = self.coords[known] - V
            filtered = [
                cand for cand in self._candidates[unknown]
                if abs(cosine_of_angle(cand - V, vk) - cos_target) < tol
            ]
            if filtered:
                if len(filtered) == 1:
                    self.coords[unknown] = filtered[0]
                    del self._candidates[unknown]
                else:
                    self._candidates[unknown] = filtered
                return True
        return False

    def _handle_distance(self, c: Constraint) -> bool:
        """
        Khoảng cách bằng L. points=[P,Q], length=L.
        Lọc candidates của P hoặc Q theo khoảng cách.
        Nếu cả hai chưa xác định → cập nhật side_length mặc định.
        """
        pts = c.points or []
        L = c.length
        if len(pts) != 2 or L is None:
            return False
        P_name, Q_name = pts
        if P_name in self.coords and Q_name in self.coords:
            return True
        # Update side_length early (before any points are placed)
        if not self.coords:
            self._side_length = L
            return True
        for unknown, known in [(P_name, Q_name), (Q_name, P_name)]:
            if unknown not in self._candidates:
                continue
            if known not in self.coords:
                continue
            K = self.coords[known]
            filtered = [
                cand for cand in self._candidates[unknown]
                if abs(dist(cand, K) - L) < 1e-5
            ]
            if filtered:
                if len(filtered) == 1:
                    self.coords[unknown] = filtered[0]
                    del self._candidates[unknown]
                else:
                    self._candidates[unknown] = filtered
                return True
        return False

    def _handle_edge_length(self, c: Constraint) -> bool:
        if c.length is not None and not self.coords:
            self._side_length = c.length
        return True

    def _handle_on_line(self, c: Constraint) -> bool:
        """
        Điểm nằm trên đường thẳng AB — thường kết hợp với điều kiện khác.
        Nếu point đã được xác định → True.  Nếu chưa → False (chờ constraint khác).
        """
        P = c.point
        seg = c.segment or []
        if not P or len(seg) != 2:
            return False
        if P in self.coords:
            return True
        A_name, B_name = seg
        if A_name not in self.coords or B_name not in self.coords:
            return False
        # Can't determine exact position without more info; skip (return True) to
        # avoid blocking the loop, but don't place the point.
        return True

    # ═══════════════════════════════════════════════════════════════════════
    # MULTI RIGHT-ANGLE PERPENDICULAR SYSTEM
    # ═══════════════════════════════════════════════════════════════════════

    def _try_perpendicular_system(self, all_constraints: list[Constraint]) -> bool:
        """
        Phát hiện mẫu "SA ⊥ (ABCD)" được biểu diễn qua hai right_angle:
            right_angle [S, A, B]
            right_angle [S, A, D]
        Khi S chưa có toạ độ, A/B/D đã biết:
          (S-A) ⊥ AB  và  (S-A) ⊥ AD  →  (S-A) ∥ AB×AD
          → S = A ± h*normalize(AB×AD)
        Sinh 2 candidates cho S và trả về True (để vòng lặp chính tiếp tục).
        """
        # Group right_angle constraints: {(unknown, vertex): [known_arm, ...]}
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)

        for con in all_constraints:
            if con.type != "right_angle":
                continue
            pts = con.points or []
            if len(pts) != 3:
                continue
            arm1, vertex, arm2 = pts
            if vertex not in self.coords:
                continue
            # Check arm1 unknown, arm2 known
            if arm1 not in self.coords and arm1 not in self._candidates:
                if arm2 in self.coords:
                    groups[(arm1, vertex)].append(arm2)
            # Check arm2 unknown, arm1 known
            if arm2 not in self.coords and arm2 not in self._candidates:
                if arm1 in self.coords:
                    groups[(arm2, vertex)].append(arm1)

        made_progress = False
        for (unknown, vertex), known_arms in groups.items():
            if unknown in self.coords or unknown in self._candidates:
                continue
            if len(known_arms) < 2:
                continue

            V = self.coords[vertex]
            vecs = [self.coords[k] - V for k in known_arms]

            # Find first pair of linearly independent vectors → cross product = normal
            normal = None
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    cross = np.cross(vecs[i], vecs[j])
                    if np.linalg.norm(cross) > 1e-8:
                        normal = normalize(cross)
                        break
                if normal is not None:
                    break

            if normal is None:
                continue  # All arms coplanar → underdetermined

            # Ensure the normal points "up" (positive z)
            if normal[2] < 0:
                normal = -normal

            h = self._side_length
            self._candidates[unknown] = [V + h * normal, V - h * normal]
            logger.debug(
                "Perpendicular system: %s = %s ± %g * %s",
                unknown, vertex, h, normal,
            )
            made_progress = True

        return made_progress

    # ═══════════════════════════════════════════════════════════════════════
    # CANDIDATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════

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

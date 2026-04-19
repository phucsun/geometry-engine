"""
ConstraintValidator — verifies that solved 3-D coordinates satisfy every constraint.

After the engine places all points, run the validator to catch numerical drift
or unsolvable systems early.  The output is a list of human-readable strings
describing each violated constraint.
"""
from __future__ import annotations

import numpy as np

from .models import Constraint
from .utils import dist, are_perpendicular, polygon_normal, plane_from_points, normalize


_TOL = 1e-5   # absolute tolerance for geometric checks


class ConstraintValidator:
    """
    Usage::

        validator = ConstraintValidator(coords, tol=1e-5)
        violations = validator.validate(constraints)
    """

    def __init__(self, coords: dict[str, np.ndarray], tol: float = _TOL) -> None:
        self.coords = coords
        self.tol = tol

    def validate(self, constraints: list[Constraint]) -> list[str]:
        """Return a list of violation messages (empty → all OK)."""
        violations: list[str] = []
        for c in constraints:
            msg = self._check(c)
            if msg:
                violations.append(msg)
        return violations

    # ── Per-constraint checks ─────────────────────────────────────────────────

    def _check(self, c: Constraint) -> str | None:
        checker = {
            "square":                self._chk_square,
            "rectangle":             self._chk_quad_sides,
            "rhombus":               self._chk_rhombus,
            "equilateral_triangle":  self._chk_equilateral,
            "right_angle":           self._chk_right_angle,
            "midpoint":              self._chk_midpoint,
            "ratio_point":           self._chk_ratio_point,
            "regular_tetrahedron":   self._chk_regular_tetrahedron,
            "apex":                  self._chk_apex_equidistant,
            "regular_pyramid":       self._chk_apex_equidistant,
            "pyramid":               self._chk_apex_equidistant,
            "centroid":              self._chk_centroid,
            "perpendicular_to_plane": self._chk_perpendicular_to_plane,
            "angle":                 self._chk_angle,
            "symmetric":             self._chk_symmetric,
            "foot_perpendicular":    self._chk_foot_perpendicular,
            "foot_on_plane":         self._chk_foot_on_plane,
            "regular_hexagon":       self._chk_regular_polygon,
            "regular_octahedron":    self._chk_regular_octahedron,
        }.get(c.type)
        return checker(c) if checker else None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get(self, *names: str) -> list[np.ndarray] | None:
        """Return coordinate arrays for *names*, or None if any is missing."""
        result = []
        for n in names:
            if n not in self.coords:
                return None
            result.append(self.coords[n])
        return result

    def _sides(self, positions: list[np.ndarray]) -> list[float]:
        n = len(positions)
        return [dist(positions[i], positions[(i + 1) % n]) for i in range(n)]

    # ── Checkers ──────────────────────────────────────────────────────────────

    def _chk_square(self, c: Constraint) -> str | None:
        pts = c.points or []
        pos = self._get(*pts)
        if pos is None or len(pos) != 4:
            return None
        sides = self._sides(pos)
        s = sides[0]
        if not all(abs(x - s) < self.tol for x in sides):
            return f"square {pts}: sides not equal {[round(x,6) for x in sides]}"
        # Check right angles at each corner
        for i in range(4):
            A = pos[(i - 1) % 4]
            V = pos[i]
            B = pos[(i + 1) % 4]
            if not are_perpendicular(A - V, B - V, tol=self.tol):
                dot = np.dot(A - V, B - V)
                return f"square {pts}: angle at {pts[i]} not 90° (dot={dot:.4f})"
        return None

    def _chk_quad_sides(self, c: Constraint) -> str | None:
        pts = c.points or []
        pos = self._get(*pts)
        if pos is None or len(pos) != 4:
            return None
        sides = self._sides(pos)
        # For rectangle: opposite sides equal
        if not (abs(sides[0] - sides[2]) < self.tol and abs(sides[1] - sides[3]) < self.tol):
            return f"rectangle {pts}: opposite sides not equal {[round(x,6) for x in sides]}"
        return None

    def _chk_rhombus(self, c: Constraint) -> str | None:
        pts = c.points or []
        pos = self._get(*pts)
        if pos is None or len(pos) != 4:
            return None
        sides = self._sides(pos)
        s = sides[0]
        if not all(abs(x - s) < self.tol for x in sides):
            return f"rhombus {pts}: sides not equal {[round(x,6) for x in sides]}"
        return None

    def _chk_equilateral(self, c: Constraint) -> str | None:
        pts = c.points or []
        pos = self._get(*pts)
        if pos is None or len(pos) != 3:
            return None
        d01 = dist(pos[0], pos[1])
        d12 = dist(pos[1], pos[2])
        d20 = dist(pos[2], pos[0])
        if not (abs(d01 - d12) < self.tol and abs(d12 - d20) < self.tol):
            return (
                f"equilateral_triangle {pts}: sides not equal "
                f"({d01:.6f}, {d12:.6f}, {d20:.6f})"
            )
        return None

    def _chk_right_angle(self, c: Constraint) -> str | None:
        pts = c.points or []
        if len(pts) != 3:
            return None
        arm1_name, vertex_name, arm2_name = pts
        pos = self._get(arm1_name, vertex_name, arm2_name)
        if pos is None:
            return None
        arm1, vertex, arm2 = pos
        v1 = arm1 - vertex
        v2 = arm2 - vertex
        if not are_perpendicular(v1, v2, tol=self.tol):
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            cosine = np.dot(v1, v2) / (n1 * n2 + 1e-30)
            angle = float(np.degrees(np.arccos(np.clip(abs(cosine), 0, 1))))
            return (
                f"right_angle {pts}: angle at {vertex_name} = "
                f"{90 - angle:.4f}° off from 90°"
            )
        return None

    def _chk_midpoint(self, c: Constraint) -> str | None:
        J = c.point
        seg = c.segment or []
        if not J or len(seg) != 2:
            return None
        P_name, Q_name = seg
        pos = self._get(J, P_name, Q_name)
        if pos is None:
            return None
        J_pos, P_pos, Q_pos = pos
        expected = (P_pos + Q_pos) / 2.0
        err = dist(J_pos, expected)
        if err > self.tol:
            return f"midpoint {J} of {seg}: error = {err:.6f}"
        return None

    def _chk_ratio_point(self, c: Constraint) -> str | None:
        G = c.point
        seg = c.segment or []
        t = c.ratio
        if not G or len(seg) != 2 or t is None:
            return None
        A_name, B_name = seg
        pos = self._get(G, A_name, B_name)
        if pos is None:
            return None
        G_pos, A_pos, B_pos = pos
        expected = A_pos + t * (B_pos - A_pos)
        err = dist(G_pos, expected)
        if err > self.tol:
            return f"ratio_point {G} = {A_name}+{t}*{B_name}: error = {err:.6f}"
        return None

    def _chk_regular_tetrahedron(self, c: Constraint) -> str | None:
        pts = c.points or []
        if len(pts) != 4:
            return None
        pos = self._get(*pts)
        if pos is None:
            return None
        edges = [
            dist(pos[i], pos[j])
            for i in range(4)
            for j in range(i + 1, 4)
        ]
        s = edges[0]
        if not all(abs(e - s) < self.tol for e in edges):
            return f"regular_tetrahedron {pts}: edges not equal {[round(e,6) for e in edges]}"
        return None

    def _chk_centroid(self, c: Constraint) -> str | None:
        G = c.point
        ref = c.points or []
        if not G or not ref:
            return None
        pos = self._get(G, *ref)
        if pos is None:
            return None
        G_pos = pos[0]
        refs_pos = pos[1:]
        expected = np.mean(refs_pos, axis=0)
        err = dist(G_pos, expected)
        if err > self.tol:
            return f"centroid {G} of {ref}: error = {err:.6f}"
        return None

    def _chk_perpendicular_to_plane(self, c: Constraint) -> str | None:
        S = c.point
        foot_name = c.from_point
        ref = c.points or []
        if not S or not foot_name or len(ref) < 3:
            return None
        pos = self._get(S, foot_name, *ref)
        if pos is None:
            return None
        S_pos, foot_pos = pos[0], pos[1]
        ref_positions = pos[2:]
        _, normal = plane_from_points(ref_positions)
        SA = S_pos - foot_pos
        if np.linalg.norm(SA) < 1e-10:
            return None
        cross = np.cross(SA, normal)
        if np.linalg.norm(cross) > self.tol:
            return f"perpendicular_to_plane {S}: SA not parallel to plane normal"
        return None

    def _chk_angle(self, c: Constraint) -> str | None:
        pts = c.points or []
        deg = c.degrees
        if len(pts) != 3 or deg is None:
            return None
        arm1_name, vertex_name, arm2_name = pts
        pos = self._get(arm1_name, vertex_name, arm2_name)
        if pos is None:
            return None
        arm1, vertex, arm2 = pos
        v1 = arm1 - vertex
        v2 = arm2 - vertex
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            return None
        cos_val = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        actual_deg = float(np.degrees(np.arccos(cos_val)))
        if abs(actual_deg - deg) > 0.1:
            return f"angle {pts}: expected {deg}° got {actual_deg:.4f}°"
        return None

    def _chk_symmetric(self, c: Constraint) -> str | None:
        Pp = c.point
        P = c.from_point
        ref = c.points or []
        if not Pp or not P or not ref:
            return None
        pos = self._get(Pp, P, *ref)
        if pos is None:
            return None
        Pp_pos, P_pos = pos[0], pos[1]
        ref_positions = pos[2:]
        if len(ref) == 1:
            M = ref_positions[0]
            expected = 2.0 * M - P_pos
        elif len(ref) == 2:
            from .utils import reflect_over_line
            expected = reflect_over_line(P_pos, ref_positions[0], ref_positions[1])
        else:
            from .utils import reflect_over_plane
            plane_pt, normal = plane_from_points(ref_positions)
            expected = reflect_over_plane(P_pos, plane_pt, normal)
        err = dist(Pp_pos, expected)
        if err > self.tol:
            return f"symmetric {Pp} of {P} over {ref}: error = {err:.6f}"
        return None

    def _chk_foot_perpendicular(self, c: Constraint) -> str | None:
        H = c.point
        S = c.from_point
        seg = c.segment or []
        if not H or not S or len(seg) != 2:
            return None
        A_name, B_name = seg
        pos = self._get(H, S, A_name, B_name)
        if pos is None:
            return None
        H_pos, S_pos, A_pos, B_pos = pos
        from .utils import project_point_onto_line
        expected = project_point_onto_line(S_pos, A_pos, B_pos)
        err = dist(H_pos, expected)
        if err > self.tol:
            return f"foot_perpendicular {H}: error = {err:.6f}"
        return None

    def _chk_foot_on_plane(self, c: Constraint) -> str | None:
        H = c.point
        S = c.from_point
        ref = c.points or []
        if not H or not S or len(ref) < 3:
            return None
        pos = self._get(H, S, *ref)
        if pos is None:
            return None
        H_pos, S_pos = pos[0], pos[1]
        ref_positions = pos[2:]
        plane_pt, normal = plane_from_points(ref_positions)
        from .utils import project_point_onto_plane
        expected = project_point_onto_plane(S_pos, plane_pt, normal)
        err = dist(H_pos, expected)
        if err > self.tol:
            return f"foot_on_plane {H}: error = {err:.6f}"
        return None

    def _chk_regular_polygon(self, c: Constraint) -> str | None:
        pts = c.points or []
        if len(pts) < 3:
            return None
        pos = self._get(*pts)
        if pos is None:
            return None
        sides = self._sides(pos)
        s = sides[0]
        if not all(abs(x - s) < self.tol for x in sides):
            return f"{c.type} {pts}: sides not equal {[round(x,6) for x in sides]}"
        return None

    def _chk_regular_octahedron(self, c: Constraint) -> str | None:
        pts = c.points or []
        if len(pts) != 6:
            return None
        pos = self._get(*pts)
        if pos is None:
            return None
        T, B, E1, E2, E3, E4 = pos
        equator = [E1, E2, E3, E4]
        edges = []
        for e in equator:
            edges.append(dist(T, e))
            edges.append(dist(B, e))
        for i in range(4):
            edges.append(dist(equator[i], equator[(i + 1) % 4]))
        s = edges[0]
        if not all(abs(e - s) < self.tol for e in edges):
            return f"regular_octahedron {pts}: edges not equal"
        return None

    def _chk_apex_equidistant(self, c: Constraint) -> str | None:
        """For regular pyramid: apex is equidistant from all base vertices."""
        pts = c.points or []
        if len(pts) < 3:
            return None
        apex_name = pts[0]
        base = pts[1:]
        pos_apex = self.coords.get(apex_name)
        if pos_apex is None:
            return None
        base_dists = []
        for p in base:
            if p not in self.coords:
                return None
            base_dists.append(dist(pos_apex, self.coords[p]))
        if not base_dists:
            return None
        s = base_dists[0]
        if not all(abs(d - s) < self.tol for d in base_dists):
            return (
                f"{c.type} {pts}: apex not equidistant from base "
                f"{[round(d,6) for d in base_dists]}"
            )
        return None

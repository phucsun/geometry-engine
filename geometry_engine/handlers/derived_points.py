"""Handlers for derived construction points in geometry drawings.

These methods support midpoints, centroids, perpendicular feet, projections,
reflections, intersections, and triangle centers used after anchor shapes exist.
"""
from __future__ import annotations

import logging

from geometry_engine.errors import SolverError
from geometry_engine.models import Constraint
from geometry_engine.utils import (
    angle_bisector_foot,
    centroid,
    circumcenter,
    circumscribed_sphere_center,
    incenter,
    intersect_line_plane,
    intersect_two_lines,
    midpoint,
    orthocenter,
    plane_from_points,
    project_point_onto_line,
    project_point_onto_plane,
    ratio_point,
    reflect_over_line,
    reflect_over_plane,
    reflect_over_point,
)

logger = logging.getLogger(__name__)


class DerivedPointHandlers:
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


    def _handle_circumcenter(self, c: Constraint) -> bool:
        """
        Tâm đường tròn ngoại tiếp tam giác.
        point=O, points=[A, B, C].
        """
        O = c.point
        pts = c.points or []
        if not O or len(pts) != 3:
            raise SolverError("'circumcenter' needs 'point' and 'points=[A,B,C]'")
        if O in self.coords:
            return True
        if not all(p in self.coords for p in pts):
            return False
        A, B, C = (self.coords[p] for p in pts)
        self.coords[O] = circumcenter(A, B, C)
        return True


    def _handle_orthocenter(self, c: Constraint) -> bool:
        """
        Trực tâm tam giác.
        point=H, points=[A, B, C].
        """
        H = c.point
        pts = c.points or []
        if not H or len(pts) != 3:
            raise SolverError("'orthocenter' needs 'point' and 'points=[A,B,C]'")
        if H in self.coords:
            return True
        if not all(p in self.coords for p in pts):
            return False
        A, B, C = (self.coords[p] for p in pts)
        self.coords[H] = orthocenter(A, B, C)
        return True


    def _handle_incenter(self, c: Constraint) -> bool:
        """
        Tâm đường tròn nội tiếp tam giác.
        point=I, points=[A, B, C].
        """
        I = c.point
        pts = c.points or []
        if not I or len(pts) != 3:
            raise SolverError("'incenter' needs 'point' and 'points=[A,B,C]'")
        if I in self.coords:
            return True
        if not all(p in self.coords for p in pts):
            return False
        A, B, C = (self.coords[p] for p in pts)
        self.coords[I] = incenter(A, B, C)
        return True


    def _handle_equidistant(self, c: Constraint) -> bool:
        """
        Điểm cách đều n điểm đã biết — tâm mặt cầu ngoại tiếp.
        point=O, points=[A, B, C, D, …].
        Dùng least-squares để xác định O.
        """
        O = c.point
        pts = c.points or []
        if not O or len(pts) < 2:
            raise SolverError("'equidistant' needs 'point' and at least 2 'points'")
        if O in self.coords:
            return True
        if not all(p in self.coords for p in pts):
            return False
        positions = [self.coords[p] for p in pts]
        center = circumscribed_sphere_center(positions)
        if center is None:
            logger.warning("equidistant: underdetermined for %s", pts)
            return False
        self.coords[O] = center
        return True


    def _handle_angle_bisector(self, c: Constraint) -> bool:
        """
        Chân đường phân giác từ đỉnh giữa xuống cạnh đối diện.
        point=D, points=[A, B, C]:
          B là đỉnh có góc cần phân giác, D là chân trên AC.
          AD/DC = |AB|/|BC|  (định lý phân giác).
        """
        D = c.point
        pts = c.points or []
        if not D or len(pts) != 3:
            raise SolverError("'angle_bisector' needs 'point' and 'points=[A,B,C]'")
        if D in self.coords:
            return True
        A_n, B_n, C_n = pts
        if not all(p in self.coords for p in pts):
            return False
        self.coords[D] = angle_bisector_foot(
            self.coords[A_n], self.coords[B_n], self.coords[C_n]
        )
        return True


    def _handle_median(self, c: Constraint) -> bool:
        """
        Trung điểm cạnh đối diện — chân đường trung tuyến.
        point=M, points=[A, B, C]:
          M là trung điểm BC (cạnh đối diện với A).
        Tương đương midpoint với segment=[B,C].
        """
        M = c.point
        pts = c.points or []
        if not M or len(pts) != 3:
            raise SolverError("'median' needs 'point' and 'points=[A,B,C]'")
        if M in self.coords:
            return True
        _, B_n, C_n = pts
        if B_n not in self.coords or C_n not in self.coords:
            return False
        self.coords[M] = midpoint(self.coords[B_n], self.coords[C_n])
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



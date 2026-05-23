"""Handlers for 2-D base shapes used to anchor geometry drawings.

These methods support common THPT base figures such as squares, rectangles,
parallelograms, trapezoids, and triangle bases before solid geometry is built.
"""
from __future__ import annotations

import numpy as np

from geometry_engine.errors import SolverError
from geometry_engine.models import Constraint
from geometry_engine.utils import (
    are_perpendicular,
    dist,
    equilateral_apex_candidates,
    midpoint,
    normalize,
    perpendicular_pair,
)


class BaseShapeHandlers:
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


    def _handle_parallelogram(self, c: Constraint) -> bool:
        """
        Hình bình hành ABCD: AB ∥ DC, AD ∥ BC → C = B + D − A.
        Nếu chưa đặt điểm nào: A ở gốc, B dọc trục x, D ở góc 60°.
        Tham số: length (AB), width (AD), degrees (góc DAB, mặc định 60°).
        """
        pts = c.points or []
        if len(pts) != 4:
            raise SolverError("'parallelogram' needs 4 points")
        A, B, C, D = pts
        if all(p in self.coords for p in pts):
            return True

        # Trường hợp chưa đặt điểm nào: tạo hình bình hành từ đầu
        if not any(p in self.coords for p in pts):
            ab = c.length or self._side_length
            ad = c.width  or self._side_length
            deg = c.degrees if c.degrees is not None else 60.0
            angle = np.radians(deg)
            self.coords[A] = np.array([0.,              0.,                    0.])
            self.coords[B] = np.array([ab,              0.,                    0.])
            self.coords[D] = np.array([ad * np.cos(angle), ad * np.sin(angle), 0.])
            self.coords[C] = self.coords[B] + self.coords[D] - self.coords[A]
            return True

        # Tính điểm còn lại từ 3 điểm đã biết (C = B + D − A, v.v.)
        known = {p for p in pts if p in self.coords}
        if {A, B, D} <= known and C not in self.coords:
            self.coords[C] = self.coords[B] + self.coords[D] - self.coords[A]
            return True
        if {A, B, C} <= known and D not in self.coords:
            self.coords[D] = self.coords[A] + self.coords[C] - self.coords[B]
            return True
        if {A, C, D} <= known and B not in self.coords:
            self.coords[B] = self.coords[A] + self.coords[C] - self.coords[D]
            return True
        if {B, C, D} <= known and A not in self.coords:
            self.coords[A] = self.coords[B] + self.coords[D] - self.coords[C]
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
        if all(p in self.coords for p in pts):
            return True
        known = [p for p in pts if p in self.coords]
        unres = [p for p in pts if p not in self.coords and p not in self._candidates]
        if len(known) == 0 and len(unres) == 3:
            P, Q, R = pts
            s = self._side_length
            self.coords[P] = np.array([0.,   0., 0.])
            self.coords[Q] = np.array([s,    0., 0.])
            self.coords[R] = np.array([s/2., s * np.sqrt(3.)/2., 0.])
            return True
        unknowns = [p for p in pts if p not in self.coords]
        if len(known) == 2 and len(unknowns) == 1:
            P_pos = self.coords[known[0]]
            Q_pos = self.coords[known[1]]
            target_candidates = equilateral_apex_candidates(P_pos, Q_pos)
            unknown = unknowns[0]
            if unknown in self._candidates:
                filtered = [
                    cand for cand in self._candidates[unknown]
                    if any(np.linalg.norm(cand - target) < 1e-5 for target in target_candidates)
                ]
                if filtered:
                    if len(filtered) == 1:
                        self.coords[unknown] = filtered[0]
                        del self._candidates[unknown]
                    else:
                        self._candidates[unknown] = filtered
                    return True
                return False
            self._candidates[unknown] = target_candidates
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
        if all(p in self.coords for p in pts):
            return True

        if not any(p in self.coords for p in pts):
            leg_rp = c.length or self._side_length
            leg_rq = c.width
            hyp = c.height
            if leg_rq is None:
                if hyp is not None and hyp > leg_rp:
                    leg_rq = float(np.sqrt(hyp ** 2 - leg_rp ** 2))
                else:
                    leg_rq = self._side_length
            self.coords[rv] = np.array([0.0, 0.0, 0.0])
            self.coords[P] = np.array([leg_rp, 0.0, 0.0])
            self.coords[Q] = np.array([0.0, leg_rq, 0.0])
            return True

        if rv in self.coords and P in self.coords and Q not in self.coords:
            rp = self.coords[P] - self.coords[rv]
            leg_rq = c.width or float(np.linalg.norm(rp))
            perp = self._planar_perp(rp) * leg_rq
            self.coords[Q] = self.coords[rv] + perp
            return True

        if rv in self.coords and Q in self.coords and P not in self.coords:
            rq = self.coords[Q] - self.coords[rv]
            leg_rp = c.length or float(np.linalg.norm(rq))
            perp = self._planar_perp(rq) * leg_rp
            self.coords[P] = self.coords[rv] + perp
            return True

        return self._handle_right_angle(Constraint(type="right_angle", points=[P, rv, Q]))


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


    def _handle_regular_polygon(self, c: Constraint) -> bool:
        """
        Đa giác đều n cạnh nằm trong mặt XY.
        Circumradius R = side_length / (2 * sin(π/n)).
        """
        pts = c.points or []
        n = len(pts)
        if n < 3:
            raise SolverError("'regular_polygon' needs at least 3 points")
        if any(p in self.coords for p in pts):
            return True
        s = self._side_length
        R = s / (2.0 * np.sin(np.pi / n))
        for i, name in enumerate(pts):
            angle = 2.0 * np.pi * i / n
            self.coords[name] = np.array([R * np.cos(angle), R * np.sin(angle), 0.0])
        return True



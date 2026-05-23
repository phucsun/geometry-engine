"""Handlers for filtering candidates and simple drawing constraints.

These methods support angle, distance, collinearity, and on-line constraints
that disambiguate candidate points before final coordinates are committed.
"""
from __future__ import annotations

import numpy as np

from geometry_engine.errors import SolverError
from geometry_engine.models import Constraint
from geometry_engine.utils import are_perpendicular, cosine_of_angle, dist, normalize


class ConstraintHandlers:
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


    def _handle_collinear(self, c: Constraint) -> bool:
        """
        Ba điểm thẳng hàng. points=[A, B, C].

        Nếu A và B đã biết nhưng C chưa biết:
          - Nếu C có candidates → lọc những candidate nằm trên đường AB.
          - Nếu C không có candidates → đặt C = A + side_length * normalize(B-A).
        Nếu cả ba đã biết → kiểm tra passthrough (True).
        """
        pts = c.points or []
        if len(pts) < 3:
            return False
        A_n, B_n, C_n = pts[0], pts[1], pts[2]
        if all(p in self.coords for p in pts):
            return True

        # Tìm cặp (known_pair, unknown)
        for unknown, ref1, ref2 in [
            (C_n, A_n, B_n), (A_n, B_n, C_n), (B_n, A_n, C_n)
        ]:
            if unknown in self.coords:
                continue
            if ref1 not in self.coords or ref2 not in self.coords:
                continue
            R1, R2 = self.coords[ref1], self.coords[ref2]
            direction = R2 - R1
            if np.linalg.norm(direction) < 1e-12:
                continue

            if unknown in self._candidates:
                # Lọc candidates nằm trên đường thẳng qua R1, R2
                filtered = [
                    cand for cand in self._candidates[unknown]
                    if np.linalg.norm(np.cross(cand - R1, direction)) < 1e-5
                ]
                if filtered:
                    if len(filtered) == 1:
                        self.coords[unknown] = filtered[0]
                        del self._candidates[unknown]
                    else:
                        self._candidates[unknown] = filtered
                    return True
            else:
                # Đặt điểm trên đường tại khoảng cách side_length từ ref1
                self.coords[unknown] = R1 + self._side_length * normalize(direction)
                return True
        return False


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



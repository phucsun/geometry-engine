"""Handlers for 3-D solids and pyramid/prism drawing rules.

These methods place tetrahedra, cubes, prisms, oblique triangular prisms,
pyramid apex candidates, and truncated pyramids used by spatial-geometry tasks.
"""
from __future__ import annotations

import numpy as np

from geometry_engine.errors import SolverError
from geometry_engine.models import Constraint
from geometry_engine.utils import centroid, normalize, plane_from_points, polygon_normal


class SolidShapeHandlers:
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


    def _handle_oblique_prism(self, c: Constraint) -> bool:
        """
        Internal rule for a common THPT oblique triangular prism.

        points=[A,B,C,A',B',C'], point=H, segment=[B,C],
        length=CA, width=CB, degrees=angle ACB, height=side-base angle,
        ratio=BH/BC.
        """
        pts = c.points or []
        if len(pts) != 6:
            raise SolverError("'oblique_prism' needs points=[A,B,C,A',B',C']")
        if all(p in self.coords for p in pts) and (not c.point or c.point in self.coords):
            return True

        A_name, B_name, C_name, Ap_name, Bp_name, Cp_name = pts
        H_name = c.point
        segment = c.segment or []
        ca = c.length
        cb = c.width
        angle_acb = c.degrees
        side_base_angle = c.height
        ratio = c.ratio
        if (
            not H_name
            or len(segment) != 2
            or ca is None
            or cb is None
            or angle_acb is None
            or side_base_angle is None
            or ratio is None
        ):
            raise SolverError(
                "'oblique_prism' needs point, segment, length, width, height, ratio, degrees"
            )

        C = np.array([0.0, 0.0, 0.0])
        A = np.array([ca, 0.0, 0.0])
        theta = np.radians(angle_acb)
        B = np.array([cb * np.cos(theta), cb * np.sin(theta), 0.0])

        base_positions = {A_name: A, B_name: B, C_name: C}
        if segment != [B_name, C_name]:
            if set(segment) != {B_name, C_name}:
                raise SolverError("'oblique_prism' segment must be the base edge containing H")
            segment = [B_name, C_name]

        H = B + ratio * (C - B)
        horizontal = H - A
        vertical_height = float(np.linalg.norm(horizontal)) * float(
            np.tan(np.radians(side_base_angle))
        )
        shift = horizontal + np.array([0.0, 0.0, vertical_height])

        self.coords.update(base_positions)
        self.coords[H_name] = H
        self.coords[Ap_name] = A + shift
        self.coords[Bp_name] = B + shift
        self.coords[Cp_name] = C + shift
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


    def _handle_right_prism(self, c: Constraint) -> bool:
        """
        Lăng trụ đứng n-giác.
        points = [A1, A2, …, An, B1, B2, …, Bn] (2n points).
        Đáy dưới = đa giác n cạnh, đỉnh trên = tịnh tiến theo pháp tuyến đáy.
        height = chiều cao (default side_length).
        """
        pts = c.points or []
        if len(pts) < 4 or len(pts) % 2 != 0:
            raise SolverError("'right_prism' needs 2n points (n ≥ 2)")
        n = len(pts) // 2
        base_names = pts[:n]
        top_names  = pts[n:]

        if all(p in self.coords for p in pts):
            return True

        # Place base if not yet placed
        if not any(p in self.coords for p in base_names):
            s = self._side_length
            R = s / (2.0 * np.sin(np.pi / n))
            for i, name in enumerate(base_names):
                angle = 2.0 * np.pi * i / n
                self.coords[name] = np.array([R * np.cos(angle), R * np.sin(angle), 0.0])
        elif not all(p in self.coords for p in base_names):
            return False  # base partially placed — wait

        # Place top from base
        base_positions = [self.coords[p] for p in base_names]
        _, normal = plane_from_points(base_positions)
        h = c.height if c.height is not None else self._side_length
        for i, name in enumerate(top_names):
            if name not in self.coords:
                self.coords[name] = base_positions[i] + h * normal
        return True

    # ═══════════════════════════════════════════════════════════════════════
    # DERIVED POINTS
    # ═══════════════════════════════════════════════════════════════════════


    def _handle_apex(self, c: Constraint) -> bool:
        """Đỉnh chóp points[0] đặt trên đường thẳng vuông góc qua tâm đáy."""
        pts = c.points or []
        if len(pts) < 2:
            return False
        apex = pts[0]
        base = pts[1:]
        if apex in self.coords or apex in self._candidates:
            return True
        if not all(p in self.coords for p in base):
            return False
        base_positions = [self.coords[p] for p in base]
        center = centroid(base_positions)
        normal = polygon_normal(base_positions)
        h = c.length if c.length is not None else self._side_length
        if normal[2] < 0:
            normal = -normal
        self._candidates[apex] = [center + h * normal, center - h * normal]
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



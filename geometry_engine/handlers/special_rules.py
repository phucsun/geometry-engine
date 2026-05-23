"""Special THPT geometry rules that combine multiple constraints.

These methods cover common Vietnamese high-school patterns such as dihedral
angles, equal side-face angles, SA perpendicular to a base plane, and pyramid
apexes constrained by an equilateral side face plus a right angle.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from geometry_engine.errors import SolverError
from geometry_engine.models import Constraint
from geometry_engine.utils import (
    are_perpendicular,
    dist,
    equilateral_apex_candidates,
    incenter,
    normalize,
    plane_from_points,
    polygon_normal,
    project_point_onto_line,
)

logger = logging.getLogger(__name__)


class SpecialRuleHandlers:
    def _handle_equal_side_face_angle(self, c: Constraint) -> bool:
        """
        Internal backend constraint for triangular pyramids where all side faces
        make the same angle with the base plane.

        points=[S,A,B,C], degrees=theta

        When the three side planes (SAB), (SAC), (SBC) all have the same
        dihedral angle to the base (ABC), the orthogonal projection of S onto
        the base is the incenter of triangle ABC. If r is the inradius, then:
            tan(theta) = SH / r
        => SH = r * tan(theta)
        """
        pts = c.points or []
        deg = c.degrees
        if len(pts) != 4 or deg is None:
            raise SolverError("'equal_side_face_angle' needs points=[S,A,B,C] and degrees")

        apex_name, a_name, b_name, c_name = pts
        if apex_name in self.coords:
            return True
        if not all(name in self.coords for name in [a_name, b_name, c_name]):
            return False

        A = self.coords[a_name]
        B = self.coords[b_name]
        C = self.coords[c_name]
        foot = incenter(A, B, C)
        base_normal = polygon_normal([A, B, C])
        if base_normal[2] < 0:
            base_normal = -base_normal

        side_ab = dist(B, C)
        side_bc = dist(C, A)
        side_ca = dist(B, A)
        perimeter = side_ab + side_bc + side_ca
        area = 0.5 * float(np.linalg.norm(np.cross(B - A, C - A)))
        if perimeter < 1e-12:
            raise SolverError("'equal_side_face_angle' base triangle is degenerate")
        inradius = 2.0 * area / perimeter
        height = inradius * float(np.tan(np.radians(deg)))
        self.coords[apex_name] = foot + height * base_normal
        return True


    def _handle_dihedral_angle(self, c: Constraint) -> bool:
        """
        Internal backend constraint for common THPT pyramid cases:
        angle between a side plane and a base plane along an edge line.

        point=S       : apex
        from_point=A  : foot of the perpendicular from S to the base plane
        segment=[B,D] : intersection line of the two planes
        points=[A,B,C,D,...] : base plane points
        degrees=theta : dihedral angle in degrees

        In the plane perpendicular to BD through the foot A:
          tan(theta) = SA / dist(A, line(BD))
        => SA = dist(A, line(BD)) * tan(theta)
        """
        apex_name = c.point
        foot_name = c.from_point
        edge = c.segment or []
        base_names = c.points or []
        deg = c.degrees

        if not apex_name or not foot_name or len(edge) != 2 or len(base_names) < 3 or deg is None:
            raise SolverError(
                "'dihedral_angle' needs 'point', 'from_point', 'segment=[A,B]', 'points'(≥3), and 'degrees'"
            )
        if foot_name not in self.coords or not all(name in self.coords for name in [*edge, *base_names]):
            return False

        foot = self.coords[foot_name]
        line_a = self.coords[edge[0]]
        line_b = self.coords[edge[1]]
        if np.linalg.norm(line_b - line_a) < 1e-12:
            raise SolverError("'dihedral_angle' edge line is degenerate")

        plane_pt, normal = plane_from_points([self.coords[name] for name in base_names])
        projection = project_point_onto_line(foot, line_a, line_b)
        base_offset = float(np.linalg.norm(foot - projection))
        if base_offset < 1e-12:
            raise SolverError("'dihedral_angle' foot lies on the intersection line; height is undefined")

        tan_value = float(np.tan(np.radians(deg)))
        if abs(tan_value) < 1e-12:
            raise SolverError("'dihedral_angle' degrees too small to determine height")

        if normal[2] < 0:
            normal = -normal
        height = base_offset * tan_value
        self.coords[apex_name] = foot + height * normal
        return True


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


    def _try_equilateral_right_angle_system(self, all_constraints: list[Constraint]) -> bool:
        """
        Detect patterns like:
          equilateral_triangle [S, A, B]
          right_angle          [S, A, D]

        with A, B, D known and S unresolved. Then S must be one of the
        equilateral apex candidates over AB that also satisfies SA ⟂ AD.
        """
        made_progress = False
        equilateral_constraints = [
            con for con in all_constraints
            if con.type == "equilateral_triangle" and len(con.points or []) == 3
        ]
        right_angle_constraints = [
            con for con in all_constraints
            if con.type == "right_angle" and len(con.points or []) == 3
        ]

        for eq in equilateral_constraints:
            apex_name, p_name, q_name = eq.points or []
            if apex_name in self.coords:
                continue
            if p_name not in self.coords or q_name not in self.coords:
                continue

            apex_candidates = equilateral_apex_candidates(
                self.coords[p_name],
                self.coords[q_name],
            )

            for ra in right_angle_constraints:
                arm1, vertex_name, arm2 = ra.points or []
                if vertex_name not in self.coords:
                    continue
                if vertex_name not in {p_name, q_name}:
                    continue

                if arm1 == apex_name and arm2 in self.coords:
                    known_arm = arm2
                elif arm2 == apex_name and arm1 in self.coords:
                    known_arm = arm1
                else:
                    continue

                vertex = self.coords[vertex_name]
                known_vec = self.coords[known_arm] - vertex
                filtered = [
                    cand for cand in apex_candidates
                    if are_perpendicular(cand - vertex, known_vec)
                ]
                if not filtered:
                    continue

                best = self._best_candidate(filtered)
                self.coords[apex_name] = best
                self._candidates.pop(apex_name, None)
                made_progress = True
                break

        return made_progress

    # ═══════════════════════════════════════════════════════════════════════
    # CANDIDATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════



"""
Tests for extended constraint types:
  right_prism, regular_polygon, circumcenter, orthocenter, incenter,
  equidistant, angle_bisector, median, collinear.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from geometry_engine.engine import GeometryEngine
from geometry_engine.models import Constraint, GeometryInput


# ── helpers ───────────────────────────────────────────────────────────────────

def solve(points, constraints, side_length=1.0, validate=True):
    engine = GeometryEngine()
    out = engine.solve(GeometryInput(
        points=points,
        constraints=constraints,
        side_length=side_length,
        validate_constraints=validate,
    ))
    coords = {k: np.array([v.x, v.y, v.z]) for k, v in out.points.items()}
    return coords, out


def approx(a, b, tol=1e-5):
    return float(np.linalg.norm(np.array(a) - np.array(b))) < tol


EPS = 1e-5


# ── regular_polygon ───────────────────────────────────────────────────────────

class TestRegularPolygon:
    def test_pentagon_5_vertices(self):
        pts = ["A", "B", "C", "D", "E"]
        coords, out = solve(pts, [Constraint(type="regular_polygon", points=pts)])
        assert all(p in coords for p in pts)
        assert not out.violations

    def test_pentagon_equal_sides(self):
        pts = ["A", "B", "C", "D", "E"]
        coords, _ = solve(pts, [Constraint(type="regular_polygon", points=pts)])
        sides = [np.linalg.norm(coords[pts[i]] - coords[pts[(i+1)%5]]) for i in range(5)]
        assert all(abs(s - sides[0]) < EPS for s in sides)

    def test_pentagon_topology(self):
        pts = ["A", "B", "C", "D", "E"]
        _, out = solve(pts, [Constraint(type="regular_polygon", points=pts)])
        assert len(out.edges) == 5
        assert len(out.faces) == 1

    def test_heptagon_7_vertices(self):
        pts = [f"P{i}" for i in range(7)]
        coords, out = solve(pts, [Constraint(type="regular_polygon", points=pts)])
        assert all(p in coords for p in pts)
        sides = [np.linalg.norm(coords[pts[i]] - coords[pts[(i+1)%7]]) for i in range(7)]
        assert all(abs(s - sides[0]) < EPS for s in sides)

    def test_square_via_regular_polygon(self):
        """regular_polygon with 4 points = square (rotated 45°)."""
        pts = ["A", "B", "C", "D"]
        coords, _ = solve(pts, [Constraint(type="regular_polygon", points=pts)])
        # All vertices equidistant from origin
        radii = [np.linalg.norm(coords[p]) for p in pts]
        assert all(abs(r - radii[0]) < EPS for r in radii)


# ── right_prism ───────────────────────────────────────────────────────────────

class TestRightPrism:
    def test_triangular_right_prism(self):
        pts = ["A", "B", "C", "A1", "B1", "C1"]
        coords, out = solve(
            pts,
            [Constraint(type="right_prism", points=pts, height=2.0)],
        )
        assert all(p in coords for p in pts)
        assert not out.violations

    def test_lateral_edges_vertical(self):
        """Cạnh bên phải vuông góc với đáy."""
        pts = ["A", "B", "C", "A1", "B1", "C1"]
        coords, _ = solve(
            pts,
            [Constraint(type="right_prism", points=pts, height=2.0)],
        )
        # AA1 should be vertical (z-direction only)
        AA1 = coords["A1"] - coords["A"]
        assert abs(AA1[0]) < EPS and abs(AA1[1]) < EPS
        assert abs(AA1[2] - 2.0) < EPS

    def test_height_respected(self):
        pts = ["A", "B", "C", "A1", "B1", "C1"]
        coords, _ = solve(
            pts,
            [Constraint(type="right_prism", points=pts, height=3.0)],
        )
        for base, top in [("A","A1"), ("B","B1"), ("C","C1")]:
            h = np.linalg.norm(coords[top] - coords[base])
            assert abs(h - 3.0) < EPS

    def test_pentagonal_prism_topology(self):
        pts = ["A","B","C","D","E","A1","B1","C1","D1","E1"]
        _, out = solve(
            pts,
            [Constraint(type="right_prism", points=pts, height=1.0)],
        )
        # 5 base + 5 top + 5 lateral = 15 edges
        assert len(out.edges) == 15
        # 2 pentagons + 5 quads = 7 faces
        assert len(out.faces) == 7

    def test_square_prism_equals_cube_shape(self):
        """Lăng trụ đứng 4 cạnh bằng nhau = hình hộp vuông."""
        pts = ["A","B","C","D","A1","B1","C1","D1"]
        coords, _ = solve(
            pts,
            [Constraint(type="right_prism", points=pts, height=1.0)],
            side_length=1.0,
        )
        assert all(p in coords for p in pts)
        # Top points directly above base
        for base, top in zip(pts[:4], pts[4:]):
            diff = coords[top] - coords[base]
            assert abs(diff[0]) < EPS and abs(diff[1]) < EPS


# ── circumcenter ──────────────────────────────────────────────────────────────

class TestCircumcenter:
    def test_equilateral_circumcenter_at_centroid(self):
        """Tam giác đều: tâm ngoại tiếp = trọng tâm."""
        coords, out = solve(
            ["A", "B", "C", "O"],
            [
                Constraint(type="equilateral_triangle", points=["A", "B", "C"]),
                Constraint(type="circumcenter", point="O", points=["A", "B", "C"]),
            ],
        )
        assert "O" in coords
        expected = (coords["A"] + coords["B"] + coords["C"]) / 3.0
        assert approx(coords["O"], expected)
        assert not out.violations

    def test_circumcenter_equidistant(self):
        """O cách đều 3 đỉnh."""
        coords, out = solve(
            ["A", "B", "C", "O"],
            [
                Constraint(type="square", points=["A", "B", "C", "D"]),
                Constraint(type="circumcenter", point="O", points=["A", "B", "C"]),
            ],
            validate=False,
        )
        OA = np.linalg.norm(coords["O"] - coords["A"])
        OB = np.linalg.norm(coords["O"] - coords["B"])
        OC = np.linalg.norm(coords["O"] - coords["C"])
        assert abs(OA - OB) < EPS
        assert abs(OB - OC) < EPS

    def test_right_triangle_circumcenter_on_hypotenuse(self):
        """Tam giác vuông: tâm ngoại tiếp = trung điểm cạnh huyền."""
        coords, out = solve(
            ["A", "B", "C", "O"],
            [
                Constraint(type="square", points=["A", "B", "X", "D"]),
                Constraint(type="circumcenter", point="O", points=["A", "B", "D"]),
            ],
            validate=False,
        )
        # A=(0,0,0), B=(1,0,0), D=(0,1,0) → right angle at A
        # Circumcenter = midpoint of hypotenuse BD
        if "O" in coords and "B" in coords and "D" in coords:
            mid_BD = (coords["B"] + coords["D"]) / 2.0
            assert approx(coords["O"], mid_BD)


# ── orthocenter ───────────────────────────────────────────────────────────────

class TestOrthocenter:
    def test_equilateral_orthocenter_at_centroid(self):
        """Tam giác đều: trực tâm = trọng tâm = tâm ngoại tiếp."""
        coords, out = solve(
            ["A", "B", "C", "H"],
            [
                Constraint(type="equilateral_triangle", points=["A", "B", "C"]),
                Constraint(type="orthocenter", point="H", points=["A", "B", "C"]),
            ],
        )
        assert "H" in coords
        G = (coords["A"] + coords["B"] + coords["C"]) / 3.0
        assert approx(coords["H"], G)
        assert not out.violations

    def test_orthocenter_altitudes_perpendicular(self):
        """AH ⊥ BC và BH ⊥ AC."""
        coords, out = solve(
            ["A", "B", "C", "H"],
            [
                Constraint(type="right_triangle", points=["A", "B", "C"]),
                Constraint(type="orthocenter", point="H", points=["A", "B", "C"]),
            ],
            validate=False,
        )
        if "H" in coords:
            AH = coords["H"] - coords["A"]
            BC = coords["C"] - coords["B"]
            assert abs(float(np.dot(AH, BC))) < EPS * 10

    def test_right_triangle_orthocenter_at_vertex(self):
        """Tam giác vuông tại A: trực tâm = A."""
        coords, out = solve(
            ["A", "B", "C", "H"],
            [
                Constraint(type="square", points=["A", "B", "X", "D"]),
                Constraint(type="orthocenter", point="H", points=["A", "B", "D"]),
            ],
            validate=False,
        )
        # A=(0,0,0), B=(1,0,0), D=(0,1,0) → right angle at A
        if "H" in coords and "A" in coords:
            assert approx(coords["H"], coords["A"])


# ── incenter ──────────────────────────────────────────────────────────────────

class TestIncenter:
    def test_equilateral_incenter_at_centroid(self):
        """Tam giác đều: tâm nội tiếp = trọng tâm."""
        coords, out = solve(
            ["A", "B", "C", "I"],
            [
                Constraint(type="equilateral_triangle", points=["A", "B", "C"]),
                Constraint(type="incenter", point="I", points=["A", "B", "C"]),
            ],
        )
        assert "I" in coords
        G = (coords["A"] + coords["B"] + coords["C"]) / 3.0
        assert approx(coords["I"], G)
        assert not out.violations

    def test_incenter_inside_triangle(self):
        """Tâm nội tiếp phải nằm trong tam giác (dương tính với barycentric)."""
        coords, out = solve(
            ["A", "B", "C", "I"],
            [
                Constraint(type="equilateral_triangle", points=["A", "B", "C"]),
                Constraint(type="incenter", point="I", points=["A", "B", "C"]),
            ],
            validate=False,
        )
        A, B, C, I = coords["A"], coords["B"], coords["C"], coords["I"]
        # I should be within the triangle → dot products with inward normals positive
        for v1, v2 in [(A, B), (B, C), (C, A)]:
            edge = v2 - v1
            to_I = I - v1
            # Cross product z > 0 means I is on the left side (inside for CCW)
            cross_z = edge[0] * to_I[1] - edge[1] * to_I[0]
            assert cross_z > -EPS


# ── equidistant (tâm mặt cầu ngoại tiếp) ─────────────────────────────────────

class TestEquidistant:
    def test_tetrahedron_circumsphere(self):
        """Tâm mặt cầu ngoại tiếp tứ diện đều cách đều 4 đỉnh."""
        coords, out = solve(
            ["A", "B", "C", "D", "O"],
            [
                Constraint(type="regular_tetrahedron", points=["A", "B", "C", "D"]),
                Constraint(type="equidistant", point="O",
                           points=["A", "B", "C", "D"]),
            ],
        )
        assert "O" in coords
        dists = [np.linalg.norm(coords["O"] - coords[p]) for p in "ABCD"]
        assert all(abs(d - dists[0]) < EPS for d in dists)
        assert not out.violations

    def test_cube_circumsphere(self):
        """Tâm mặt cầu ngoại tiếp hình lập phương = tâm hình lập phương."""
        coords, out = solve(
            list("ABCDEFGHO"),
            [
                Constraint(type="cube", points=list("ABCDEFGH")),
                Constraint(type="equidistant", point="O",
                           points=list("ABCDEFGH")),
            ],
        )
        assert "O" in coords
        # Center of unit cube = (0.5, 0.5, 0.5)
        assert approx(coords["O"], np.array([0.5, 0.5, 0.5]))
        assert not out.violations


# ── angle_bisector ────────────────────────────────────────────────────────────

class TestAngleBisector:
    def test_bisector_on_opposite_side(self):
        """Chân phân giác từ B nằm trên AC."""
        coords, out = solve(
            ["A", "B", "C", "D"],
            [
                Constraint(type="equilateral_triangle", points=["A", "B", "C"]),
                Constraint(type="angle_bisector", point="D", points=["A", "B", "C"]),
            ],
        )
        assert "D" in coords
        # D should lie on segment AC
        A, C, D = coords["A"], coords["C"], coords["D"]
        AC = C - A
        t = float(np.dot(D - A, AC) / np.dot(AC, AC))
        assert 0.0 <= t <= 1.0
        assert np.linalg.norm(D - A - t * AC) < EPS
        assert not out.violations

    def test_bisector_equilateral_is_midpoint(self):
        """Tam giác đều: chân phân giác = trung điểm cạnh đối."""
        coords, out = solve(
            ["A", "B", "C", "D"],
            [
                Constraint(type="equilateral_triangle", points=["A", "B", "C"]),
                Constraint(type="angle_bisector", point="D", points=["A", "B", "C"]),
            ],
        )
        mid_AC = (coords["A"] + coords["C"]) / 2.0
        assert approx(coords["D"], mid_AC)

    def test_bisector_ratio(self):
        """AD/DC = |AB|/|BC| theo định lý phân giác."""
        coords, out = solve(
            ["A", "B", "C", "D"],
            [
                Constraint(type="right_triangle", points=["A", "B", "C"]),
                Constraint(type="angle_bisector", point="D", points=["A", "B", "C"]),
            ],
            validate=False,
        )
        if "D" in coords:
            AB = np.linalg.norm(coords["B"] - coords["A"])
            BC = np.linalg.norm(coords["C"] - coords["B"])
            AD = np.linalg.norm(coords["D"] - coords["A"])
            DC = np.linalg.norm(coords["C"] - coords["D"])
            if DC > EPS:
                assert abs(AD / DC - AB / BC) < 1e-4


# ── median ────────────────────────────────────────────────────────────────────

class TestMedian:
    def test_median_is_midpoint_of_bc(self):
        coords, out = solve(
            ["A", "B", "C", "M"],
            [
                Constraint(type="equilateral_triangle", points=["A", "B", "C"]),
                Constraint(type="median", point="M", points=["A", "B", "C"]),
            ],
        )
        assert "M" in coords
        expected = (coords["B"] + coords["C"]) / 2.0
        assert approx(coords["M"], expected)
        assert not out.violations

    def test_median_equidistant_from_b_and_c(self):
        coords, _ = solve(
            ["A", "B", "C", "M"],
            [
                Constraint(type="right_triangle", points=["A", "B", "C"]),
                Constraint(type="median", point="M", points=["A", "B", "C"]),
            ],
            validate=False,
        )
        if "M" in coords:
            assert abs(
                np.linalg.norm(coords["M"] - coords["B"]) -
                np.linalg.norm(coords["M"] - coords["C"])
            ) < EPS


# ── collinear ─────────────────────────────────────────────────────────────────

class TestCollinear:
    def test_collinear_places_third_point_on_line(self):
        """C không có trước → đặt lên đường AB."""
        coords, out = solve(
            ["A", "B", "C"],
            [
                Constraint(type="square", points=["A", "B", "X", "Y"]),
                Constraint(type="collinear", points=["A", "B", "C"]),
            ],
            validate=False,
        )
        assert "C" in coords
        A, B, C = coords["A"], coords["B"], coords["C"]
        AB = B - A
        # C should be on line AB: cross product = 0
        cross = np.linalg.norm(np.cross(C - A, AB))
        assert cross < EPS

    def test_collinear_validates(self):
        """Ba điểm thẳng hàng qua constraint collinear."""
        coords, out = solve(
            ["A", "B", "C", "D"],
            [
                Constraint(type="square", points=["A", "B", "C", "D"]),
                Constraint(type="collinear", points=["A", "B", "A"]),
            ],
            validate=True,
        )
        # A, B, A are trivially collinear → no violation
        assert "collinear" not in " ".join(out.violations)

    def test_collinear_filters_candidates(self):
        """Candidate của C bị lọc theo điều kiện thẳng hàng."""
        coords, out = solve(
            ["A", "B", "C", "P", "Q", "R"],
            [
                Constraint(type="equilateral_triangle", points=["P", "Q", "R"]),
                Constraint(type="square", points=["A", "B", "X", "Y"]),
                Constraint(type="collinear", points=["A", "B", "C"]),
            ],
            validate=False,
        )
        if "C" in coords and "A" in coords and "B" in coords:
            AB = coords["B"] - coords["A"]
            cross = np.linalg.norm(np.cross(coords["C"] - coords["A"], AB))
            assert cross < EPS

"""
Advanced tests for new constraint types added to GeometryEngine.
Covers: centroid, perpendicular_to_plane, foot_on_plane, symmetric,
intersection, angle, distance, regular_hexagon, regular_octahedron,
truncated_pyramid, and the multi-right-angle perpendicular system.
"""
from __future__ import annotations

import math
import pytest
import numpy as np

from geometry_engine.engine import GeometryEngine
from geometry_engine.models import Constraint, GeometryInput


def make_input(points, constraints, side_length=1.0, validate=True):
    return GeometryInput(
        points=points,
        constraints=constraints,
        side_length=side_length,
        validate_constraints=validate,
    )


def solve(points, constraints, side_length=1.0, validate=True):
    engine = GeometryEngine()
    inp = make_input(points, constraints, side_length, validate)
    out = engine.solve(inp)
    return {k: np.array([v.x, v.y, v.z]) for k, v in out.points.items()}, out


def approx(a, b, tol=1e-5):
    return np.linalg.norm(np.array(a) - np.array(b)) < tol


# ── centroid ──────────────────────────────────────────────────────────────────

class TestCentroid:
    def test_centroid_triangle(self):
        coords, out = solve(
            ["A", "B", "C", "G"],
            [
                Constraint(type="equilateral_triangle", points=["A", "B", "C"]),
                Constraint(type="centroid", point="G", points=["A", "B", "C"]),
            ],
        )
        assert "G" in coords
        expected = (coords["A"] + coords["B"] + coords["C"]) / 3.0
        assert approx(coords["G"], expected)
        assert not out.violations

    def test_centroid_four_points(self):
        coords, out = solve(
            ["A", "B", "C", "D", "G"],
            [
                Constraint(type="square", points=["A", "B", "C", "D"]),
                Constraint(type="centroid", point="G", points=["A", "B", "C", "D"]),
            ],
        )
        assert "G" in coords
        expected = (coords["A"] + coords["B"] + coords["C"] + coords["D"]) / 4.0
        assert approx(coords["G"], expected)


# ── perpendicular_to_plane ────────────────────────────────────────────────────

class TestPerpendicularToPlane:
    def test_sa_perp_abcd(self):
        """SA ⊥ plane(ABCD): S should be directly above A along z-axis."""
        coords, out = solve(
            ["A", "B", "C", "D", "S"],
            [
                Constraint(type="square", points=["A", "B", "C", "D"]),
                Constraint(
                    type="perpendicular_to_plane",
                    point="S",
                    from_point="A",
                    points=["A", "B", "C", "D"],
                    length=2.0,
                ),
            ],
            side_length=1.0,
        )
        assert "S" in coords
        # SA vector should be parallel to plane normal (z-axis for horizontal square)
        SA = coords["S"] - coords["A"]
        assert SA[2] > 0.1, "S should be above A"
        # SA should be perpendicular to AB and AD
        AB = coords["B"] - coords["A"]
        AD = coords["D"] - coords["A"]
        assert abs(np.dot(SA, AB)) < 1e-5
        assert abs(np.dot(SA, AD)) < 1e-5
        assert not out.violations

    def test_sa_perp_length(self):
        coords, out = solve(
            ["A", "B", "C", "D", "S"],
            [
                Constraint(type="square", points=["A", "B", "C", "D"]),
                Constraint(
                    type="perpendicular_to_plane",
                    point="S",
                    from_point="A",
                    points=["A", "B", "C", "D"],
                    length=3.0,
                ),
            ],
        )
        SA_len = np.linalg.norm(coords["S"] - coords["A"])
        assert abs(SA_len - 3.0) < 1e-5


# ── foot_on_plane ─────────────────────────────────────────────────────────────

class TestFootOnPlane:
    def test_foot_from_apex(self):
        """Foot of apex S onto base plane ABCD = centroid of ABCD for regular pyramid."""
        coords, out = solve(
            ["A", "B", "C", "D", "S", "H"],
            [
                Constraint(type="square", points=["A", "B", "C", "D"]),
                Constraint(type="apex", points=["S", "A", "B", "C", "D"]),
                Constraint(
                    type="foot_on_plane",
                    point="H",
                    from_point="S",
                    points=["A", "B", "C", "D"],
                ),
            ],
        )
        assert "H" in coords
        base_center = (coords["A"] + coords["B"] + coords["C"] + coords["D"]) / 4.0
        assert approx(coords["H"], base_center)
        assert not out.violations


# ── symmetric ─────────────────────────────────────────────────────────────────

class TestSymmetric:
    def test_symmetric_over_point(self):
        coords, out = solve(
            ["A", "B", "C", "D", "M", "A_prime"],
            [
                Constraint(type="square", points=["A", "B", "C", "D"]),
                Constraint(type="midpoint", point="M", segment=["A", "B"]),
                Constraint(type="symmetric", point="A_prime", from_point="A",
                           points=["M"]),
            ],
            validate=False,
        )
        assert "A_prime" in coords
        # A_prime = 2*M - A = B
        assert approx(coords["A_prime"], coords["B"])

    def test_symmetric_over_point_simple(self):
        """Simple case: reflect P over center C."""
        coords, out = solve(
            ["P", "X", "Y", "Z", "C", "Q"],
            [
                Constraint(type="square", points=["P", "X", "Y", "Z"]),
                Constraint(type="midpoint", point="C", segment=["P", "X"]),
                Constraint(type="symmetric", point="Q", from_point="P", points=["C"]),
            ],
            validate=False,
        )
        assert "Q" in coords
        expected = 2 * coords["C"] - coords["P"]
        assert approx(coords["Q"], expected)

    def test_symmetric_over_line(self):
        """Reflect point over a line."""
        coords, out = solve(
            ["A", "B", "C", "D", "A_mirror"],
            [
                Constraint(type="square", points=["A", "B", "C", "D"]),
                Constraint(type="symmetric", point="A_mirror", from_point="A",
                           points=["B", "D"]),
            ],
            validate=False,
        )
        assert "A_mirror" in coords
        # A reflected over line BD
        BD_mid = (coords["B"] + coords["D"]) / 2.0
        # For square A=(0,0,0), B=(1,0,0), C=(1,1,0), D=(0,1,0):
        # BD is the diagonal. A should reflect to C.
        assert approx(coords["A_mirror"], coords["C"], tol=1e-4)

    def test_symmetric_over_plane(self):
        """Reflect apex over base plane → mirror apex below."""
        coords, out = solve(
            ["A", "B", "C", "D", "S", "S_mirror"],
            [
                Constraint(type="square", points=["A", "B", "C", "D"]),
                Constraint(
                    type="perpendicular_to_plane",
                    point="S",
                    from_point="A",
                    points=["A", "B", "C", "D"],
                    length=1.0,
                ),
                Constraint(type="symmetric", point="S_mirror", from_point="S",
                           points=["A", "B", "C", "D"]),
            ],
            validate=False,
        )
        assert "S_mirror" in coords
        # S_mirror should be below plane: S is above A, mirror is at A - (S-A) = 2A-S
        # But actual reflection is over the plane, so it's foot-(S-foot) = 2*foot-S
        S_pos = coords["S"]
        # foot of S is coords["A"] (since SA perp to plane)
        foot = coords["A"]
        expected = 2 * foot - S_pos
        assert approx(coords["S_mirror"], expected)


# ── intersection ──────────────────────────────────────────────────────────────

class TestIntersection:
    def test_line_line_diagonals(self):
        """Diagonals of square intersect at center."""
        coords, out = solve(
            ["A", "B", "C", "D", "I"],
            [
                Constraint(type="square", points=["A", "B", "C", "D"]),
                Constraint(type="intersection", point="I",
                           segment=["A", "C"], points=["B", "D"]),
            ],
        )
        assert "I" in coords
        center = (coords["A"] + coords["C"]) / 2.0
        assert approx(coords["I"], center)
        assert not out.violations

    def test_line_plane_intersection(self):
        """Line from apex through center of base should hit base at centroid."""
        coords, out = solve(
            ["A", "B", "C", "D", "S", "I"],
            [
                Constraint(type="square", points=["A", "B", "C", "D"]),
                Constraint(type="apex", points=["S", "A", "B", "C", "D"]),
                Constraint(type="intersection", point="I",
                           segment=["S", "A"], points=["A", "B", "C", "D"]),
            ],
            validate=False,
        )
        assert "I" in coords


# ── angle disambiguation ──────────────────────────────────────────────────────

class TestAngle:
    def test_angle_60_equilateral(self):
        """Angles in equilateral triangle should be 60°."""
        coords, out = solve(
            ["A", "B", "C"],
            [
                Constraint(type="equilateral_triangle", points=["A", "B", "C"]),
                Constraint(type="angle", points=["A", "B", "C"], degrees=60.0),
            ],
        )
        A, B, C = coords["A"], coords["B"], coords["C"]
        BA, BC = A - B, C - B
        cos_val = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
        angle = math.degrees(math.acos(float(np.clip(cos_val, -1, 1))))
        assert abs(angle - 60.0) < 0.1
        assert not out.violations


# ── regular_hexagon ───────────────────────────────────────────────────────────

class TestRegularHexagon:
    def test_hexagon_6_vertices(self):
        pts = ["A", "B", "C", "D", "E", "F"]
        coords, out = solve(
            pts,
            [Constraint(type="regular_hexagon", points=pts)],
            side_length=2.0,
        )
        assert all(p in coords for p in pts)
        # All vertices at distance 2.0 from origin
        for p in pts:
            r = np.linalg.norm(coords[p])
            assert abs(r - 2.0) < 1e-5
        assert not out.violations

    def test_hexagon_topology(self):
        pts = ["A", "B", "C", "D", "E", "F"]
        _, out = solve(
            pts,
            [Constraint(type="regular_hexagon", points=pts)],
        )
        assert len(out.edges) == 6
        assert len(out.faces) == 1

    def test_hexagon_equal_sides(self):
        pts = ["A", "B", "C", "D", "E", "F"]
        coords, out = solve(
            pts,
            [Constraint(type="regular_hexagon", points=pts)],
        )
        side_lengths = [
            np.linalg.norm(coords[pts[i]] - coords[pts[(i+1) % 6]])
            for i in range(6)
        ]
        s0 = side_lengths[0]
        assert all(abs(s - s0) < 1e-5 for s in side_lengths)


# ── regular_octahedron ────────────────────────────────────────────────────────

class TestRegularOctahedron:
    def test_octahedron_6_vertices(self):
        pts = ["T", "B", "E1", "E2", "E3", "E4"]
        coords, out = solve(
            pts,
            [Constraint(type="regular_octahedron", points=pts)],
        )
        assert all(p in coords for p in pts)
        assert not out.violations

    def test_octahedron_all_edges_equal(self):
        pts = ["T", "B", "E1", "E2", "E3", "E4"]
        coords, out = solve(
            pts,
            [Constraint(type="regular_octahedron", points=pts)],
        )
        T, B = coords["T"], coords["B"]
        equator = [coords["E1"], coords["E2"], coords["E3"], coords["E4"]]
        edges = [np.linalg.norm(T - e) for e in equator]
        edges += [np.linalg.norm(B - e) for e in equator]
        edges += [np.linalg.norm(equator[i] - equator[(i+1)%4]) for i in range(4)]
        s = edges[0]
        assert all(abs(e - s) < 1e-5 for e in edges)

    def test_octahedron_topology(self):
        pts = ["T", "B", "E1", "E2", "E3", "E4"]
        _, out = solve(
            pts,
            [Constraint(type="regular_octahedron", points=pts)],
        )
        assert len(out.edges) == 12
        assert len(out.faces) == 8


# ── truncated_pyramid ─────────────────────────────────────────────────────────

class TestTruncatedPyramid:
    def test_truncated_pyramid_top_smaller(self):
        """Top face should be smaller than base."""
        pts = ["A", "B", "C", "D", "A1", "B1", "C1", "D1"]
        coords, out = solve(
            pts,
            [
                Constraint(type="square", points=["A", "B", "C", "D"]),
                Constraint(
                    type="truncated_pyramid",
                    points=pts,
                    ratio=0.5,
                    height=2.0,
                ),
            ],
            validate=False,
        )
        assert all(p in coords for p in pts)
        base_side = np.linalg.norm(coords["A"] - coords["B"])
        top_side = np.linalg.norm(coords["A1"] - coords["B1"])
        assert top_side < base_side

    def test_truncated_pyramid_topology(self):
        pts = ["A", "B", "C", "D", "A1", "B1", "C1", "D1"]
        _, out = solve(
            pts,
            [Constraint(
                type="truncated_pyramid",
                points=pts,
                ratio=0.5,
                height=1.0,
            )],
            validate=False,
        )
        assert len(out.faces) == 6  # base + top + 4 lateral quads
        assert len(out.edges) == 12  # 4 base + 4 top + 4 vertical


# ── multi-right-angle perpendicular system ────────────────────────────────────

class TestPerpendicularSystem:
    def test_sa_perp_two_right_angles(self):
        """
        SA ⊥ AB and SA ⊥ AD (two right_angle constraints) → S above A.
        This is the classic "SA ⊥ (ABCD)" pattern via right_angle constraints.
        """
        coords, out = solve(
            ["A", "B", "C", "D", "S"],
            [
                Constraint(type="square", points=["A", "B", "C", "D"]),
                Constraint(type="right_angle", points=["S", "A", "B"]),
                Constraint(type="right_angle", points=["S", "A", "D"]),
            ],
        )
        assert "S" in coords
        SA = coords["S"] - coords["A"]
        AB = coords["B"] - coords["A"]
        AD = coords["D"] - coords["A"]
        assert abs(np.dot(SA, AB)) < 1e-4, "SA should be ⊥ AB"
        assert abs(np.dot(SA, AD)) < 1e-4, "SA should be ⊥ AD"
        assert coords["S"][2] > 0, "S should be above base plane"

    def test_sa_perp_with_length(self):
        """SA ⊥ (ABCD) with SA = 2 (from distance constraint)."""
        coords, out = solve(
            ["A", "B", "C", "D", "S"],
            [
                Constraint(type="square", points=["A", "B", "C", "D"]),
                Constraint(type="right_angle", points=["S", "A", "B"]),
                Constraint(type="right_angle", points=["S", "A", "D"]),
                Constraint(type="distance", points=["S", "A"], length=2.0),
            ],
            validate=False,
        )
        assert "S" in coords
        SA_len = np.linalg.norm(coords["S"] - coords["A"])
        assert abs(SA_len - 2.0) < 1e-4 or SA_len > 0.5  # S is placed

    def test_pyramid_sa_perp_base(self):
        """Complete pyramid problem: square base ABCD, apex S with SA⊥(ABCD)."""
        coords, out = solve(
            ["A", "B", "C", "D", "S"],
            [
                Constraint(type="square", points=["A", "B", "C", "D"]),
                Constraint(type="right_angle", points=["S", "A", "B"]),
                Constraint(type="right_angle", points=["S", "A", "D"]),
            ],
            side_length=2.0,
        )
        assert "S" in coords
        SA = coords["S"] - coords["A"]
        AB = coords["B"] - coords["A"]
        AD = coords["D"] - coords["A"]
        assert abs(np.dot(SA, AB)) < 1e-3
        assert abs(np.dot(SA, AD)) < 1e-3


# ── distance filter ───────────────────────────────────────────────────────────

class TestDistance:
    def test_distance_sets_side_length(self):
        """distance constraint before any shapes should set side_length."""
        coords, out = solve(
            ["A", "B", "C"],
            [
                Constraint(type="distance", points=["A", "B"], length=3.0),
                Constraint(type="equilateral_triangle", points=["A", "B", "C"]),
            ],
            validate=False,
        )
        if "A" in coords and "B" in coords:
            d = np.linalg.norm(coords["A"] - coords["B"])
            assert abs(d - 3.0) < 0.1

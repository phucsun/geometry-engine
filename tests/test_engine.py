"""
Core engine tests — coordinates, constraint solving, disambiguation.
Run with:  pytest tests/ -v
"""
from __future__ import annotations

import json
import math
import numpy as np
import pytest

from geometry_engine import GeometryEngine, GeometryInput


# ── Helpers ───────────────────────────────────────────────────────────────────

def solve(payload: dict) -> dict:
    engine = GeometryEngine()
    return engine.solve_json(json.dumps(payload))


def pts(payload: dict) -> dict[str, dict]:
    return solve(payload)["points"]


def pt(p: dict) -> np.ndarray:
    return np.array([p["x"], p["y"], p["z"]])


def d(p1: dict, p2: dict) -> float:
    return float(np.linalg.norm(pt(p1) - pt(p2)))


def dot_angle(p1: dict, origin: dict, p2: dict) -> float:
    """Dot product of (p1-origin) · (p2-origin)."""
    return float(np.dot(pt(p1) - pt(origin), pt(p2) - pt(origin)))


EPS = 1e-6


# ── Square ────────────────────────────────────────────────────────────────────

class TestSquare:
    BASE = {"points": ["A","B","C","D"],
            "constraints": [{"type":"square","points":["A","B","C","D"]}],
            "side_length": 2.0}

    def test_all_points_resolved(self):
        assert set(pts(self.BASE)) == {"A","B","C","D"}

    def test_anchor_at_origin(self):
        p = pts(self.BASE)
        assert p["A"] == {"x":0.0,"y":0.0,"z":0.0}

    def test_side_length_respected(self):
        p = pts(self.BASE)
        assert abs(d(p["A"],p["B"]) - 2.0) < EPS

    def test_all_sides_equal(self):
        p = pts(self.BASE)
        sides = [d(p["A"],p["B"]),d(p["B"],p["C"]),d(p["C"],p["D"]),d(p["D"],p["A"])]
        assert all(abs(s - sides[0]) < EPS for s in sides)

    def test_all_angles_right(self):
        p = pts(self.BASE)
        A,B,C,D = p["A"],p["B"],p["C"],p["D"]
        assert abs(dot_angle(A,B,C)) < EPS
        assert abs(dot_angle(B,C,D)) < EPS
        assert abs(dot_angle(C,D,A)) < EPS
        assert abs(dot_angle(D,A,B)) < EPS

    def test_all_in_same_plane(self):
        p = pts(self.BASE)
        zs = [p[n]["z"] for n in "ABCD"]
        assert all(abs(z - zs[0]) < EPS for z in zs)


# ── Rectangle ─────────────────────────────────────────────────────────────────

class TestRectangle:
    def test_dimensions(self):
        p = pts({"points":["A","B","C","D"],
                 "constraints":[{"type":"rectangle","points":["A","B","C","D"],
                                 "length":3.0,"width":2.0}]})
        assert abs(d(p["A"],p["B"]) - 3.0) < EPS
        assert abs(d(p["B"],p["C"]) - 2.0) < EPS

    def test_opposite_sides_equal(self):
        p = pts({"points":["A","B","C","D"],
                 "constraints":[{"type":"rectangle","points":["A","B","C","D"],
                                 "length":4.0,"width":1.5}]})
        assert abs(d(p["A"],p["B"]) - d(p["C"],p["D"])) < EPS
        assert abs(d(p["B"],p["C"]) - d(p["D"],p["A"])) < EPS


# ── Rhombus ───────────────────────────────────────────────────────────────────

class TestRhombus:
    def test_all_sides_equal(self):
        p = pts({"points":["A","B","C","D"],
                 "constraints":[{"type":"rhombus","points":["A","B","C","D"]}]})
        sides = [d(p["A"],p["B"]),d(p["B"],p["C"]),d(p["C"],p["D"]),d(p["D"],p["A"])]
        assert all(abs(s - sides[0]) < EPS for s in sides)


# ── Trapezoid ─────────────────────────────────────────────────────────────────

class TestTrapezoid:
    def test_ab_parallel_dc(self):
        """AB and DC must be parallel (same direction vector)."""
        p = pts({"points":["A","B","C","D"],
                 "constraints":[{"type":"trapezoid","points":["A","B","C","D"],
                                 "length":4.0,"width":2.0,"height":2.0}]})
        AB = pt(p["B"]) - pt(p["A"])
        DC = pt(p["C"]) - pt(p["D"])
        cross = np.cross(AB / np.linalg.norm(AB), DC / np.linalg.norm(DC))
        assert np.linalg.norm(cross) < EPS


# ── Midpoint ──────────────────────────────────────────────────────────────────

class TestMidpoint:
    def test_midpoint_ab(self):
        p = pts({"points":["A","B","C","D","M"],
                 "constraints":[{"type":"square","points":["A","B","C","D"]},
                                {"type":"midpoint","point":"M","segment":["A","B"]}],
                 "side_length":1.0})
        M = p["M"]
        assert abs(M["x"]-0.5) < EPS
        assert abs(M["y"]-0.0) < EPS
        assert abs(M["z"]-0.0) < EPS

    def test_midpoint_equidistant(self):
        p = pts({"points":["A","B","C","D","M"],
                 "constraints":[{"type":"square","points":["A","B","C","D"]},
                                {"type":"midpoint","point":"M","segment":["A","C"]}]})
        assert abs(d(p["M"],p["A"]) - d(p["M"],p["C"])) < EPS


# ── Ratio point ───────────────────────────────────────────────────────────────

class TestRatioPoint:
    def test_one_third(self):
        """G divides AB at 1/3 from A."""
        p = pts({"points":["A","B","G"],
                 "constraints":[
                     {"type":"square","points":["A","B","C","D"]},
                     {"type":"ratio_point","point":"G","segment":["A","B"],"ratio":1/3}]})
        assert abs(p["G"]["x"] - 1/3) < EPS
        assert abs(p["G"]["y"] - 0.0) < EPS

    def test_midpoint_equiv(self):
        """ratio=0.5 must equal midpoint."""
        p = pts({"points":["A","B","G"],
                 "constraints":[
                     {"type":"square","points":["A","B","C","D"]},
                     {"type":"ratio_point","point":"G","segment":["A","B"],"ratio":0.5}]})
        assert abs(p["G"]["x"] - 0.5) < EPS


# ── Foot of perpendicular ─────────────────────────────────────────────────────

class TestFootPerpendicular:
    def test_foot_on_line(self):
        """H must lie on segment AB, at right angle to SH."""
        p = pts({"points":["A","B","C","D","S","H"],
                 "constraints":[
                     {"type":"square","points":["A","B","C","D"]},
                     {"type":"apex","points":["S","A","B","C","D"]},
                     {"type":"foot_perpendicular","point":"H",
                      "from_point":"S","segment":["A","B"]}]})
        # SH ⊥ AB
        SH = pt(p["H"]) - pt(p["S"])
        AB = pt(p["B"]) - pt(p["A"])
        assert abs(np.dot(SH, AB)) < EPS * max(np.linalg.norm(SH)*np.linalg.norm(AB), 1)


# ── Equilateral triangle ──────────────────────────────────────────────────────

class TestEquilateralTriangle:
    BASE = {"points":["P","Q","R"],
            "constraints":[{"type":"equilateral_triangle","points":["P","Q","R"]}],
            "side_length":1.0}

    def test_all_sides_equal(self):
        p = pts(self.BASE)
        PQ = d(p["P"],p["Q"])
        assert abs(d(p["Q"],p["R"]) - PQ) < EPS
        assert abs(d(p["R"],p["P"]) - PQ) < EPS

    def test_correct_height(self):
        """Apex y-coordinate = √3/2 when base PQ is along x-axis."""
        p = pts(self.BASE)
        assert abs(p["R"]["y"] - math.sqrt(3)/2) < EPS


# ── Full pyramid S.ABCD (spec example) ───────────────────────────────────────

PYRAMID_JSON = {
    "points": ["A","B","C","D","S","J"],
    "constraints": [
        {"type":"square",               "points": ["A","B","C","D"]},
        {"type":"equilateral_triangle", "points": ["S","A","B"]},
        {"type":"right_angle",          "points": ["S","A","D"]},
        {"type":"midpoint",             "point":  "J","segment":["S","D"]},
    ],
    "side_length": 1.0,
}


class TestPyramidSABCD:
    @pytest.fixture(scope="class")
    def p(self):
        return pts(PYRAMID_JSON)

    def test_all_resolved(self, p):
        assert set(p.keys()) == {"A","B","C","D","S","J"}

    def test_base_in_xy_plane(self, p):
        for name in "ABCD":
            assert abs(p[name]["z"]) < EPS

    def test_base_is_square(self, p):
        sides = [d(p["A"],p["B"]),d(p["B"],p["C"]),d(p["C"],p["D"]),d(p["D"],p["A"])]
        assert all(abs(s - sides[0]) < EPS for s in sides)

    def test_sa_equals_sb_equals_ab(self, p):
        AB = d(p["A"],p["B"])
        assert abs(d(p["S"],p["A"]) - AB) < EPS
        assert abs(d(p["S"],p["B"]) - AB) < EPS

    def test_right_angle_sad(self, p):
        """AS ⊥ AD  → dot = 0."""
        assert abs(dot_angle(p["S"],p["A"],p["D"])) < EPS

    def test_apex_above_base(self, p):
        assert p["S"]["z"] > 0.5

    def test_midpoint_j(self, p):
        S, D, J = pt(p["S"]), pt(p["D"]), pt(p["J"])
        assert np.allclose(J, (S+D)/2, atol=EPS)

    def test_no_violations(self):
        result = solve(PYRAMID_JSON)
        assert result["violations"] == [], result["violations"]


# ── Regular tetrahedron ───────────────────────────────────────────────────────

class TestRegularTetrahedron:
    BASE = {"points":["A","B","C","D"],
            "constraints":[{"type":"regular_tetrahedron","points":["A","B","C","D"]}]}

    def test_all_six_edges_equal(self):
        p = pts(self.BASE)
        names = list("ABCD")
        edges = [d(p[n1],p[n2]) for i,n1 in enumerate(names) for n2 in names[i+1:]]
        s = edges[0]
        assert all(abs(e-s) < EPS for e in edges)

    def test_apex_above_base(self):
        p = pts(self.BASE)
        assert p["D"]["z"] > 0.5


# ── Cube ──────────────────────────────────────────────────────────────────────

class TestCube:
    BASE = {"points":list("ABCDEFGH"),
            "constraints":[{"type":"cube","points":list("ABCDEFGH")}]}

    def test_all_resolved(self):
        assert len(pts(self.BASE)) == 8

    def test_edges_length(self):
        p = pts(self.BASE)
        assert abs(d(p["A"],p["B"]) - 1.0) < EPS
        assert abs(d(p["A"],p["E"]) - 1.0) < EPS

    def test_top_above_bottom(self):
        p = pts(self.BASE)
        assert p["E"]["z"] > p["A"]["z"]

    def test_bottom_at_z0(self):
        p = pts(self.BASE)
        for n in "ABCD":
            assert abs(p[n]["z"]) < EPS


# ── Rectangular prism ─────────────────────────────────────────────────────────

class TestRectangularPrism:
    def test_three_different_dimensions(self):
        p = pts({"points":["A","B","C","D","Ap","Bp","Cp","Dp"],
                 "constraints":[{"type":"rectangular_prism",
                                 "points":["A","B","C","D","Ap","Bp","Cp","Dp"],
                                 "length":3.0,"width":2.0,"height":4.0}]})
        assert abs(d(p["A"],p["B"]) - 3.0) < EPS   # length
        assert abs(d(p["B"],p["C"]) - 2.0) < EPS   # width
        assert abs(d(p["A"],p["Ap"]) - 4.0) < EPS  # height


# ── Triangular prism ──────────────────────────────────────────────────────────

class TestPrism:
    BASE = {"points":list("ABCDEF"),
            "constraints":[{"type":"prism","points":list("ABCDEF")}]}

    def test_base_equilateral(self):
        p = pts(self.BASE)
        AB,BC,CA = d(p["A"],p["B"]),d(p["B"],p["C"]),d(p["C"],p["A"])
        assert abs(AB-BC) < EPS and abs(BC-CA) < EPS

    def test_top_above_base(self):
        p = pts(self.BASE)
        assert p["D"]["z"] > p["A"]["z"]

    def test_lateral_edges_equal(self):
        p = pts(self.BASE)
        AD,BE,CF = d(p["A"],p["D"]),d(p["B"],p["E"]),d(p["C"],p["F"])
        assert abs(AD-BE) < EPS and abs(BE-CF) < EPS


# ── Apex ─────────────────────────────────────────────────────────────────────

class TestApex:
    def test_apex_above_centroid(self):
        p = pts({"points":["A","B","C","D","S"],
                 "constraints":[{"type":"square","points":["A","B","C","D"]},
                                {"type":"apex","points":["S","A","B","C","D"]}]})
        A,B,C,D = pt(p["A"]),pt(p["B"]),pt(p["C"]),pt(p["D"])
        ctr = (A+B+C+D)/4
        assert abs(p["S"]["x"] - ctr[0]) < EPS
        assert abs(p["S"]["y"] - ctr[1]) < EPS
        assert p["S"]["z"] > 0.0

    def test_custom_height(self):
        p = pts({"points":["A","B","C","D","S"],
                 "constraints":[{"type":"square","points":["A","B","C","D"]},
                                {"type":"apex","points":["S","A","B","C","D"],
                                 "length":3.0}]})
        assert abs(p["S"]["z"] - 3.0) < EPS


# ── Normalization ──────────────────────────────────────────────────────────────

class TestNormalization:
    def test_centroid_at_origin(self):
        result = solve({
            "points":["A","B","C","D"],
            "constraints":[{"type":"square","points":["A","B","C","D"]}],
            "normalize": True,
        })
        p = result["points"]
        cx = sum(p[n]["x"] for n in "ABCD") / 4
        cy = sum(p[n]["y"] for n in "ABCD") / 4
        cz = sum(p[n]["z"] for n in "ABCD") / 4
        assert abs(cx) < EPS and abs(cy) < EPS and abs(cz) < EPS

    def test_max_radius_is_one(self):
        result = solve({
            "points":["A","B","C","D"],
            "constraints":[{"type":"square","points":["A","B","C","D"]}],
            "normalize": True,
        })
        p = result["points"]
        radii = [math.sqrt(p[n]["x"]**2 + p[n]["y"]**2 + p[n]["z"]**2) for n in "ABCD"]
        assert abs(max(radii) - 1.0) < EPS


# ── Output schema ─────────────────────────────────────────────────────────────

class TestOutputSchema:
    def test_has_edges_and_faces(self):
        result = solve({"points":["A","B","C","D"],
                        "constraints":[{"type":"square","points":["A","B","C","D"]}]})
        assert "edges" in result
        assert "faces" in result

    def test_edges_have_p1_p2(self):
        result = solve({"points":["A","B","C","D"],
                        "constraints":[{"type":"square","points":["A","B","C","D"]}]})
        for e in result["edges"]:
            assert "p1" in e and "p2" in e

    def test_faces_have_vertices(self):
        result = solve({"points":["A","B","C","D"],
                        "constraints":[{"type":"square","points":["A","B","C","D"]}]})
        for f in result["faces"]:
            assert "vertices" in f

    def test_unknown_constraint_skipped(self):
        p = pts({"points":["A","B","C","D"],
                 "constraints":[{"type":"square","points":["A","B","C","D"]},
                                {"type":"unknown_xyz","points":["A","B"]}]})
        assert len(p) == 4

    def test_violations_list_present(self):
        result = solve({"points":["A","B","C","D"],
                        "constraints":[{"type":"square","points":["A","B","C","D"]}]})
        assert "violations" in result
        assert result["violations"] == []

    def test_unresolved_points_list(self):
        result = solve({"points":["A","B","C","D","Z"],
                        "constraints":[{"type":"square","points":["A","B","C","D"]}]})
        assert "Z" in result["unresolved_points"]

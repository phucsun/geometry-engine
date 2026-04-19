"""
Tests for TopologyBuilder — edges and faces derived from constraints.
"""
from __future__ import annotations

import json
import pytest
from geometry_engine import GeometryEngine


def solve(payload: dict) -> dict:
    return GeometryEngine().solve_json(json.dumps(payload))


def edge_set(result: dict) -> set[frozenset[str]]:
    return {frozenset([e["p1"], e["p2"]]) for e in result["edges"]}


def face_list(result: dict) -> list[list[str]]:
    return [f["vertices"] for f in result["faces"]]


# ── Square topology ───────────────────────────────────────────────────────────

class TestSquareTopology:
    BASE = {"points":["A","B","C","D"],
            "constraints":[{"type":"square","points":["A","B","C","D"]}]}

    def test_has_four_edges(self):
        assert len(edge_set(solve(self.BASE))) == 4

    def test_correct_edges(self):
        es = edge_set(solve(self.BASE))
        expected = {frozenset(["A","B"]),frozenset(["B","C"]),
                    frozenset(["C","D"]),frozenset(["D","A"])}
        assert es == expected

    def test_has_one_face(self):
        fl = face_list(solve(self.BASE))
        assert len(fl) == 1
        assert set(fl[0]) == {"A","B","C","D"}


# ── Equilateral triangle topology ────────────────────────────────────────────

class TestTriangleTopology:
    BASE = {"points":["P","Q","R"],
            "constraints":[{"type":"equilateral_triangle","points":["P","Q","R"]}]}

    def test_three_edges(self):
        assert len(edge_set(solve(self.BASE))) == 3

    def test_one_triangular_face(self):
        fl = face_list(solve(self.BASE))
        assert len(fl) == 1
        assert set(fl[0]) == {"P","Q","R"}


# ── Pyramid topology ──────────────────────────────────────────────────────────

class TestPyramidTopology:
    BASE = {"points":["S","A","B","C","D"],
            "constraints":[
                {"type":"square","points":["A","B","C","D"]},
                {"type":"pyramid","points":["S","A","B","C","D"]},
            ]}

    def test_edge_count(self):
        """Square base (4) + lateral edges to apex (4) = 8 unique edges."""
        es = edge_set(solve(self.BASE))
        assert len(es) == 8

    def test_lateral_edges_present(self):
        es = edge_set(solve(self.BASE))
        for base_pt in "ABCD":
            assert frozenset(["S", base_pt]) in es

    def test_face_count(self):
        """1 base quad + 4 triangular lateral faces = 5."""
        fl = face_list(solve(self.BASE))
        assert len(fl) == 5

    def test_base_face_present(self):
        fl = face_list(solve(self.BASE))
        base_faces = [f for f in fl if set(f) == {"A","B","C","D"}]
        assert len(base_faces) == 1


# ── Cube topology ─────────────────────────────────────────────────────────────

class TestCubeTopology:
    BASE = {"points":list("ABCDEFGH"),
            "constraints":[{"type":"cube","points":list("ABCDEFGH")}]}

    def test_edge_count(self):
        assert len(edge_set(solve(self.BASE))) == 12

    def test_face_count(self):
        assert len(face_list(solve(self.BASE))) == 6

    def test_all_faces_are_quads(self):
        fl = face_list(solve(self.BASE))
        assert all(len(f) == 4 for f in fl)


# ── Prism topology ────────────────────────────────────────────────────────────

class TestPrismTopology:
    BASE = {"points":list("ABCDEF"),
            "constraints":[{"type":"prism","points":list("ABCDEF")}]}

    def test_edge_count(self):
        """3 (base) + 3 (top) + 3 (lateral) = 9."""
        assert len(edge_set(solve(self.BASE))) == 9

    def test_face_count(self):
        """2 triangles + 3 quads = 5."""
        assert len(face_list(solve(self.BASE))) == 5


# ── Tetrahedron topology ──────────────────────────────────────────────────────

class TestTetrahedronTopology:
    BASE = {"points":list("ABCD"),
            "constraints":[{"type":"regular_tetrahedron","points":list("ABCD")}]}

    def test_edge_count(self):
        assert len(edge_set(solve(self.BASE))) == 6

    def test_face_count(self):
        assert len(face_list(solve(self.BASE))) == 4

    def test_all_faces_are_triangles(self):
        fl = face_list(solve(self.BASE))
        assert all(len(f) == 3 for f in fl)


# ── Derived points have no topology ──────────────────────────────────────────

class TestDerivedPointsHaveNoTopology:
    def test_midpoint_adds_no_edges(self):
        result = solve({"points":["A","B","C","D","M"],
                        "constraints":[{"type":"square","points":["A","B","C","D"]},
                                       {"type":"midpoint","point":"M","segment":["A","B"]}]})
        # Only 4 edges from square; midpoint M introduces no new structural edge
        es = edge_set(result)
        assert frozenset(["A","M"]) not in es
        assert frozenset(["M","B"]) not in es

    def test_ratio_point_adds_no_edges(self):
        result = solve({"points":["A","B","C","D","G"],
                        "constraints":[{"type":"square","points":["A","B","C","D"]},
                                       {"type":"ratio_point","point":"G",
                                        "segment":["A","B"],"ratio":0.25}]})
        es = edge_set(result)
        assert frozenset(["A","G"]) not in es

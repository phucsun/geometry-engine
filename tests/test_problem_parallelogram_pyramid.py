"""
Bài toán: Hình chóp S.ABCD có đáy là hình bình hành ABCD.
  M = trung điểm SD
  N = trọng tâm tam giác SAB
  I = giao điểm đường thẳng MN với mặt phẳng (SBC)
  Tính tỉ số IN / IM.

Đáp án lý thuyết (bất biến affine): IN/IM = 2/3.

Test này kiểm tra:
  1. Engine mô hình hoá đúng tất cả các điểm.
  2. M là trung điểm SD.
  3. N là trọng tâm SAB.
  4. I nằm trên đường thẳng MN.
  5. I nằm trên mặt phẳng (SBC).
  6. IN/IM = 2/3 (chính xác đến 1e-5).
  7. Kết quả bất biến với nhiều dạng hình bình hành khác nhau.
"""
from __future__ import annotations

import numpy as np
import pytest

from geometry_engine.engine import GeometryEngine
from geometry_engine.models import Constraint, GeometryInput


EPS = 1e-5


def solve_problem(parallelogram_kwargs: dict | None = None, apex_height: float = 2.0):
    """
    Giải bài toán với tham số hình bình hành tuỳ chỉnh.
    Trả về dict tọa độ {tên: ndarray}.
    """
    pg_kwargs = parallelogram_kwargs or {}
    engine = GeometryEngine()
    result = engine.solve(GeometryInput(
        points=["A", "B", "C", "D", "S", "M", "N", "I"],
        constraints=[
            Constraint(type="parallelogram", points=["A", "B", "C", "D"],
                       **pg_kwargs),
            Constraint(type="apex", points=["S", "A", "B", "C", "D"],
                       length=apex_height),
            Constraint(type="midpoint",  point="M", segment=["S", "D"]),
            Constraint(type="centroid",  point="N", points=["S", "A", "B"]),
            Constraint(type="intersection", point="I",
                       segment=["M", "N"], points=["S", "B", "C"]),
        ],
        side_length=2.0,
        validate_constraints=True,
    ))
    assert result.violations == [], f"Violations: {result.violations}"
    return {k: np.array([v.x, v.y, v.z]) for k, v in result.points.items()}


# ── Kiểm tra từng điểm ────────────────────────────────────────────────────────

class TestPointPlacement:
    def test_all_points_resolved(self):
        pts = solve_problem()
        for name in ["A", "B", "C", "D", "S", "M", "N", "I"]:
            assert name in pts, f"Điểm {name} chưa được giải"

    def test_abcd_is_parallelogram(self):
        """AB = DC (vector)."""
        pts = solve_problem()
        AB = pts["B"] - pts["A"]
        DC = pts["C"] - pts["D"]
        assert np.linalg.norm(AB - DC) < EPS

    def test_m_is_midpoint_sd(self):
        pts = solve_problem()
        expected_M = (pts["S"] + pts["D"]) / 2.0
        assert np.linalg.norm(pts["M"] - expected_M) < EPS

    def test_n_is_centroid_sab(self):
        pts = solve_problem()
        expected_N = (pts["S"] + pts["A"] + pts["B"]) / 3.0
        assert np.linalg.norm(pts["N"] - expected_N) < EPS

    def test_i_on_line_mn(self):
        """I phải nằm trên đường thẳng qua M và N."""
        pts = solve_problem()
        MN = pts["N"] - pts["M"]
        MI = pts["I"] - pts["M"]
        # MI phải song song với MN: cross product = 0
        cross = np.linalg.norm(np.cross(MI, MN))
        assert cross < EPS * 10, f"I không nằm trên đường thẳng MN (cross={cross:.6f})"

    def test_i_on_plane_sbc(self):
        """I phải nằm trên mặt phẳng (SBC)."""
        pts = solve_problem()
        S, B, C, I = pts["S"], pts["B"], pts["C"], pts["I"]
        # Pháp tuyến mp (SBC)
        normal = np.cross(B - S, C - S)
        dot = abs(float(np.dot(I - S, normal)))
        assert dot < EPS * 100, f"I không nằm trên mp (SBC) (dot={dot:.6f})"


# ── Tỉ số IN/IM ───────────────────────────────────────────────────────────────

class TestRatio:
    def test_ratio_default_parallelogram(self):
        """Hình bình hành mặc định (góc 60°): IN/IM = 2/3."""
        pts = solve_problem()
        IN = np.linalg.norm(pts["I"] - pts["N"])
        IM = np.linalg.norm(pts["I"] - pts["M"])
        ratio = IN / IM
        assert abs(ratio - 2.0 / 3.0) < EPS, \
            f"IN/IM = {ratio:.6f}, kỳ vọng 2/3 ≈ {2/3:.6f}"

    def test_ratio_rectangle_base(self):
        """Đáy là hình chữ nhật (trường hợp riêng): IN/IM = 2/3."""
        pts = solve_problem({"length": 2.0, "width": 1.0, "degrees": 90.0})
        IN = np.linalg.norm(pts["I"] - pts["N"])
        IM = np.linalg.norm(pts["I"] - pts["M"])
        assert abs(IN / IM - 2.0 / 3.0) < EPS

    def test_ratio_oblique_60_degrees(self):
        """Hình bình hành xiên 60°: IN/IM = 2/3."""
        pts = solve_problem({"length": 3.0, "width": 2.0, "degrees": 60.0})
        IN = np.linalg.norm(pts["I"] - pts["N"])
        IM = np.linalg.norm(pts["I"] - pts["M"])
        assert abs(IN / IM - 2.0 / 3.0) < EPS

    def test_ratio_oblique_45_degrees(self):
        """Hình bình hành xiên 45°: IN/IM = 2/3."""
        pts = solve_problem({"length": 4.0, "width": 2.0, "degrees": 45.0})
        IN = np.linalg.norm(pts["I"] - pts["N"])
        IM = np.linalg.norm(pts["I"] - pts["M"])
        assert abs(IN / IM - 2.0 / 3.0) < EPS

    def test_ratio_tall_pyramid(self):
        """Chóp cao hơn: IN/IM = 2/3 (bất biến với chiều cao SA)."""
        pts = solve_problem(apex_height=5.0)
        IN = np.linalg.norm(pts["I"] - pts["N"])
        IM = np.linalg.norm(pts["I"] - pts["M"])
        assert abs(IN / IM - 2.0 / 3.0) < EPS

    def test_ratio_short_pyramid(self):
        """Chóp thấp: IN/IM = 2/3."""
        pts = solve_problem(apex_height=0.5)
        IN = np.linalg.norm(pts["I"] - pts["N"])
        IM = np.linalg.norm(pts["I"] - pts["M"])
        assert abs(IN / IM - 2.0 / 3.0) < EPS

    def test_i_is_beyond_n_from_m(self):
        """
        I nằm ngoài đoạn MN, về phía N (t=3 > 1).
        Có nghĩa N nằm giữa M và I.
        """
        pts = solve_problem()
        MN = pts["N"] - pts["M"]
        MI = pts["I"] - pts["M"]
        # t = MI / MN theo hướng MN
        MN_norm = float(np.linalg.norm(MN))
        t = float(np.dot(MI, MN)) / (MN_norm ** 2)
        assert t > 1.0, f"I phải ở phía ngoài N (t={t:.4f}, cần t > 1)"
        assert abs(t - 3.0) < EPS * 100, f"t = {t:.6f}, kỳ vọng t = 3"


# ── Kiểm tra hình bình hành mới ───────────────────────────────────────────────

class TestParallelogramConstraint:
    def test_default_placement(self):
        """Hình bình hành mặc định: AB vector = DC vector."""
        from geometry_engine.engine import GeometryEngine
        from geometry_engine.models import GeometryInput, Constraint
        engine = GeometryEngine()
        out = engine.solve(GeometryInput(
            points=["A", "B", "C", "D"],
            constraints=[Constraint(type="parallelogram",
                                    points=["A", "B", "C", "D"])],
        ))
        pts = {k: np.array([v.x, v.y, v.z]) for k, v in out.points.items()}
        AB = pts["B"] - pts["A"]
        DC = pts["C"] - pts["D"]
        assert np.linalg.norm(AB - DC) < EPS
        assert out.violations == []

    def test_compute_fourth_point(self):
        """Nếu biết A, B, D thì C = B + D − A."""
        engine = GeometryEngine()
        out = engine.solve(GeometryInput(
            points=["A", "B", "C", "D"],
            constraints=[
                Constraint(type="equilateral_triangle", points=["A", "B", "D"]),
                Constraint(type="parallelogram", points=["A", "B", "C", "D"]),
            ],
        ))
        pts = {k: np.array([v.x, v.y, v.z]) for k, v in out.points.items()}
        expected_C = pts["B"] + pts["D"] - pts["A"]
        assert np.linalg.norm(pts["C"] - expected_C) < EPS
        assert out.violations == []

    def test_topology(self):
        """Hình bình hành: 4 cạnh, 1 mặt."""
        engine = GeometryEngine()
        out = engine.solve(GeometryInput(
            points=["A", "B", "C", "D"],
            constraints=[Constraint(type="parallelogram",
                                    points=["A", "B", "C", "D"])],
        ))
        assert len(out.edges) == 4
        assert len(out.faces) == 1

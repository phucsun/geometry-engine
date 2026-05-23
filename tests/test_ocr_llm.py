from __future__ import annotations

import math

from geometry_engine import GeometryEngine
from geometry_engine.models import GeometryInput
from ocr_llm import analyze_image, analyze_problem_text


class FakeChain:
    def invoke(self, payload):
        assert "problem_text" in payload
        return {
            "points": ["A", "B", "C", "D"],
            "constraints": [{"type": "square", "points": ["A", "B", "C", "D"]}],
            "side_length": 2.0,
        }


def test_analyze_problem_text_validates_llm_payload(monkeypatch):
    monkeypatch.setattr("ocr_llm.analyzer._build_analysis_chain", lambda **_kwargs: FakeChain())

    result = analyze_problem_text("Cho hình vuông ABCD cạnh 2.")

    assert isinstance(result, GeometryInput)
    assert result.points == ["A", "B", "C", "D"]
    assert result.constraints[0].type == "square"
    assert result.side_length == 2.0


def test_analyze_image_runs_ocr_then_llm(monkeypatch):
    monkeypatch.setattr("ocr_llm.analyzer.run_ocr", lambda *_args, **_kwargs: "OCR text")
    monkeypatch.setattr("ocr_llm.analyzer._build_analysis_chain", lambda **_kwargs: FakeChain())

    ocr_text, geometry_input = analyze_image("unused.png")

    assert ocr_text == "OCR text"
    assert geometry_input.points == ["A", "B", "C", "D"]


def test_analyze_problem_text_repairs_parallelogram_pyramid_payload(monkeypatch):
    class BadParallelogramPyramidChain:
        def invoke(self, payload):
            return {
                "points": ["S", "A", "B", "C", "D", "M", "N", "I"],
                "constraints": [
                    {"type": "midpoint", "point": "M", "segment": ["S", "D"]},
                    {"type": "centroid", "point": "N", "segment": ["S", "A", "B"]},
                    {"type": "intersection", "point": "I", "segment": ["M", "N"]},
                ],
                "side_length": 1.0,
                "normalize": False,
                "validate_constraints": True,
            }

    monkeypatch.setattr(
        "ocr_llm.analyzer._build_analysis_chain",
        lambda **_kwargs: BadParallelogramPyramidChain(),
    )

    result = analyze_problem_text(
        "Cho hình chóp S. ABCD có đáy là hình bình hành. "
        "Gọi M là trung điểm của SD, N là trọng tâm tam giác SAB. "
        "Đường thẳng MN cắt mặt phẳng (SBC) tại điểm I."
    )

    constraints = [constraint.model_dump(exclude_none=True) for constraint in result.constraints]

    assert constraints == [
        {"type": "parallelogram", "points": ["A", "B", "C", "D"]},
        {"type": "apex", "points": ["S", "A", "B", "C", "D"]},
        {"type": "midpoint", "point": "M", "segment": ["S", "D"]},
        {"type": "centroid", "points": ["S", "A", "B"], "point": "N"},
        {"type": "intersection", "points": ["S", "B", "C"], "point": "I", "segment": ["M", "N"]},
    ]

    output = GeometryEngine().solve(result)
    assert output.unresolved_points == []
    assert output.violations == []


def test_analyze_problem_text_repairs_dihedral_angle_payload(monkeypatch):
    class BadDihedralChain:
        def invoke(self, payload):
            return {
                "points": ["S", "A", "B", "C", "D", "M", "N"],
                "constraints": [
                    {"type": "apex", "points": ["S", "A", "B", "C", "D"]},
                    {"type": "square", "points": ["A", "B", "C", "D"]},
                    {"type": "perpendicular_to_plane", "point": "S", "points": ["A", "B", "C", "D"]},
                    {"type": "angle", "points": ["S", "B", "D"], "degrees": 60.0},
                    {"type": "midpoint", "point": "M", "segment": ["S", "B"]},
                    {"type": "midpoint", "point": "N", "segment": ["S", "C"]},
                ],
                "side_length": 1.0,
                "normalize": False,
                "validate_constraints": True,
            }

    monkeypatch.setattr(
        "ocr_llm.analyzer._build_analysis_chain",
        lambda **_kwargs: BadDihedralChain(),
    )

    result = analyze_problem_text(
        "Cho hình chóp S.ABCD có đáy là hình vuông cạnh a, SA \\perp (ABCD); "
        "góc giữa hai mặt phẳng (SBD),(ABCD) bằng 60^0. "
        "Gọi M,N lần lượt là trung điểm các cạnh SB,SC."
    )

    constraints = [constraint.model_dump(exclude_none=True) for constraint in result.constraints]

    assert constraints[0] == {"type": "square", "points": ["A", "B", "C", "D"]}
    assert constraints[1] == {
        "type": "perpendicular_to_plane",
        "point": "S",
        "from_point": "A",
        "points": ["A", "B", "C", "D"],
    }
    assert constraints[2] == {
        "type": "dihedral_angle",
        "point": "S",
        "from_point": "A",
        "segment": ["B", "D"],
        "points": ["A", "B", "C", "D"],
        "degrees": 60.0,
    }
    assert not any(constraint["type"] == "angle" for constraint in constraints)


def test_analyze_problem_text_keeps_plain_angle_constraint(monkeypatch):
    class PlainAngleChain:
        def invoke(self, payload):
            return {
                "points": ["S", "B", "D"],
                "constraints": [
                    {"type": "angle", "points": ["S", "B", "D"], "degrees": 60.0},
                ],
                "side_length": 1.0,
            }

    monkeypatch.setattr(
        "ocr_llm.analyzer._build_analysis_chain",
        lambda **_kwargs: PlainAngleChain(),
    )

    result = analyze_problem_text("Cho tam giác SBD có góc SBD bằng 60 độ.")

    constraints = [constraint.model_dump(exclude_none=True) for constraint in result.constraints]
    assert constraints == [{"type": "angle", "points": ["S", "B", "D"], "degrees": 60.0}]


def test_analyze_problem_text_normalizes_symbolic_lengths(monkeypatch):
    class SymbolicLengthChain:
        def invoke(self, payload):
            return {
                "points": ["A", "B", "C", "D", "S"],
                "constraints": [
                    {"type": "square", "points": ["A", "B", "C", "D"], "length": "a"},
                    {"type": "distance", "points": ["S", "A"], "length": "2a"},
                    {"type": "angle", "points": ["S", "B", "D"], "degrees": "60"},
                ],
                "side_length": "a",
                "normalize": False,
                "validate_constraints": True,
            }

    monkeypatch.setattr(
        "ocr_llm.analyzer._build_analysis_chain",
        lambda **_kwargs: SymbolicLengthChain(),
    )

    result = analyze_problem_text("Cho hình vuông ABCD cạnh a và SA = 2a.")

    constraints = [constraint.model_dump(exclude_none=True) for constraint in result.constraints]
    assert result.side_length == 1.0
    assert constraints[0]["length"] == 1.0
    assert constraints[1]["length"] == 2.0
    assert constraints[2]["degrees"] == 60.0


def test_analyze_problem_text_repairs_equal_side_face_angle_payload(monkeypatch):
    class BadFaceAngleChain:
        def invoke(self, payload):
            return {
                "points": ["S", "A", "B", "C"],
                "constraints": [
                    {"type": "apex", "points": ["S", "A", "B", "C"]},
                    {"type": "right_triangle", "points": ["A", "B", "C"]},
                    {"type": "angle", "points": ["S", "A", "B"], "degrees": 60.0},
                    {"type": "angle", "points": ["S", "A", "C"], "degrees": 60.0},
                    {"type": "angle", "points": ["S", "B", "C"], "degrees": 60.0},
                    {"type": "edge_length", "segment": ["A", "B"], "length": "a"},
                    {"type": "edge_length", "segment": ["B", "C"], "length": "2a"},
                ],
                "side_length": "a",
                "normalize": False,
                "validate_constraints": True,
            }

    monkeypatch.setattr(
        "ocr_llm.analyzer._build_analysis_chain",
        lambda **_kwargs: BadFaceAngleChain(),
    )

    result = analyze_problem_text(
        "Cho hình chóp S.ABC có đáy là tam giác vuông tại A, AB = a, BC = 2a. "
        "Các mặt bên (SAB), (SAC), (SBC) cùng tạo với mặt đáy góc 60 độ."
    )

    constraints = [constraint.model_dump(exclude_none=True) for constraint in result.constraints]
    assert constraints[0]["type"] == "right_triangle"
    assert constraints[0]["points"] == ["A", "B", "C"]
    assert constraints[0]["length"] == 1.0
    assert abs(constraints[0]["width"] - math.sqrt(3.0)) < 1e-9
    assert constraints[1] == {"type": "equal_side_face_angle", "points": ["S", "A", "B", "C"], "degrees": 60.0}
    assert constraints[2] == {"type": "apex", "points": ["S", "A", "B", "C"]}
    assert not any(constraint["type"] == "edge_length" for constraint in constraints)
    assert not any(constraint["type"] == "angle" for constraint in constraints)


def test_analyze_problem_text_removes_three_point_angles_from_side_face_phrase(monkeypatch):
    class WrongThreePointAnglesChain:
        def invoke(self, payload):
            return {
                "points": ["S", "A", "B", "C"],
                "constraints": [
                    {"type": "apex", "points": ["S", "A", "B", "C"]},
                    {"type": "right_triangle", "points": ["A", "B", "C"]},
                    {"type": "angle", "points": ["S", "A", "B"], "degrees": 60.0},
                    {"type": "angle", "points": ["S", "A", "C"], "degrees": 60.0},
                    {"type": "angle", "points": ["S", "B", "C"], "degrees": 60.0},
                ],
                "side_length": 1.0,
                "normalize": False,
                "validate_constraints": True,
            }

    monkeypatch.setattr(
        "ocr_llm.analyzer._build_analysis_chain",
        lambda **_kwargs: WrongThreePointAnglesChain(),
    )

    result = analyze_problem_text(
        "Cho hình chóp S.ABC có đáy là tam giác vuông tại A. "
        "Các mặt bên (SAB), (SAC), (SBC) cùng tạo với mặt đáy góc 60 độ."
    )
    constraints = [constraint.model_dump(exclude_none=True) for constraint in result.constraints]

    assert not any(constraint["type"] == "angle" for constraint in constraints)
    assert any(
        constraint["type"] == "equal_side_face_angle"
        and constraint["points"] == ["S", "A", "B", "C"]
        and abs(constraint["degrees"] - 60.0) < 1e-9
        for constraint in constraints
    )


def test_analyze_problem_text_repairs_latex_side_face_phrase_and_drops_wrong_perpendicular(monkeypatch):
    class LatexFaceAngleChain:
        def invoke(self, payload):
            return {
                "points": ["S", "A", "B", "C"],
                "constraints": [
                    {"type": "apex", "points": ["S", "A", "B", "C"]},
                    {"type": "right_triangle", "points": ["A", "B", "C"]},
                    {"type": "perpendicular_to_plane", "point": "S", "from_point": "A", "points": ["A", "B", "C"]},
                    {"type": "angle", "points": ["S", "A", "B"], "degrees": 60.0},
                    {"type": "angle", "points": ["S", "A", "C"], "degrees": 60.0},
                    {"type": "angle", "points": ["S", "B", "C"], "degrees": 60.0},
                    {"type": "edge_length", "segment": ["A", "B"], "length": "a"},
                    {"type": "edge_length", "segment": ["B", "C"], "length": "2a"},
                ],
                "side_length": "a",
                "normalize": False,
                "validate_constraints": True,
            }

    monkeypatch.setattr(
        "ocr_llm.analyzer._build_analysis_chain",
        lambda **_kwargs: LatexFaceAngleChain(),
    )

    result = analyze_problem_text(
        "Cho hình chóp $S.ABC$ có đáy là tam giác vuông tại $A$, $AB = a$, $BC = 2a$. "
        "Các mặt bên $(SAB), (SAC), (SBC)$ cùng tạo với mặt đáy góc $60^\\circ$ "
        "và hình chiếu vuông góc của $S$ lên mặt phẳng $(ABC)$ nằm trong tam giác $ABC$."
    )
    constraints = [constraint.model_dump(exclude_none=True) for constraint in result.constraints]

    assert not any(constraint["type"] == "angle" for constraint in constraints)
    assert not any(constraint["type"] == "perpendicular_to_plane" for constraint in constraints)
    assert any(
        constraint["type"] == "equal_side_face_angle"
        and constraint["points"] == ["S", "A", "B", "C"]
        and abs(constraint["degrees"] - 60.0) < 1e-9
        for constraint in constraints
    )

    output = GeometryEngine().solve(result)
    assert output.unresolved_points == []
    assert output.violations == []


def test_analyze_problem_text_repairs_oblique_triangular_prism_payload(monkeypatch):
    class BadObliquePrismChain:
        def invoke(self, payload):
            return {
                "points": ["A", "B", "C", "A'", "B'", "C'", "H"],
                "constraints": [
                    {"type": "prism", "points": ["A", "B", "C", "A'", "B'", "C'"]},
                    {"type": "rectangle", "points": ["C", "A", "C'", "A'"]},
                    {"type": "perpendicular_to_plane", "point": "A'", "from_point": "A", "points": ["A", "B", "C"]},
                    {"type": "perpendicular_to_plane", "point": "B'", "from_point": "B", "points": ["A", "B", "C"]},
                    {"type": "perpendicular_to_plane", "point": "C'", "from_point": "C", "points": ["A", "B", "C"]},
                    {"type": "perpendicular_to_plane", "point": "H", "from_point": "A", "points": ["A", "B", "C"]},
                    {"type": "right_angle", "points": ["B", "C", "A"]},
                    {"type": "angle", "points": ["A", "C", "B"], "degrees": 30.0},
                ],
                "side_length": "a",
                "normalize": False,
                "validate_constraints": True,
            }

    monkeypatch.setattr(
        "ocr_llm.analyzer._build_analysis_chain",
        lambda **_kwargs: BadObliquePrismChain(),
    )

    result = analyze_problem_text(
        "Cho hình lăng trụ ABC.A'B'C', đáy ABC có AC=a√3, BC=3a, ACB=30^0. "
        "Cạnh bên hợp với mặt phẳng đáy góc 60^0 và mặt phẳng (A'BC) vuông góc với mặt phẳng (ABC). "
        "Điểm H trên cạnh BC sao cho HC=3BH và mặt phẳng (A'AH) vuông góc với mặt phẳng (ABC)."
    )
    constraints = [constraint.model_dump(exclude_none=True) for constraint in result.constraints]

    assert constraints[0] == {
        "type": "oblique_prism",
        "points": ["A", "B", "C", "A'", "B'", "C'"],
        "point": "H",
        "segment": ["B", "C"],
        "length": math.sqrt(3.0),
        "width": 3.0,
        "height": 60.0,
        "ratio": 0.25,
        "degrees": 30.0,
    }
    assert not any(constraint["type"] in {"rectangle", "perpendicular_to_plane", "right_angle", "angle"} for constraint in constraints)

    output = GeometryEngine().solve(result)
    assert output.unresolved_points == []
    assert output.violations == []


def test_analyze_problem_text_repairs_right_triangular_prism_payload(monkeypatch):
    class BadRightPrismChain:
        def invoke(self, payload):
            return {
                "points": ["A", "B", "C", "A'", "B'", "C'", "M"],
                "constraints": [
                    {"type": "rectangle", "points": ["A", "B", "A'", "B'"]},
                    {"type": "rectangle", "points": ["B", "C", "B'", "C'"]},
                    {"type": "rectangle", "points": ["A", "C", "A'", "C'"]},
                    {"type": "right_triangle", "points": ["A", "B", "C"]},
                    {"type": "perpendicular_to_plane", "point": "M", "from_point": "C", "points": ["A'", "B", "C"]},
                    {"type": "midpoint", "point": "M", "segment": ["C", "C'"]},
                ],
                "side_length": 1.0,
                "normalize": False,
                "validate_constraints": True,
            }

    monkeypatch.setattr(
        "ocr_llm.analyzer._build_analysis_chain",
        lambda **_kwargs: BadRightPrismChain(),
    )

    result = analyze_problem_text(
        "Cho hình lăng trụ đứng ABC.A'B'C' có đáy ABC là tam giác vuông tại B, "
        "AB = 1, AA' = 2, M là trung điểm CC'. "
        "Khoảng cách từ điểm M đến mặt phẳng (A'BC) bằng bao nhiêu?"
    )
    constraints = [constraint.model_dump(exclude_none=True) for constraint in result.constraints]

    assert constraints[0] == {
        "type": "right_triangle",
        "points": ["B", "A", "C"],
        "length": 1.0,
    }
    assert constraints[1] == {
        "type": "right_prism",
        "points": ["A", "B", "C", "A'", "B'", "C'"],
        "height": 2.0,
    }
    assert constraints[2] == {"type": "midpoint", "point": "M", "segment": ["C", "C'"]}
    assert not any(constraint["type"] == "rectangle" for constraint in constraints)
    assert not any(constraint["type"] == "perpendicular_to_plane" for constraint in constraints)

    output = GeometryEngine().solve(result)
    assert output.unresolved_points == []
    assert output.violations == []

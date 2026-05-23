from __future__ import annotations

from ocr_llm.problem_types import ProblemType, detect_problem_type
from ocr_llm.prompts import prompt_context_for


def test_detect_problem_types_for_supported_families():
    cases = [
        (
            "Cho hình chóp S.ABCD có đáy là hình vuông, tam giác SAB đều và SAD=90.",
            ProblemType.SQUARE_PYRAMID,
        ),
        (
            "Góc giữa hai mặt phẳng (SBD),(ABCD) bằng 60 độ.",
            ProblemType.DIHEDRAL_PYRAMID,
        ),
        (
            "Các mặt bên (SAB), (SAC), (SBC) cùng tạo với mặt đáy góc 60.",
            ProblemType.EQUAL_SIDE_FACE_ANGLE_PYRAMID,
        ),
        (
            "Cho hình lăng trụ đứng ABC.A'B'C' có đáy ABC là tam giác vuông tại B.",
            ProblemType.RIGHT_TRIANGULAR_PRISM,
        ),
        (
            "Cho hình lăng trụ ABC.A'B'C', cạnh bên hợp với mặt phẳng đáy góc 60.",
            ProblemType.OBLIQUE_TRIANGULAR_PRISM,
        ),
    ]

    for text, expected in cases:
        assert detect_problem_type(text) == expected


def test_prompt_context_is_scoped_to_detected_problem_type():
    right_prism_context = prompt_context_for(ProblemType.RIGHT_TRIANGULAR_PRISM)
    oblique_prism_context = prompt_context_for(ProblemType.OBLIQUE_TRIANGULAR_PRISM)

    assert "right_prism" in right_prism_context["supported_constraints"]
    assert "oblique_prism" not in right_prism_context["supported_constraints"]
    assert "cạnh bên hợp" not in right_prism_context["problem_rules"]
    assert "cạnh bên hợp" in oblique_prism_context["problem_rules"]

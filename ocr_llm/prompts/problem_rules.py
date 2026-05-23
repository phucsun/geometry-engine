"""Prompt rule snippets selected by detected problem type."""
from __future__ import annotations

from ocr_llm.problem_types import PROBLEM_TYPE_SPECS, ProblemType

COMMON_EXAMPLE = """
Output tối thiểu gồm `points`, `constraints`, `side_length`, `normalize`, `validate_constraints`.
"""

_RULES: dict[ProblemType, str] = {
    ProblemType.SQUARE_PYRAMID: """
- `hình chóp S.ABCD` -> thêm `apex` với points `[S,A,B,C,D]`.
- `đáy ABCD là hình vuông` -> `square(A,B,C,D)`.
- `tam giác SAB đều` -> `equilateral_triangle(S,A,B)`.
- `SAD = 90` -> `right_angle(S,A,D)`, không phải perpendicular_to_plane.
- Trung điểm `J` của `SD` -> `midpoint(point=J, segment=[S,D])`.
""",
    ProblemType.DIHEDRAL_PYRAMID: """
- `SA ⟂ (ABCD)` -> `perpendicular_to_plane(point=S, from_point=A, points=[A,B,C,D])`.
- `góc giữa hai mặt phẳng (SBD),(ABCD)` không map thành `angle(S,B,D)`. Backend repair sẽ tạo rule nội bộ.
""",
    ProblemType.EQUAL_SIDE_FACE_ANGLE_PYRAMID: """
- `các mặt bên ... cùng tạo với mặt đáy góc θ` không map thành `angle` 3 điểm.
- Giữ dữ kiện đáy, cạnh, hình chóp; backend repair sẽ tạo rule nội bộ.
""",
    ProblemType.RIGHT_TRIANGULAR_PRISM: """
- `lăng trụ đứng ABC.A'B'C'` không tách thành 3 rectangle riêng lẻ.
- `tam giác vuông tại B` -> `right_triangle(points=[B,A,C])`.
- `AA' = h` là chiều cao lăng trụ; backend repair sẽ tạo `right_prism`.
- Câu hỏi khoảng cách từ điểm đến mặt phẳng không phải `perpendicular_to_plane`.
""",
    ProblemType.OBLIQUE_TRIANGULAR_PRISM: """
- `lăng trụ ABC.A'B'C'` với cạnh bên hợp đáy là lăng trụ xiên, không phải right_prism.
- Không tạo các `perpendicular_to_plane` cho A',B',C' trừ khi đề nói cạnh bên vuông góc đáy.
- Giữ độ dài đáy, góc đáy, tỉ lệ điểm trên cạnh; backend repair sẽ tạo rule nội bộ.
""",
    ProblemType.GENERIC_SHAPES: """
- Dùng constraint đơn giản nhất khớp trực tiếp đề bài.
- Không biến câu hỏi cần tính thành dữ kiện dựng hình.
""",
}

_EXAMPLES: dict[ProblemType, str] = {
    ProblemType.RIGHT_TRIANGULAR_PRISM: """
Đề: Cho lăng trụ đứng ABC.A'B'C', đáy ABC vuông tại B, AB=1, AA'=2, M là trung điểm CC'.
Output constraints chính: right_triangle([B,A,C]), midpoint(M,[C,C']).
""",
    ProblemType.OBLIQUE_TRIANGULAR_PRISM: """
Đề: Lăng trụ ABC.A'B'C', AC=a√3, BC=3a, ACB=30, cạnh bên hợp đáy 60, HC=3BH.
Output giữ dữ kiện đáy/góc/tỉ lệ; không tạo rectangle/perpendicular_to_plane giả.
""",
    ProblemType.SQUARE_PYRAMID: """
Đề: Chóp S.ABCD đáy vuông, tam giác SAB đều, SAD=90, J trung điểm SD.
Output constraints chính: square, apex, equilateral_triangle, right_angle, midpoint.
""",
}


def rules_for(problem_type: ProblemType) -> str:
    return _RULES[problem_type]


def example_for(problem_type: ProblemType) -> str:
    return _EXAMPLES.get(problem_type, COMMON_EXAMPLE)


def constraints_for(problem_type: ProblemType) -> str:
    return ", ".join(PROBLEM_TYPE_SPECS[problem_type].allowed_constraints)


def prompt_context_for(problem_type: ProblemType) -> dict[str, str]:
    """Return the small prompt context for one detected problem family."""
    return {
        "problem_type": problem_type.value,
        "supported_constraints": constraints_for(problem_type),
        "problem_rules": rules_for(problem_type),
        "problem_example": example_for(problem_type),
    }

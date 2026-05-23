"""OCR + LLM analysis for geometry problems.

This module only turns an image or problem text into GeometryInput. It does
not call GeometryEngine; orchestration belongs in root-level backend files.
"""
from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

from geometry_engine.models import GeometryInput
from ocr_llm.problem_types import ProblemType, detect_problem_type
from ocr_llm.prompts import BASE_PROMPT_TEMPLATE, constraints_for, prompt_context_for
from ocr_llm.repairs import _repair_geometry_payload


DEFAULT_ANALYZER_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OCR_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
logger = logging.getLogger(__name__)

SUPPORTED_CONSTRAINTS = (
    "square, rectangle, parallelogram, rhombus, trapezoid, "
    "equilateral_triangle, isosceles_triangle, right_triangle, "
    "regular_tetrahedron, cube, rectangular_prism, prism, right_prism, "
    "regular_hexagon, regular_octahedron, regular_polygon, midpoint, "
    "ratio_point, centroid, circumcenter, orthocenter, incenter, equidistant, "
    "angle_bisector, median, foot_perpendicular, foot_on_plane, "
    "perpendicular_to_plane, symmetric, intersection, apex, regular_pyramid, "
    "pyramid, truncated_pyramid, right_angle, angle, distance, edge_length, "
    "on_line, collinear, parallel, perpendicular, coplanar"
)

# Dùng model OCR trên ảnh để lấy văn bản đề bài
def run_ocr(
    image_path: str | Path,
    *,
    model_name: str = DEFAULT_OCR_MODEL,
) -> str:
    """Read the problem statement from an image using a Groq vision model."""
    ChatGroq = _import_chat_groq()
    image_b64 = image_to_base64(image_path)
    llm_ocr = ChatGroq(
        model_name=model_name,
        temperature=0.1,
        groq_api_key=_resolve_groq_api_key(),
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Hãy đọc chính xác toàn bộ nội dung đề toán trong ảnh. "
                        "Chỉ đọc, không giải và trả về chính xác văn bản của đề bài."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ],
        }
    ]
    return llm_ocr.invoke(messages).content


# Xây dựng chuỗi phân tích LLM với prompt và parser
def _build_analysis_chain(*, model_name: str, problem_type: ProblemType):
    ChatGroq = _import_chat_groq()
    ChatPromptTemplate, JsonOutputParser = _import_langchain_core()

    parser = JsonOutputParser(pydantic_object=GeometryInput)
    prompt = ChatPromptTemplate.from_template(
        BASE_PROMPT_TEMPLATE,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    llm = ChatGroq(
        model_name=model_name,
        temperature=0.1,
        groq_api_key=_resolve_groq_api_key(),
    )
    return prompt | llm | parser


# Phân tích đề bài đã được OCR thành GeometryInput
def analyze_problem_text(
    problem_text: str,
    *,
    model_name: str = DEFAULT_ANALYZER_MODEL,
) -> GeometryInput:
    """Analyze OCR/plain text and return validated GeometryInput."""
    problem_type = detect_problem_type(problem_text)
    chain = _build_analysis_chain(model_name=model_name, problem_type=problem_type)
    result = chain.invoke(
        {
            "problem_text": problem_text,
            **prompt_context_for(problem_type),
        }
    )
    payload = _repair_geometry_payload(_to_plain_payload(result), problem_text)
    return _validate_geometry_input(payload)


# Gộp hai bước trên: OCR ảnh rồi phân tích văn bản thành GeometryInput
def analyze_image(
    image_path: str | Path,
    *,
    analyzer_model: str = DEFAULT_ANALYZER_MODEL,
    ocr_model: str = DEFAULT_OCR_MODEL,
) -> tuple[str, GeometryInput]:
    """Run OCR on an image and analyze the text into GeometryInput."""
    ocr_text = run_ocr(image_path, model_name=ocr_model)
    geometry_input = analyze_problem_text(
        ocr_text,
        model_name=analyzer_model,
    )
    if logger.isEnabledFor(logging.INFO):
        logger.info("OCR text:\n%s\nGeometryInput:\n%s", ocr_text, geometry_input)
    return ocr_text, geometry_input


# Chuyển đổi ảnh thành base64 để gửi qua API
def image_to_base64(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")

def _to_plain_payload(result: Any) -> dict[str, Any] | str:
    if isinstance(result, dict | str):
        return result
    # chuyển đổi các object có model_dump (như GeometryInput) thành dict để validate
    if hasattr(result, "model_dump"):
        return result.model_dump()
    raise TypeError(f"Unsupported LLM parser result type: {type(result)!r}")


def _validate_geometry_input(payload: dict[str, Any] | str) -> GeometryInput:
    # nếu payload đã là JSON string hoặc dict thì parse thẳng
    if isinstance(payload, str):
        return GeometryInput.model_validate_json(payload)
    # nếu là GeometryInput đã được parser rồi thì chỉ cần validate lại
    return GeometryInput.model_validate(payload)


# kiểm tra xem có biến môi trường GROQ_API_KEY không
def _resolve_groq_api_key() -> str:
    _load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing Groq API key. Add GROQ_API_KEY to your .env file."
        )
    return key


# kiểm tra xem có cài đặt python-dotenv không và load .env nếu có
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency python-dotenv. Run: pip install -r requirements.txt"
        ) from exc
    load_dotenv()


# kiểm tra xem có cài đặt langchain-groq trước khi import
def _import_chat_groq():
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency langchain-groq. Run: pip install -r requirements.txt"
        ) from exc
    return ChatGroq


# kiểm tra xem có cài đặt langchain-core trước khi import
def _import_langchain_core():
    try:
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError as exc:
        raise RuntimeError(
            "Missing LangChain dependencies. Run: pip install -r requirements.txt"
        ) from exc
    return ChatPromptTemplate, JsonOutputParser

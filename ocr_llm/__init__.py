"""OCR + LLM module for extracting GeometryInput from problem images."""

from .analyzer import (
    DEFAULT_ANALYZER_MODEL,
    DEFAULT_OCR_MODEL,
    SUPPORTED_CONSTRAINTS,
    analyze_image,
    analyze_problem_text,
    image_to_base64,
    run_ocr,
)

__all__ = [
    "DEFAULT_ANALYZER_MODEL",
    "DEFAULT_OCR_MODEL",
    "SUPPORTED_CONSTRAINTS",
    "analyze_image",
    "analyze_problem_text",
    "image_to_base64",
    "run_ocr",
]

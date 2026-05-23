"""Root pipeline that connects OCR/LLM analysis to GeometryEngine.

This is the backend-facing orchestration layer:
image -> ocr_llm -> GeometryInput -> geometry_engine -> GeometryOutput.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from geometry_engine import GeometryEngine
from geometry_engine.models import GeometryOutput
from ocr_llm import analyze_image


def solve_image(
    image_path: str | Path,
) -> GeometryOutput:
    """Analyze a problem image and return GeometryEngine output."""
    _ocr_text, geometry_input = analyze_image(image_path)
    return GeometryEngine().solve(geometry_input)


def solve_image_json(
    image_path: str | Path,
    *,
    pretty: bool = False,
) -> str:
    """Analyze a problem image and serialize GeometryEngine output as JSON."""
    output = solve_image(image_path)
    return json.dumps(output.model_dump(), ensure_ascii=False, indent=2 if pretty else None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OCR/LLM analysis, solve with GeometryEngine, and print GeometryOutput JSON."
    )
    parser.add_argument("image", help="Path to the problem image")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    print(
        solve_image_json(
            args.image,
            pretty=args.pretty,
        )
    )


if __name__ == "__main__":
    main()

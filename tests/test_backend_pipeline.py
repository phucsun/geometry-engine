from __future__ import annotations

from geometry_engine.models import Constraint, GeometryInput, GeometryOutput

import backend_pipeline


def test_solve_image_connects_ocr_llm_to_geometry_engine(monkeypatch):
    geometry_input = GeometryInput(
        points=["A", "B", "C", "D"],
        constraints=[Constraint(type="square", points=["A", "B", "C", "D"])],
        side_length=2.0,
    )

    monkeypatch.setattr(
        "backend_pipeline.analyze_image",
        lambda *_args, **_kwargs: ("OCR text", geometry_input),
    )

    output = backend_pipeline.solve_image("unused.png")

    assert isinstance(output, GeometryOutput)
    assert set(output.points) == {"A", "B", "C", "D"}
    assert output.violations == []


def test_solve_image_json_returns_geometry_output_json(monkeypatch):
    geometry_input = GeometryInput(
        points=["A", "B", "C", "D"],
        constraints=[Constraint(type="square", points=["A", "B", "C", "D"])],
    )
    monkeypatch.setattr(
        "backend_pipeline.analyze_image",
        lambda *_args, **_kwargs: ("OCR text", geometry_input),
    )

    result = backend_pipeline.solve_image_json("unused.png", pretty=True)

    assert '"points"' in result
    assert '"violations": []' in result

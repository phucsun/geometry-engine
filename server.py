"""
FastAPI server for the GeometryEngine.

Pipeline position:
    Mobile App  →  POST /solve or POST /solve-image  →  AI Server  →  GeometryOutput JSON

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Or via CLI:
    python -m geometry_engine serve
"""
from __future__ import annotations

import mimetypes
from pathlib import Path
import tempfile
import os

import backend_pipeline
from fastapi import FastAPI, HTTPException, Request
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from geometry_engine import GeometryEngine
from geometry_engine.models import GeometryInput, GeometryOutput

logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GeometryEngine API",
    description=(
        "Accepts either GeometryInput JSON or an uploaded problem image and returns "
        "3-D coordinates, edges, and faces for downstream rendering."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = GeometryEngine()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}


@app.post(
    "/solve",
    response_model=GeometryOutput,
    summary="Solve geometry constraints → 3-D model",
    description=(
        "Accepts a JSON body conforming to **GeometryInput** and returns "
        "resolved 3-D coordinates plus structural edges and faces."
    ),
)
def solve(input_data: GeometryInput) -> GeometryOutput:
    try:
        return _engine.solve(input_data)
    except Exception as exc:
        logger.exception("Solver error")
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/solve-image",
    response_model=GeometryOutput,
    summary="Solve a geometry problem from an uploaded image",
    description=(
        "Accepts a multipart image upload, runs OCR/LLM analysis, then returns "
        "the solved 3-D coordinates plus structural edges and faces."
    ),
)
async def solve_image(image: UploadFile = File(...)) -> GeometryOutput:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    suffix = _resolve_upload_suffix(image)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name
        return backend_pipeline.solve_image(temp_path)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Image solve error")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await image.close()
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _resolve_upload_suffix(image: UploadFile) -> str:
    filename = image.filename or ""
    suffix = Path(filename).suffix
    if suffix:
        return suffix
    guessed = mimetypes.guess_extension(image.content_type or "")
    return guessed or ".png"


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error for %s", request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"},
    )


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

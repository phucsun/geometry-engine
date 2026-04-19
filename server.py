"""
FastAPI server for the GeometryEngine.

Pipeline position:
    Mobile App  →  POST /solve  →  AI Server  →  GeometryOutput JSON  →  Mobile App

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Or via CLI:
    python -m geometry_engine serve
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
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
        "Receives geometry constraint JSON from an LLM and returns 3-D coordinates, "
        "edges and faces for AR rendering in Unity."
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

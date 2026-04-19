"""
CLI for GeometryEngine.

Usage
-----
# Solve from stdin (pipe JSON):
    echo '{...}' | python -m geometry_engine solve

# Solve from a file:
    python -m geometry_engine solve problem.json

# Pretty-print output:
    python -m geometry_engine solve problem.json --pretty

# Validate only (print violations):
    python -m geometry_engine validate problem.json

# Start the API server:
    python -m geometry_engine serve [--host HOST] [--port PORT]
"""
from __future__ import annotations

import json
import sys
import argparse

from . import GeometryEngine
from .models import GeometryInput


def cmd_solve(args: argparse.Namespace) -> None:
    raw = _read_input(args.file)
    engine = GeometryEngine()
    result = engine.solve_json(raw)

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent, ensure_ascii=False))

    violations = result.get("violations", [])
    if violations:
        print("\n⚠  Constraint violations:", file=sys.stderr)
        for v in violations:
            print(f"   • {v}", file=sys.stderr)
        sys.exit(1)


def cmd_validate(args: argparse.Namespace) -> None:
    raw = _read_input(args.file)
    data = GeometryInput.model_validate_json(raw)
    engine = GeometryEngine()
    output = engine.solve(data)

    if output.violations:
        print("FAIL — constraint violations:")
        for v in output.violations:
            print(f"  • {v}")
        sys.exit(1)
    else:
        print(f"OK — all constraints satisfied ({len(data.constraints)} checked).")


def cmd_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn  # type: ignore[import]
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn", file=sys.stderr)
        sys.exit(1)
    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)


def _read_input(filepath: str | None) -> str:
    if filepath:
        with open(filepath, encoding="utf-8") as fh:
            return fh.read()
    return sys.stdin.read()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="geometry_engine",
        description="GeometryEngine — constraint-based 3-D geometry solver",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # solve
    p_solve = sub.add_parser("solve", help="Solve geometry constraints → JSON")
    p_solve.add_argument("file", nargs="?", help="Input JSON file (default: stdin)")
    p_solve.add_argument("--pretty", action="store_true", help="Indent output")
    p_solve.set_defaults(func=cmd_solve)

    # validate
    p_val = sub.add_parser("validate", help="Check that solved coords satisfy constraints")
    p_val.add_argument("file", nargs="?", help="Input JSON file (default: stdin)")
    p_val.set_defaults(func=cmd_validate)

    # serve
    p_srv = sub.add_parser("serve", help="Start the FastAPI server")
    p_srv.add_argument("--host", default="0.0.0.0")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--reload", action="store_true")
    p_srv.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

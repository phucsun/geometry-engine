"""
Normalizer — centers the solved geometry and scales it for AR display.

Unity AR scenes typically use a model that fits inside a unit sphere so that
it looks consistent regardless of the original coordinate values.
"""
from __future__ import annotations

import numpy as np

from .models import GeometryOutput, Point3D


class Normalizer:
    """
    Transforms a ``GeometryOutput`` so that:

    1. The centroid of all points is moved to the origin.
    2. The coordinates are scaled so the farthest point sits at radius 1.0.

    Edges and faces are label-based (point names), so they require no changes.
    """

    def normalize(self, output: GeometryOutput) -> GeometryOutput:
        if not output.points:
            return output

        positions = {
            name: np.array([p.x, p.y, p.z])
            for name, p in output.points.items()
        }

        # 1. Center at origin
        center = np.mean(list(positions.values()), axis=0)
        centered = {name: pos - center for name, pos in positions.items()}

        # 2. Scale to unit sphere
        max_r = max(float(np.linalg.norm(pos)) for pos in centered.values())
        scale = 1.0 / max_r if max_r > 1e-10 else 1.0
        normalized = {name: pos * scale for name, pos in centered.items()}

        new_points = {
            name: Point3D(
                x=round(float(pos[0]), 10),
                y=round(float(pos[1]), 10),
                z=round(float(pos[2]), 10),
            )
            for name, pos in normalized.items()
        }

        return GeometryOutput(
            points=new_points,
            edges=output.edges,
            faces=output.faces,
            unresolved_points=output.unresolved_points,
            violations=output.violations,
        )

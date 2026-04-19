"""
Low-level vector math utilities for the GeometryEngine.
All functions operate on numpy float64 arrays of shape (3,).
"""
from __future__ import annotations

import numpy as np


# ── Basics ────────────────────────────────────────────────────────────────────

def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return v / norm


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5


def centroid(pts: list[np.ndarray]) -> np.ndarray:
    return np.mean(pts, axis=0)


def ratio_point(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """P = A + t*(B-A).  t=0→A, t=1→B, t=0.5→midpoint."""
    return a + t * (b - a)


# ── Perpendicular directions ──────────────────────────────────────────────────

def perpendicular_pair(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return two unit vectors (v, w) ⊥ u forming a right-handed orthonormal basis
    {u, v, w}.
    """
    u = normalize(u)
    ref = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = normalize(np.cross(u, ref))
    w = np.cross(u, v)  # already unit since u ⊥ v
    return v, w


# ── Projections ───────────────────────────────────────────────────────────────

def project_point_onto_line(
    P: np.ndarray, A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """Foot of perpendicular from P to the infinite line through A, B."""
    AB = B - A
    sq = float(np.dot(AB, AB))
    if sq < 1e-24:
        return A.copy()
    t = float(np.dot(P - A, AB)) / sq
    return A + t * AB


def project_point_onto_plane(
    P: np.ndarray, plane_pt: np.ndarray, plane_normal: np.ndarray
) -> np.ndarray:
    """Foot of perpendicular from P to the plane defined by plane_pt and plane_normal."""
    n = normalize(plane_normal)
    return P - float(np.dot(P - plane_pt, n)) * n


def reflect_over_point(P: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Symmetric image of P over center: P' = 2*center - P."""
    return 2.0 * center - P


def reflect_over_line(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Symmetric image of P over the infinite line AB."""
    foot = project_point_onto_line(P, A, B)
    return reflect_over_point(P, foot)


def reflect_over_plane(
    P: np.ndarray, plane_pt: np.ndarray, plane_normal: np.ndarray
) -> np.ndarray:
    """Symmetric image of P over the plane."""
    foot = project_point_onto_plane(P, plane_pt, plane_normal)
    return reflect_over_point(P, foot)


# ── Intersections ─────────────────────────────────────────────────────────────

def intersect_two_lines(
    A: np.ndarray, B: np.ndarray,
    C: np.ndarray, D: np.ndarray,
    tol: float = 1e-6,
) -> np.ndarray | None:
    """
    Intersection of lines AB and CD in 3-D.
    Returns the intersection point, or None if lines are parallel / skew.
    """
    d1 = B - A
    d2 = D - C
    # Solve A + t*d1 = C + s*d2  →  [d1 | -d2] [t;s] = C-A
    M = np.column_stack([d1, -d2])          # 3×2
    b = C - A
    ts, _, rank, _ = np.linalg.lstsq(M, b, rcond=None)
    if rank < 2:
        return None  # parallel or coincident
    P1 = A + ts[0] * d1
    P2 = C + ts[1] * d2
    if np.linalg.norm(P1 - P2) < tol:
        return (P1 + P2) * 0.5
    return None  # skew lines


def intersect_line_plane(
    P: np.ndarray,
    direction: np.ndarray,
    plane_pt: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray | None:
    """
    Intersection of line P+t*direction with the plane.
    Returns the point, or None if line is parallel to the plane.
    """
    denom = float(np.dot(direction, plane_normal))
    if abs(denom) < 1e-12:
        return None
    t = float(np.dot(plane_pt - P, plane_normal)) / denom
    return P + t * direction


# ── Equilateral triangle apex candidates ─────────────────────────────────────

def equilateral_apex_candidates(P: np.ndarray, Q: np.ndarray) -> list[np.ndarray]:
    """
    4 candidate apex positions R so that PQR is equilateral.
    Sorted: highest z first (above-base option comes first).
    """
    M = midpoint(P, Q)
    side = dist(P, Q)
    h = side * np.sqrt(3.0) / 2.0
    u = normalize(Q - P)
    v, w = perpendicular_pair(u)

    candidates = [M + h * v, M - h * v, M + h * w, M - h * w]
    candidates.sort(
        key=lambda c: (-round(c[2], 8), -round(c[1], 8), -round(c[0], 8))
    )
    return candidates


# ── Angle checks ─────────────────────────────────────────────────────────────

def are_perpendicular(v1: np.ndarray, v2: np.ndarray, tol: float = 1e-6) -> bool:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return False
    return abs(float(np.dot(v1, v2))) / (n1 * n2) < tol


def are_parallel(v1: np.ndarray, v2: np.ndarray, tol: float = 1e-6) -> bool:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return True
    return float(np.linalg.norm(np.cross(v1 / n1, v2 / n2))) < tol


def cosine_of_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """cos(angle between v1 and v2), clamped to [-1, 1]."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))


# ── Polygon plane normal ──────────────────────────────────────────────────────

def polygon_normal(positions: list[np.ndarray]) -> np.ndarray:
    """
    Outward normal of a convex polygon (Newell's method).
    Biased toward positive z so it points 'upward'.
    """
    if len(positions) < 3:
        return np.array([0.0, 0.0, 1.0])

    n = np.zeros(3)
    for i in range(len(positions)):
        cur = positions[i]
        nxt = positions[(i + 1) % len(positions)]
        n[0] += (cur[1] - nxt[1]) * (cur[2] + nxt[2])
        n[1] += (cur[2] - nxt[2]) * (cur[0] + nxt[0])
        n[2] += (cur[0] - nxt[0]) * (cur[1] + nxt[1])

    mag = np.linalg.norm(n)
    if mag < 1e-10:
        return np.array([0.0, 0.0, 1.0])

    n = n / mag
    if n[2] < -1e-8 or (abs(n[2]) < 1e-8 and n[1] < -1e-8):
        n = -n
    return n


def plane_from_points(positions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroid, normal) of the plane containing the given points."""
    c = centroid(positions)
    n = polygon_normal(positions)
    return c, n

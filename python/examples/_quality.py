"""Uniform tet-mesh quality, computed the same way for every mesher.

The comparison showcase meshes one geometry with rapidmesh, gmsh and tetgen and
shows their quality side by side. To keep the numbers apples-to-apples we ignore
each tool's own quality readout and recompute everything here from the bare
``(points, tets)`` with one set of formulas:

- ``min_dihedral_deg``  smallest dihedral angle over all tets (the sliver
  detector: a flat sliver has a dihedral near 0 or 180 deg)
- ``max_radius_edge``    largest circumradius / shortest-edge ratio (the
  Delaunay refinement quality measure; lower is rounder)
- ``min_edge`` / ``max_edge``  edge-length extremes

All vectorized over the tet array; no per-tet Python loop.
"""

from __future__ import annotations

import numpy as np


def _dihedral_angles_deg(p: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """The six dihedral angles (deg) of every tet, shape ``(n_tets, 6)``.

    The dihedral along an edge is the angle between the two faces sharing it;
    we get it from the two outward face normals. For a tet with vertices
    0,1,2,3 the four faces (opposite each vertex) have normals n0..n3, and the
    dihedral on the edge shared by faces i,j is ``pi - angle(n_i, n_j)``.
    """
    a, b, c, d = p[tets[:, 0]], p[tets[:, 1]], p[tets[:, 2]], p[tets[:, 3]]
    # Face normal opposite each vertex (face = the other three vertices).
    # Orientation/sign does not matter: we normalize and take pi - angle.
    n0 = np.cross(c - b, d - b)  # face bcd, opposite a
    n1 = np.cross(d - a, c - a)  # face acd, opposite b
    n2 = np.cross(b - a, d - a)  # face abd, opposite c
    n3 = np.cross(c - a, b - a)  # face abc, opposite d
    normals = [n0, n1, n2, n3]
    norms = [np.linalg.norm(n, axis=1) for n in normals]
    # The six edges pair up the six (i<j) face combinations.
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    out = np.empty((tets.shape[0], 6), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        denom = norms[i] * norms[j]
        denom = np.where(denom == 0.0, np.inf, denom)
        cos = np.sum(normals[i] * normals[j], axis=1) / denom
        cos = np.clip(cos, -1.0, 1.0)
        out[:, k] = 180.0 - np.degrees(np.arccos(cos))
    return out


def _circumradius(p: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Circumradius of every tet, shape ``(n_tets,)``.

    R = |a-d||b-d||c-d| ... use the standard determinant form via the squared
    edge vectors from one vertex. We use the formula
    R = sqrt(|A|^2 |B x C|^2 + ...)/(2|det|) built from edge vectors A,B,C off
    vertex 0; equivalently R = |v| with the circumcenter solve below.
    """
    a, b, c, d = p[tets[:, 0]], p[tets[:, 1]], p[tets[:, 2]], p[tets[:, 3]]
    A = b - a
    B = c - a
    C = d - a
    # 2 * volume * 6 = det[A B C]
    det = np.einsum("ij,ij->i", A, np.cross(B, C))
    a2 = np.sum(A * A, axis=1)
    b2 = np.sum(B * B, axis=1)
    c2 = np.sum(C * C, axis=1)
    # circumcenter offset from vertex 0:
    # o = (a2 (B x C) + b2 (C x A) + c2 (A x B)) / (2 det)
    num = (a2[:, None] * np.cross(B, C)
           + b2[:, None] * np.cross(C, A)
           + c2[:, None] * np.cross(A, B))
    safe = np.where(det == 0.0, np.inf, det)
    o = num / (2.0 * safe[:, None])
    return np.linalg.norm(o, axis=1)


def _edge_lengths(p: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """All six edge lengths per tet, shape ``(n_tets, 6)``."""
    e = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    out = np.empty((tets.shape[0], 6), dtype=np.float64)
    for k, (i, j) in enumerate(e):
        out[:, k] = np.linalg.norm(p[tets[:, i]] - p[tets[:, j]], axis=1)
    return out


def quality(points, tets) -> dict:
    """Uniform quality dict for a ``(points, tets)`` mesh: ``n_points``,
    ``n_tets``, ``min_dihedral_deg``, ``max_radius_edge``, ``min_edge``,
    ``max_edge``. Empty meshes return zeros."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(tets, dtype=np.int64).reshape(-1, 4)
    if t.shape[0] == 0:
        return {
            "n_points": int(p.shape[0]),
            "n_tets": 0,
            "min_dihedral_deg": 0.0,
            "max_radius_edge": 0.0,
            "min_edge": 0.0,
            "max_edge": 0.0,
        }
    dih = _dihedral_angles_deg(p, t)
    edges = _edge_lengths(p, t)
    R = _circumradius(p, t)
    shortest = edges.min(axis=1)
    shortest = np.where(shortest == 0.0, np.inf, shortest)
    radius_edge = R / shortest
    return {
        "n_points": int(p.shape[0]),
        "n_tets": int(t.shape[0]),
        "min_dihedral_deg": float(dih.min()),
        "max_radius_edge": float(radius_edge.max()),
        "min_edge": float(edges.min()),
        "max_edge": float(edges.max()),
    }

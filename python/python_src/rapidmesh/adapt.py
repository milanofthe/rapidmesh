"""gmsh-style Doerfler-marking adaptive refinement for surface meshes.

This is the MARK -> REFINE half of the adaptive loop (SOLVE and ESTIMATE belong
to the solver): given per-triangle error indicators, Doerfler-mark the bulk and
turn the marked elements into a background size field -- gmsh-style -- which the
mesher's grading gradient-limits and the Ruppert refinement re-triangulates, so
every adapted mesh stays sliver-free with a guaranteed minimum angle.

Typical loop::

    mesh = geom.surface_mesh(maxh=h)
    for _ in range(n_adapt):
        eta = solver.error_indicators(mesh)      # a-posteriori, per triangle
        mesh = refine_dorfler(geom, mesh, eta, theta=0.5, maxh=h)
"""
import numpy as np


def dorfler_mark(eta, theta=0.5):
    """Doerfler (bulk) marking. Returns the indices of the smallest element set
    whose summed SQUARED indicator reaches ``theta`` of the total
    (``theta`` in (0, 1]; 0.5 is the common choice). The squared convention
    treats ``eta`` as an energy-norm error contribution per element."""
    eta = np.abs(np.asarray(eta, dtype=float))
    e2 = eta * eta
    total = float(e2.sum())
    if total <= 0.0 or eta.size == 0:
        return np.empty(0, dtype=int)
    order = np.argsort(e2)[::-1]
    cum = np.cumsum(e2[order])
    k = int(np.searchsorted(cum, theta * total)) + 1
    return np.sort(order[:k])


def _tri_local_h(points, faces):
    """Mean edge length per triangle -- the local element size."""
    p = np.asarray(points, float)
    f = np.asarray(faces)
    e = np.stack([
        np.linalg.norm(p[f[:, 1]] - p[f[:, 0]], axis=1),
        np.linalg.norm(p[f[:, 2]] - p[f[:, 1]], axis=1),
        np.linalg.norm(p[f[:, 0]] - p[f[:, 2]], axis=1),
    ], axis=1)
    return e.mean(axis=1)


def mark_size_field(geom, mesh, eta, *, theta=0.5, factor=2.0, h_min=0.0):
    """Doerfler-mark by ``eta`` and register the marked elements as point size
    sources at ``local_h / factor`` (clamped to ``h_min`` if > 0). Mutates the
    geometry's size field and returns the marked element indices; re-mesh
    afterwards (e.g. ``geom.surface_mesh(...)``) to realise the refinement."""
    p = np.asarray(mesh.points, float)
    f = np.asarray(mesh.faces)
    marked = dorfler_mark(eta, theta)
    if marked.size == 0:
        return marked
    hloc = _tri_local_h(p, f)[marked] / float(factor)
    if h_min > 0.0:
        hloc = np.maximum(hloc, h_min)
    centroids = p[f[marked]].mean(axis=1)
    geom.refine_near_points([tuple(c) for c in centroids], hloc.tolist())
    return marked


def refine_dorfler(geom, mesh, eta, *, theta=0.5, factor=2.0, h_min=0.0, **mesh_kw):
    """One ESTIMATE -> MARK -> REFINE step: Doerfler-mark, build the background
    size field, and return the remeshed surface. ``mesh_kw`` is forwarded to
    :meth:`Geometry.surface_mesh` (e.g. ``maxh``, ``grading``,
    ``target_triangles``). The mesher's grading limits the size gradient and the
    Ruppert refinement keeps the result sliver-free.

    ``eta`` must be a proper a-posteriori estimator -- the per-element error,
    which SHRINKS as an element refines (e.g. ``area * gradient``, ``h**k``).
    Then the marked set stays small and the loop converges. A pointwise field
    that does NOT shrink on refinement (e.g. a bare ``exp(-r**2)``) makes Doerfler
    re-mark the same region every pass, so the refinement never localizes -- it
    "refines everywhere". ``grading`` trades the apron width of the refined zone
    against the minimum angle: steeper grading localizes more tightly but pushes
    some triangles below the 20 deg Ruppert target (still sliver-free)."""
    mark_size_field(geom, mesh, eta, theta=theta, factor=factor, h_min=h_min)
    return geom.surface_mesh(**mesh_kw)

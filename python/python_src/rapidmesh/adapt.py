"""gmsh-style Doerfler-marking adaptive refinement for surface meshes.

This is the MARK -> REFINE half of the adaptive loop (SOLVE and ESTIMATE belong
to the solver): given per-triangle error indicators, Doerfler-mark the bulk and
turn the marked elements into a background size field -- gmsh-style -- which the
mesher's grading gradient-limits and the Ruppert refinement re-triangulates, so
every adapted mesh stays sliver-free with a guaranteed minimum angle.

The marking and size-field construction run in Rust (``rapidmesh_tet::adapt``);
these are thin wrappers so a Rust solver drives the same loop natively.

Typical loop::

    mesh = geom.surface_mesh(maxh=h)
    for _ in range(n_adapt):
        eta = solver.error_indicators(mesh)      # a-posteriori, per triangle
        mesh = refine_dorfler(geom, mesh, eta, theta=0.5, maxh=h)
"""
import numpy as np

from . import _native


def dorfler_mark(eta, theta=None):
    """Doerfler (bulk) marking. Returns the indices of the smallest element set
    whose summed SQUARED indicator reaches ``theta`` of the total
    (``theta`` in (0, 1], default 0.5). The squared convention treats ``eta``
    as an energy-norm error contribution per element."""
    return _native.dorfler_mark(np.asarray(eta, dtype=float).ravel().tolist(), theta)


def mark_size_field(geom, mesh, eta, *, theta=None, factor=None, h_min=None):
    """Doerfler-mark by ``eta`` and register the marked elements as point size
    sources at ``local_h / factor`` (default 2, clamped to ``h_min`` if > 0).
    Mutates the geometry's size field and returns the marked element indices;
    re-mesh afterwards (e.g. ``geom.surface_mesh(...)``) to realise the
    refinement."""
    return geom._native.mark_dorfler(
        mesh._native, np.asarray(eta, dtype=float).ravel().tolist(), theta, factor, h_min
    )


def refine_dorfler(geom, mesh, eta, *, theta=None, factor=None, h_min=None, **mesh_kw):
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

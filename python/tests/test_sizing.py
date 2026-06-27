"""Permutation tests for the error/size control surface: the global knobs
(maxh, tol_edge/tol_surf, maxh_edge/surf/vol), the per-dimension hierarchy
(g.edge()/g.surf()/g.region()), and the per-entity hierarchy
(g.region(..).surf(..).edge(..) with id/tag/normal/near/between selectors).

Each check asserts a MONOTONIC effect (a finer bound makes more elements, a
selector refines only its target), which is robust to absolute counts. Run with
``pytest`` or directly (``python python/tests/test_sizing.py``).
"""
import math

import numpy as np
import rapidmesh as rm


def _box(maxh, setup=None, **mesh_kw):
    g = rm.Geometry(maxh=maxh)
    g.box(4.0, 4.0, 4.0, (0.0, 0.0, 0.0))
    if setup is not None:
        setup(g)
    return g.mesh(maxh=maxh, **mesh_kw)


def _ntets(maxh, setup=None, **kw):
    return int(_box(maxh, setup, **kw).stats["n_tets"])


def _cyl_faces(tol_edge=1e-2, tol_surf=1e-2, maxh_surf=math.inf):
    g = rm.Geometry(maxh=2.0)
    g.cylinder(1.0, 2.0, position=(0.0, 0.0, 0.0))
    m = g.surface_mesh(maxh=2.0, tol_edge=tol_edge, tol_surf=tol_surf, maxh_surf=maxh_surf)
    return len(m.faces)


def _box_faces(maxh, setup=None):
    g = rm.Geometry(maxh=maxh)
    g.box(4.0, 4.0, 4.0, (0.0, 0.0, 0.0))
    if setup is not None:
        setup(g)
    return len(g.surface_mesh(maxh=maxh).faces)


_ALL_BOX_NORMALS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _refine_all_faces(g, h=0.5):
    for n in _ALL_BOX_NORMALS:
        g.surf(normal=n).maxh = h


# ---- global knobs --------------------------------------------------------

def test_global_maxh_refines():
    assert _ntets(0.8) > _ntets(2.0)


def test_maxh_vol_refines_interior():
    assert _ntets(4.0, maxh_vol=0.8) > _ntets(4.0)


def test_maxh_edge_refines_edges():
    assert _ntets(4.0, maxh_edge=0.8) > _ntets(4.0)


def test_tol_surf_refines_curved_surface():
    assert _cyl_faces(tol_surf=1e-3) > _cyl_faces(tol_surf=1e-2)


def test_tol_edge_refines_curved_edges():
    assert _cyl_faces(tol_edge=1e-3) > _cyl_faces(tol_edge=1e-2)


def test_maxh_surf_refines_curved_surface():
    assert _cyl_faces(maxh_surf=0.2) > _cyl_faces(maxh_surf=2.0)


# ---- per-dimension hierarchy (unfiltered == the global knob) -------------

def test_g_edge_equals_maxh_edge():
    hier = _ntets(4.0, setup=lambda g: setattr(g.edge(), "maxh", 0.8))
    flat = _ntets(4.0, maxh_edge=0.8)
    assert hier == flat


def test_g_region_equals_maxh_vol():
    hier = _ntets(4.0, setup=lambda g: setattr(g.region(), "maxh", 0.8))
    flat = _ntets(4.0, maxh_vol=0.8)
    assert hier == flat


# ---- per-entity hierarchy ------------------------------------------------

def _points_on_edge(mesh, a, b):
    P = np.asarray(mesh.points)
    ab = np.array(b) - np.array(a)
    l2 = float(ab @ ab)
    t = np.clip((P - a) @ ab / l2, 0.0, 1.0)
    q = np.array(a) + t[:, None] * ab
    return int(np.count_nonzero(np.einsum("ij,ij->i", P - q, P - q) < 1e-12))


def test_per_edge_near_refines_only_that_edge():
    # Edge along x at y=z=0 of the [0,4]^3 box; refine just it.
    a, b = (0.0, 0.0, 0.0), (4.0, 0.0, 0.0)
    base = _box(4.0)
    fine = _box(4.0, setup=lambda g: setattr(g.edge(near=(2.0, 0.0, 0.0)), "maxh", 0.4))
    assert _points_on_edge(fine, a, b) > _points_on_edge(base, a, b)


def test_per_surface_normal_selects_faces():
    # Refining the +z face only must add fewer tets than refining all faces.
    one = _ntets(4.0, setup=lambda g: setattr(g.surf(normal=(0.0, 0.0, 1.0)), "maxh", 0.5))
    allf = _ntets(4.0, setup=lambda g: setattr(g.surf(), "maxh", 0.5))
    assert one <= allf


# ---- global cap vs per-entity consistency (no path silently ignores a knob) ---

def test_global_surf_refines_volume():
    # The global surface cap must refine the VOLUME, not just the surface tiling
    # (regression guard: cap_surf was omitted from the domain sizing field).
    assert _ntets(4.0, setup=lambda g: setattr(g.surf(), "maxh", 0.5)) > 4 * _ntets(4.0)


def test_global_surf_equals_per_entity_all_faces():
    # The global cap and the equivalent per-entity override on EVERY face produce
    # the same volume field, hence the same mesh.
    glob = _ntets(4.0, setup=lambda g: setattr(g.surf(), "maxh", 0.5))
    per_entity = _ntets(4.0, setup=_refine_all_faces)
    assert glob == per_entity


def test_per_entity_surf_refines_surface_export():
    # A per-entity surf override must reach surface_mesh() too (regression guard:
    # the surface export built its domain without the per-entity overrides).
    assert _box_faces(4.0, setup=_refine_all_faces) > 4 * _box_faces(4.0)


def test_maxh_vol_refines_box_interior():
    # The global volume cap must densify the interior, not merely a boundary band.
    assert _ntets(4.0, maxh_vol=0.5) > 10 * _ntets(4.0)


def test_hierarchical_composition_runs_and_refines():
    # region -> surf(+x face) -> edge(near its y=0 edge): refine just that edge.
    a, b = (4.0, 0.0, 0.0), (4.0, 0.0, 4.0)

    def setup(g):
        g.region().maxh = 1.5
        g.region(1).surf(normal=(1.0, 0.0, 0.0)).edge(near=(4.0, 0.0, 2.0)).maxh = 0.3

    base = _box(4.0, setup=lambda g: setattr(g.region(), "maxh", 1.5))
    fine = _box(4.0, setup=setup)
    assert _points_on_edge(fine, a, b) > _points_on_edge(base, a, b)


def test_most_specific_wins():
    # Global coarse edges, one edge fine: the fine override must take effect.
    a, b = (0.0, 0.0, 0.0), (4.0, 0.0, 0.0)

    def setup(g):
        g.edge().maxh = 3.0
        g.edge(near=(2.0, 0.0, 0.0)).maxh = 0.4

    base = _box(4.0, setup=lambda g: setattr(g.edge(), "maxh", 3.0))
    fine = _box(4.0, setup=setup)
    assert _points_on_edge(fine, a, b) > _points_on_edge(base, a, b)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} sizing permutation tests passed")

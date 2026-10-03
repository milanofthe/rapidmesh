"""The bottom-up surface stage: every face meshed alone on the shared
samples of its edges."""
import collections

import numpy as np
import pytest

import rapidmesh as rm


def _stack():
    g = rm.Geometry(maxh=16.0)
    g.box(100, 80, 40)
    g.box(100, 80, 0.2, position=(0, 0, 40))
    g.plate((20, 10, 0), (0, 20, 0), (0, 0, 40), tag=5)
    return g


def test_a_thin_stack_with_a_port_closes():
    s = _stack().surface_mesh()
    F, R = np.asarray(s.faces), np.asarray(s.face_regions)
    for r in set(R.ravel()) - {0}:
        count = collections.Counter()
        for t, (a, b) in zip(F, R):
            sides = ([True, False] if a == b else [True]) if a == r else ([False] if b == r else [])
            for flip in sides:
                tt = (t[0], t[2], t[1]) if flip else tuple(t)
                for k in range(3):
                    x, y = tt[k], tt[(k + 1) % 3]
                    count[(min(x, y), max(x, y))] += 1 if x < y else -1
        assert not any(count.values()), f"region {r} is open"
    port = np.asarray(s.face_tags) == 5
    assert abs(np.asarray(s.areas())[port].sum() - 800.0) < 1e-9
    # The 0.2 layer costs its faces only.
    assert len(F) < 2000


@pytest.mark.parametrize("shape", ["cylinder", "sphere", "torus", "helix"])
def test_curved_faces_close_without_defects(shape):
    g = rm.Geometry(maxh=0.3)
    {
        "cylinder": lambda: g.cylinder(1.0, 2.0),
        "sphere": lambda: g.sphere(1.0),
        "torus": lambda: g.torus(1.0, 0.3),
        "helix": lambda: g.helix(0.8, 0.5, 2.5, 0.15),
    }[shape]()
    m = g.mesh()
    d = m.diagnostics
    assert d["watertight"]
    assert all(x["kind"] == "sliver" for x in d["defects"])
    assert m.stats["min_dihedral_deg"] > 10.0


def test_a_thin_stack_with_a_port_fills_region_by_region():
    m = _stack().mesh()
    d = m.diagnostics
    assert d["watertight"]
    assert all(x["kind"] == "sliver" for x in d["defects"])
    T = np.asarray(m.tets)
    count = collections.Counter()
    for t in T:
        for k in range(4):
            count[tuple(sorted(np.delete(t, k)))] += 1
    assert max(count.values()) <= 2


def test_a_box_is_refined_to_its_size_and_improved():
    g = rm.Geometry(maxh=0.2)
    g.box(2, 3, 1)
    m = g.mesh()
    d = m.diagnostics
    assert d["watertight"] and not d["defects"]
    assert m.stats["min_dihedral_deg"] > 15.0
    P, T = np.asarray(m.points), np.asarray(m.tets)
    E = np.concatenate([np.linalg.norm(P[T[:, i]] - P[T[:, j]], axis=1) for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]])
    assert 0.15 < np.median(E) < 0.3


def _volumes(m):
    P, T, R = np.asarray(m.points), np.asarray(m.tets), np.asarray(m.tet_regions)
    a, b, c, d = (P[T[:, i]] for i in range(4))
    v = np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0
    return {int(r): float(v[R == r].sum()) for r in set(R.tolist())}


def test_a_contact_wedge_is_filled():
    g = rm.Geometry(maxh=0.25)
    g.box(3, 3, 2, position=(-1.5, -1.5, 0))
    g.cylinder(0.5, 2, position=(-0.5, 0, 0), maxh=0.12)
    g.cylinder(0.5, 2, position=(0.5, 0, 0), maxh=0.12)
    m = g.mesh()
    d = m.diagnostics
    assert d["watertight"]
    assert not [x for x in d["defects"] if x["kind"] != "sliver"]


def test_a_thin_gap_and_a_trace_keep_their_volumes():
    # Two plates 0.02 apart and a trace on a substrate: no contact, nothing
    # filled, so every region keeps its volume.
    g = rm.Geometry(maxh=0.5)
    g.box(4, 4, 2, position=(-2, -2, -1))
    g.box(2, 2, 0.1, position=(-1, -1, 0.02))
    g.box(2, 2, 0.1, position=(-1, -1, -0.1))
    g.box(1, 0.2, 0.05, position=(-0.5, -0.1, 0.12))
    m = g.mesh()
    v = sorted(_volumes(m).values())
    expect = sorted([2 * 2 * 0.1, 2 * 2 * 0.1, 1 * 0.2 * 0.05, 4 * 4 * 2 - 0.8 - 0.01])
    assert np.allclose(v, expect, rtol=1e-9, atol=1e-12)


def test_periodic_sides_carry_the_same_triangles():
    g = rm.Geometry(maxh=0.6)
    g.box(2.0, 2.0, 3.0)
    g.box(2.0, 2.0, 0.5, maxh=0.3)
    g.cylinder(0.3, 3.0, position=(1.0, 1.0, 0.0), maxh=0.2)
    g.periodic(g.surf(normal=(-1, 0, 0)), g.surf(normal=(1, 0, 0)))
    m = g.mesh()
    P, F = np.asarray(m.points), np.asarray(m.faces)

    def side(x, shift):
        on = np.isclose(P[F][:, :, 0], x).all(axis=1)
        return {tuple(sorted(tuple(np.round(P[v] - shift, 9)) for v in t)) for t in F[on]}

    left, right = side(0.0, (0, 0, 0)), side(2.0, (2.0, 0, 0))
    assert left and left == right
    assert m.diagnostics["watertight"]


def test_touching_bodies_keep_their_material():
    """Two cylinders touching along a line: no volume between them, so the
    tets of the wedge around the contact stay in the region around (flat
    ones are expected there); each cylinder keeps its own volume and the
    contact line is no leak."""
    import math

    g = rm.Geometry(maxh=0.25)
    g.box(3, 3, 2, position=(-1.5, -1.5, 0))
    g.cylinder(0.5, 2, position=(-0.5, 0, 0), maxh=0.12)
    g.cylinder(0.5, 2, position=(0.5, 0, 0), maxh=0.12)
    m = g.mesh()
    p, t, r = np.asarray(m.points), np.asarray(m.tets, np.int64), np.asarray(m.tet_regions)
    x = p[t]
    v = np.abs(np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0])) / 6
    c = x.mean(1)
    exact = math.pi * 0.25 * 2
    for region, cx in ((2, -0.5), (3, 0.5)):
        k = r == region
        assert abs(v[k].sum() / exact - 1) < 0.02
        assert (np.hypot(c[k, 0] - cx, c[k, 1]) < 0.53).all()
    d = m.diagnostics
    assert d["watertight"] and d["n_loose_faces"] == 0

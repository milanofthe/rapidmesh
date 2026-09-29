"""Periodic face pairs (#31): the sides of a unit cell carry the same
triangles, and every point on a master face has its image."""
import numpy as np
import pytest
import rapidmesh as rm


def _cell():
    g = rm.Geometry(maxh=0.6)
    g.box(2.0, 2.0, 3.0)
    g.box(2.0, 2.0, 0.5, maxh=0.3)
    g.xy_plate(1.0, 1.0, position=(0.5, 0.5, 0.5), tag=3)
    return g


def test_sides_carry_the_same_points():
    g = _cell()
    sx = g.periodic(g.surf(normal=(-1, 0, 0)), g.surf(normal=(1, 0, 0)))
    sy = g.periodic(g.surf(normal=(0, -1, 0)), g.surf(normal=(0, 1, 0)))
    assert sx == (2.0, 0.0, 0.0) and sy == (0.0, 2.0, 0.0)
    m = g.mesh()
    pp = m.periodic_points
    assert pp.shape[1] == 2 and len(pp) > 0
    d = m.points[pp[:, 1]] - m.points[pp[:, 0]]
    fits = np.isclose(d, sx).all(axis=1) | np.isclose(d, sy).all(axis=1)
    assert fits.all()
    # the two sides of a pair are meshed alike: equal point counts
    on = lambda axis, v: np.isclose(m.points[:, axis], v)
    assert on(0, 0.0).sum() == on(0, 2.0).sum()
    assert on(1, 0.0).sum() == on(1, 2.0).sum()
    dg = m.diagnostics
    assert dg["watertight"] and not dg["defects"]


def test_refuses_what_does_not_fit():
    g = _cell()
    with pytest.raises(ValueError):
        g.periodic(g.surf(normal=(-1, 0, 0)), g.surf(normal=(0, 0, 1)))
    with pytest.raises(ValueError):
        g.periodic(g.region(), g.surf(normal=(1, 0, 0)))
    g.periodic(g.surf(normal=(-1, 0, 0)), g.surf(normal=(1, 0, 0)))
    with pytest.raises(ValueError):
        g.mesh(optimize=True)

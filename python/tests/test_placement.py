import math

import numpy as np
import pytest

import rapidmesh as rm


def volume(m, region):
    p = np.asarray(m.points)
    t = np.asarray(m.tets)[np.asarray(m.tet_regions) == region]
    a, b, c, d = (p[t[:, k]] for k in range(4))
    return float(np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))).sum() / 6)


def test_transforms_copies_and_arrays():
    g = rm.Geometry(maxh=0.3)
    b = g.box(1, 2, 3)
    g.translate(b, dx=5)
    g.rotate(b, math.pi / 2, center=(5, 0, 0))
    g.mirror(b, normal=(0, 0, 1))
    g.stretch(b, fz=2)
    [(lo, hi)] = g._topology().region_bbox()
    assert np.allclose(lo, [3, 0, -6]) and np.allclose(hi, [5, 1, 0])
    row = g.array(b, 3, spacing=(0, 0, 10))
    assert len(row) == 3 and row[0] == b and all(isinstance(s, rm.Solid) for s in row)
    assert row[1].roles == b.roles
    m = g.mesh()
    assert all(abs(volume(m, s.region) - 12) < 1e-9 for s in row)
    with pytest.raises(ValueError):
        g.array(b, 2)
    with pytest.raises(ValueError):
        g.rotate(b, 1.0, axis=(0, 0, 0))


def test_intersect_and_extrude():
    g = rm.Geometry(maxh=0.3)
    a = g.box(2, 2, 2)
    t = g.box(2, 2, 2, position=(1, 1, 1))
    g.intersect(a, t)
    assert abs(volume(g.mesh(), a.region) - 1) < 1e-9
    g = rm.Geometry(maxh=0.3)
    s = g.xy_plate(2, 1, tag=3)
    assert isinstance(s, rm.Sheet) and s.tag == 3
    p = g.extrude(s, 1.5, axis=(0.2, 0.1, 1))
    m = g.mesh()
    assert abs(volume(m, p.region) - 3) < 1e-9
    d = g.disc(0.4, position=(5, 0, 0))
    c = g.extrude(d, 1.0)
    assert abs(volume(g.mesh(), c.region) - math.pi * 0.16) < 0.03
    with pytest.raises(ValueError, match="along its axis"):
        g.extrude(g.copy(d), 1.0, axis=(1, 0, 1))


def test_sheet_booleans_cut_and_extrude():
    g = rm.Geometry(maxh=0.3)
    plate = g.xy_plate(4, 4)
    hole = g.disc(1.0, (2, 2, 0), tag=2)
    g.sheet_boolean("difference", plate, hole)
    block = g.extrude(plate, 1.0)
    m = g.mesh()
    assert m.diagnostics["watertight"]
    assert abs(volume(m, block.region) - (16 - math.pi)) < 0.02 * math.pi
    assert 2 not in set(m.face_tags.tolist())
    with pytest.raises(ValueError):
        g.sheet_boolean("xor", plate, hole)


def test_polygon_plate_in_any_plane():
    g = rm.Geometry(maxh=0.5)
    g.polygon_plate([(0, 0), (2, 0), (2, 1), (0, 1)], (0, 3, 0), axes=((1, 0, 0), (0, 0, 1)))
    sm = g.surface_mesh()
    assert np.all(sm.points[:, 1] == 3.0)
    assert sm.points[:, 2].max() == 1.0

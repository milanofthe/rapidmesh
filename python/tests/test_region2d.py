"""The 2D region API: local width, variable inward offset, constraints, booleans."""
import math

import pytest

import rapidmesh as rm


def _area(ring):
    n = len(ring)
    return abs(sum(ring[k][0] * ring[(k + 1) % n][1] - ring[(k + 1) % n][0] * ring[k][1]
                   for k in range(n))) / 2


def test_local_width_reads_across_the_shape():
    # A 600 x 20 strip is 20 wide along its long sides. At its END the facing wall is the far
    # end, 600 away -- the ends do not make the strip narrow.
    r = rm.Region2D(outer=[(0, 0), (600, 0), (600, 20), (0, 20)])
    assert math.isclose(r.local_width((300, 0), (0, 1)), 20.0, abs_tol=1e-9)
    assert math.isclose(r.local_width((0, 10), (1, 0)), 600.0, abs_tol=1e-9)


def test_offset_chains_follow_the_local_width():
    r = rm.Region2D(outer=[(0, 0), (600, 0), (600, 20), (0, 20)])
    chains = r.offset_chains(lambda w: w / 8.0, scales=(1.0, 3.0))
    assert len(chains) == 2
    for ch in chains:
        assert ch[0] == ch[-1], "a row around a simple shape closes"
    # a number instead of a callable is a plain inward buffer
    sq = rm.Region2D(outer=[(0, 0), (10, 0), (10, 10), (0, 10)])
    buf = sq.offset_chains(2.0)
    assert len(buf) == 1
    assert math.isclose(_area(buf[0][:-1]), 36.0, abs_tol=1e-9)


def test_constraints_reach_the_mesh():
    """A chain lands ON the mesh: its rows become node lines.

    Not a triangle count -- the count goes DOWN here, because the rows are filled with long
    thin elements instead of an isotropic patch. What has to hold is that the mesher put nodes
    exactly where the chain runs."""
    r = rm.Region2D(outer=[(0, 0), (600, 0), (600, 20), (0, 20)])
    plain = rm.mesh_2d([r], h=5.0)
    on_row = lambda m, y: sum(1 for p in m.points if abs(p[1] - y) < 1e-9)
    assert on_row(plain, 2.5) == 0, "nothing sits on the row without the chain"

    r.constraints = r.offset_chains(lambda w: w / 8.0, scales=(1.0, 3.0))
    banded = rm.mesh_2d([r], h=5.0)
    for y in (2.5, 7.5, 12.5, 17.5):
        assert on_row(banded, y) > 10, f"row y={y} is not in the mesh"


def test_booleans():
    a = rm.Region2D(outer=[(0, 0), (10, 0), (10, 10), (0, 10)], tag=1)
    b = rm.Region2D(outer=[(10, 0), (20, 0), (20, 10), (10, 10)], tag=2)
    merged = rm.union_regions([a, b])
    assert len(merged) == 1, "abutting regions are one shape"
    assert math.isclose(_area(merged[0][0]), 200.0, abs_tol=1e-6)
    over = rm.Region2D(outer=[(5, 0), (15, 0), (15, 10), (5, 10)])
    inter = rm.overlay_regions([a], [over], "intersect")
    assert math.isclose(_area(inter[0][0]), 50.0, abs_tol=1e-6)


def test_band_diagonals_option():
    """The edge band's diagonals alternate by default, lean one way along
    the outline with ``"along"``; an unknown value is refused."""
    strip = [(0.0, 0.0), (100.0, 0.0), (100.0, 10.0), (0.0, 10.0)]
    for lean in (None, "alternate", "along"):
        m = rm.mesh_2d([(strip, 1, [])], 5.0, band_diagonals=lean)
        assert len(m.tris) > 0
    with pytest.raises(ValueError, match="band diagonals"):
        rm.mesh_2d([(strip, 1, [])], 5.0, band_diagonals="diagonal")


def test_width_size_follows_the_trace():
    """``width_size`` caps the size by the local width: a 4 wide strip
    meshed at size 50 gets cells about 4 long."""
    import numpy as np
    strip = [(0.0, 0.0), (200.0, 0.0), (200.0, 4.0), (0.0, 4.0)]
    m = rm.mesh_2d([(strip, 1, [])], 50.0, width_size=1.0)
    p, t = np.asarray(m.points), np.asarray(m.tris)
    edges = np.concatenate([np.linalg.norm(p[t[:, k]] - p[t[:, (k + 1) % 3]], axis=1)
                            for k in range(3)])
    assert np.max(edges) < 1.5 * 4.0


def test_snap_joins_nearly_coincident_chains():
    """Two chains 0.05 apart at size 1 snap into one with ``snap``."""
    import numpy as np
    r = rm.Region2D(outer=[(0, 0), (20, 0), (20, 10), (0, 10)])
    r.constraints = [[(2, 3), (10, 3), (18, 3)], [(2, 3.05), (18, 3.05)]]
    worst = lambda snap: float(np.min(rm.mesh_2d([r], 1.0, snap=snap).min_angles()))
    assert worst(None) < 10.0
    assert worst(0.25) > 20.0

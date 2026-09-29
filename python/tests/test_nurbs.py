import numpy as np
import pytest

import rapidmesh as rm


def bump(height=0.3):
    c = np.zeros((5, 5, 3))
    for i in range(5):
        for j in range(5):
            lift = height if 0 < i < 4 and 0 < j < 4 else 0.0
            c[i, j] = (0.4 + 0.3 * i, 0.4 + 0.3 * j, 1.0 + lift)
    return c


def test_nurbs_plate_meshes_clean():
    g = rm.Geometry(maxh=0.25)
    g.box(2, 2, 2)
    g.nurbs_plate(bump(), tag=5)
    m = g.mesh()
    assert m.diagnostics["n_slivers"] == 0
    assert not m.diagnostics["defects"]
    assert m.stats["min_dihedral_deg"] > 15


def test_nurbs_plate_rejects_a_bad_net():
    g = rm.Geometry(maxh=0.25)
    with pytest.raises(Exception, match="degrees below"):
        g.nurbs_plate(np.zeros((2, 2, 3)), degree=(3, 3))
    with pytest.raises(ValueError):
        g.nurbs_plate(np.zeros((4, 4)))

import pytest

import rapidmesh as rm


def test_chamfer_in_air_and_countersink():
    g = rm.Geometry(maxh=0.15)
    air = g.box(3, 3, 2, position=(-1.5, -1.5, -0.5))
    block = g.box(1.0, 1.0, 0.5, position=(-0.5, -0.5, 0.0))
    hole = g.cylinder(0.2, 1.0, position=(0, 0, -0.25), void=True)
    block = g.chamfer(block, 0.08, edges=[("+z", "+x"), ("+z", "-x")])
    [sink] = g.chamfer(block, 0.05, edges=("+z", hole, "side"), void=True)
    assert [r for r in block.roles if r.startswith("chamfer")] == ["chamfer0", "chamfer1"]
    g.surf(solid=block, role="chamfer0").name = "bevel"
    g.surf(solid=sink, role="chamfer").name = "sink"
    m = g.mesh()
    assert not m.diagnostics["defects"]
    assert m.diagnostics["n_slivers"] == 0
    assert len(m.sets()["faces"]["bevel"]) > 0
    assert len(m.sets()["faces"]["sink"]) > 0


def test_chamfer_rejects_what_it_cannot_cut():
    g = rm.Geometry(maxh=0.3)
    s = g.prism([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)], 1.0)
    with pytest.raises(ValueError, match="not convex"):
        g.chamfer(s, 0.1)
    b = g.box(1, 1, 1, position=(5, 0, 0))
    with pytest.raises(ValueError, match="no edge"):
        g.chamfer(b, 0.1, edges=("+z", "-z"))
    with pytest.raises(ValueError, match="is not one of"):
        g.chamfer(b, 0.1, edges="top")

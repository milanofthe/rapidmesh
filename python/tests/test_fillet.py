import pytest

import rapidmesh as rm


def test_fillets_mesh_clean():
    cases = {
        "cube": (0.1, lambda g: g.fillet(g.box(1, 1, 1), 0.2)),
        "rod": (0.08, lambda g: g.fillet(g.cylinder(0.5, 1.0), 0.15, edges=("side", "top"))),
    }
    for name, (maxh, build) in cases.items():
        g = rm.Geometry(maxh=maxh)
        build(g)
        m = g.mesh()
        assert not m.diagnostics["defects"], name
        assert m.diagnostics["n_slivers"] == 0, name
        assert m.stats["min_dihedral_deg"] > 15, name


def test_fillet_roles_and_rounded_bore():
    g = rm.Geometry(maxh=0.1)
    g.box(3, 3, 2, position=(-1.5, -1.5, -0.5))
    plate = g.box(1.2, 1.2, 0.5, position=(-0.6, -0.6, 0.0))
    bore = g.cylinder(0.25, 1.0, position=(0, 0, -0.25), void=True)
    plate = g.fillet(plate, 0.1, edges=[("+z", "+x"), ("+z", "-x")])
    [rim] = g.fillet(plate, 0.06, edges=("+z", bore, "side"), void=True)
    assert [r for r in plate.roles if r.startswith("fillet")] == ["fillet0", "fillet1"]
    g.surf(solid=rim, role="fillet").name = "round"
    m = g.mesh()
    assert not m.diagnostics["defects"]
    assert len(m.sets()["faces"]["round"]) > 0


def test_fillet_rejects_what_it_cannot_round():
    g = rm.Geometry(maxh=0.3)
    s = g.prism([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)], 1.0)
    with pytest.raises(ValueError, match="not convex"):
        g.fillet(s, 0.1)
    b = g.sphere(0.5, position=(5, 0, 0))
    with pytest.raises(ValueError, match="no edge"):
        g.fillet(b, 0.1)

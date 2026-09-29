import pytest

import rapidmesh as rm


def test_revolved_profiles_mesh_clean():
    profiles = {
        "coax": [(0.3, 0), (1.0, 0), (1.0, 1.0), (0.6, 1.0), (0.6, 2.0), (0.3, 2.0)],
        "vase": [(0, 0), (0.8, 0), rm.Spline([(1.1, 0.6), (0.5, 1.3)]), (0.4, 2.0), (0, 2.0)],
        "ring": [(1.0, -0.3, 1.0), (1.0, 0.3, 1.0)],
    }
    for name, profile in profiles.items():
        g = rm.Geometry(maxh=0.25)
        g.revolve(profile)
        m = g.mesh()
        assert m.diagnostics["n_slivers"] == 0, name
        assert not m.diagnostics["defects"], name
        assert m.stats["min_dihedral_deg"] > 15, name


def test_revolve_roles_and_part_turns():
    g = rm.Geometry(maxh=0.3)
    s = g.revolve([(0, 0), (1.0, 0), (1.0, 1.0), (0, 1.0)], angle=90)
    assert s.roles == ("edge0", "edge1", "edge2", "edge3", "start", "end")
    g.surf(solid=s, role="edge1").name = "barrel"
    g.surf(solid=s, role="start").name = "cut"
    sets = g.mesh().sets()["faces"]
    assert len(sets["barrel"]) > 0 and len(sets["cut"]) > 0


def test_revolve_rejects_bad_input():
    g = rm.Geometry(maxh=0.3)
    with pytest.raises(ValueError):
        g.revolve([(-1.0, 0), (1.0, 0), (1.0, 1.0)])
    with pytest.raises(ValueError):
        g.revolve([rm.Spline([(1, 1), (2, 2)]), (0, 0), (1, 0)])

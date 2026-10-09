"""The solver view of a mesh (#29): topology with orientation signs and face
permutations, the classification of faces and edges on the geometry, named
sets, and the MSH/VTU files (the MSH read back by gmsh when installed)."""
import numpy as np
import pytest
import rapidmesh as rm

FACE_PERMS = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2))
TET_FACE_LOCAL = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
TET_EDGE_LOCAL = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _mesh():
    """A dielectric block in an air box with a tagged port sheet."""
    g = rm.Geometry(maxh=0.35)
    air = g.box(3.0, 2.0, 2.0, position=(0.0, 0.0, 0.0))
    diel = g.box(1.0, 1.0, 0.5, position=(1.0, 0.5, 0.0))
    g.label(air, "air")
    g.label(diel, "substrate")
    g.xz_plate(0.8, 1.0, position=(2.1, 1.0, 0.5), tag=7)
    g.label(7, "port")
    return g.mesh()


@pytest.fixture(scope="module")
def mesh():
    return _mesh()


def test_incidence_signs_and_permutations(mesh):
    t = mesh.topology
    tets = mesh.tets.astype(np.int64)
    # Edges: the global edge of a local edge is its sorted pair, the sign
    # tells the direction.
    for k, (a, b) in enumerate(TET_EDGE_LOCAL):
        e = t["edges"][t["tet_edges"][:, k]]
        va, vb = tets[:, a], tets[:, b]
        assert np.array_equal(e[:, 0], np.minimum(va, vb))
        assert np.array_equal(t["tet_edge_sign"][:, k] > 0, va < vb)
    # Faces: the permutation maps the ascending face to the local order.
    perms = np.array(FACE_PERMS)
    for k, loc in enumerate(TET_FACE_LOCAL):
        f = t["faces"][t["tet_faces"][:, k]]
        p = perms[t["tet_face_perm"][:, k]]
        back = np.take_along_axis(f, p, axis=1)
        assert np.array_equal(back, tets[:, list(loc)])
        assert np.array_equal(t["tet_face_perm"][:, k] < 3, t["tet_face_sign"][:, k] > 0)
    # A face between two tets is seen with opposite signs.
    inner = t["face_tets"][:, 1] >= 0
    ft = t["face_tets"][inner]
    s = []
    for side in (0, 1):
        tt = ft[:, side]
        k = np.argmax(t["tet_faces"][tt] == np.flatnonzero(inner)[:, None], axis=1)
        s.append(t["tet_face_sign"][tt, k])
    assert np.all(s[0] == -s[1])
    assert np.all(t["volume"] > 0)


def test_classification(mesh):
    t = mesh.topology
    fr = t["face_regions"]
    # Every boundary and every interface face lies on a geometric face; a
    # face inside one region does only on an embedded sheet.
    on = t["face_patch"] >= 0
    assert np.all(on[fr[:, 0] != fr[:, 1]])
    sheet = on & (fr[:, 0] == fr[:, 1])
    assert np.all(t["face_tag"][sheet] == 7)
    assert sheet.sum() > 0
    # Edges on geometric edges have their ends classified on vertices or
    # on that edge.
    assert (t["edge_curve"] >= 0).sum() > 0


def test_sets(mesh):
    sets = mesh.sets()
    cells = sets["cells"]
    assert set(cells) == {"air", "substrate"}
    assert len(cells["air"]) + len(cells["substrate"]) == len(mesh.tets)
    port = sets["faces"]["port"]
    t = mesh.topology
    area = t["face_area"][port].sum()
    assert abs(area - 0.8) < 1e-9, area
    assert len(sets["faces"]["boundary"]) > 0


def test_msh_reads_back(mesh, tmp_path):
    gmsh = pytest.importorskip("gmsh")
    path = mesh.write_msh(tmp_path / "m.msh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        tags, _, _ = gmsh.model.mesh.getNodes()
        used = np.unique(mesh.tets)
        assert len(tags) == len(used)
        types, etags, _ = gmsh.model.mesh.getElements(3)
        assert sum(len(e) for e in etags) == len(mesh.tets)
        names = {gmsh.model.getPhysicalName(d, g) for d, g in gmsh.model.getPhysicalGroups()}
        assert {"air", "substrate", "port"} <= names
        # The port group holds the sheet triangles.
        port = [g for d, g in gmsh.model.getPhysicalGroups(2) if gmsh.model.getPhysicalName(2, g) == "port"]
        ents = gmsh.model.getEntitiesForPhysicalGroup(2, port[0])
        n = sum(len(gmsh.model.mesh.getElements(2, int(e))[1][0]) for e in ents)
        assert n == len(mesh.sets()["faces"]["port"])
    finally:
        gmsh.finalize()


def test_vtu(mesh, tmp_path):
    path = mesh.write_vtu(tmp_path / "m.vtu")
    text = path.read_text()
    assert f'NumberOfCells="{len(mesh.tets) + len(mesh.faces)}"' in text


def _tee():
    """Two sheets meeting in a T: the upright one stands on a line inside
    the flat one, so three triangles meet at every edge of that line."""
    g = rm.Geometry(maxh=0.25)
    g.xy_plate(2.0, 1.0, position=(0.0, 0.0, 0.0), tag=1)
    g.xz_plate(2.0, 1.0, position=(0.0, 0.5, 0.0), tag=2)
    g.label(1, "ground")
    g.label(2, "wall")
    return g.surface_mesh()


def test_surface_junctions_and_sets(tmp_path):
    sm = _tee()
    t = sm.topology
    off, tris = t["edge_tris_offsets"], t["edge_tris"]
    count = np.diff(off)
    junction = np.flatnonzero(count == 3)
    assert len(junction) > 0
    # Joined across the tags, every junction edge carries two functions.
    rwg = sm.rwg_edges(connect_tags=True)
    per_edge = {}
    for v0, v1, _, _ in rwg:
        per_edge[(v0, v1)] = per_edge.get((v0, v1), 0) + 1
    for e in junction:
        a, b = t["edges"][e]
        assert per_edge[(a, b)] == 2
    # Apart, the junction keeps the pair of the ground sheet only.
    apart = sm.rwg_edges()
    assert len(apart) == len(rwg) - len(junction)
    # The junction edges lie on a geometric edge; the sets follow the tags.
    assert np.all(t["edge_curve"][junction] >= 0)
    sets = sm.sets()
    area = t["area"]
    assert abs(area[sets["faces"]["ground"]].sum() - 2.0) < 1e-9
    assert abs(area[sets["faces"]["wall"]].sum() - 2.0) < 1e-9
    # Barycentric gradients sum to zero on every triangle.
    assert np.abs(t["grad"].sum(axis=1)).max() < 1e-9


def test_surface_msh_reads_back(tmp_path):
    gmsh = pytest.importorskip("gmsh")
    sm = _tee()
    path = sm.write_msh(tmp_path / "s.msh")
    sm.write_vtu(tmp_path / "s.vtu")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        _, et, _ = gmsh.model.mesh.getElements(2)
        assert sum(len(e) for e in et) == len(sm.faces)
        names = {gmsh.model.getPhysicalName(d, g) for d, g in gmsh.model.getPhysicalGroups()}
        assert {"ground", "wall"} <= names
    finally:
        gmsh.finalize()


def test_named_faces_and_edges(tmp_path):
    """Ports on the end faces of a block and a named edge: sets and physical
    groups, the end faces counted by area."""
    g = rm.Geometry(maxh=0.3)
    g.box(3.0, 1.0, 0.5, position=(0.0, 0.0, 0.0))
    g.surf(normal=(-1, 0, 0)).name = "port1"
    g.surf(normal=(1, 0, 0)).name = "port2"
    g.edge(near=(1.5, 0.0, 0.0)).name = "rail"
    m = g.mesh()
    t = m.topology
    sets = m.sets()
    for name in ("port1", "port2"):
        assert abs(t["face_area"][sets["faces"][name]].sum() - 0.5) < 1e-9
    rail = sets["edges"]["rail"]
    e = t["edges"][rail]
    p = m.points[e]
    assert np.allclose(p[:, :, 1], 0.0) and np.allclose(p[:, :, 2], 0.0)
    assert abs(np.linalg.norm(p[:, 0] - p[:, 1], axis=1).sum() - 3.0) < 1e-9
    with pytest.raises(ValueError):
        g.surf(normal=(1, 1, 1), normal_tol=0.99).name = "nothing"
    gmsh = pytest.importorskip("gmsh")
    path = m.write_msh(tmp_path / "ports.msh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        groups = {gmsh.model.getPhysicalName(d, g): (d, g) for d, g in gmsh.model.getPhysicalGroups()}
        assert groups["port1"][0] == 2 and groups["rail"][0] == 1
        d, gtag = groups["port2"]
        ents = gmsh.model.getEntitiesForPhysicalGroup(d, gtag)
        n = sum(len(gmsh.model.mesh.getElements(2, int(x))[1][0]) for x in ents)
        assert n == len(sets["faces"]["port2"])
    finally:
        gmsh.finalize()


def test_a_region_left_out_leaves_its_walls(mesh, tmp_path):
    sub = mesh.sets()["cells"]["substrate"]
    region = int(mesh.tet_regions[sub[0]])
    m = mesh.without_regions([region])
    assert set(m.tet_regions.tolist()) == set(mesh.tet_regions.tolist()) - {region}
    assert len(m.tets) == len(mesh.tets) - len(sub)
    assert len(np.unique(m.tets)) == len(m.points)
    assert "substrate" not in m.sets()["cells"]
    assert (m.face_regions == 0).any(axis=1).any()
    assert m.diagnostics["watertight"]
    back = rm.load_msh(m.write_msh(tmp_path / "holes.msh"))
    assert len(back.tets) == len(m.tets)
    assert set(back.sets()["cells"]) == {"air"}

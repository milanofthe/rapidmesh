import numpy as np
import pytest

import rapidmesh as rm


def test_a_written_mesh_loads_back(tmp_path):
    g = rm.Geometry(maxh=0.4)
    air = g.box(2, 2, 2)
    g.label(air, "air")
    rod = g.cylinder(0.3, 1.0, position=(1, 1, 0.5))
    g.label(rod, "rod")
    m = g.mesh()
    path = tmp_path / "cell.msh"
    m.write_msh(path)
    back = rm.load_msh(path)
    assert len(back.tets) == len(m.tets)
    assert np.allclose(np.sort(np.asarray(back.points), axis=0), np.sort(np.asarray(m.points), axis=0))
    cells = back.sets()["cells"]
    assert len(cells["rod"]) == len(m.sets()["cells"]["rod"])
    again = tmp_path / "again.msh"
    back.write_msh(again)
    assert again.read_text() == path.read_text(), "the rewritten file differs"


def test_gmsh_files_load(tmp_path):
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [1], 7, name="block")
        gmsh.model.addPhysicalGroup(2, [1, 2], 9, name="sides")
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)
        gmsh.model.mesh.generate(3)
        path = tmp_path / "box.msh"
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    m = rm.load_msh(path)
    p = np.asarray(m.points)
    t = np.asarray(m.tets)
    a, b, c, d = (p[t[:, k]] for k in range(4))
    # The mesher's orientation (det of the edge vectors negative).
    vol = -np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6
    assert (vol > 0).all() and abs(vol.sum() - 1) < 1e-9
    assert len(m.sets()["cells"]["block"]) == len(t)
    assert len(m.sets()["faces"]["sides"]) > 0


def test_written_tets_are_positive_for_gmsh(tmp_path):
    """Read with a minimal MSH 4.1 parser, independent of rapidmesh: every
    tet has a positive volume in gmsh's convention."""
    g = rm.Geometry(maxh=0.4)
    g.sphere(1.0)
    path = tmp_path / "ball.msh"
    g.mesh().write_msh(path)
    lines = iter(path.read_text().splitlines())
    nodes, tets = {}, []
    for line in lines:
        if line == "$Nodes":
            blocks = int(next(lines).split()[0])
            for _ in range(blocks):
                n = int(next(lines).split()[3])
                tags = [int(next(lines)) for _ in range(n)]
                for t in tags:
                    nodes[t] = np.array([float(x) for x in next(lines).split()])
        if line == "$Elements":
            blocks = int(next(lines).split()[0])
            for _ in range(blocks):
                _, _, ty, n = map(int, next(lines).split())
                for _ in range(n):
                    el = list(map(int, next(lines).split()))[1:]
                    if ty == 4:
                        tets.append(el)
    a, b, c, d = (np.array([nodes[t[k]] for t in tets]) for k in range(4))
    vol = np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6
    assert len(tets) > 0 and (vol > 0).all()

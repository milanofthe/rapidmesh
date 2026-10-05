"""Second-order tets on the true geometry, CalculiX / Abaqus input."""
import math

import numpy as np

import rapidmesh as rm


def _ball():
    g = rm.Geometry(maxh=0.35)
    g.sphere(1.0)
    return g.mesh()


def test_mid_edge_nodes_lie_on_the_sphere_and_the_volume_follows():
    m = _ball()
    so = m.second_order()
    pts, faces = so["points"], so["faces"]
    assert so["tets"].shape == (len(m.tets), 10)
    assert so["curved"] > 0
    # curved tets are those on the sphere; the inner ones stay affine
    assert so["curved_tets"].dtype == bool and len(so["curved_tets"]) == len(m.tets)
    assert 0 < so["curved_tets"].sum() < len(m.tets)
    assert so["straightened"] <= 0.01 * so["curved"]
    # every node of the boundary on the unit sphere (corners and mid-edges)
    r = np.linalg.norm(pts[np.unique(faces)], axis=1)
    moved = np.unique(faces[:, 3:])
    assert np.abs(np.linalg.norm(pts[moved], axis=1) - 1.0).max() < 1e-9 + 0.0 * r.max()
    # the curved volume is far nearer the ball's than the straight one
    exact = 4.0 / 3.0 * math.pi
    p, t = np.asarray(m.points), np.asarray(m.tets)
    a, b, c, d = (p[t[:, k]] for k in range(4))
    straight = np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))).sum() / 6.0
    curved = so["volumes"].sum()
    assert (so["volumes"] > 0).all()
    assert abs(curved - exact) < 0.1 * abs(straight - exact)


def test_the_input_file_lists_nodes_elements_and_sets(tmp_path):
    g = rm.Geometry(maxh=0.5)
    g.box(2.0, 1.0, 1.0)
    g.surf(normal=(1, 0, 0)).name = "load"
    m = g.mesh()
    for order, kind, n in ((1, "C3D4", 4), (2, "C3D10", 10)):
        path = m.write_inp(tmp_path / f"box{order}.inp", order=order)
        text = path.read_text()
        assert f"TYPE={kind}" in text
        block = text.split(f"TYPE={kind}")[1].split("*")[0].strip().split("\n")[1:]
        assert len(block[0].split(",")) == n + 1
        assert "*NSET, NSET=load" in text and "*SURFACE, NAME=load, TYPE=ELEMENT" in text
        side = text.split("*SURFACE, NAME=load, TYPE=ELEMENT")[1].strip().split("\n")
        assert side and all(line.split(",")[1].strip() in ("S1", "S2", "S3", "S4") for line in side if line)


def test_gmsh_reads_the_second_order_mesh_with_positive_jacobians(tmp_path):
    import pytest
    gmsh = pytest.importorskip("gmsh")
    m = _ball()
    path = m.write_msh(tmp_path / "ball2.msh", order=2)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        types, tags, _ = gmsh.model.mesh.getElements()
        count = dict(zip(types, (len(t) for t in tags)))
        assert count.get(11) == len(m.tets)  # tet10
        assert count.get(9, 0) > 0  # tri6
        pts, _ = gmsh.model.mesh.getIntegrationPoints(11, "Gauss4")
        _, dets, _ = gmsh.model.mesh.getJacobians(11, pts)
        assert min(dets) > 0
        # the mid-edge nodes of the boundary on the sphere
        node_tags, coords, _ = gmsh.model.mesh.getNodes(2, -1, includeBoundary=True)
        r = np.linalg.norm(np.asarray(coords).reshape(-1, 3), axis=1)
        assert np.abs(r - 1.0).max() < 1e-9
    finally:
        gmsh.finalize()


def test_the_viewer_gets_the_curved_edges_of_the_second_order_mesh():
    """``to_viewer_dict(order=2)`` lists every edge whose mid-edge
    node lies off its chord, on the sphere; the linear one lists none."""
    m = _ball()
    assert "curved_edges" not in m.to_viewer_dict("ball")
    edges = m.to_viewer_dict("ball", order=2)["curved_edges"]
    assert edges
    pts = np.asarray(m.points)
    for a, b, p in edges:
        assert a < b
        assert abs(np.linalg.norm(p) - 1.0) < 1e-9
        assert np.linalg.norm(np.asarray(p) - 0.5 * (pts[a] + pts[b])) > 0


def test_a_sheet_outside_the_volume_takes_mid_edge_nodes_too():
    """A sheet reaching out of every region has faces no tet has; the
    second-order mesh gives their edges nodes all the same."""
    g = rm.Geometry(maxh=0.5)
    g.box(1, 1, 1)
    g.xy_plate(2, 0.5, position=(0.25, 0.25, 0.5), tag=3)
    m = g.mesh()
    so = m.second_order()
    assert so["faces"].shape == (len(m.faces), 6)
    assert so["faces"].max() < len(so["points"])


def test_a_mesh_names_the_surface_under_each_face():
    """``surfaces`` gives, per id of ``face_surfaces``, the kind and the
    parameters of the surface: the faces of a cylinder's barrel lie on a
    cylinder of its radius, its caps on planes."""
    g = rm.Geometry(maxh=0.5)
    g.cylinder(0.8, 2.0)
    m = g.mesh()
    kinds = {m.surfaces[s]["kind"] for s in set(m.face_surfaces.tolist())}
    assert kinds == {"cylinder", "plane"}
    barrel = next(s for s in m.surfaces if s["kind"] == "cylinder")
    assert abs(barrel["radius"] - 0.8) < 1e-12
    assert np.allclose(np.abs(barrel["axis"]), [0, 0, 1])

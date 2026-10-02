"""OpenFOAM polyMesh output and finite volume quality."""
import re

import numpy as np

import rapidmesh as rm


def _read(path):
    """The entries of a polyMesh list file: count, then the lines between the parentheses."""
    text = path.read_text()
    body = text[text.index("}") + 1:]
    m = re.search(r"(\d+)\s*\(\n(.*)\n\)", body, re.S)
    n, rows = int(m.group(1)), m.group(2).split("\n") if m.group(2) else []
    assert len(rows) == n
    return rows


def test_a_box_writes_a_closed_poly_mesh(tmp_path):
    g = rm.Geometry(maxh=0.5)
    g.box(2.0, 1.0, 1.0)
    g.surf(normal=(1, 0, 0)).name = "outlet"
    m = g.mesh()
    out = m.write_foam(tmp_path)
    pts = np.array([[float(x) for x in r.strip("()").split()] for r in _read(out / "points")])
    faces = [[int(x) for x in r[2:-1].split()] for r in _read(out / "faces")]
    owner = [int(x) for x in _read(out / "owner")]
    nei = [int(x) for x in _read(out / "neighbour")]
    n_cells = len(m.tets)
    assert len(pts) == len(m.points) and len(faces) == len(owner)
    assert set(owner) | set(nei) == set(range(n_cells))
    # upper-triangular: owner below neighbour, sorted
    assert all(o < n for o, n in zip(owner, nei))
    assert sorted(zip(owner, nei)) == list(zip(owner, nei))
    # every cell closed: its faces' area vectors, turned out of it, sum to 0
    total = np.zeros((n_cells, 3))
    for i, f in enumerate(faces):
        a, b, c = pts[f]
        s = 0.5 * np.cross(b - a, c - a)
        total[owner[i]] += s
        if i < len(nei):
            total[nei[i]] -= s
    assert np.abs(total).max() < 1e-12
    boundary = (out / "boundary").read_text()
    assert "outlet" in boundary and "boundary" in boundary
    q = m.fvm_quality()
    assert q["severely_non_orthogonal"] == 0
    assert q["max_non_orthogonality"] < 70.0
    assert len(q["skewness"]) == len(q["non_orthogonality"])


def test_the_polyhedral_dual_is_closed_and_conserves_the_volume(tmp_path):
    g = rm.Geometry(maxh=0.2)
    g.box(2.0, 1.0, 1.0)
    m = g.mesh()
    tets = m.fvm_quality()
    poly = m.fvm_quality(polyhedral=True)
    assert poly["cells"] == len(m.points)
    assert poly["cells"] < tets["cells"] / 3
    assert abs(poly["volumes"].sum() - 2.0) < 1e-12 and (poly["volumes"] > 0).all()
    assert poly["severely_non_orthogonal"] == 0
    assert poly["max_openness"] < 1e-12 and tets["max_openness"] < 1e-12
    out = m.write_foam(tmp_path, polyhedral=True)
    faces = _read(out / "faces")
    assert len(faces) == poly["faces"]
    assert any(not f.startswith("3(") for f in faces)

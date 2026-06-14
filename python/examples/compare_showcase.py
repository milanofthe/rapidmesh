"""Mesh every comparison geometry with all three meshers and export the data
the landing page needs.

For each geometry in ``compare_geoms.GEOMS`` and each of rapidmesh / gmsh /
tetgen, this builds the same shape at the same target size, times the mesh
generation, recomputes quality uniformly (``_quality``), and writes a viewer
JSON in the standard schema to::

    site/static/meshes/compare/<id>.<mesher>.json

plus a ``compare/manifest.json`` indexing the geometries and per-mesher stats.

Honest framing baked into the data: tetgen has no CAD kernel, so it
tetrahedralizes *gmsh's surface* of the same geometry (recorded as
``on_surface_of: "gmsh"`` in its stats). rapidmesh and gmsh each run their full
native pipeline from the geometry spec. Quality is recomputed here for all
three with identical formulas, so the numbers are apples-to-apples.

Run from the repo root:

    python python/examples/compare_showcase.py [ids...]
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from _quality import quality
from compare_geoms import GEOMS, CompareGeom

OUT = Path(__file__).resolve().parents[2] / "site" / "static" / "meshes" / "compare"
MESHERS = ("rapidmesh", "gmsh", "tetgen")


# --------------------------------------------------------------- viewer JSON


def _viewer_dict(name: str, mesher: str, points, tets, q: dict, millis: int,
                 extra: dict | None = None) -> dict:
    """A mesh in the shared viewer schema. ``faces=[]``: the renderer builds
    the surface hull from the tets. ``tet_regions`` is a flat single region."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    tt = np.asarray(tets, dtype=np.int64).reshape(-1, 4)
    stats = {
        "n_points": q["n_points"],
        "n_tets": q["n_tets"],
        "min_dihedral_deg": q["min_dihedral_deg"],
        "max_radius_edge": q["max_radius_edge"],
        "max_edge": q["max_edge"],
        "millis": int(millis),
    }
    if extra:
        stats.update(extra)
    return {
        "name": name,
        "mesher": mesher,
        "points": pts.tolist(),
        "tets": tt.tolist(),
        "tet_regions": [1] * tt.shape[0],
        "faces": [],
        "stats": stats,
    }


# ------------------------------------------------------------------ meshers


def mesh_rapidmesh(geom: CompareGeom):
    """(points, tets, millis) from rapidmesh's native pipeline."""
    g = geom.build_rapidmesh()
    t0 = time.perf_counter()
    m = g.mesh(maxh=geom.target_h)
    millis = (time.perf_counter() - t0) * 1e3
    return np.asarray(m.points), np.asarray(m.tets, dtype=np.int64), millis


def _gmsh_extract():
    """Compact (points, tets) and the surface (sverts, stris) from the current
    gmsh model after generate(3). Node tags are remapped to dense indices."""
    import gmsh

    tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(tags)}

    _, tet_nodes = gmsh.model.mesh.getElementsByType(4)  # 4-node tets
    tet_nodes = np.asarray(tet_nodes, dtype=np.int64).reshape(-1, 4)
    tets = np.vectorize(tag_to_idx.get)(tet_nodes)

    _, tri_nodes = gmsh.model.mesh.getElementsByType(2)  # 3-node tris (surface)
    tri_nodes = np.asarray(tri_nodes, dtype=np.int64).reshape(-1, 3)
    tris_global = np.vectorize(tag_to_idx.get)(tri_nodes)
    # compact the surface to only its own vertices for tetgen
    used = np.unique(tris_global)
    remap = {int(g): i for i, g in enumerate(used)}
    sverts = coords[used]
    stris = np.vectorize(remap.get)(tris_global)
    return coords, tets, sverts, stris


def mesh_gmsh(geom: CompareGeom):
    """(points, tets, millis, (sverts, stris)) from gmsh's native pipeline;
    the surface mesh is returned for the tetgen run."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(geom.id)
        geom.build_gmsh(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        h = geom.target_h
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        t0 = time.perf_counter()
        gmsh.model.mesh.generate(3)
        millis = (time.perf_counter() - t0) * 1e3
        pts, tets, sverts, stris = _gmsh_extract()
        return pts, tets, millis, (sverts, stris)
    finally:
        gmsh.finalize()


def mesh_tetgen(geom: CompareGeom, surface):
    """(points, tets, millis) from tetgen on gmsh's surface triangulation."""
    import tetgen

    sverts, stris = surface
    h = geom.target_h
    # regular-tet volume of edge h ~ h^3/(6 sqrt2); allow a little slack.
    maxvol = (h ** 3) / (6 * math.sqrt(2.0)) * 1.4
    tg = tetgen.TetGen(np.asarray(sverts, dtype=np.float64),
                       np.asarray(stris, dtype=np.int32))
    t0 = time.perf_counter()
    tg.tetrahedralize(order=1, mindihedral=0.0, minratio=1.2, maxvolume=maxvol)
    millis = (time.perf_counter() - t0) * 1e3
    grid = tg.grid
    pts = np.asarray(grid.points, dtype=np.float64)
    # UnstructuredGrid cells: VTK_TETRA stored as [4, a,b,c,d, 4, ...]
    cells = np.asarray(grid.cells, dtype=np.int64).reshape(-1, 5)
    tets = cells[:, 1:]
    return pts, tets, millis


# --------------------------------------------------------------------- main


def main(argv: list[str]) -> None:
    wanted = set(argv) if argv else None
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_geoms = []

    for geom in GEOMS:
        if wanted and geom.id not in wanted:
            continue
        per_mesher: dict[str, dict] = {}
        surface = None

        # rapidmesh (a pyo3 PanicException is a BaseException, so catch broadly
        # to isolate a single geometry's robustness failure from the run)
        try:
            pts, tets, ms = mesh_rapidmesh(geom)
            q = quality(pts, tets)
            d = _viewer_dict(geom.name, "rapidmesh", pts, tets, q, ms)
            (OUT / f"{geom.id}.rapidmesh.json").write_text(json.dumps(d))
            per_mesher["rapidmesh"] = {"file": f"meshes/compare/{geom.id}.rapidmesh.json",
                                       "stats": d["stats"]}
            print(f"{geom.id:<14} rapidmesh {q['n_tets']:>7} tets  "
                  f"min-dih {q['min_dihedral_deg']:5.1f}  {ms:7.0f} ms")
        except BaseException as e:  # noqa: BLE001
            msg = str(e).splitlines()[0] if str(e) else ""
            print(f"{geom.id:<14} rapidmesh FAILED: {type(e).__name__}: {msg}")

        # gmsh (also yields the surface for tetgen)
        try:
            pts, tets, ms, surface = mesh_gmsh(geom)
            q = quality(pts, tets)
            d = _viewer_dict(geom.name, "gmsh", pts, tets, q, ms)
            (OUT / f"{geom.id}.gmsh.json").write_text(json.dumps(d))
            per_mesher["gmsh"] = {"file": f"meshes/compare/{geom.id}.gmsh.json",
                                  "stats": d["stats"]}
            print(f"{geom.id:<14} gmsh      {q['n_tets']:>7} tets  "
                  f"min-dih {q['min_dihedral_deg']:5.1f}  {ms:7.0f} ms")
        except Exception as e:  # noqa: BLE001
            print(f"{geom.id:<14} gmsh      FAILED: {type(e).__name__}: {e}")

        # tetgen on gmsh's surface
        if surface is not None:
            try:
                pts, tets, ms = mesh_tetgen(geom, surface)
                q = quality(pts, tets)
                d = _viewer_dict(geom.name, "tetgen", pts, tets, q, ms,
                                 extra={"on_surface_of": "gmsh"})
                (OUT / f"{geom.id}.tetgen.json").write_text(json.dumps(d))
                per_mesher["tetgen"] = {"file": f"meshes/compare/{geom.id}.tetgen.json",
                                        "stats": d["stats"]}
                print(f"{geom.id:<14} tetgen    {q['n_tets']:>7} tets  "
                      f"min-dih {q['min_dihedral_deg']:5.1f}  {ms:7.0f} ms")
            except Exception as e:  # noqa: BLE001
                print(f"{geom.id:<14} tetgen    FAILED: {type(e).__name__}: {e}")

        if per_mesher:
            manifest_geoms.append({
                "id": geom.id,
                "name": geom.name,
                "category": geom.category,
                "target_h": geom.target_h,
                "meshers": per_mesher,
            })
        sys.stdout.flush()

    # rebuild the manifest from whatever JSON is present (so partial runs keep
    # the site consistent); merge with any geometries we did not touch.
    existing = {}
    mpath = OUT / "manifest.json"
    if mpath.exists() and wanted:
        for g in json.loads(mpath.read_text()).get("geometries", []):
            existing[g["id"]] = g
    for g in manifest_geoms:
        existing[g["id"]] = g
    ordered = [existing[gd.id] for gd in GEOMS if gd.id in existing]
    mpath.write_text(json.dumps({"geometries": ordered}, indent=1))
    print(f"manifest: {len(ordered)} geometries")


if __name__ == "__main__":
    main(sys.argv[1:])

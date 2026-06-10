"""Meshes the comparison geometries with gmsh and tetgen and writes viewer
JSONs (gmsh_<name>.json / tetgen_<name>.json) in the rapidmesh schema, so the
viewer shows the meshers side by side.

tetgen consumes the IDENTICAL assembled PLC (bench/plc/<name>.json, written by
the rapidmesh exporters) with region seeds and per-region size constraints.
gmsh cannot mesh a discrete PLC without reparametrization, so the scenes are
rebuilt analytically with OCC (same primitives, same sizes); embedded sheets
go in via fragment, per-region sizing via fields.

Size conversion: rapidmesh maxh is a target max edge length; the tetgen
per-region constraint is a max tet VOLUME, converted as the volume of the
regular tet with edge maxh, h^3/(6*sqrt(2)).

Usage: python bench/compare_meshers.py [--mesher gmsh|tetgen|both] [names...]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

BENCH = Path(__file__).parent
MESHES = BENCH.parent / "viewer" / "public" / "meshes"
PLC = BENCH / "plc"

RADIUS_EDGE_BOUND = 2.0


# --------------------------------------------------------------- quality


def quality_stats(points: np.ndarray, tets: np.ndarray) -> dict:
    """min dihedral (deg), max radius-edge, max edge, matching rapidmesh's
    quality_stats definitions."""
    p = points[tets]  # (n, 4, 3)
    # Edges (6 per tet).
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    evecs = np.stack([p[:, b] - p[:, a] for a, b in pairs], axis=1)  # (n,6,3)
    elens = np.linalg.norm(evecs, axis=2)  # (n, 6)
    max_edge = float(elens.max())
    # Circumradius: solve 2 (p_i - p_0) . c = |p_i|^2 - |p_0|^2.
    a = 2.0 * (p[:, 1:] - p[:, :1])  # (n, 3, 3)
    b = (p[:, 1:] ** 2).sum(axis=2) - (p[:, :1] ** 2).sum(axis=2)  # (n, 3)
    cc = np.linalg.solve(a, b[..., None])[..., 0]  # (n, 3)
    cr = np.linalg.norm(cc - p[:, 0], axis=1)  # (n,)
    max_re = float((cr / elens.min(axis=1)).max())
    # Dihedral angle at each of the 6 edges, projection-based exactly like
    # the Rust quality_stats: angle between the projections of the two
    # opposite vertices onto the plane normal to the edge.
    min_dih = math.inf
    others = {(0, 1): (2, 3), (0, 2): (1, 3), (0, 3): (1, 2),
              (1, 2): (0, 3), (1, 3): (0, 2), (2, 3): (0, 1)}
    for (i, j), (k, l) in others.items():
        a, b = p[:, i], p[:, j]
        t = b - a
        t = t / np.linalg.norm(t, axis=1, keepdims=True)
        u = p[:, k] - a
        u = u - (u * t).sum(axis=1, keepdims=True) * t
        v = p[:, l] - a
        v = v - (v * t).sum(axis=1, keepdims=True) * t
        nu = np.linalg.norm(u, axis=1)
        nv = np.linalg.norm(v, axis=1)
        ok = nu * nv > 0.0
        cosang = np.clip((u[ok] * v[ok]).sum(axis=1) / (nu[ok] * nv[ok]), -1.0, 1.0)
        min_dih = min(min_dih, float(np.degrees(np.arccos(cosang)).min()))
    return {
        "min_dihedral_deg": min_dih,
        "max_radius_edge": max_re,
        "max_edge": max_edge,
    }


# ----------------------------------------------------------- face deriva.


def orient_tets(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Reorders tets into the rapidmesh orientation convention (the viewer
    winds tet faces assuming it; wrong-handed tets shade flat)."""
    p = points[tets]
    vol = np.einsum(
        "ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])
    )
    out = tets.copy()
    flip = vol > 0.0
    out[flip, 2], out[flip, 3] = tets[flip, 3], tets[flip, 2]
    return out


def derive_faces(points: np.ndarray, tets: np.ndarray, regions: np.ndarray) -> list:
    """Boundary and region-interface triangles from the tet mesh (tag 0),
    wound like rapidmesh faces: the normal points away from the
    higher-priority (inner) region, regions = [front, back]."""
    n = tets.shape[0]
    faces = {}
    face_idx = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
    for ti in range(n):
        t = tets[ti]
        for fi, (i, j, k) in enumerate(face_idx):
            key = tuple(sorted((int(t[i]), int(t[j]), int(t[k]))))
            faces.setdefault(key, []).append((ti, fi))
    out = []
    for key, owners in faces.items():
        if len(owners) == 1:
            inner, outer = owners[0], None
            r_in, r_out = int(regions[owners[0][0]]), 0
        else:
            (t0, f0), (t1, f1) = owners
            if int(regions[t0]) == int(regions[t1]):
                continue
            if int(regions[t0]) > int(regions[t1]):
                inner, r_in, r_out = (t0, f0), int(regions[t0]), int(regions[t1])
            else:
                inner, r_in, r_out = (t1, f1), int(regions[t1]), int(regions[t0])
        # Wind (a, b, c) so the normal points away from the inner tet's
        # opposite vertex.
        ti, fi = inner
        t = tets[ti]
        i, j, k = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)][fi]
        a, b, c = int(t[i]), int(t[j]), int(t[k])
        d = int(t[fi])
        nrm = np.cross(points[b] - points[a], points[c] - points[a])
        if float(nrm @ (points[d] - points[a])) > 0.0:
            b, c = c, b
        out.append({"tri": [a, b, c], "tag": 0, "regions": [r_out, r_in]})
    return out


def write_viewer_json(mesher, name, points, tets, regions, faces, millis):
    q = quality_stats(points, tets)
    data = {
        "name": name,
        "mesher": mesher,
        "points": points.tolist(),
        "tets": tets.tolist(),
        "tet_regions": [int(r) for r in regions],
        "faces": faces,
        "stats": {
            "n_points": int(points.shape[0]),
            "n_tets": int(tets.shape[0]),
            **q,
            "millis": int(millis),
        },
    }
    path = MESHES / f"{mesher}_{name}.json"
    path.write_text(json.dumps(data))
    print(
        f"  {mesher} {name}: {tets.shape[0]} tets, "
        f"min dihedral {q['min_dihedral_deg']:.1f} deg, "
        f"radius-edge {q['max_radius_edge']:.2f}, {int(millis)} ms -> {path.name}"
    )


# ------------------------------------------------------------------ tetgen


def maxh_to_volume(h: float) -> float:
    return h**3 / (6.0 * math.sqrt(2.0))


def run_tetgen(name: str) -> None:
    import tetgen

    spec = json.loads((PLC / f"{name}.json").read_text())
    points = np.asarray(spec["vertices"], dtype=np.float64)
    tris = np.asarray(spec["triangles"], dtype=np.int32)
    tg = tetgen.TetGen(points, tris)
    has_volume = any(r["maxh"] is not None for r in spec["regions"])
    for r in spec["regions"]:
        vol = maxh_to_volume(r["maxh"]) if r["maxh"] is not None else 0.0
        tg.add_region(int(r["tag"]), tuple(r["seed"]), vol)
    switches = f"pzQq{spec['radius_edge_bound']}A" + ("a" if has_volume else "")
    t0 = time.perf_counter()
    out = tg.tetrahedralize(switches=switches)
    millis = (time.perf_counter() - t0) * 1000.0
    nodes, elems, attrib = out[0], out[1], out[2]
    regions = np.asarray(attrib).ravel().astype(np.int64)
    if regions.size != elems.shape[0]:
        regions = np.ones(elems.shape[0], dtype=np.int64)
    elems = orient_tets(np.asarray(nodes), np.asarray(elems))
    faces = derive_faces(np.asarray(nodes), elems, regions)
    write_viewer_json("tetgen", name, nodes, elems, regions, faces, millis)


# ------------------------------------------------------------------- gmsh


def gmsh_scene(name: str, gmsh) -> tuple:
    """Builds scene `name` with OCC. Returns (solids, sheets, maxh) where
    solids = [(vol_tag, region_tag, region_maxh|None)] in rapidmesh priority
    order (later solids win) and sheets = [surface tags pre-fragment]."""
    occ = gmsh.model.occ

    def box(p0, p1):
        return occ.addBox(p0[0], p0[1], p0[2], p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])

    def rect(p0, du, dv):
        # Axis-aligned rectangle sheet in a z = const plane.
        return occ.addRectangle(p0[0], p0[1], p0[2], du[0] + dv[0], du[1] + dv[1])

    solids = []
    sheets = []

    if name == "em_scene":
        b = box([0, 0, 0], [4, 4, 4])
        d = box([1, 1, 1], [3, 3, 2])
        solids.append((b, 1, None))
        solids.append((d, 2, 0.45))
        sheets.append(rect([1.5, 1.5, 2.0], [1, 0, 0], [0, 1, 0]))
        sheets.append(rect([0.5, 0.5, 3.0], [1, 0, 0], [0, 1, 0]))
        maxh = 0.9
    elif name == "via":
        b = box([-2, -2, 0], [2, 2, 1])
        c = occ.addCylinder(0, 0, 0, 0, 0, 1, 0.75)
        solids.append((b, 1, None))
        solids.append((c, 2, 0.3))
        maxh = 0.6
    elif name == "microstrip":
        b = box([0, 0, 0], [6, 3, 3])
        s = box([0, 0, 0], [6, 3, 0.5])
        solids.append((b, 1, None))
        solids.append((s, 2, 0.35))
        sheets.append(rect([0.0, 1.25, 0.5], [6, 0, 0], [0, 0.5, 0]))
        maxh = 0.8
    elif name == "sphere":
        b = box([-2, -2, -2], [2, 2, 2])
        s = occ.addSphere(0, 0, 0, 1.0)
        solids.append((b, 1, None))
        solids.append((s, 2, 0.4))
        maxh = 0.8
    elif name == "l_prism":
        b = box([-1, -1, -1], [4, 3, 2])
        pts = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 2), (0, 2)]
        ptags = [occ.addPoint(x, y, 0) for x, y in pts]
        lines = [
            occ.addLine(ptags[i], ptags[(i + 1) % len(ptags)]) for i in range(len(ptags))
        ]
        loop = occ.addCurveLoop(lines)
        surf = occ.addPlaneSurface([loop])
        ext = occ.extrude([(2, surf)], 0, 0, 1)
        vol = next(tag for dim, tag in ext if dim == 3)

        solids.append((b, 1, None))
        solids.append((vol, 2, None))
        maxh = 0.5
    elif name == "density_transition":
        b = box([0, 0, 0], [4, 4, 4])
        d = box([1, 1, 1], [3, 3, 2])
        solids.append((b, 1, None))
        solids.append((d, 2, 0.45))
        maxh = 1.4
    else:
        raise ValueError(f"unknown scene {name}")

    return solids, sheets, maxh


def run_gmsh(name: str) -> None:
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add(name)
        occ = gmsh.model.occ
        solids, sheets, maxh = gmsh_scene(name, gmsh)

        # Conformal decomposition of all volumes and embedded sheets. The
        # fragment map (input object -> output volumes) gives the region of
        # every output volume; assigning in input order makes later solids
        # win, matching rapidmesh region priority. Center-of-mass
        # classification would be WRONG here: the CoM of a non-convex
        # difference volume (box minus cylinder) lies inside the cut-out.
        objs = [(3, tag) for tag, *_ in solids] + [(2, s) for s in sheets]
        _, out_map = occ.fragment(objs, [])
        occ.synchronize()

        vol_region = {}
        for (vtag, rtag, _), mapped in zip(solids, out_map):
            for dim, tag in mapped:
                if dim == 3:
                    vol_region[tag] = rtag

        # Sizing: global maxh, finer per-region targets via MathEval-style
        # constant fields restricted to the fine volumes.
        gmsh.option.setNumber("Mesh.MeshSizeMax", maxh)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        fields = []
        for _, rtag, rmaxh in solids:
            if rmaxh is None:
                continue
            vols = [tag for tag, reg in vol_region.items() if reg == rtag]
            if not vols:
                continue
            f = gmsh.model.mesh.field.add("Constant")
            gmsh.model.mesh.field.setNumber(f, "VIn", rmaxh)
            gmsh.model.mesh.field.setNumber(f, "VOut", maxh)
            gmsh.model.mesh.field.setNumbers(f, "VolumesList", vols)
            fields.append(f)
        if fields:
            fmin = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(fmin)

        t0 = time.perf_counter()
        gmsh.model.mesh.generate(3)
        millis = (time.perf_counter() - t0) * 1000.0

        # Nodes and tets.
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        idx = {int(t): i for i, t in enumerate(node_tags)}
        points = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        all_tets = []
        all_regions = []
        for dim, tag in gmsh.model.getEntities(3):
            etypes, etags, enodes = gmsh.model.mesh.getElements(3, tag)
            for et, en in zip(etypes, enodes):
                if et != 4:  # linear tet
                    continue
                conn = np.asarray(en, dtype=np.int64).reshape(-1, 4)
                conn = np.vectorize(idx.__getitem__)(conn)
                all_tets.append(conn)
                all_regions.append(np.full(conn.shape[0], vol_region[tag], dtype=np.int64))
        tets = np.vstack(all_tets)
        regions = np.concatenate(all_regions)
        tets = orient_tets(points, tets)
        faces = derive_faces(points, tets, regions)
        write_viewer_json("gmsh", name, points, tets, regions, faces, millis)
    finally:
        gmsh.finalize()


# -------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesher", choices=["gmsh", "tetgen", "both"], default="both")
    ap.add_argument("names", nargs="*", help="geometry names (default: all PLC exports)")
    args = ap.parse_args()

    available = sorted(p.stem for p in PLC.glob("*.json"))
    names = args.names or available
    scene_names = {
        "em_scene", "via", "microstrip", "sphere", "l_prism", "density_transition",
    }

    for name in names:
        if args.mesher in ("tetgen", "both"):
            if name not in available:
                print(f"  tetgen {name}: SKIP (no PLC export, run the bench binary)")
            else:
                try:
                    run_tetgen(name)
                except Exception as e:
                    print(f"  tetgen {name}: FAILED ({e})", file=sys.stderr)
        if args.mesher in ("gmsh", "both"):
            if name not in scene_names:
                print(f"  gmsh {name}: SKIP (only analytic scenes)")
            else:
                try:
                    run_gmsh(name)
                except Exception as e:
                    print(f"  gmsh {name}: FAILED ({e})", file=sys.stderr)


if __name__ == "__main__":
    main()

"""From-scratch validation campaign for the report.

Builds a systematic corpus -- 2D plates, 3D primitives, then booleans -- each
meshed at several densities, records quality (min dihedral angle for volumes,
min planar angle for 2D plates; element counts; timing), and renders one figure
per case via the unchanged viewer (``report/viewer.py``).

Run:  python report/validate.py            # full campaign
      python report/validate.py --quick    # primitives + a few booleans only

Outputs:
  report/validation/results.json           # all (case x density) stats
  report/validation/meshes/*.json          # viewer JSONs (one per rendered case)
  report/figures/val/*.png                 # transparent renders (one per case)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rapidmesh as rm

REPO = Path(__file__).resolve().parents[1]
VALID = REPO / "report" / "validation"
MESHES = VALID / "meshes"
FIGS = REPO / "report" / "figures" / "val"

# ----------------------------------------------------------------------------
# geometry builders.  Each takes (g, h) and configures the Geometry; the
# campaign meshes with maxh=h.  kind "vol" -> g.mesh(), "surf" -> g.surface_mesh().
# ----------------------------------------------------------------------------
def _hexagon(r=1.0):
    return [(r * math.cos(k * math.pi / 3), r * math.sin(k * math.pi / 3)) for k in range(6)]


_LPOLY = [(0, 0), (2, 0), (2, 0.7), (0.7, 0.7), (0.7, 2), (0, 2)]
_RING_OUT = [(1.2 * math.cos(k * math.pi / 12), 1.2 * math.sin(k * math.pi / 12)) for k in range(24)]
_RING_IN = [(0.55 * math.cos(-k * math.pi / 12), 0.55 * math.sin(-k * math.pi / 12)) for k in range(24)]
_TRI = [(0.0, 0.0), (2.0, 0.0), (1.0, 1.7)]


def _ngon(n, r=1.1, rot=0.0):
    return [(r * math.cos(rot + 2 * math.pi * k / n), r * math.sin(rot + 2 * math.pi * k / n)) for k in range(n)]


def _star(n=5, ro=1.2, ri=0.5):
    pts = []
    for k in range(2 * n):
        rr = ro if k % 2 == 0 else ri
        a = math.pi / 2 + math.pi * k / n
        pts.append((rr * math.cos(a), rr * math.sin(a)))
    return pts


_SQ_OUT = [(-1.1, -1.1), (1.1, -1.1), (1.1, 1.1), (-1.1, 1.1)]
_SQ_IN = [(0.45, -0.45), (0.45, 0.45), (-0.45, 0.45), (-0.45, -0.45)]  # CW hole

# (name, category, kind, base_h, builder)
CASES = [
    # --- 2D plates (planar surface meshing) --------------------------------
    ("plate_rect", "2D", "surf", 0.18, lambda g, h: g.xy_plate(2.0, 1.2, maxh=h)),
    ("disc",       "2D", "surf", 0.16, lambda g, h: g.disc(1.0, segments=64, maxh=h)),
    ("hexagon",    "2D", "surf", 0.16, lambda g, h: g.polygon_plate(_hexagon(1.1), maxh=h)),
    ("l_polygon",  "2D", "surf", 0.14, lambda g, h: g.polygon_plate(_LPOLY, maxh=h)),
    ("annulus",    "2D", "surf", 0.13, lambda g, h: g.polygon_plate(_RING_OUT, holes=[_RING_IN], maxh=h)),
    # --- 3D primitives -----------------------------------------------------
    ("box",      "Primitive", "vol", 0.30, lambda g, h: g.box(2, 1.2, 1, position=(-1, -0.6, -0.5))),
    ("sphere",   "Primitive", "vol", 0.24, lambda g, h: g.sphere(1.0)),
    ("cylinder", "Primitive", "vol", 0.22, lambda g, h: g.cylinder(0.6, 1.6, position=(0, 0, -0.8), segments=40, uniform=True)),
    ("cone",     "Primitive", "vol", 0.22, lambda g, h: g.cone(0.8, 0.0, 1.5, position=(0, 0, -0.75), segments=40)),
    ("frustum",  "Primitive", "vol", 0.22, lambda g, h: g.cone(0.8, 0.4, 1.3, position=(0, 0, -0.65), segments=40)),
    ("torus",    "Primitive", "vol", 0.18, lambda g, h: g.torus(1.0, 0.35, segments=48, tube_segments=24)),
    ("wedge",    "Primitive", "vol", 0.16, lambda g, h: g.wedge(2.0, 1.0, 1.0, position=(-1, -0.5, 0), top_x=0.0)),
    ("prism_l",  "Primitive", "vol", 0.16, lambda g, h: g.prism(_LPOLY, 0.7, position=(-1, -1, -0.35))),
    # --- booleans: unions (g.union) and differences (void cuts) ------------
    ("union_box_sphere", "Boolean", "vol", 0.18,
        lambda g, h: g.union(g.box(1.6, 1.6, 1.0, position=(-0.8, -0.8, -0.5)),
                             g.sphere(0.7, position=(0.7, 0, 0)))),
    ("diff_box_cyl", "Boolean", "vol", 0.18,
        lambda g, h: (g.box(2, 2, 1, position=(-1, -1, -0.5)),
                      g.cylinder(0.45, 1.4, position=(0, 0, -0.7), segments=36, void=True))),
    ("diff_box_sphere", "Boolean", "vol", 0.18,
        lambda g, h: (g.box(1.8, 1.8, 1.2, position=(-0.9, -0.9, -0.6)),
                      g.sphere(0.7, position=(0, 0, 0.6), void=True))),
    ("fused_two", "Boolean", "vol", 0.18,
        lambda g, h: g.union(g.sphere(0.8, position=(-0.4, 0, 0)),
                             g.sphere(0.8, position=(0.4, 0, 0)))),
    ("fused_three", "Boolean", "vol", 0.16,
        lambda g, h: g.union(g.sphere(0.7, position=(0, 0, 0)),
                             g.sphere(0.7, position=(0.9, 0, 0)),
                             g.sphere(0.7, position=(0.45, 0.78, 0)))),
    ("capsule", "Boolean", "vol", 0.16,
        lambda g, h: g.union(g.cylinder(0.6, 1.2, position=(0, 0, -0.6), segments=40, uniform=True),
                             g.sphere(0.6, position=(0, 0, -0.6)),
                             g.sphere(0.6, position=(0, 0, 0.6)))),
    ("union_box_cyl", "Boolean", "vol", 0.18,
        lambda g, h: g.union(g.box(1.4, 1.4, 0.8, position=(-0.7, -0.7, -0.4)),
                             g.cylinder(0.4, 1.8, position=(0, 0, -0.9), segments=36, uniform=True))),
    ("diff_cyl_box", "Boolean", "vol", 0.18,
        lambda g, h: (g.cylinder(0.9, 1.4, position=(0, 0, -0.7), segments=44, uniform=True),
                      g.box(0.6, 2.2, 2.2, position=(-0.3, -1.1, -1.1), void=True))),
    ("drilled_block", "Boolean", "vol", 0.2,
        lambda g, h: (g.box(2, 2, 1, position=(-1, -1, -0.5)),
                      g.cylinder(0.3, 1.4, position=(-0.5, -0.5, -0.7), segments=28, void=True),
                      g.cylinder(0.3, 1.4, position=(0.5, 0.5, -0.7), segments=28, void=True))),
    # --- more 2D plates ----------------------------------------------------
    ("triangle",  "2D", "surf", 0.14, lambda g, h: g.polygon_plate(_TRI, maxh=h)),
    ("pentagon",  "2D", "surf", 0.15, lambda g, h: g.polygon_plate(_ngon(5, 1.1), maxh=h)),
    ("star",      "2D", "surf", 0.12, lambda g, h: g.polygon_plate(_star(5, 1.2, 0.5), maxh=h)),
    ("square_hole", "2D", "surf", 0.12, lambda g, h: g.polygon_plate(_SQ_OUT, holes=[_SQ_IN], maxh=h)),
    # --- more 3D primitives ------------------------------------------------
    ("slab",      "Primitive", "vol", 0.18, lambda g, h: g.box(2.2, 1.6, 0.35, position=(-1.1, -0.8, -0.175))),
    ("hex_prism", "Primitive", "vol", 0.16, lambda g, h: g.prism(_ngon(6, 1.0), 0.8, position=(0, 0, -0.4))),
    ("tri_prism", "Primitive", "vol", 0.16, lambda g, h: g.prism(_TRI, 0.8, position=(-1, -0.6, -0.4))),
    ("star_prism", "Primitive", "vol", 0.12, lambda g, h: g.prism(_star(5, 1.1, 0.45), 0.6, position=(0, 0, -0.3))),
    ("ellipsoidish", "Primitive", "vol", 0.16,
        lambda g, h: g.cone(0.9, 0.9, 1.4, position=(0, 0, -0.7), segments=40)),  # straight cylinder via cone r1=r2
    # --- more booleans -----------------------------------------------------
    ("tube", "Boolean", "vol", 0.16,
        lambda g, h: (g.cylinder(0.9, 1.4, position=(0, 0, -0.7), segments=44, uniform=True),
                      g.cylinder(0.5, 1.6, position=(0, 0, -0.8), segments=36, void=True))),
    ("fused_deep", "Boolean", "vol", 0.16,
        lambda g, h: g.union(g.sphere(0.8, position=(-0.3, 0, 0)),
                             g.sphere(0.8, position=(0.3, 0, 0)))),
    ("fused_unequal", "Boolean", "vol", 0.16,
        lambda g, h: g.union(g.sphere(0.85, position=(-0.35, 0, 0)),
                             g.sphere(0.5, position=(0.55, 0, 0)))),
    ("sphere_minus_box", "Boolean", "vol", 0.16,
        lambda g, h: (g.sphere(1.0),
                      g.box(0.7, 2.4, 2.4, position=(-0.35, -1.2, -1.2), void=True))),
    ("box_minus_2sph", "Boolean", "vol", 0.18,
        lambda g, h: (g.box(2, 1.4, 1.4, position=(-1, -0.7, -0.7)),
                      g.sphere(0.55, position=(-0.6, 0, 0), void=True),
                      g.sphere(0.55, position=(0.6, 0, 0), void=True))),
    ("cross_cyl", "Boolean", "vol", 0.16,
        lambda g, h: g.union(g.cylinder(0.4, 2.2, position=(0, 0, -1.1), segments=36, uniform=True),
                             g.cylinder(0.4, 2.2, position=(0, -1.1, 0), axis=(0, 1, 0), segments=36, uniform=True))),
    ("via", "Boolean", "vol", 0.2,
        lambda g, h: (g.box(2, 2, 1, position=(-1, -1, -0.5)),
                      g.cylinder(0.3, 1.4, position=(0, 0, -0.7), segments=28, uniform=True))),
    ("nested_spheres", "Boolean", "vol", 0.16,
        lambda g, h: (g.sphere(1.0),
                      g.sphere(0.55))),
]

DENSITY_FACTORS = [1.0, 0.62, 0.40]   # coarse / medium / fine (x base_h)
RENDER_FACTOR = 0.62                   # which density gets the figure


def _min_triangle_angle(points: np.ndarray, faces: np.ndarray) -> float:
    """Smallest interior angle (deg) over a 2D/surface triangulation."""
    P = np.asarray(points, float)
    best = 180.0
    for a, b, c in np.asarray(faces, int):
        va, vb, vc = P[a], P[b], P[c]
        for p, q, r in ((va, vb, vc), (vb, vc, va), (vc, va, vb)):
            u, w = q - p, r - p
            nu, nw = np.linalg.norm(u), np.linalg.norm(w)
            if nu < 1e-12 or nw < 1e-12:
                continue
            cosang = np.clip(np.dot(u, w) / (nu * nw), -1, 1)
            best = min(best, math.degrees(math.acos(cosang)))
    return best


def _surface_viewer_dict(sm, name: str) -> dict:
    """Viewer JSON for a surface-only mesh (no tets); now a method on SurfaceMesh."""
    return sm.to_viewer_dict(name)


def run(quick: bool = False) -> list[dict]:
    MESHES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    from report.viewer import render

    cases = CASES
    if quick:
        cases = [c for c in CASES if c[1] != "Boolean" or c[0] in ("fused_two", "diff_box_cyl")]

    results: list[dict] = []
    for name, cat, kind, base_h, build in cases:
        for fac in DENSITY_FACTORS:
            h = base_h * fac
            g = rm.Geometry(maxh=h)
            try:
                build(g, h)
                if kind == "surf":
                    m = g.surface_mesh(maxh=h)
                    quality = _min_triangle_angle(m.points, m.faces)
                    rec = {"name": name, "category": cat, "kind": kind, "h": round(h, 4),
                           "n_points": int(m.stats["n_points"]), "n_elems": int(m.stats["n_faces"]),
                           "quality_deg": round(quality, 2), "millis": int(m.stats["millis"]),
                           "metric": "min angle"}
                    vd = _surface_viewer_dict(m, name)
                else:
                    m = g.mesh(maxh=h)
                    s = m.stats
                    rec = {"name": name, "category": cat, "kind": kind, "h": round(h, 4),
                           "n_points": int(s["n_points"]), "n_elems": int(s["n_tets"]),
                           "quality_deg": round(float(s["min_dihedral_deg"]), 2),
                           "n_regions": int(s.get("n_regions", 1)),
                           "millis": int(s["millis"]), "metric": "min dihedral"}
                    vd = m.to_viewer_dict(name)
                results.append(rec)
                print(f"  {name:18s} h={h:.3f}  {rec['n_elems']:6d} elems  "
                      f"{rec['metric']}={rec['quality_deg']:5.1f}  {rec['millis']}ms")

                # render one figure per case at the chosen density
                if abs(fac - RENDER_FACTOR) < 1e-9:
                    mp = MESHES / f"{name}.json"
                    mp.write_text(json.dumps(vd))
                    try:
                        if kind == "vol":
                            # 3D: clip at 70% in y to reveal the interior tets
                            render(str(mp), str(FIGS / f"{name}.png"), azim=32,
                                   elev=18, clip=0.7, clip_axis=1, edges=True,
                                   width=1000, height=820)
                        else:
                            render(str(mp), str(FIGS / f"{name}.png"), azim=30,
                                   elev=24, edges=False, width=1000, height=820)
                    except Exception as e:
                        print(f"    render FAILED {name}: {e}")
            except Exception as e:
                print(f"  {name:18s} h={h:.3f}  MESH FAILED: {e}")
                results.append({"name": name, "category": cat, "kind": kind,
                                "h": round(h, 4), "error": str(e)})

    VALID.mkdir(parents=True, exist_ok=True)
    (VALID / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\n{len([r for r in results if 'error' not in r])} ok / {len(results)} runs "
          f"-> {VALID / 'results.json'}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(REPO))
    run(quick=args.quick)

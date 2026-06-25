"""The unified geometry corpus: every geometry we define anywhere, in one place,
used as a single benchmark by BOTH the gallery (``render_gallery.py``) and the
quality/conformity bench (``bench`` below).

Sources aggregated:
  * ``report/validate.py``                -- 2D plates, primitives, booleans,
  * ``python/examples/showcase.py``       -- mechanical / CAD showcase models,
  * ``python/examples/rapidfem_geometries.py`` -- RF/EM structures.

Each entry is ``(name, category, kind, make)`` where ``make()`` returns a meshed
``Mesh`` (kind ``"vol"``) or ``SurfaceMesh`` (kind ``"surf"``), meshed at call
time by the constrained per-region mesher.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "python" / "examples")):
    if p not in sys.path:
        sys.path.insert(0, p)

import rapidmesh as rm  # noqa: E402
from report import validate as V  # noqa: E402
import showcase as _SC  # noqa: E402
import rapidfem_geometries as _RF  # noqa: E402


def _from_validate(name, cat, kind, base_h, build):
    h = base_h

    def make():
        g = rm.Geometry(maxh=h)
        build(g, h)
        return g.surface_mesh(maxh=h) if kind == "surf" else g.mesh(maxh=h)

    return (name, cat, kind, make)


def _from_showcase(model):
    return (model.id, "Showcase", "vol", lambda m=model: m.build().mesh())


_RF_FNS = (
    "coax_step", "microstrip_line", "dielectric_resonator", "iris_filter", "patch_antenna",
    # magnetics: coils / transformers (the EM target geometries)
    "solenoid", "cored_inductor", "transformer", "toroid_core",
)


def _from_rf(fn_name):
    fn = getattr(_RF, fn_name)
    return (f"rf_{fn_name}", "RF", "vol", lambda f=fn: f())


def _sizing_cases():
    """Graded-sizing stress cases: LINEAR sizing (a fine size source, the field
    growing linearly outward under the grading Lipschitz bound) and COARSE-INTERIOR
    sizing (fine boundary, coarse bulk -- the classic FEM layout), in 2D and 3D."""
    cases = []

    def plate_linear():
        g = rm.Geometry()
        g.xy_plate(2.0, 1.2, position=(-1.0, -0.6, 0.0))
        g.refine_near_points([(-1.0, -0.6, 0.0)], 0.04)  # fine at one corner
        return g.surface_mesh(maxh=0.4, grading=0.3)
    cases.append(("plate_linear", "Sizing", "surf", plate_linear))

    def plate_coarse_interior():
        g = rm.Geometry()
        g.xy_plate(2.0, 1.2, position=(-1.0, -0.6, 0.0))
        return g.surface_mesh(maxh_edge=0.05, maxh_surf=0.35)  # fine rim, coarse bulk
    cases.append(("plate_coarse_interior", "Sizing", "surf", plate_coarse_interior))

    def disc_coarse_interior():
        g = rm.Geometry()
        g.disc(1.0, segments=80)
        return g.surface_mesh(maxh_edge=0.05, maxh_surf=0.4)
    cases.append(("disc_coarse_interior", "Sizing", "surf", disc_coarse_interior))

    def box_linear():
        g = rm.Geometry()
        g.box(2.0, 1.2, 1.0, position=(-1.0, -0.6, -0.5))
        g.refine_near_points([(-1.0, -0.6, -0.5)], 0.08)  # fine at one corner
        return g.mesh(maxh=0.5, grading=0.3)
    cases.append(("box_linear", "Sizing", "vol", box_linear))

    def box_coarse_interior():
        g = rm.Geometry()
        g.box(2.0, 1.2, 1.0, position=(-1.0, -0.6, -0.5))
        return g.mesh(maxh_surf=0.12, maxh_vol=0.45)  # fine boundary, coarse bulk
    cases.append(("box_coarse_interior", "Sizing", "vol", box_coarse_interior))

    def sphere_coarse_interior():
        g = rm.Geometry()
        g.icosphere(1.0, subdivisions=3)
        return g.mesh(maxh_surf=0.14, maxh_vol=0.5)
    cases.append(("sphere_coarse_interior", "Sizing", "vol", sphere_coarse_interior))

    return cases


import math  # noqa: E402


def _spiral(turns=3.0, r0=0.18, pitch=0.16, width=0.07, ppt=44):
    """A planar spiral TRACK of constant width -> one closed outline (out along
    the outer edge, back along the inner). The signature RF-inductor shape: thin
    track, many near-parallel walls, a hard 2D meshing stress."""
    n = int(turns * ppt)
    cen = [(r0 + pitch * i / ppt, 2 * math.pi * i / ppt) for i in range(n + 1)]
    out = [((r + width / 2) * math.cos(t), (r + width / 2) * math.sin(t)) for r, t in cen]
    inn = [((r - width / 2) * math.cos(t), (r - width / 2) * math.sin(t)) for r, t in cen]
    return out + inn[::-1]


def _blob(n=170):
    """An organic closed curve: radius = sum of low harmonics. Smooth, non-convex,
    no straight edges -- the curvature-graded 2D case."""
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        r = 1.0 * (1 + 0.18 * math.cos(3 * a + 0.5) + 0.10 * math.cos(5 * a + 1.2) + 0.05 * math.cos(8 * a + 2.0))
        pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


def _gear(teeth=14, r_out=1.0, r_in=0.78):
    """A cog: alternating radii -> many sharp reentrant corners."""
    pts = []
    for k in range(teeth):
        a0 = 2 * math.pi * k / teeth
        for frac, r in [(0.04, r_in), (0.18, r_out), (0.5, r_out), (0.64, r_in)]:
            a = a0 + 2 * math.pi * frac / teeth
            pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


def _comb(fingers=5, w=1.8, base=0.25, fh=0.65):
    """An interdigital-capacitor comb: a base bar with rectangular fingers (deep
    slots, thin walls)."""
    seg = w / (2 * fingers + 1)
    pts = [(0.0, 0.0), (w, 0.0)]
    for i in range(fingers):
        x1 = w - (2 * i + 1) * seg
        x2 = w - (2 * i + 2) * seg
        pts += [(x1, base), (x1, base + fh), (x2, base + fh), (x2, base)]
    pts.append((0.0, base))
    return [(x - w / 2, y - 0.3) for x, y in pts]


def _shape_cases():
    """Complex / organic 2D polygon shapes (RF-passive-like and free-form), plus
    extra graded 2D cases -- the 2D mesher's hardest inputs (thin tracks, sharp
    reentrant corners, smooth curvature, strong grading)."""
    cases = []

    def add(name, h, fn, **mesh):
        def make(_fn=fn, _h=h, _m=mesh):
            g = rm.Geometry(maxh=_h)
            g.polygon_plate(_fn())
            return g.surface_mesh(**(_m or {"maxh": _h}))
        cases.append((name, "Shape", "surf", make))

    add("spiral_inductor", 0.05, _spiral, maxh=0.05)
    add("organic_blob", 0.08, _blob, maxh=0.08)
    add("gear", 0.07, _gear, maxh=0.07)
    add("interdigital_comb", 0.06, _comb, maxh=0.06)

    # graded 2D: a fine source point with the field growing linearly outward.
    def blob_graded():
        g = rm.Geometry()
        g.polygon_plate(_blob())
        g.refine_near_points([(1.18, 0.0, 0.0)], 0.025)
        return g.surface_mesh(maxh=0.25, grading=0.3)
    cases.append(("blob_graded", "Sizing", "surf", blob_graded))

    # coarse-interior 2D: fine rim, coarse bulk (the classic FEM 2D layout).
    def gear_coarse_interior():
        g = rm.Geometry()
        g.polygon_plate(_gear())
        return g.surface_mesh(maxh_edge=0.035, maxh_surf=0.3)
    cases.append(("gear_coarse_interior", "Sizing", "surf", gear_coarse_interior))

    return cases


def _graded_cases():
    """Grading woven into the existing geometry families on the SAME primitives the
    corpus already uses, one knob each: coarse vs fine interior, a single finer
    face, a single finer edge (also at a CSG feature), point sources on a curved
    hull, and a multi-region split. Exercises the sizing field per region / per
    face / per edge, not just globally."""
    cases = []

    def vol(name, fn):
        cases.append((name, "Graded", "vol", fn))

    # innen GRÖBER: fine shell, coarse bulk (the classic FEM layout) on a cylinder.
    def cyl_coarse_interior():
        g = rm.Geometry()
        g.cylinder(0.8, 2.0, position=(0.0, 0.0, -1.0), segments=48)
        return g.mesh(maxh_surf=0.13, maxh_vol=0.55)
    vol("cyl_coarse_interior", cyl_coarse_interior)

    # innen FEINER: a point source at the centre, the field grows linearly outward.
    def box_core_fine():
        g = rm.Geometry()
        g.box(2.0, 2.0, 2.0, position=(-1.0, -1.0, -1.0))
        g.refine_near_points([(0.0, 0.0, 0.0)], 0.09)
        return g.mesh(maxh=0.55, grading=0.3)
    vol("box_core_fine", box_core_fine)

    # einzelne FLÄCHE: only the +z face of a box is refined, the rest stays coarse.
    def box_face_fine():
        g = rm.Geometry(maxh=0.5)
        g.box(2.0, 2.0, 2.0, position=(-1.0, -1.0, -1.0))
        g.surf(normal=(0.0, 0.0, 1.0)).maxh = 0.11
        return g.mesh(maxh=0.5)
    vol("box_face_fine", box_face_fine)

    # einzelne KANTE: one vertical edge of a box is refined, graded into the bulk.
    def box_edge_fine():
        g = rm.Geometry(maxh=0.5)
        g.box(2.0, 2.0, 2.0, position=(-1.0, -1.0, -1.0))
        g.edge(near=(1.0, 1.0, 0.0)).maxh = 0.07
        return g.mesh(maxh=0.5, grading=0.3)
    vol("box_edge_fine", box_edge_fine)

    # einzelne KANTE at a CSG feature: a corner notch carved from a slab, fine
    # along the reentrant edge it creates.
    def notch_edge_fine():
        g = rm.Geometry(maxh=0.45)
        g.box(2.0, 2.0, 1.0, position=(-1.0, -1.0, -0.5))
        g.box(1.0, 1.0, 1.0, position=(0.0, 0.0, 0.0), void=True)
        g.edge(near=(0.0, 0.0, 0.0)).maxh = 0.07
        return g.mesh(maxh=0.45, grading=0.3)
    vol("notch_edge_fine", notch_edge_fine)

    # curved hull, point-refined: one patch of an icosphere is fine, graded out.
    def sphere_patch_fine():
        g = rm.Geometry()
        g.icosphere(1.0, subdivisions=3)
        g.refine_near_points([(0.0, 0.0, 1.0)], 0.05)
        return g.mesh(maxh=0.45, grading=0.3)
    vol("sphere_patch_fine", sphere_patch_fine)

    # MULTI-REGION: two stacked boxes sharing a face, upper region fine, lower coarse.
    def stacked_two_region():
        g = rm.Geometry()
        g.box(2.0, 2.0, 1.0, position=(-1.0, -1.0, 0.0), maxh=0.5)
        g.box(2.0, 2.0, 1.0, position=(-1.0, -1.0, 1.0), maxh=0.16)
        return g.mesh()
    vol("stacked_two_region", stacked_two_region)

    return cases


#: The whole corpus: validate cases + showcase models + RF structures + sizing.
CORPUS = (
    [_from_validate(*c) for c in V.CASES]
    + [_from_showcase(m) for m in _SC.MODELS]
    + [_from_rf(n) for n in _RF_FNS]
    + _sizing_cases()
    + _shape_cases()
    + _graded_cases()
)


def names() -> list[str]:
    return [e[0] for e in CORPUS]


def bench(only=None) -> list[dict]:
    """Runs every geometry through the mesher, recording quality + timing.
    A geometry that panics (e.g. an assembly degeneracy) is recorded, not fatal,
    so the benchmark always completes. Returns one record per geometry.
    """
    rows: list[dict] = []
    for name, cat, kind, make in CORPUS:
        if only is not None and name not in only:
            continue
        rec = {"name": name, "category": cat, "kind": kind}
        t0 = time.time()
        try:
            m = make()
            s = m.stats
            rec.update(
                status="ok",
                n_elems=int(s["n_faces"]) if kind == "surf" else int(s["n_tets"]),
                n_points=int(s["n_points"]),
                min_dihedral=None if kind == "surf" else round(float(s["min_dihedral_deg"]), 2),
                millis=int((time.time() - t0) * 1000),
            )
            # Located diagnostics (volume meshes): the conformity/quality map.
            if kind == "vol":
                d = m.diagnostics
                rec.update(
                    watertight=bool(d["watertight"]),
                    n_slivers=int(d["n_slivers"]),
                    n_straddlers=int(d["n_straddlers"]),
                    n_nonmanifold=int(d["n_nonmanifold_edges"]),
                    max_surf_dev=round(float(d["max_surface_deviation"]), 6),
                    n_defects=len(d["defects"]),
                )
        except BaseException as e:  # noqa: BLE001 - a panic must not abort the bench
            rec.update(status="FAIL", error=f"{type(e).__name__}: {str(e)[:80]}", millis=int((time.time() - t0) * 1000))
        rows.append(rec)
    return rows


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-render", action="store_true", help="skip the gallery render (metrics only)")
    args = ap.parse_args()

    print(f"corpus: {len(CORPUS)} geometries")
    rows = bench()
    out = REPO / "report" / "validation" / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))

    ok = [r for r in rows if r["status"] == "ok"]
    vol = [r for r in ok if r["kind"] == "vol"]
    print(f"\n{len(ok)}/{len(rows)} meshed ok ({len(rows) - len(ok)} failed)")
    leaky = [r for r in vol if not r.get("watertight", True)]
    strad = [r for r in vol if r.get("n_straddlers", 0) > 0]
    sliv = [r for r in vol if r.get("n_slivers", 0) > 0]
    print(f"volume: {len(vol)} | not watertight: {len(leaky)} | with straddlers: {len(strad)} | with slivers: {len(sliv)}")
    # the territory map: per-geometry headline
    print(f"\n{'geometry':<26}{'tets':>7}{'minDih':>8}{'sliv':>6}{'strad':>6}{'wtr':>5}{'maxDev':>9}{'ms':>7}")
    for r in sorted(vol, key=lambda x: (x.get("watertight", True), x.get("n_straddlers", 0) == 0, x.get("min_dihedral", 99))):
        print(f"{r['name']:<26}{r['n_elems']:>7}{r.get('min_dihedral', 0):>8.1f}"
              f"{r.get('n_slivers', 0):>6}{r.get('n_straddlers', 0):>6}"
              f"{'Y' if r.get('watertight', True) else 'N':>5}{r.get('max_surf_dev', 0):>9.4f}{r['millis']:>7}")
    for r in rows:
        if r["status"] != "ok":
            print(f"  FAIL {r['name']:<22} {r['error']}")
    print(f"\n-> {out}")

    # The gallery is part of every run: a benchmark with no fresh images leaves the
    # human loop open (you can't eyeball what the metrics changed). Render unless
    # explicitly skipped.
    if not args.no_render:
        print("\nrendering gallery (corpus)...")
        from report import render_gallery
        render_gallery.render_corpus()
        print(f"-> {REPO / 'report' / 'figures' / 'gallery' / 'corpus'}")

"""The unified geometry corpus: every geometry we define anywhere, in one place,
used as a single benchmark by BOTH the gallery (``render_gallery.py``) and the
quality/conformity bench (``bench`` below).

Sources aggregated:
  * ``report/validate.py``                -- 2D plates, primitives, booleans,
  * ``python/examples/showcase.py``       -- mechanical / CAD showcase models,
  * ``python/examples/rapidfem_geometries.py`` -- RF/EM structures.

Each entry is ``(name, category, kind, make)`` where ``make()`` returns a meshed
``Mesh`` (kind ``"vol"``) or ``SurfaceMesh`` (kind ``"surf"``). ``make`` meshes at
call time, so setting ``RAPIDMESH_CDT`` before it selects the constrained path.
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


#: The whole corpus: validate cases + showcase models + RF structures.
CORPUS = (
    [_from_validate(*c) for c in V.CASES]
    + [_from_showcase(m) for m in _SC.MODELS]
    + [_from_rf(n) for n in _RF_FNS]
)


def names() -> list[str]:
    return [e[0] for e in CORPUS]


def bench(meshers=("default", "cdt"), only=None) -> list[dict]:
    """Runs every geometry through every mesher, recording quality + timing.
    A geometry that panics (e.g. an assembly degeneracy) is recorded, not fatal,
    so the benchmark always completes. Returns one record per (geometry, mesher).
    """
    import os

    rows: list[dict] = []
    for name, cat, kind, make in CORPUS:
        if only is not None and name not in only:
            continue
        for mesher in meshers:
            if mesher == "cdt":
                os.environ["RAPIDMESH_CDT"] = "1"
            else:
                os.environ.pop("RAPIDMESH_CDT", None)
            rec = {"name": name, "category": cat, "kind": kind, "mesher": mesher}
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
            os.environ.pop("RAPIDMESH_CDT", None)
    return rows


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--mesher", default="cdt", choices=["default", "cdt"], help="which mesher to map")
    args = ap.parse_args()

    print(f"corpus: {len(CORPUS)} geometries, mesher={args.mesher}")
    rows = bench(meshers=(args.mesher,))
    out = REPO / "report" / "validation" / f"benchmark_{args.mesher}.json"
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

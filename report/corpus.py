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


_RF_FNS = ("coax_step", "microstrip_line", "dielectric_resonator", "iris_filter", "patch_antenna")


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
            except BaseException as e:  # noqa: BLE001 - a panic must not abort the bench
                rec.update(status="FAIL", error=f"{type(e).__name__}: {str(e)[:80]}", millis=int((time.time() - t0) * 1000))
            rows.append(rec)
            os.environ.pop("RAPIDMESH_CDT", None)
    return rows


if __name__ == "__main__":
    import json

    print(f"corpus: {len(CORPUS)} geometries")
    rows = bench(meshers=("default",))
    out = REPO / "report" / "validation" / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"default mesher: {ok}/{len(rows)} ok")
    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['name']:<20} {r['error']}")
    print(f"-> {out}")

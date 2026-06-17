"""Render a PNG for every test geometry and every sizing permutation, so the
results can be eyeballed. Reuses the validation corpus (``validate.CASES``) and
the unchanged viewer (``report/viewer.py``).

For each corpus geometry it renders BOTH meshers side by side:
  * ``default`` -- the current ``cvt::mesh`` (oversampling + restricted boundary),
  * ``cdt``     -- the constrained ``mesh_cdt`` (RAPIDMESH_CDT), the spec path,
so the two can be compared directly. It then renders the hierarchical
per-entity sizing permutations on a box.

Run:  python report/render_gallery.py            # everything
      python report/render_gallery.py --corpus   # only the corpus
      python report/render_gallery.py --sizing   # only the permutations

Outputs (transparent PNGs):
  report/figures/gallery/corpus/<name>__<mesher>.png
  report/figures/gallery/sizing/<label>.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import rapidmesh as rm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from report import validate as V  # noqa: E402
from report.viewer import render  # noqa: E402

GAL = REPO / "report" / "figures" / "gallery"
MESHES = REPO / "report" / "validation" / "meshes"


def _set_cdt(on: bool) -> None:
    if on:
        os.environ["RAPIDMESH_CDT"] = "1"
    else:
        os.environ.pop("RAPIDMESH_CDT", None)


def _render_mesh(name: str, kind: str, build, h: float, out_png: Path) -> str:
    """Builds + meshes one geometry and renders it. Returns a short status line."""
    g = rm.Geometry(maxh=h)
    build(g, h)
    t0 = time.time()
    if kind == "surf":
        m = g.surface_mesh(maxh=h)
        vd = V._surface_viewer_dict(m, name)
        n = len(m.faces)
        clip = None
    else:
        m = g.mesh(maxh=h)
        vd = m.to_viewer_dict(name)
        n = int(m.stats["n_tets"])
        clip = 0.7
    dt = time.time() - t0
    mp = MESHES / f"gal_{out_png.stem}.json"
    mp.write_text(json.dumps(vd))
    render(
        str(mp), str(out_png),
        azim=32, elev=20,
        clip=clip, clip_axis=1, edges=(kind != "surf"),
        width=900, height=740,
    )
    return f"{out_png.name}: {n} elems, {dt:.1f}s"


def render_corpus() -> None:
    out = GAL / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    MESHES.mkdir(parents=True, exist_ok=True)
    for name, _cat, kind, base_h, build in V.CASES:
        h = base_h * V.RENDER_FACTOR
        for mesher in ("default", "cdt"):
            _set_cdt(mesher == "cdt")
            png = out / f"{name}__{mesher}.png"
            try:
                status = _render_mesh(name, kind, build, h, png)
                print(f"  [{mesher:7s}] {status}")
            except Exception as e:  # a mesher gap should not stop the gallery
                print(f"  [{mesher:7s}] {name}: FAILED ({type(e).__name__}: {e})")
    _set_cdt(False)


# --- hierarchical per-entity sizing permutations (on a 4x4x4 box) -----------
def _box(g, h):
    g.box(4.0, 4.0, 4.0, position=(0.0, 0.0, 0.0))


SIZING_PERMS = [
    ("00_global_coarse", 2.0, lambda g: None),
    ("01_global_fine", 0.8, lambda g: None),
    ("02_maxh_vol", 4.0, lambda g: setattr(g.region(), "maxh", 0.6)),
    ("03_maxh_surf", 4.0, lambda g: setattr(g.surf(), "maxh", 0.6)),
    ("04_maxh_edge", 4.0, lambda g: setattr(g.edge(), "maxh", 0.6)),
    ("05_surf_plus_z", 4.0, lambda g: setattr(g.surf(normal=(0.0, 0.0, 1.0)), "maxh", 0.5)),
    ("06_edge_near", 4.0, lambda g: setattr(g.edge(near=(2.0, 0.0, 0.0)), "maxh", 0.3)),
    ("07_region_surf_edge", 4.0, lambda g: (
        setattr(g.region(), "maxh", 1.5),
        setattr(g.region(1).surf(normal=(1.0, 0.0, 0.0)).edge(near=(4.0, 0.0, 2.0)), "maxh", 0.3),
    )),
    ("08_most_specific", 4.0, lambda g: (
        setattr(g.edge(), "maxh", 3.0),
        setattr(g.edge(near=(2.0, 0.0, 0.0)), "maxh", 0.4),
    )),
]


def render_sizing() -> None:
    out = GAL / "sizing"
    out.mkdir(parents=True, exist_ok=True)
    _set_cdt(False)
    for label, h, setup in SIZING_PERMS:
        png = out / f"{label}.png"
        try:
            g = rm.Geometry(maxh=h)
            _box(g, h)
            setup(g)
            m = g.mesh(maxh=h)
            vd = m.to_viewer_dict(label)
            mp = MESHES / f"gal_{label}.json"
            mp.write_text(json.dumps(vd))
            render(str(mp), str(png), azim=32, elev=20, clip=0.7, clip_axis=1, edges=True, width=900, height=740)
            print(f"  {png.name}: {int(m.stats['n_tets'])} tets")
        except Exception as e:
            print(f"  {label}: FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="store_true", help="only the geometry corpus")
    ap.add_argument("--sizing", action="store_true", help="only the sizing permutations")
    args = ap.parse_args()
    do_all = not (args.corpus or args.sizing)
    if args.corpus or do_all:
        print("== corpus (default vs cdt) ==")
        render_corpus()
    if args.sizing or do_all:
        print("== sizing permutations ==")
        render_sizing()
    print(f"\ngallery -> {GAL}")

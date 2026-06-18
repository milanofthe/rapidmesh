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
from report import corpus as C  # noqa: E402
from report import validate as V  # noqa: E402
from report.viewer import render  # noqa: E402

GAL = REPO / "report" / "figures" / "gallery"
MESHES = REPO / "report" / "validation" / "meshes"


def _set_cdt(on: bool) -> None:
    if on:
        os.environ["RAPIDMESH_CDT"] = "1"
    else:
        os.environ.pop("RAPIDMESH_CDT", None)


def _render_mesh(name: str, kind: str, make, out_png: Path) -> str:
    """Meshes one corpus geometry (via its `make`) and renders it in DIAGNOSTIC
    style: the boundary surface (region-coloured, so face assignment is visible)
    with its triangulation edges and the located defect markers (slivers, straddlers,
    non-manifold edges) overlaid. No clip and no interior tet wireframe -- the focus
    is surface conformity, face assignment and where the defects sit, not the bulk."""
    t0 = time.time()
    m = make()
    if kind == "surf":
        vd = V._surface_viewer_dict(m, name)
        n = len(m.faces)
    else:
        vd = m.to_viewer_dict(name)
        n = int(m.stats["n_tets"])
    dt = time.time() - t0
    mp = MESHES / f"gal_{out_png.stem}.json"
    mp.write_text(json.dumps(vd))
    render(
        str(mp), str(out_png),
        azim=32, elev=20,
        clip=None,            # no clip: show the whole boundary, not a cut
        tets=True,            # surface fill ON (region colours -> face assignment)
        edges=False,          # no interior tet wireframe
        wireframe=True,       # surface triangulation edges
        defects=True,         # the metric overlay: located defect markers
        width=1100, height=900,
    )
    return f"{out_png.name}: {n} elems, {dt:.1f}s"


def render_corpus() -> None:
    """Renders every geometry in the unified corpus with the NEW constrained
    mesher (`mesh_cdt`), one image `<name>.png`. The corpus directory is cleared
    first so no stale image from the retired oversampling path survives."""
    out = GAL / "corpus"
    if out.exists():
        for old in out.glob("*.png"):
            old.unlink()
    out.mkdir(parents=True, exist_ok=True)
    MESHES.mkdir(parents=True, exist_ok=True)
    _set_cdt(True)  # the new constrained path is the gallery's mesher
    for name, _cat, kind, make in C.CORPUS:
        png = out / f"{name}.png"
        try:
            status = _render_mesh(name, kind, make, png)
            print(f"  {status}")
        except BaseException as e:  # a mesher gap / panic must not stop the gallery
            print(f"  {name}: FAILED ({type(e).__name__}: {str(e)[:70]})")
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
    if out.exists():
        for old in out.glob("*.png"):
            old.unlink()
    out.mkdir(parents=True, exist_ok=True)
    _set_cdt(True)  # new constrained mesher
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
            render(str(mp), str(png), azim=32, elev=20, clip=0.7, clip_axis=1, edges=True, width=1100, height=900)
            print(f"  {png.name}: {int(m.stats['n_tets'])} tets")
        except BaseException as e:
            print(f"  {label}: FAILED ({type(e).__name__}: {e})")
    _set_cdt(False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="store_true", help="only the geometry corpus")
    ap.add_argument("--sizing", action="store_true", help="only the sizing permutations")
    args = ap.parse_args()
    do_all = not (args.corpus or args.sizing)
    if args.corpus or do_all:
        print("== corpus (mesh_cdt) ==")
        render_corpus()
    if args.sizing or do_all:
        print("== sizing permutations (mesh_cdt) ==")
        render_sizing()
    print(f"\ngallery -> {GAL}")

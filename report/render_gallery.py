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

# Defect-marker colours (match the viewer's DEFECT_COLORS), for the legend.
DEFECT_LEGEND = [
    ("sliver", (255, 191, 0)),            # amber
    ("straddler", (255, 26, 204)),        # magenta
    ("nonmanifold_edge", (255, 26, 26)),  # red
]


def _font(size: int):
    from PIL import ImageFont
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _annotate(png: Path, name: str, kind: str, n: int, diag: dict | None) -> None:
    """Overlays the metrics text (top-left) and the defect-colour legend
    (bottom-left) onto a rendered diagnostic PNG."""
    from PIL import Image, ImageDraw

    im = Image.open(png).convert("RGBA")
    dr = ImageDraw.Draw(im)
    title_f, body_f = _font(26), _font(19)
    # Neutral mid-gray for ALL non-semantic text (title, metrics, legend labels) so
    # it reads on both light AND dark backgrounds; only the colour-coded text
    # (watertight yes/no) and the defect swatches keep their meaning colours.
    fg = dim = (128, 128, 128, 255)

    def txt(xy, s, font, fill):
        dr.text(xy, s, font=font, fill=fill)

    # ---- metrics block (top-left) ----
    lines: list[tuple[str, tuple]] = [(name, fg)]
    if diag is not None:
        wt = diag["watertight"]
        lines += [
            (f"{n} tets   min-dih {diag['min_dihedral_deg']:.1f} deg", dim),
            (f"watertight: {'yes' if wt else 'NO'}", (20, 120, 40, 255) if wt else (190, 30, 30, 255)),
            (f"slivers {diag['n_slivers']}   straddlers {diag['n_straddlers']}   "
             f"non-manifold {diag['n_nonmanifold_edges']}", dim),
            (f"max surface dev {diag['max_surface_deviation']:.4f}", dim),
        ]
    else:
        lines.append((f"{n} surface faces", dim))
    x, y = 16, 14
    txt((x, y), lines[0][0], title_f, lines[0][1])
    y += 34
    for s, col in lines[1:]:
        txt((x, y), s, body_f, col)
        y += 24

    # ---- defect legend (bottom-left), only the kinds actually present ----
    present = {d["kind"] for d in diag["defects"]} if diag else set()
    legend = [(lbl, c) for (lbl, c) in DEFECT_LEGEND if lbl in present]
    if legend:
        ly = im.height - 16 - 26 * len(legend)
        txt((16, ly - 26), "defects:", body_f, dim)
        for lbl, (r, g, b) in legend:
            dr.rectangle([16, ly + 4, 32, ly + 20], fill=(r, g, b, 255), outline=(0, 0, 0, 210))
            txt((40, ly), lbl, body_f, fg)
            ly += 26
    im.save(png)


def _set_cdt(on: bool) -> None:
    if on:
        os.environ["RAPIDMESH_CDT"] = "1"
    else:
        os.environ.pop("RAPIDMESH_CDT", None)


def _render_mesh(name: str, kind: str, make, out_normal: Path, out_debug: Path) -> str:
    """Meshes one corpus geometry ONCE and renders TWO views from it:
      * NORMAL (`out_normal`): the clean figure -- region-coloured solid (clipped
        so the interior is visible) for volumes, or the surface wireframe for
        surface meshes, on a transparent background, no overlays.
      * DEBUG (`out_debug`): the DIAGNOSTIC view -- surface triangulation with the
        located defect markers (slivers, straddlers, non-manifold edges) and a
        metrics text overlay, on a dark background.
    Meshing is the expensive step (heavy CSG geometries), so it runs once and both
    PNGs come from the same viewer JSON."""
    t0 = time.time()
    m = make()
    if kind == "surf":
        vd = V._surface_viewer_dict(m, name)
        n = len(m.faces)
        diag = None
    else:
        vd = m.to_viewer_dict(name)
        n = int(m.stats["n_tets"])
        diag = m.diagnostics
    dt = time.time() - t0
    mp = MESHES / f"gal_{name}.json"
    mp.write_text(json.dumps(vd))
    # ---- normal view: clean, region-coloured SURFACE MESH (fill + triangulation),
    # clipped for volumes so the interior shows. The surface mesh (fill + wire) is
    # ALWAYS on, including for surface-only (2D) geometries. ----
    render(
        str(mp), str(out_normal),
        azim=32, elev=20,
        transparent=True,
        clip=0.55 if kind == "vol" else None,   # cutaway so interior tets show
        clip_axis=1,
        tets=True,                # surface fill (region-coloured) -- always
        edges=(kind == "vol"),    # interior tet edges on the cutaway (volumes)
        wireframe=True,           # surface triangulation -- always (fixes 2D tris)
        defects=False,
        width=1100, height=900,
    )
    # ---- debug view: the surface triangulation (WIREFRAME only, no fill) + located
    # defect markers + metrics overlay, on a TRANSPARENT background (annotation text
    # gets a dark stroke for legibility on any surface). ----
    render(
        str(mp), str(out_debug),
        azim=32, elev=20,
        transparent=True,
        clip=None,
        tets=False,           # NO solid fill -- just the mesh
        edges=False,
        wireframe=True,       # surface triangulation
        defects=True,         # located defect markers
        width=1100, height=900,
    )
    _annotate(out_debug, name, kind, n, diag)
    return f"{name}: {n} elems, {dt:.1f}s"


def render_corpus() -> None:
    """Renders every geometry in the unified corpus with the NEW constrained
    mesher (`mesh_cdt`) into TWO directories: `corpus/` (normal view) and
    `corpus_debug/` (diagnostic view). Both are cleared first so no stale image
    from the retired oversampling path survives."""
    out = GAL / "corpus"
    dbg = GAL / "corpus_debug"
    for d in (out, dbg):
        if d.exists():
            for old in d.glob("*.png"):
                old.unlink()
        d.mkdir(parents=True, exist_ok=True)
    MESHES.mkdir(parents=True, exist_ok=True)
    _set_cdt(True)  # the new constrained path is the gallery's mesher
    for name, _cat, kind, make in C.CORPUS:
        try:
            status = _render_mesh(name, kind, make, out / f"{name}.png", dbg / f"{name}.png")
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

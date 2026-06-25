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
import subprocess
import sys
import time
from pathlib import Path

import rapidmesh as rm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from report import corpus as C  # noqa: E402
from report import validate as V  # noqa: E402

GAL = REPO / "report" / "figures" / "gallery"
MESHES = REPO / "report" / "validation" / "meshes"
# Headless WebGPU rasterizer (replaces the playwright/chromium screenshotter).
RENDER_NODE = REPO / "report" / "render-node"
RASTERIZE = RENDER_NODE / "rasterize.mjs"
RENDER_W, RENDER_H = 1500, 1230

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


def _timing_lines(wall: float | None, timings: dict | None) -> list[str]:
    """Meshing wall-clock + per-phase breakdown for the debug overlay."""
    if wall is None:
        return []
    out = [f"mesh {wall:.2f}s"]
    t = timings or {}

    def g(*keys):
        return next((t[k] for k in keys if k in t), 0.0)

    surf = g("mesh_cdt.surface", "mesh.surface")
    seed = g("mesh_cdt.seed", "mesh.seed")
    lloyd = g("mesh_cdt.lloyd", "mesh.lloyd")
    tet = g("mesh_cdt.tetrahedralize", "mesh.build_final")
    opt = g("optimize.total")
    parts = [(lbl, v) for lbl, v in
             (("surf", surf), ("seed", seed), ("lloyd", lloyd), ("tet", tet), ("opt", opt)) if v > 0]
    if parts:
        out.append("  " + "  ".join(f"{lbl} {v * 1e3:.0f}" for lbl, v in parts) + " ms")
    return out


def _annotate(png: Path, name: str, kind: str, n: int, diag: dict | None,
              wall: float | None = None, timings: dict | None = None) -> None:
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
    for s in _timing_lines(wall, timings):
        lines.append((s, dim))
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


def _build_bundle() -> None:
    """Bundle the shared browser render pipeline (mesh_adapter + scene_build +
    canvas3d_webgpu) into render-node/bundle.mjs so the Node rasterizer runs the
    exact same code as the live viewer."""
    r = subprocess.run(["node", "build-bundle.mjs"], cwd=str(RENDER_NODE), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"bundle build failed: {r.stderr[:400]}")


def _rasterize(jobs: list[dict]) -> None:
    """Render every queued job in ONE headless Node process (batch; used for the
    quick sizing permutations)."""
    if not jobs:
        return
    jp = RENDER_NODE / "jobs.json"
    jp.write_text(json.dumps(jobs))
    r = subprocess.run(["node", str(RASTERIZE), str(jp)], capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.returncode != 0:
        print(f"  RASTERIZER ERR: {r.stderr[:600]}")


class _Rasterizer:
    """A persistent headless Node rasterizer: ONE GPU device for the whole run,
    fed jobs over stdin so each image appears the moment its mesh is ready (no
    batch wait, no per-image process)."""

    def __init__(self):
        _build_bundle()
        self.proc = subprocess.Popen(
            ["node", str(RASTERIZE), "--stream"], cwd=str(RENDER_NODE),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def render(self, job: dict) -> bool:
        """Render one job, blocking until its PNG is written. Returns success."""
        self.proc.stdin.write(json.dumps(job) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("rasterizer process died")
            line = line.strip()
            if line.startswith("DONE "):
                return True
            if line.startswith("FAIL "):
                print(f"    raster {line}")
                return False

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


def _jobs_for(name: str, kind: str, mp: Path, out_normal: Path, out_debug: Path):
    base = {"mesh": str(mp), "width": RENDER_W, "height": RENDER_H, "featEdges": False}
    # NORMAL: region-coloured fill (+ interior tet edges on the cutaway for volumes)
    # + surface triangulation. Volumes are clipped so the interior shows.
    normal = {**base, "out": str(out_normal), "clip": 0.55 if kind == "vol" else None,
              "fills": True, "surfWire": True, "intWire": kind == "vol", "defects": False, "lineHalfPx": 0.6}
    # DEBUG: wireframe only (thicker, so it reads on transparent) + defect markers.
    debug = {**base, "out": str(out_debug), "clip": None,
             "fills": False, "surfWire": True, "intWire": False, "defects": True, "lineHalfPx": 1.2}
    return normal, debug


def render_corpus() -> None:
    """Renders every geometry in the unified corpus with the NEW constrained
    per-region mesher into TWO directories: `corpus/` (normal view) and
    `corpus_debug/` (diagnostic view), via a PERSISTENT headless WebGPU
    rasterizer (the exact viewer pipeline, no browser). Each geometry's two PNGs
    are produced -- and the debug view annotated with metrics + meshing-time
    breakdown -- the moment its mesh is ready. Both dirs are cleared first."""
    out = GAL / "corpus"
    dbg = GAL / "corpus_debug"
    for d in (out, dbg):
        if d.exists():
            for old in d.glob("*.png"):
                old.unlink()
        d.mkdir(parents=True, exist_ok=True)
    MESHES.mkdir(parents=True, exist_ok=True)
    _set_cdt(True)                              # constrained mesher
    os.environ["RAPIDMESH_PERREGION"] = "1"     # per-region decomposition (straddler-free)
    ras = _Rasterizer()
    try:
        for name, _cat, kind, make in C.CORPUS:
            try:
                t0 = time.time()
                m = make()
                wall = time.time() - t0
                if kind == "surf":
                    vd, n, diag = V._surface_viewer_dict(m, name), len(m.faces), None
                else:
                    vd, n, diag = m.to_viewer_dict(name), int(m.stats["n_tets"]), m.diagnostics
                timings = getattr(m, "timings", None)
                mp = MESHES / f"gal_{name}.json"
                mp.write_text(json.dumps(vd))
                del m
                normal, debug = _jobs_for(name, kind, mp, out / f"{name}.png", dbg / f"{name}.png")
                ras.render(normal)
                if ras.render(debug) and (dbg / f"{name}.png").exists():
                    _annotate(dbg / f"{name}.png", name, kind, n, diag, wall, timings)
                print(f"  {name}: {n} elems, {wall:.1f}s")
            except BaseException as e:  # a mesher gap / panic must not stop the gallery
                print(f"  {name}: FAILED ({type(e).__name__}: {str(e)[:70]})")
    finally:
        ras.close()
        os.environ.pop("RAPIDMESH_PERREGION", None)
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
    _build_bundle()
    jobs = []
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
            jobs.append({"mesh": str(mp), "out": str(png), "clip": 0.7, "fills": True,
                         "surfWire": True, "intWire": True, "featEdges": False, "defects": False,
                         "lineHalfPx": 0.6, "width": RENDER_W, "height": RENDER_H})
            print(f"  {png.name}: {int(m.stats['n_tets'])} tets")
        except BaseException as e:
            print(f"  {label}: FAILED ({type(e).__name__}: {e})")
    _rasterize(jobs)
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

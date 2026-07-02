"""Renders the probe-suite meshes (viewer/public/meshes/probe_*.json, written
by `cargo run --release -p rapidmesh --example probe_suite`) into PNGs via the
headless WebGPU rasterizer: a NORMAL view (region fill + cutaway) and a DEBUG
view (wireframe + located defect markers: sliver amber, straddler magenta,
non-manifold red).

Run:  python bench/render_probe.py
Outputs:  bench/renders/<name>.png  and  bench/renders/<name>_debug.png
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MESHES = REPO / "viewer" / "public" / "meshes"
RENDER_NODE = REPO / "report" / "render-node"
RASTERIZE = RENDER_NODE / "rasterize.mjs"
OUT = REPO / "bench" / "renders"
W, H = 1100, 900


def jobs_for(mp: Path, out_normal: Path, out_debug: Path) -> list[dict]:
    base = {"mesh": str(mp), "width": W, "height": H, "featEdges": False}
    normal = {**base, "out": str(out_normal), "clip": 0.55, "fills": True,
              "surfWire": True, "intWire": True, "defects": False, "lineHalfPx": 0.6}
    debug = {**base, "out": str(out_debug), "clip": None, "fills": False,
             "surfWire": True, "intWire": False, "defects": True, "lineHalfPx": 1.2}
    return [normal, debug]


def annotate(png: Path, stats: dict, n_defects: int) -> None:
    """Best-effort metrics banner on the debug view (PIL, optional)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    img = Image.open(png).convert("RGBA")
    d = ImageDraw.Draw(img)
    txt = (f"tets {stats['n_tets']}   min-dihedral {stats['min_dihedral_deg']:.2f} deg   "
           f"radius-edge {stats['max_radius_edge']:.1f}   defects {n_defects}   "
           f"{stats['millis']} ms")
    d.rectangle([0, 0, img.width, 26], fill=(20, 20, 25, 230))
    d.text((8, 6), txt, fill=(240, 240, 240, 255))
    img.save(png)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meshes = sorted(MESHES.glob("probe_*.json"))
    if not meshes:
        print("no probe_*.json found -- run the probe_suite example first")
        sys.exit(1)
    jobs: list[dict] = []
    meta: list[tuple[Path, dict, int]] = []
    for mp in meshes:
        name = mp.stem.removeprefix("probe_")
        doc = json.loads(mp.read_text())
        dbg = OUT / f"{name}_debug.png"
        jobs += jobs_for(mp, OUT / f"{name}.png", dbg)
        meta.append((dbg, doc.get("stats", {}), len(doc.get("defects", []))))
    jp = RENDER_NODE / "jobs.json"
    jp.write_text(json.dumps(jobs))
    r = subprocess.run(["node", str(RASTERIZE), str(jp)], capture_output=True,
                       text=True, cwd=str(RENDER_NODE))
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.returncode != 0:
        print(f"RASTERIZER ERR: {r.stderr[:800]}")
        sys.exit(1)
    for dbg, stats, nd in meta:
        if dbg.exists() and stats:
            annotate(dbg, stats, nd)
    print(f"rendered {len(meshes)} meshes -> {OUT}")


if __name__ == "__main__":
    main()

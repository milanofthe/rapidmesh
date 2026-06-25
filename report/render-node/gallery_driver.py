"""Drive the headless WebGPU rasterizer from Python (the render_gallery path).

For each geometry: mesh it, write the viewer JSON, queue a NORMAL + a DEBUG job,
run the Node rasterizer ONCE for all jobs (no per-image browser), then overlay
the metrics text + defect legend (reusing render_gallery._annotate) on the debug
PNGs. Feature edges are OFF -- the gallery never showed them.
"""
import os, sys, json, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "report"))
sys.path.insert(0, str(REPO / "python" / "examples"))
os.environ["RAPIDMESH_CDT"] = "1"
os.environ["RAPIDMESH_PERREGION"] = "1"

import rapidmesh as rm  # noqa: E402
from report import render_gallery as RG  # noqa: E402

HERE = Path(__file__).resolve().parent
MESHES = REPO / "report" / "validation" / "meshes"
OUT = REPO / "report" / "figures" / "webgpu"


def render(items, width=1500, height=1230):
    """items: list of (name, mesh_obj). Renders normal+debug PNGs with overlay."""
    jobs, meta = [], []
    for name, m in items:
        vd = m.to_viewer_dict(name)
        mp = MESHES / f"gal_{name}.json"
        mp.write_text(json.dumps(vd))
        normal = OUT / f"g_{name}.png"
        debug = OUT / f"g_{name}_debug.png"
        jobs.append({"mesh": str(mp), "out": str(normal), "clip": 0.55,
                     "fills": True, "surfWire": True, "intWire": True, "featEdges": False, "defects": False,
                     "lineHalfPx": 0.6, "width": width, "height": height})
        # Debug = wireframe-only: thicker lines so the dark mesh stays visible on
        # the transparent background (no fills to carry the body).
        jobs.append({"mesh": str(mp), "out": str(debug), "clip": None,
                     "fills": False, "surfWire": True, "intWire": False, "featEdges": False, "defects": True,
                     "lineHalfPx": 1.2, "width": width, "height": height})
        meta.append((name, debug, int(m.stats["n_tets"]), m.diagnostics))
    jp = HERE / "jobs.json"
    jp.write_text(json.dumps(jobs))
    out = subprocess.run(["node", str(HERE / "rasterize.mjs"), str(jp)], capture_output=True, text=True)
    print(out.stdout.strip())
    if out.returncode != 0:
        print("RASTERIZER ERR:", out.stderr[:400])
    # overlay metrics + defect legend on the debug PNGs (reuse render_gallery)
    for name, debug, n, diag in meta:
        RG._annotate(debug, name, "vol", n, diag)
    return [d for _, d, _, _ in meta]


if __name__ == "__main__":
    import showcase as SC
    from report import validate as V
    sc = {x.id: x for x in SC.MODELS}
    vc = {c[0]: (c[4], c[3]) for c in V.CASES}

    def mesh_showcase(nm):
        return sc[nm].build().mesh()

    def mesh_validate(nm):
        bf, h = vc[nm]; g = rm.Geometry(maxh=h); bf(g, h); return g.mesh(maxh=h)

    items = [("bearing", mesh_showcase("bearing")), ("diff_box_cyl", mesh_validate("diff_box_cyl"))]
    debugs = render(items)
    print("debug PNGs:", *[str(d) for d in debugs])

"""Render only the rapidpassives corpus cases (normal + debug), reusing the
gallery rasterizer. Output: figures/gallery/passives{,_debug}/."""
import json
import time

import corpus as C
import render_gallery as RG
from report import validate as V


def main():
    out = RG.GAL / "passives"
    dbg = RG.GAL / "passives_debug"
    for d in (out, dbg):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.png"):
            old.unlink()
    RG.MESHES.mkdir(parents=True, exist_ok=True)
    ras = RG._Rasterizer()
    try:
        for name, _cat, kind, make in C.CORPUS:
            if not name.startswith("rp_"):
                continue
            try:
                t0 = time.time()
                m = make()
                wall = time.time() - t0
                if kind == "surf":
                    vd, n, diag = V._surface_viewer_dict(m, name), len(m.faces), None
                else:
                    vd, n, diag = m.to_viewer_dict(name), int(m.stats["n_tets"]), m.diagnostics
                timings = getattr(m, "timings", None)
                mp = RG.MESHES / f"gal_{name}.json"
                mp.write_text(json.dumps(vd))
                del m
                normal, debug = RG._jobs_for(name, kind, mp, out / f"{name}.png", dbg / f"{name}.png")
                ras.render(normal)
                if ras.render(debug) and (dbg / f"{name}.png").exists():
                    RG._annotate(dbg / f"{name}.png", name, kind, n, diag, wall, timings)
                print(f"  {name}: {n} elems, {wall:.1f}s")
            except BaseException as e:
                print(f"  {name}: FAILED ({type(e).__name__}: {str(e)[:70]})")
    finally:
        ras.close()


if __name__ == "__main__":
    main()

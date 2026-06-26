"""Copy the built viewer (site/build, minus the showcase sample meshes) into the
rapidmesh package as _viewer_dist/, so `mesh.show()` ships the viewer in the wheel.

Run after `cd site && npm run build` whenever the viewer changes.
"""
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "site" / "build"
DST = REPO / "python" / "python_src" / "rapidmesh" / "_viewer_dist"

if not (SRC / "embed.html").is_file():
    raise SystemExit(f"build the viewer first: cd site && npm run build  (missing {SRC/'embed.html'})")
if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True)
for item in ("CNAME", "_app", "embed.html", "favicon.svg", "index.html"):
    s = SRC / item
    if s.is_dir():
        shutil.copytree(s, DST / item)
    elif s.is_file():
        shutil.copy2(s, DST / item)
print(f"synced viewer -> {DST.relative_to(REPO)} (sample meshes excluded)")

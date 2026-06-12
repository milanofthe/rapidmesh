"""Meshes every showcase model (python/examples/showcase.py) and exports it
into the showcase site (site/static/meshes/<id>.json plus manifest.json).

Run from the repo root:

    python python/examples/export_showcase.py [ids...]

With ids only those models are re-exported; the manifest is rebuilt from the
full registry either way, so partial runs keep the site consistent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from showcase import MODELS

OUT = Path(__file__).resolve().parents[2] / "site" / "static" / "meshes"
TET_CAP = 150_000


def main(argv: list[str]) -> None:
    wanted = set(argv) if argv else None
    OUT.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        if wanted and m.id not in wanted:
            continue
        try:
            mesh = m.build().mesh()
        except Exception as e:  # noqa: BLE001 - export the rest regardless
            print(f"{m.id:<18} FAILED: {type(e).__name__}: {e}")
            continue
        s = mesh.stats
        if s["n_tets"] > TET_CAP:
            print(f"{m.id:<18} {s['n_tets']:>8} tets exceeds cap, not exported")
            continue
        (OUT / f"{m.id}.json").write_text(json.dumps(mesh.to_viewer_dict(m.id)))
        kb = (OUT / f"{m.id}.json").stat().st_size // 1024
        print(f"{m.id:<18} {s['n_tets']:>8} tets  min-dih "
              f"{s['min_dihedral_deg']:5.1f}  {s['millis']:>6} ms  {kb:>6} kB")
        sys.stdout.flush()

    manifest = {"models": []}
    for m in MODELS:
        path = OUT / f"{m.id}.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())["stats"]
        manifest["models"].append({
            "id": m.id,
            "name": m.name,
            "file": f"meshes/{m.id}.json",
            "stats": {
                "n_tets": s["n_tets"],
                "n_points": s["n_points"],
                "min_dihedral_deg": round(s["min_dihedral_deg"], 1),
            },
        })
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {len(manifest['models'])} models")


if __name__ == "__main__":
    main(sys.argv[1:])

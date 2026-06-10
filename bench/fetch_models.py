"""Fetches public surface-mesh test models into bench/models/ (gitignored).

Source: github.com/alecjacobson/common-3d-test-models (mixed public-domain /
CC models, OBJ format). The bench harness imports, validates (watertight,
consistently oriented) and meshes whatever lands here; models that fail
validation are skipped with a message.
"""

import sys
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data"

# Small to mid-size models first; comment in the larger ones as needed.
MODELS = [
    "spot.obj",        # ~5.9k faces, watertight cow
    "cheburashka.obj", # ~13.3k faces
    "fandisk.obj",     # ~13k faces, CAD classic with creases
    "stanford-bunny.obj",  # ~70k faces (the classic; hole-status checked by validation)
    "armadillo.obj",   # ~50k faces
]

def main() -> None:
    out = Path(__file__).parent / "models"
    out.mkdir(exist_ok=True)
    for name in MODELS:
        dst = out / name
        if dst.exists():
            print(f"{name}: already present")
            continue
        url = f"{RAW}/{name}"
        print(f"{name}: fetching {url}")
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as e:
            print(f"{name}: FAILED ({e})", file=sys.stderr)

if __name__ == "__main__":
    main()

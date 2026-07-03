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

# The SOTA-mesher reference set (small to mid-size first). Every model runs
# through validate_closed on import; non-watertight classics (bunny has holes
# in some distributions) are skipped there, not here.
MODELS = [
    "spot.obj",         # ~5.9k faces, watertight cow (organic, smooth)
    "cheburashka.obj",  # ~13.3k faces (organic)
    "fandisk.obj",      # ~13k faces, CAD classic with creases
    "stanford-bunny.obj",  # ~70k faces (the classic; hole-status checked by validation)
    "armadillo.obj",    # ~50k faces (organic, high detail)
    "homer.obj",        # ~12k faces (organic)
    "max-planck.obj",   # ~100k faces (scan bust)
    "nefertiti.obj",    # ~100k faces (scan bust)
    "igea.obj",         # ~270k faces (scan bust, Venus/Igea)
    "beast.obj",        # ~64k faces (organic)
    "cow.obj",          # ~5.8k faces (organic classic)
    "teapot.obj",       # Utah teapot (open surface in most distributions; validation decides)
    "bimba.obj",        # ~150k faces (scan bust)
    "lucy.obj",         # large scan (validation/size decides)
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

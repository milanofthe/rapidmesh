"""High-facet benchmark: meshes the organic / high-facet comparison geometries
with rapidmesh and prints per-stage timings, so we can see whether the stages
that scale with boundary-facet count F (domain build, classification, inside
ray-casts) dominate here, the way the parametric low-facet bench (bench.rs)
showed they do NOT. This decides whether the facet BVH (WP9-1) is worth it.

    python python/examples/bench_facets.py
"""

from __future__ import annotations

import time

from compare_geoms import GEOMS

# High-facet / organic models from the showcase corpus.
HIGH_F = ["bunny", "blob", "gear", "fused_spheres", "nested_spheres"]

STAGES = ["mesh.domain", "mesh.surface", "mesh.seed", "mesh.lloyd", "mesh.build_final", "mesh.tilings", "mesh.region"]


def main():
    by_id = {g.id: g for g in GEOMS}
    print("=== high-facet rapidmesh bench (times in ms) ===\n")
    hdr = f"{'geom':<14} {'tets':>8} {'srf_faces':>9} | " + " ".join(f"{s.split('.')[1]:>8}" for s in STAGES) + f" | {'total':>8}"
    print(hdr)
    for gid in HIGH_F:
        geom = by_id.get(gid)
        if geom is None:
            continue
        g = geom.build_rapidmesh()
        t0 = time.perf_counter()
        m = g.mesh(maxh=geom.target_h)
        total = (time.perf_counter() - t0) * 1e3
        tim = m.timings  # dict {stage: seconds}
        cells = [f"{tim.get(s, 0.0) * 1e3:>8.1f}" for s in STAGES]
        print(f"{gid:<14} {len(m.tets):>8} {len(m.faces):>9} | " + " ".join(cells) + f" | {total:>8.1f}")


if __name__ == "__main__":
    main()

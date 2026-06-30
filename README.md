# rapidmesh

Tetrahedral mesh generator for 3D electromagnetic FEM (Maxwell, H(curl)/Nédélec)
in pure Rust. Solid primitives and **exact** CSG booleans build a non-manifold
boundary representation; each region's interior is filled by a constrained
tetrahedralization against a frozen, watertight surface, then optimized for
dihedral-angle quality. A Python builder API drives the whole pipeline.

Replaces the gmsh dependency of [rapidfem](https://github.com/milanofthe/rapidfem).

## Pipeline

1. **Geometry** — primitives (box, cylinder, sphere, cone, torus, prism, sweep,
   loft) assembled into a tagged piecewise-linear complex.
2. **Exact CSG** — a robust arrangement of the input surfaces (exact predicates,
   no float snapping) yields a non-manifold B-rep; material interfaces stay
   exactly conforming.
3. **Surface mesh** — every B-rep face is meshed on its analytic carrier (planes,
   revolution barrels, spheres) into a watertight, region-tagged surface.
4. **Volume mesh** — the surface is frozen and each region is tetrahedralized
   **separately** under a gradient-limited sizing field, with the frozen surface
   as a hard constraint: watertight by construction, with no tets straddling a
   region interface.
5. **Optimization** — dihedral-angle-targeted smoothing, topological operations,
   and a dedicated sliver stage.

## Workspace

| Crate | Purpose |
| --- | --- |
| `rapidmesh-exact` | Exact arithmetic: expansions, interval filter, implicit points, staged predicates |
| `rapidmesh-geom` | Solid primitives, the tagged PLC, surface back-references |
| `rapidmesh-csg` | Exact mesh arrangements, multi-operand boolean expressions |
| `rapidmesh-brep` | Non-manifold boundary-representation layer between CSG and the mesher |
| `rapidmesh-tet` | Surface + per-region constrained tetrahedralization, the shared 2D core (`surf2d`), sizing fields, quality optimization |
| `rapidmesh-topo` | Mesh topology + element geometry, and the embedding endpoints (`mesh_2d` / `mesh_layers` / `mesh_3d`) that bundle mesh + topology + geometry in one call |
| `rapidmesh` | Facade builder API and mesh export |
| `rapidmesh-testutil` | Shared test utilities (dev-dependency only) |

The Python extension lives in `python/` (PyO3 + maturin).

## 2D meshing (planar / MoM)

The same `surf2d` core that meshes each 3D surface patch is also the standalone 2D
planar mesher — a graded, sliver-free constrained Delaunay triangulation of tagged
polygons-with-holes, used by [rapidmom](https://github.com/milanofthe/rapidmom) for
2.5D method-of-moments and by the WebGL landing page. There is **one** entry point
per level; everything (including the triangle budget) is set through parameters.

```rust
use rapidmesh_topo::{mesh_layers, mesh_2d, Region2D, Mesh2DOptions};

// A tagged polygon (outer CCW, holes CW); `tag` flows to every triangle it owns.
let sq = Region2D::new(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], 7);

// Single layer, field-driven: `h(x)` is the desired edge length at any point.
let m = mesh_2d(&[sq], |_p| 0.05, &Mesh2DOptions::default());
m.points;     // Vec<[f64; 2]>          vertices
m.tris;       // Vec<[u32; 3]>          triangles (CCW)
m.tri_tags;   // Vec<i64>               region tag per triangle
m.topo;       // TriTopology            edges, incidence, RWG edges, boundary edges
m.geom;       // TriGeometry            areas, centroids, edge lengths/midpoints

// MANY layers under ONE global triangle budget. Within a group, overlapping /
// abutting regions are unioned into one RWG-connected component; regions of
// different groups never merge (free to overlap in the plane). The budget is
// shared across every patch of every group — distributed emergently, not by an
// a-priori per-layer split. `opts.target_count > 0` switches on the budget.
let opts = Mesh2DOptions { target_count: 20_000, ..Default::default() };
let layers: Vec<Mesh2D> = mesh_layers(&[group0, group1], |_p| 0.05, &opts);
```

### API reference

| Item | Crate | Role |
| --- | --- | --- |
| `mesh_layers(groups, h, opts) -> Vec<Mesh2D>` | `rapidmesh-topo` | Grouped endpoint: per-group union, then ONE global triangle budget across all groups |
| `mesh_2d(regions, h, opts) -> Mesh2D` | `rapidmesh-topo` | Single-layer convenience (`mesh_layers` with one group) |
| `Region2D { outer, holes, tag }` | `rapidmesh-topo` | A tagged polygon-with-holes in the xy plane |
| `Mesh2DOptions { min_angle_deg, cvt_iters, max_passes, target_count }` | `rapidmesh-topo` | Quality bound, work bounds, and the triangle budget (`0` = field-driven) |
| `Mesh2D { points, tris, tri_tags, topo, geom, regions, opts }` | `rapidmesh-topo` | The full 2D bundle; `remesh(h)` re-meshes the same regions (AMR) |
| `surf2d::mesh_polygon(loops, h, &params, on_pass)` | `rapidmesh-tet` | THE loops core: contour loops + sizing field → `(points, tris)` |
| `surf2d::PolyMeshParams { step, min_angle_deg, target_count, cvt_iters, max_passes }` | `rapidmesh-tet` | Knobs for `mesh_polygon` (`step` seeds the CVT grid; `target_count` caps the count) |

Under a budget the count is hit in a single pass: a uniform size
`h ≈ √(area / budget)` seeds the CVT grid, the Ruppert cap trims any
over-refinement, and `mesh_layers` packs all patches into one triangulation so the
worst (most over-sized / angle-violating) elements across the whole layout draw
from the shared count first.

Enable with the `mesher` feature on `rapidmesh-topo`.

## Python

```python
import rapidmesh as rm

g = rm.Geometry(maxh=0.4)
g.box(4, 4, 2)
g.cylinder(radius=0.8, height=2, position=(2, 2, 0), void=True)  # a bore

mesh = g.mesh()
mesh.points        # (n_points, 3) float64
mesh.tets          # (n_tets, 4)   uint64
mesh.tet_regions   # region tag per tet
mesh.faces         # surface faces (region interfaces, tagged sheets)
```

See [python/README.md](python/README.md) for the builder API, the returned mesh
arrays, and the observability surface (timings, metrics, quality, log).

## Visualization

- `site/` — the auto-cycling 3D mesh gallery (SvelteKit + WebGL2), deployed to
  `mesh.rapidpassives.org`. See [site/README.md](site/README.md).
- `report/render-node/` — a headless WebGPU rasterizer that renders the same
  scenes to PNG without a browser (drives the report gallery).
- `viewer/` — a standalone side-by-side mesh comparison viewer (rapidmesh / gmsh
  / tetgen), fed by `cargo run --release --bin export_meshes`.

## License

[MIT](LICENSE).

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
| `rapidmesh-tet` | Surface + per-region constrained tetrahedralization, sizing fields, quality optimization |
| `rapidmesh` | Facade builder API and mesh export |
| `rapidmesh-testutil` | Shared test utilities (dev-dependency only) |

The Python extension lives in `python/` (PyO3 + maturin).

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

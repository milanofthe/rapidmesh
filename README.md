# rapidmesh

Tetrahedral mesh generator for 3D EM FEM (Maxwell, H(curl)/Nédélec) in pure Rust.
Primitives + exact CSG booleans, constrained Delaunay tetrahedralization with
exact boundary conformity, sizing-field-driven refinement, dihedral-angle-targeted
quality optimization. STEP import planned.

Replaces the gmsh dependency of [rapidfem](https://github.com/milanofthe/rapidfem).


## Workspace

| Crate | Purpose |
| --- | --- |
| `rapidmesh-exact` | Exact arithmetic: expansions, interval filter, implicit points, staged predicates |
| `rapidmesh-geom` | Primitives, tagged PLC representation, surface back-references |
| `rapidmesh-csg` | Exact mesh arrangements, multi-operand boolean expressions |
| `rapidmesh-tet` | CDT, Delaunay refinement, sizing fields, quality pass |
| `rapidmesh` | Facade API and mesh export |

Proprietary. All rights reserved.

## Viewer

Standalone mesh comparison viewer (WebGL2, extracted from rapidfem):

```powershell
cargo run --release --bin export_meshes   # writes viewer/public/meshes/*.json
cd viewer; npm install; npm run dev        # http://localhost:5199 (or vite default)
```

One panel per mesher (rapidmesh / gmsh / tetgen, whichever JSON exists), synced cameras, surface/wireframe/tet-fill toggles, movable clip plane.

# rapidmesh

Tetrahedral mesh generator for 3D EM FEM (Maxwell, H(curl)/Nédélec) in pure Rust.
Primitives + exact CSG booleans, constrained Delaunay tetrahedralization with
exact boundary conformity, sizing-field-driven refinement, dihedral-angle-targeted
quality optimization. STEP import planned.

Replaces the gmsh dependency of [rapidfem](https://github.com/milanofthe/rapidfem).

See [DESIGN.md](DESIGN.md) for architecture, research basis, and roadmap.

## Workspace

| Crate | Purpose |
| --- | --- |
| `rapidmesh-exact` | Exact arithmetic: expansions, interval filter, implicit points, staged predicates |
| `rapidmesh-geom` | Primitives, tagged PLC representation, surface back-references |
| `rapidmesh-csg` | Exact mesh arrangements, multi-operand boolean expressions |
| `rapidmesh-tet` | CDT, Delaunay refinement, sizing fields, quality pass |
| `rapidmesh` | Facade API and mesh export |

Proprietary. All rights reserved.

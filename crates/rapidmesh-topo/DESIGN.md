# `rapidmesh-topo` — analysis-ready cell complex

The solver-agnostic, dimension-uniform view of a mesh: the 0/1/2/3-cells with
their incidence and per-element geometry. This is what makes rapidmesh a *general
mesher optimized for FEM/MoM embedding* rather than a file emitter — solvers
consume connectivity + element geometry directly and never rebuild topology or
round-trip a `.msh`/JSON.

It is **not** a solver and **not** FEM-specific. Basis-aware machinery
(RWG / Nédélec DOF maps, quadrature rules) layers *on top* of this crate.

## Integration — there are exactly two endpoints

A Rust solver embeds rapidmesh through **one** of two front doors (enable the
`mesher` feature). Each returns a complete bundle — do not assemble pieces by hand
and do not use any other accessor:

| | call | returns | for |
|---|---|---|---|
| **2D** | `rapidmesh_topo::mesh_2d(&plc, &params)` | `Mesh2D { mesh, topo, geom }` + `rwg_candidate_edges()`, `boundary_edges()`, `edges_on_line()`, `exact_face_normals()` | MoM (rapidmom) |
| **3D** | `rapidmesh_topo::mesh_3d(&plc, &params)` | `Mesh3D { mesh, topo, geom }` + `exact_face_normals()` | FEM (rapidfem) |

Already have a `SurfaceMesh`/`TetMesh`? Use `Mesh2D::build` / `Mesh3D::build`.

Ignore for embedding: `rapidmesh_tet::{mesh_cdt, surface_mesh, …}` (the internal
mesher these wrap) and `rapidmesh_tet::mom` (the Python bindings' bridge). Both
predate the endpoints; neither is the Rust front door.

## Why this exists

Today the downstream solvers rebuild what the mesher already knew:

- **rapidfem** loads a `.msh` and reconstructs `edges`, `tris`, `tet_to_edge`,
  `tet_to_tri`, `tri_to_edge`, `tri_to_tet`, `edge_lengths` from the tets
  (`rapidfem-core/src/mesh.rs:74-159`), then per-tet volume/∇λ
  (`tet_assembly.rs:71`).
- **rapidmom** rebuilds the RWG edge graph from triangles (`mom-mesh/src/rwg.rs`)
  and a `TriGeom` cache (area/centroid/inertia, `mom-core/src/geom.rs`).

All of it is derivable from mesh + geometry in one O(n) pass. `rapidmesh-topo`
owns that derivation, once, for 2D and 3D alike.

## Dimension-uniform shape

Topology is pure connectivity — identical whether a triangle mesh is planar (MoM)
or embedded in 3D (a surface). Only geometry is coordinate-aware (2 vs 3
components).

```
TriTopology   edges, tri_edges, edge_tris, edge_tags, vert_tris
TetTopology   edges, faces, tet_edges, tet_edge_sign, tet_faces, tet_face_sign,
              face_edges, face_tets, vert_edges, vert_tets

TriGeometry::build_2d(&topo, &[[f64;2]])   area, centroid, inertia
TriGeometry::build_3d(&topo, &[[f64;3]])   + normal
TetGeometry::build   (&topo, &[[f64;3]])   volume, grad (∇λ_i), face_normal, …
```

`tet_grad` (∇λ_i) is the inverse Jacobian — all a P1/Nédélec element assembly
needs. `tri_inertia` is the second area moment a multipole MoM fill needs.

## "Build from anything" — source adapters

The builders take a small trait, so rapidmom's planar `Mesh<Vec2>`, our
`SurfaceMesh`/`TetMesh`, or raw external arrays all feed the same complex:

```rust
pub trait TriSource { fn n_verts(&self)->usize; fn n_tris(&self)->usize;
                      fn tri(&self,i:usize)->[u32;3]; fn tri_tag(&self,i:usize)->i64 { 0 } }
pub trait TetSource { fn n_verts(&self)->usize; fn n_tets(&self)->usize;
                      fn tet(&self,i:usize)->[u32;4]; }

let cx = TriTopology::build(&any_tri_source);   // one path, 2D or 3D
```

The `mesher` feature provides the adapters for our own output types; external
callers impl the traits (or use the `Tris`/`Tets` slice wrappers).

## Conventions (documented + tested)

The signs are the payload — they let the solver skip orientation work.

- **Canonical edge** `(min, max)` vertex id; the local direction lives in a sign.
- **`TRI_EDGE_LOCAL`** = `[(0,1),(1,2),(2,0)]`.
- **`TET_EDGE_LOCAL`** = `[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]`.
- **`TET_FACE_LOCAL`** = `[(1,2,3),(0,3,2),(0,1,3),(0,2,1)]` — face *i* excludes
  local vertex *i*, ordered so its normal points **outward** of a positive tet
  (proved by a unit-tet test).
- **`tet_edge_sign[k]`** = `+1` if the local edge runs min→max (== canonical).
- **`tet_face_sign[k]`** = parity of the permutation from the local outward order
  to the canonical (sorted) face. Two tets sharing a face carry **opposite**
  signs (consistent orientation) — a convention test asserts this.
- **`NONE = u32::MAX`** is the missing neighbour (boundary) sentinel (matches
  rapidfem's `usize::MAX`).
- Face normal: interior `t0 → t1`, boundary outward.

## Embedding-unique hooks (a `.msh` cannot carry these)

- **Exact boundary normals / curvature** where a face lies on an analytic
  `SurfaceKind` (`Surface::normal` / `curvature_radius`) instead of the facet
  normal — the substrate for curved / higher-order elements. Opt-in.
- **AMR-cheap**: the complex is rebuild-cheap, so a solver can refine
  (`dorfler_mark`) and re-derive while carrying the solution — a true
  solver-in-the-loop adaptive cycle, impossible across files.

## Zero-copy / cross-process

Every field is POD `Vec<[T; N]>` → `bytemuck`-castable to `&[u8]`. A stable
header (counts + offsets) is simultaneously the mmap-able wire format for the
Python / cross-process path; no per-element serialization.

## Abstraction boundary

rapidmesh provides **facts about the mesh**; the solver picks the
**discretization**. Anything presupposing a basis is out of this crate and lives
in the solver repos:

- **In** (here): topology/incidence, orientation signs, element geometry
  (length, area, volume, centroid, normal, ∇λ_i, second moment) — all basis-free.
- **Out** (rapidfem / rapidmom): DOF numbering (RWG / Nédélec / Lagrange),
  quadrature rules, shape functions, assembly, materials, ports-as-excitation.

`∇λ_i` and the triangle second moment sit just inside the line: they are pure
simplex calculus / geometric moments (no basis), offered because they are the
hottest per-element O(n) loops the solvers would otherwise redo.

## Status

- **Done** — the rapidmesh side is complete:
  - `Csr`, conventions (outward-face + shared-face-sign tests).
  - `TriTopology`/`TetTopology` (Tier 1) + `TriGeometry`/`TetGeometry` (Tier 2:
    area/volume/centroid/normal/∇λ_i/inertia/edge-length).
  - `wire`: flat POD frame (`to_wire`/`from_wire`) — no serialization, bulk copy,
    mmap-friendly buffer (topology only; geometry recomputes from coords).
  - `mesher` feature: source adapters for `TetMesh`/`SurfaceMesh` plus the
    analytic exact-normal / curvature hook (`surface_normal`,
    `surface_curvature`, `exact_face_normals`).
  - 19 tests + a doctest.
- **Out of scope** (solver-side): DOF builders, quadrature, assembly, physics.

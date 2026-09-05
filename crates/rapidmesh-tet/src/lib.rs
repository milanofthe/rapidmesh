//! Tetrahedral meshing: restricted-Delaunay refinement against analytic
//! carrier oracles (see `docs/refinement-core.md`).
//!
//! Pipeline (ONE volume engine, [`refine`], behind every entry point):
//! 1. Protect the 0/1-features (B-rep corners + edge curves) Ruppert-style
//!    with shrinking balls; sample every edge on its analytic curve.
//! 2. Refine surface facets (restricted-Delaunay surface balls against the
//!    trimmed carriers) and interior cells (circumcenters under a radius-edge
//!    + size bound), with manifold pierce-repair sweeps.
//! 3. ODT (Lloyd) relaxation of the interior, then flood-fill region
//!    classification over the carrier walls.
//! 4. Quality pass targeting the minimal dihedral angle: exudation, edge
//!    removal, smoothing, sliver stages (`optimize`).

// Public surface: the 2D core (`surf2d`), adaptive marking (`adapt`), the
// sizing-field cache (`quadfield`), and `diagnostics`. The MoM/FEM topology +
// quality accessors live in the downstream `rapidmesh_topo` layer (one
// implementation for both the 2D and the 3D-surface endpoint). Everything else
// is the internal mesher engine -- `pub(crate)`, reached only through the
// re-exported entry points below (`mesh_budgeted` / `surface_mesh` / `mesh_plc` /
// `tetrahedralize` / ...). The canonical embedding front door is
// `rapidmesh_topo::{mesh_2d, mesh_3d}`.
pub mod adapt;
pub mod diagnostics;
pub mod gradefield;
pub mod quadfield;
pub mod surf2d;

pub(crate) mod brep_mesh;
pub(crate) mod conform;
pub(crate) mod constants;
pub(crate) mod curve;
pub(crate) mod cvt;
pub(crate) mod delaunay;
pub(crate) mod domain;
pub(crate) mod facetbvh;
mod geomutil;
pub(crate) mod optimize;
pub(crate) mod project;
pub(crate) mod refine;
pub(crate) mod spatial;
pub(crate) mod surfchart;

pub use adapt::dorfler_mark;
pub use conform::{
    log_metrics, log_surface_metrics, mesh_plc, mesh_plc_with, quality_stats, MeshParams,
    QualityStats, SurfaceFace, SurfaceMesh, TetMesh,
};
pub use cvt::{mesh_budgeted, surface_mesh};
pub use delaunay::{tetrahedralize, DelaunayBuilder, DelaunayTets};
pub use optimize::{optimize, OptimizeParams};
pub use refine::mesh_refine;

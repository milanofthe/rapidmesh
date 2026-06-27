//! Tetrahedral meshing: CVT (centroidal Voronoi) variational meshing.
//!
//! Pipeline (see the CVT-rewrite plan):
//! 1. Seed a BCC background lattice over the domain, graded by the sizing field
//!    (`seed`), filtered to the domain interior.
//! 2. Relax toward a centroidal Voronoi layout by Lloyd iteration on an
//!    incremental Delaunay (`cvt` driving `delaunay`), keeping boundary and
//!    interface sites on their carriers (restricted CVT) so material interfaces
//!    stay conforming; recover internal interfaces by on-plane refinement.
//! 3. Quality pass targeting the minimal dihedral angle: edge removal,
//!    smoothing, sliver removal (`optimize`).
//! 4. Optional order-2 midside snapping onto the analytic surface via the
//!    PLC surface back-references (`project`), for curved boundaries.

// Public surface: the 2D core (`surf2d`), the MoM/FEM accessors (`mom`, the
// Python bridge), adaptive marking (`adapt`), the sizing-field cache
// (`quadfield`), and `diagnostics`. Everything else is the internal mesher
// engine -- `pub(crate)`, reached only through the re-exported entry points
// below (`mesh_cdt` / `surface_mesh` / `mesh_plc` / `tetrahedralize` / ...).
// The canonical embedding front door is `rapidmesh_topo::{mesh_2d, mesh_3d}`.
pub mod adapt;
pub mod diagnostics;
pub mod mom;
pub mod quadfield;
pub mod surf2d;

pub(crate) mod brep_mesh;
pub(crate) mod cdt3;
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
pub(crate) mod seed;
pub(crate) mod sizefield;
pub(crate) mod site;
pub(crate) mod spatial;
pub(crate) mod surfchart;

pub use conform::{
    log_metrics, log_surface_metrics, mesh_plc, mesh_plc_with, quality_stats, MeshParams,
    QualityStats, SurfaceFace, SurfaceMesh, TetMesh,
};
pub use cvt::{frozen_surface, mesh_cdt, mesh_cdt_budgeted, surface_mesh, FrozenSurface};
pub use adapt::dorfler_mark;
pub use delaunay::{tetrahedralize, DelaunayBuilder, DelaunayTets};
pub use mom::EdgeAdjacency;
pub use optimize::{optimize, OptimizeParams};

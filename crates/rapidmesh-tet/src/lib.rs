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

pub mod conform;
pub mod cvt;
pub mod delaunay;
pub mod domain;
pub mod facetbvh;
pub mod optimize;
pub mod project;
pub mod seed;
pub mod site;
pub mod spatial;
pub mod surf2d;

pub use conform::{mesh_plc, mesh_plc_with, quality_stats, MeshParams, QualityStats, SurfaceFace, TetMesh};
pub use delaunay::{tetrahedralize, DelaunayBuilder, DelaunayTets};
pub use optimize::{optimize, OptimizeParams};

//! Central tuning constants for the tet mesher.
//!
//! Every numeric knob that shapes mesher BEHAVIOUR (sampling densities, relaxation
//! iteration caps, quality thresholds, spatial-structure capacities, the sizing
//! field's factors) lives here, grouped and documented, so the meshing recipe is
//! tunable from one place rather than scattered across modules. Pure structural
//! data shared by several modules (the tet face table) is here too, to remove
//! duplication. Algorithm-internal sentinels (`NONE`, bit masks, the FP splitter)
//! stay with their algorithms; constants of the lower crates (`rapidmesh-exact`,
//! `-geom`, `-csg`) stay there -- the dependency direction forbids one file across
//! crates, and they are not mesher tuning knobs.

// ---- surface vs volume sampling -------------------------------------------
/// The unified surface is meshed FINER than the volume by this factor, so the
/// restricted-Delaunay boundary recovers cleanly and volume tets cannot straddle
/// the exact PLC boundary (surface size = `OVERSAMPLE * H`, volume at `H`).
pub(crate) const OVERSAMPLE: f64 = 0.7;
/// Old-path surface oversampling for the (retired) restricted-readback mesher.
pub(crate) const SURFACE_OVERSAMPLE: f64 = 0.5;

// ---- Lloyd / CVT relaxation -----------------------------------------------
/// Volume 3D Lloyd passes (relax interior sites to their CVT layout).
pub(crate) const LLOYD_ITERS: usize = 8;
/// Lloyd converges when the largest site move drops below this fraction of the
/// spacing (and nothing new was inserted).
pub(crate) const LLOYD_CONVERGE_FRAC: f64 = 0.02;
/// 2D surface Lloyd passes for a planar/chart face (`cvt_fill`).
pub(crate) const SURF_LLOYD_ITERS: usize = 4;

// ---- seeding / domain bounds ----------------------------------------------
/// Fallback base subdivision of the bbox diagonal when no finite size cap exists.
pub(crate) const DEFAULT_SUBDIV: f64 = 8.0;
/// Relative bbox padding for the inside/classification triangle boxes.
pub(crate) const BOX_PAD_FRAC: f64 = 1e-6;
/// Interior-seed separation as a fraction of the local size (no sliver seeds).
pub(crate) const SEPARATION_FRAC: f64 = 0.45;

// ---- quality diagnostics --------------------------------------------------
/// A tet whose smallest dihedral angle is below this (degrees) is a sliver.
pub const SLIVER_DEG: f64 = 10.0;

// ---- quality optimization (optimize.rs) -----------------------------------
/// New edges up to this multiple of the local size target are legal (the same
/// slack the mesher's own max-edge contract uses).
pub(crate) const EDGE_CONTRACT: f64 = 1.5;
/// Edges shorter than this fraction of the local target are collapse candidates.
pub(crate) const COARSEN_FRACTION: f64 = 0.5;
/// Local complexes already at/above this `-max|cos(dihedral)|` quality
/// (min dihedral ~35 deg) are left alone (HXT recipe). `-cos(35 deg)`.
pub(crate) const TARGET_Q: f64 = -0.8191520442889918;
/// Degenerate-quality epsilon below which a tet is treated as flat.
pub(crate) const QUALITY_EPS: f64 = 1e-12;
/// A smoothing move shorter than this fraction of the local size is skipped.
pub(crate) const MIN_REL_MOVE: f64 = 1e-3;
/// Max edge ring size handled by edge removal.
pub(crate) const MAX_RING: usize = 12;
/// Vertex insertion targets tets whose min dihedral is below this (degrees).
pub(crate) const INSERT_BELOW_DEG: f64 = 10.0;
/// ...and allows the inserted point's radius-edge up to this.
pub(crate) const INSERT_RE_ALLOW: f64 = 16.0;

// ---- spatial structures ----------------------------------------------------
/// Point octree leaf capacity (`spatial.rs`).
pub(crate) const OCTREE_LEAF_CAP: usize = 16;
/// Point octree max depth (`spatial.rs`).
pub(crate) const OCTREE_MAX_DEPTH: u32 = 24;
/// Domain octree max refinement depth (`domain.rs`).
pub(crate) const DOMAIN_MAX_DEPTH: u32 = 18;
/// Facet BVH leaf size (`facetbvh.rs`).
pub(crate) const BVH_LEAF_MAX: usize = 4;

// ---- topology --------------------------------------------------------------
/// The four faces of a tet as local vertex-index triples (opposite vertex 0..3).
pub(crate) const TET_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];

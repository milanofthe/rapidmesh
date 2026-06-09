//! Tetrahedral meshing: CDT, Delaunay refinement, sizing, quality.
//!
//! Pipeline (see DESIGN.md):
//! 1. Constrained Delaunay tetrahedralization with exact boundary conformity;
//!    boundary recovery via implicitly represented Steiner points (indirect
//!    predicates, floating-point only — Diazzi et al., SIGGRAPH Asia 2023).
//! 2. Ruppert/Shewchuk Delaunay refinement driven by a sizing field
//!    (wavelength default + local feature size + user maxh + external size
//!    fields from the solver's error estimator).
//! 3. Quality pass targeting the minimal dihedral angle: edge removal,
//!    smoothing, sliver removal (HXT-style operator toolkit).
//! 4. Optional order-2 midside snapping onto the analytic surface via the
//!    PLC surface back-references (required for full Nédélec-2 convergence on
//!    curved boundaries).

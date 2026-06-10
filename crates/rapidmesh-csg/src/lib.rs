//! Exact mesh CSG: arrangements and multi-operand boolean expressions.
//!
//! Approach (see DESIGN.md): exact arrangement of the input triangle meshes
//! with intersection points represented implicitly (indirect predicates,
//! interval-arithmetic filtering, expansion-arithmetic fallback), CDT remeshing
//! of intersected facets with symbolic perturbation, then inside/outside
//! classification against the boolean expression. Boolean expressions are
//! evaluated as one multi-operand arrangement — never as cascaded pairwise ops
//! with float snapping in between, which is the known failure mode.
//!
//! Blueprints: Lévy (ACM TOG 2024, exact mesh CSG / Weiler model) and
//! Cherchi et al. (SIGGRAPH Asia 2022, indirect predicates).

pub mod arrange;
pub mod constraint;
pub mod tri;
pub mod tri_tri;
pub mod triangulate;

pub use arrange::{arrange, Arrangement};
pub use constraint::{Constraint, ConstraintLine};
pub use tri::Tri;
pub use tri_tri::{tri_tri_intersection, TriTriIsect};
pub use triangulate::{triangulate_facet, FacetTriangulation};

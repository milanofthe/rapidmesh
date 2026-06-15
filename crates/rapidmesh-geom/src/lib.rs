//! Geometry front-end: solid primitives and the tagged PLC.
//!
//! The tagged PLC (piecewise-linear complex: watertight triangle surface with
//! face/region tags and back-references to the originating analytic surface) is
//! the central intermediate representation of rapidmesh. Both the CSG path
//! (primitives + booleans) and the later STEP path converge on it; the tet
//! mesher consumes it. Surface back-references exist so the order-2 snapping
//! stage can project midside nodes onto the true surface.

pub mod faceted;
pub mod import;
pub mod nurbs;
pub mod plc;
pub mod polygon;
pub mod prim;
pub mod scene;

pub use faceted::{Faceted, FlatFacet, SurfaceKind};
pub use import::{import_obj, import_stl, min_height_ratio, validate_closed, ImportError};
pub use plc::{FaceTag, RegionTag, SurfaceRef, TaggedPlc};
pub use scene::Scene;
pub use polygon::{polygon_orientation, triangulate_polygon};
pub use nurbs::NurbsCurve;
pub use prim::{
    cylinder, cylinder_iso, extrude_polygon, extrude_spline_profile, frustum, frustum_iso, helix,
    icosphere, loft, mesh_solid, naca0012_profile, pipe, sheet_disk, sheet_polygon, sheet_rect,
    solid_box, sphere, torus, wedge,
};

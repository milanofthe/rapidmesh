//! Geometry front-end: solid primitives and the tagged PLC.
//!
//! The tagged PLC (piecewise-linear complex: watertight triangle surface with
//! face/region tags and back-references to the originating analytic surface) is
//! the central intermediate representation of rapidmesh. Both the CSG path
//! (primitives + booleans) and the later STEP path converge on it; the tet
//! mesher consumes it. Surface back-references exist so the order-2 snapping
//! stage can project midside nodes onto the true surface.

pub mod plc;

pub use plc::TaggedPlc;

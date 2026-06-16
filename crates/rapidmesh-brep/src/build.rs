//! Builds a [`Brep`] from the exact CSG output (`TaggedPlc` + arrangement).
//!
//! The arrangement is the source of truth for TOPOLOGY (which surfaces meet,
//! region labels, exact vertex positions); this step recovers the analytic edge
//! curves from the adjacent surfaces and the trim loops from the feature-edge
//! chains, and labels faces with their front/back regions. Exactness is preserved
//! -- the B-rep only adds analytic geometry on top of the exact topology.
//!
//! Next implementation step (see `DESIGN-brep.md`): faceted-PLC -> Brep.

use crate::Brep;
use rapidmesh_geom::TaggedPlc;

/// Build a B-rep from a tagged PLC. (Skeleton: returns the analytic surfaces; the
/// topology recovery -- vertices, edges/curves, loops, faces, regions -- lands in
/// the next step.)
pub fn from_plc(plc: &TaggedPlc) -> Brep {
    Brep { surfaces: plc.surfaces.clone(), ..Brep::new() }
}

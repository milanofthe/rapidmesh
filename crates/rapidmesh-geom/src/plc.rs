//! Tagged piecewise-linear complex: the central intermediate representation.

/// Identifies the analytic surface a PLC facet originated from, so that
/// downstream stages (order-2 midside snapping) can project points back onto
/// the exact geometry instead of the linear facet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SurfaceRef(pub u32);

/// Region (material) tag carried through CSG into the volume mesh. Every output
/// tet lies in exactly one region — conformal material interfaces are a hard
/// requirement for Maxwell FEM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionTag(pub u32);

/// Boundary/face tag for ports, PEC surfaces, ABC/PML interfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FaceTag(pub u32);

/// Watertight tagged triangle surface complex.
///
/// Coordinates are expected normalized to a unit box by the builder: the
/// Shewchuk-style robust predicates underneath do not handle exponent
/// overflow (inputs outside ~[1e-142, 1e201] lose their guarantee).
#[derive(Debug, Default, Clone)]
pub struct TaggedPlc {
    /// Vertex coordinates, xyz interleaved.
    pub vertices: Vec<[f64; 3]>,
    /// Triangle vertex indices.
    pub triangles: Vec<[u32; 3]>,
    /// Per-triangle face tag.
    pub face_tags: Vec<FaceTag>,
    /// Per-triangle back-reference to the originating analytic surface.
    pub surface_refs: Vec<SurfaceRef>,
    /// Per-triangle region tags on both sides (front, back) of the facet.
    pub region_tags: Vec<[RegionTag; 2]>,
}

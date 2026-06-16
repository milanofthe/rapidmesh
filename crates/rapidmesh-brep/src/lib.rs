//! `rapidmesh-brep`: a boundary-representation (B-rep) layer between the exact CSG
//! and the mesher.
//!
//! Faces are TRIMMED analytic surfaces, edges are analytic CURVES (including the
//! intersection curves a boolean creates), vertices are corner points. The mesher
//! re-meshes from this geometry -- distribute on each edge curve, mesh each
//! trimmed face in its (u,v) parameter space, fill the volume -- independent of
//! any input tessellation. See `DESIGN-brep.md`.
//!
//! Topology is **non-manifold** (Weiler radial-edge): an edge radially links ALL
//! faces meeting along it, and a face carries front/back material labels, so
//! multi-material interfaces and embedded sheets -- rapidmesh's core domain -- are
//! first-class.
//!
//! Deliberately MINIMAL: this layer carries only what the mesher consumes
//! (vertices, edges with an analytic curve + radial face list, faces with
//! oriented boundary loops + region/tag labels). Everything else the mesher
//! already does -- parameter-space mapping (`surfchart`), point distribution
//! (`curve`), region classification (`region_at`), volume filling -- so there is
//! no half-edge/pcurve/shell/region machinery here.

use rapidmesh_geom::{FaceTag, NurbsSurface, RegionTag, SurfaceKind};
use std::sync::Arc;

pub mod build;

type V3 = [f64; 3];

// ---- surface geometry (analytic primitive OR free-form NURBS) ------------

/// A B-rep face's underlying surface. Analytic primitives keep the lightweight
/// [`SurfaceKind`]; general CAD/boolean output is a trimmed [`NurbsSurface`]. The
/// enum is the extension point that makes the geometry layer NURBS-native -- a
/// face references a `Surface` by [`SurfaceId`] regardless of which it is, and the
/// mesher's parameter mapping branches here.
#[derive(Debug, Clone)]
pub enum Surface {
    Analytic(SurfaceKind),
    Nurbs(Arc<NurbsSurface>),
}

// ---- ids -----------------------------------------------------------------

macro_rules! id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u32);
    };
}
id!(VertexId);
id!(EdgeId);
id!(FaceId);
id!(SurfaceId);

// ---- geometry (analytic) -------------------------------------------------

/// An analytic edge curve. The edge also stores its on-PLC vertex chain
/// (`Edge::verts`), so `Polyline` needs no data and `Intersection` uses the chain
/// as the projection seed. The mesher evaluates these (it owns the curve / chart
/// machinery); the B-rep only RECOGNISES and stores the form.
#[derive(Debug, Clone)]
pub enum Curve {
    /// Straight segment through `p0` with unit direction `dir`.
    Line { p0: V3, dir: V3 },
    /// Circle: center, unit `axis` (normal), `radius`, in-plane unit `x` axis.
    Circle { center: V3, axis: V3, radius: f64, x: V3 },
    /// A 2D profile NURBS (an `Extruded` surface's profile) over `t`, lifted to 3D
    /// at extrusion height `z` on the surface frame. The airfoil outline edge.
    Profile { profile: Arc<rapidmesh_geom::nurbs::NurbsCurve>, surface: SurfaceId, t: [f64; 2], z: f64 },
    /// Intersection of two surfaces, evaluated lazily by projecting the vertex
    /// chain onto both (the mesher reuses its surface projections).
    Intersection { a: SurfaceId, b: SurfaceId },
    /// Faceted fallback: the edge IS its vertex chain (no analytic refinement).
    Polyline,
}

// ---- topology (non-manifold radial-edge) ---------------------------------

/// A corner point (an endpoint of one or more edge chains). Interior facet
/// vertices are NOT B-rep vertices.
#[derive(Debug, Clone)]
pub struct Vertex {
    pub pos: V3,
}

/// A B-rep edge: a maximal chain of PLC boundary edges between two corners, with
/// its recovered analytic `curve` and the radial list of all faces meeting it.
#[derive(Debug, Clone)]
pub struct Edge {
    /// The corner endpoints (`verts.first()` / `verts.last()`).
    pub ends: [VertexId; 2],
    /// The ordered on-PLC vertex chain (the polyline the edge follows); the curve
    /// runs from `chain[0]` to `chain[last]`.
    pub chain: Vec<V3>,
    /// The recovered analytic curve.
    pub curve: Curve,
    /// All faces meeting along this edge -- radial cycle (non-manifold): 2 for a
    /// box edge, 3+ at a multi-material interface or a sheet rim.
    pub faces: Vec<FaceId>,
}

/// An oriented boundary cycle of a face: signed edges (`forward = true` traverses
/// the edge from `ends[0]` to `ends[1]`). `loops[0]` of a face is the outer
/// boundary, the rest are holes.
#[derive(Debug, Clone, Default)]
pub struct Loop {
    pub edges: Vec<(EdgeId, bool)>,
}

/// A trimmed analytic surface. `regions` are the materials on the front
/// (`+normal`) and back sides: equal for an embedded sheet, one being
/// `RegionTag(0)` (background) for an outer wall.
#[derive(Debug, Clone)]
pub struct Face {
    pub surface: SurfaceId,
    pub loops: Vec<Loop>,
    pub regions: [RegionTag; 2],
    pub face_tag: FaceTag,
    /// Scene-solid owner (parallel to `TaggedPlc::surface_owners`).
    pub owner: u32,
}

/// The boundary-representation model: arena-allocated topology + geometry, linked
/// by ids. Built from the exact CSG arrangement (see [`build::from_plc`]),
/// consumed by the mesher.
#[derive(Debug, Clone, Default)]
pub struct Brep {
    pub vertices: Vec<Vertex>,
    pub edges: Vec<Edge>,
    pub faces: Vec<Face>,
    pub surfaces: Vec<Surface>,
}

impl Brep {
    pub fn new() -> Brep {
        Brep::default()
    }
    pub fn vertex(&self, v: VertexId) -> &Vertex {
        &self.vertices[v.0 as usize]
    }
    pub fn edge(&self, e: EdgeId) -> &Edge {
        &self.edges[e.0 as usize]
    }
    pub fn face(&self, f: FaceId) -> &Face {
        &self.faces[f.0 as usize]
    }
    pub fn surface(&self, s: SurfaceId) -> &Surface {
        &self.surfaces[s.0 as usize]
    }
}

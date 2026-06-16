//! `rapidmesh-brep`: a boundary-representation (B-rep) layer between the exact CSG
//! and the mesher.
//!
//! Faces are TRIMMED analytic surfaces, edges are analytic CURVES (including the
//! intersection curves a boolean creates), vertices are points. The mesher
//! re-meshes from this geometry -- distribute on each edge curve, mesh each
//! trimmed face in its (u,v) parameter space, fill the volume -- independent of
//! any input tessellation. See `DESIGN-brep.md`.
//!
//! Topology is **non-manifold** (Weiler radial-edge): an edge radially links ALL
//! faces meeting along it, and a face carries front/back material labels, so
//! multi-material interfaces and embedded sheets -- rapidmesh's core domain -- are
//! first-class. Geometry and topology are separated (OpenCASCADE-style), with a
//! per-face **PCurve** so a trimmed face can be meshed in its parameter space.

use rapidmesh_exact::Point3;
use rapidmesh_geom::SurfaceKind;
use std::sync::Arc;

pub mod build;

type V3 = [f64; 3];
type P2 = [f64; 2];

// ---- ids -----------------------------------------------------------------

macro_rules! id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u32);
    };
}
id!(VertexId);
id!(EdgeId);
id!(HalfEdgeId);
id!(LoopId);
id!(FaceId);
id!(ShellId);
id!(RegionId);
id!(SurfaceId);

/// Material/region tag carried from the CSG (0 = background/outside).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegionTag(pub u32);

// ---- geometry (analytic) -------------------------------------------------

/// An analytic edge curve. `Intersection` is LAZY: the curve is "where surface
/// `a` meets surface `b`", evaluated on demand by projecting onto both (reusing
/// the surface closest-point projections) -- no precomputed closed form.
#[derive(Debug, Clone)]
pub enum Curve {
    /// Straight segment through `p0` with unit direction `dir`.
    Line { p0: V3, dir: V3 },
    /// Circle: center, unit axis (normal), radius, and an in-plane unit `x` axis.
    Circle { center: V3, axis: V3, radius: f64, x: V3 },
    /// A rational B-spline curve in the parameter range `[t0, t1]`.
    Nurbs { curve: Arc<rapidmesh_geom::nurbs::NurbsCurve>, t: [f64; 2] },
    /// The intersection of two surfaces (evaluated lazily by projection).
    Intersection { a: SurfaceId, b: SurfaceId },
}

/// A boundary edge's curve in one adjacent face's (u,v) parameter space, for
/// trimming and 2D meshing of that face.
#[derive(Debug, Clone)]
pub struct PCurve {
    pub face: FaceId,
    /// Sampled (u,v) polyline of the edge in the face parameter domain; the face
    /// mesher uses it as a trim segment. (Analytic PCurves can replace this later.)
    pub uv: Vec<P2>,
}

// ---- topology (non-manifold radial-edge) ---------------------------------

#[derive(Debug, Clone)]
pub struct Vertex {
    /// Exact corner coordinate (the B-rep keeps the CSG's exact points).
    pub point: Point3,
    pub pos: V3,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub curve: Curve,
    pub ends: [VertexId; 2],
    /// Parameter range of the curve this edge spans.
    pub t: [f64; 2],
    /// All half-edges (uses) around this edge -- radial cycle (non-manifold).
    pub radial: Vec<HalfEdgeId>,
}

/// A directed use of an edge by a face loop (a "co-edge").
#[derive(Debug, Clone)]
pub struct HalfEdge {
    pub edge: EdgeId,
    /// True if this use traverses the edge from `ends[0]` to `ends[1]`.
    pub forward: bool,
    pub loop_: LoopId,
    pub pcurve: PCurve,
}

/// An oriented cycle of half-edges bounding (part of) a face.
#[derive(Debug, Clone)]
pub struct Loop {
    pub coedges: Vec<HalfEdgeId>,
    pub face: FaceId,
}

/// A trimmed analytic surface. `loops[0]` is the outer boundary, the rest holes.
/// `regions` are the materials on the front (`+normal`) and back sides -- equal
/// for an embedded sheet, one being 0 (background) for an outer wall.
#[derive(Debug, Clone)]
pub struct Face {
    pub surface: SurfaceId,
    pub loops: Vec<LoopId>,
    pub regions: [RegionTag; 2],
}

#[derive(Debug, Clone)]
pub struct Shell {
    pub faces: Vec<FaceId>,
}

/// One material volume, bounded by shells.
#[derive(Debug, Clone)]
pub struct Region {
    pub shells: Vec<ShellId>,
    pub tag: RegionTag,
}

/// The boundary-representation model: arena-allocated topology + geometry, linked
/// by ids. Built from the exact CSG arrangement (see `build`), consumed by the
/// mesher.
#[derive(Debug, Clone, Default)]
pub struct Brep {
    pub vertices: Vec<Vertex>,
    pub edges: Vec<Edge>,
    pub halfedges: Vec<HalfEdge>,
    pub loops: Vec<Loop>,
    pub faces: Vec<Face>,
    pub shells: Vec<Shell>,
    pub regions: Vec<Region>,
    pub surfaces: Vec<SurfaceKind>,
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
    pub fn surface(&self, s: SurfaceId) -> &SurfaceKind {
        &self.surfaces[s.0 as usize]
    }
}

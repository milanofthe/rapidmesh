//! Boolean operations on closed solids via the exact arrangement.
//!
//! Both operands are arranged together, every sub-triangle is classified
//! against the other solid by its exact barycenter, and a per-operation keep
//! table selects (and possibly flips) the surviving sub-triangles. Output
//! vertices stay exact ([`Point3`], possibly implicit); rounding to f64 is a
//! separate downstream decision.

use crate::arrange::arrange;
use crate::classify::{classify, Placement};
use crate::tri::Tri;
use rapidmesh_exact::Point3;
use std::collections::HashMap;

/// A closed, outward-oriented triangle mesh.
#[derive(Debug, Clone)]
pub struct Solid {
    /// The triangles. Watertightness and outward orientation are the
    /// caller's responsibility (builders guarantee it for primitives).
    pub tris: Vec<Tri>,
}

/// A regularized boolean operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    /// A ∪ B.
    Union,
    /// A ∩ B.
    Intersection,
    /// A − B.
    Difference,
}

/// The boundary surface of a boolean result, with exact vertices.
#[derive(Debug)]
pub struct BooleanResult {
    /// Globally deduplicated exact vertices.
    pub vertices: Vec<Point3>,
    /// Output triangles (outward-oriented).
    pub triangles: Vec<[usize; 3]>,
    /// Per output triangle: index of the input facet it came from
    /// (0..a.tris.len() for A, then B) — carries tags downstream.
    pub source_facet: Vec<usize>,
}

/// Keep table: `Some(flip)` if a sub-triangle with this placement survives.
///
/// Coplanar-coincident regions are kept once (from the A side) when their
/// orientation matches the result's boundary there: same-normal for union
/// and intersection, opposite-normal for difference (B touching A from
/// outside leaves A's wall intact; coincident interiors cancel).
fn keep(op: BoolOp, from_a: bool, placement: Placement) -> Option<bool> {
    match (op, placement) {
        (BoolOp::Union, Placement::Outside) => Some(false),
        (BoolOp::Union, Placement::Boundary { same_normal: true }) if from_a => Some(false),
        (BoolOp::Intersection, Placement::Inside) => Some(false),
        (BoolOp::Intersection, Placement::Boundary { same_normal: true }) if from_a => {
            Some(false)
        }
        (BoolOp::Difference, Placement::Outside) if from_a => Some(false),
        (BoolOp::Difference, Placement::Inside) if !from_a => Some(true),
        (BoolOp::Difference, Placement::Boundary { same_normal: false }) if from_a => {
            Some(false)
        }
        _ => None,
    }
}

/// Vertex pool with exact deduplication, accelerated by a spatial hash on
/// approximate coordinates (coincident points approximate to within far less
/// than the bucket size, so scanning the 27 neighboring buckets is
/// conservative).
#[derive(Default)]
struct VertexPool {
    verts: Vec<Point3>,
    buckets: HashMap<[i64; 3], Vec<usize>>,
}

const BUCKET: f64 = 1e-6;

impl VertexPool {
    fn key(a: [f64; 3]) -> [i64; 3] {
        std::array::from_fn(|k| (a[k] / BUCKET).floor() as i64)
    }

    fn insert(&mut self, p: Point3) -> usize {
        let a = p.approx().expect("valid point");
        let base = Self::key(a);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let k = [base[0] + dx, base[1] + dy, base[2] + dz];
                    if let Some(ids) = self.buckets.get(&k) {
                        for &i in ids {
                            if self.verts[i].coincides(&p) {
                                return i;
                            }
                        }
                    }
                }
            }
        }
        let id = self.verts.len();
        self.verts.push(p);
        self.buckets.entry(base).or_default().push(id);
        id
    }
}

/// Regularized boolean of two closed solids. The result boundary is exact;
/// shared/coincident surface regions are handled by the keep table.
pub fn boolean(a: &Solid, b: &Solid, op: BoolOp) -> BooleanResult {
    let mut all: Vec<Tri> = a.tris.clone();
    all.extend(b.tris.iter().cloned());
    let na = a.tris.len();
    let arr = arrange(&all);

    // Scene bounding box for ray targets.
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for t in &all {
        for v in &t.v {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
    }

    let mut pool = VertexPool::default();
    let mut triangles: Vec<[usize; 3]> = Vec::new();
    let mut source_facet: Vec<usize> = Vec::new();
    for (fi, ft) in arr.facets.iter().enumerate() {
        let from_a = fi < na;
        let other: &[Tri] = if from_a { &b.tris } else { &a.tris };
        for sub in &ft.triangles {
            let (p0, p1, p2) = (
                &ft.vertices[sub[0]],
                &ft.vertices[sub[1]],
                &ft.vertices[sub[2]],
            );
            let bary = Point3::bary(p0.clone(), p1.clone(), p2.clone());
            let placement = classify(&bary, &all[fi], other, (lo, hi));
            let Some(flip) = keep(op, from_a, placement) else {
                continue;
            };
            let i0 = pool.insert(p0.clone());
            let i1 = pool.insert(p1.clone());
            let i2 = pool.insert(p2.clone());
            triangles.push(if flip { [i0, i2, i1] } else { [i0, i1, i2] });
            source_facet.push(fi);
        }
    }
    BooleanResult {
        vertices: pool.verts,
        triangles,
        source_facet,
    }
}

//! Globally deduplicated exact vertex pool.

use rapidmesh_exact::Point3;
use std::collections::HashMap;

/// Vertex pool with exact deduplication, accelerated by a spatial hash on
/// approximate coordinates (coincident points approximate to within far less
/// than the bucket size, so scanning the 27 neighboring buckets is
/// conservative).
#[derive(Default)]
pub struct VertexPool {
    /// The deduplicated vertices.
    pub verts: Vec<Point3>,
    buckets: HashMap<[i64; 3], Vec<usize>>,
}

const BUCKET: f64 = 1e-6;

impl VertexPool {
    fn key(a: [f64; 3]) -> [i64; 3] {
        std::array::from_fn(|k| (a[k] / BUCKET).floor() as i64)
    }

    /// Returns the index of `p`, inserting it if no exactly coincident
    /// vertex exists yet.
    pub fn insert(&mut self, p: Point3) -> usize {
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

//! The tube carrier's centerline: a polyline with a midpoint-prune
//! accelerator for closest-point queries. The projection oracle of a
//! `SurfaceKind::Tube` runs millions of times per mesh (sphere tracing,
//! POCS, wall predicates); a plain scan of exact segment projections made
//! the coil geometries 10x slower than their discrete-carrier predecessor.
//! A cheap flat pass over precomputed segment midpoints bounds the search,
//! and the exact projection runs only for the handful of segments that can
//! still beat the bound -- no spatial index, no far-query pathology.

use crate::vec3::{dist, dot, sub, V3};

/// A polyline sweep centerline with midpoint-prune closest queries.
#[derive(Debug)]
pub struct TubePath {
    /// The ordered path nodes.
    pub pts: Vec<V3>,
    /// Segment midpoints (prune pass; cache-friendly flat scan).
    mids: Vec<V3>,
    /// Segment half-lengths (the midpoint bound's slack).
    half: Vec<f64>,
}

impl TubePath {
    /// Builds the midpoint prune arrays. `pts` needs at least 2 nodes.
    pub fn new(pts: Vec<V3>) -> TubePath {
        assert!(pts.len() >= 2, "tube path needs at least 2 nodes");
        let mids: Vec<V3> = pts
            .windows(2)
            .map(|w| std::array::from_fn(|k| 0.5 * (w[0][k] + w[1][k])))
            .collect();
        let half: Vec<f64> = pts.windows(2).map(|w| 0.5 * dist(w[0], w[1])).collect();
        TubePath { pts, mids, half }
    }

    /// Closest point on the polyline to `p` (exact; the grid only prunes).
    pub fn closest(&self, p: V3) -> V3 {
        let seg = |i: usize| -> (V3, f64) {
            let (a, b) = (self.pts[i], self.pts[i + 1]);
            let ab = sub(b, a);
            let len2 = dot(ab, ab);
            let t = if len2 > 0.0 {
                (dot(sub(p, a), ab) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let q: V3 = std::array::from_fn(|k| a[k] + t * ab[k]);
            let d = sub(p, q);
            (q, dot(d, d))
        };
        // Pass 1: cheap midpoint distances give an achievable upper bound
        // (midpoint IS a segment point). Pass 2: the exact test runs only for
        // segments whose midpoint could still beat it (d_mid - half < bound).
        // The oracle runs millions of times per mesh; this keeps the exact
        // evaluations to a handful without any spatial-index pathology.
        let mut bound2 = f64::MAX;
        for m in &self.mids {
            let d = sub(p, *m);
            bound2 = bound2.min(dot(d, d));
        }
        let bound = bound2.sqrt();
        let mut best = (self.pts[0], f64::MAX);
        for (i, (m, h)) in self.mids.iter().zip(&self.half).enumerate() {
            let d = sub(p, *m);
            let dm = dot(d, d).sqrt();
            if dm - h > bound {
                continue;
            }
            let c = seg(i);
            if c.1 < best.1 {
                best = c;
            }
        }
        best.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_closest_matches_linear_scan() {
        // helix-like path
        let pts: Vec<V3> = (0..=180)
            .map(|i| {
                let t = i as f64 / 28.0 * std::f64::consts::TAU;
                [0.8 * t.cos(), 0.8 * t.sin(), 0.42 * i as f64 / 28.0]
            })
            .collect();
        let tube = TubePath::new(pts.clone());
        let linear = |p: V3| -> V3 {
            let mut best = (pts[0], f64::MAX);
            for w in pts.windows(2) {
                let ab = sub(w[1], w[0]);
                let len2 = dot(ab, ab);
                let t = (dot(sub(p, w[0]), ab) / len2).clamp(0.0, 1.0);
                let q: V3 = std::array::from_fn(|k| w[0][k] + t * ab[k]);
                let d = sub(p, q);
                if dot(d, d) < best.1 {
                    best = (q, dot(d, d));
                }
            }
            best.0
        };
        let mut s = 12345u64;
        let mut frac = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..500 {
            let p: V3 = [4.0 * frac() - 2.0, 4.0 * frac() - 2.0, 4.0 * frac()];
            let (a, b) = (tube.closest(p), linear(p));
            assert!(dist(a, b) < 1e-9, "grid {a:?} vs linear {b:?} at {p:?}");
        }
    }
}

//! The sizing field shared by stages 2 (surfaces) and 3 (volumes): a
//! gradient-limited size field grown from point SOURCES.
//!
//! Each source `(p_s, h_s)` is a point that wants element size `h_s` there (a
//! distributed edge point with its local 1D spacing for the surface field; a
//! distributed surface point with its local 2D spacing for the volume field). The
//! field at any query point is `h(x) = min( maxh, min over sources of h_s *
//! (1+grad)^( dist(x,s) / h_s ) )`.
//!
//! The growth is MULTIPLICATIVE (geometric): along any ray from a source the size
//! grows by the ratio `1+grad` per element of its own length, so the field
//! coarsens FAST (a narrow transition, few elements) while every neighbouring pair
//! stays within `1+grad` (no density jump, no sliver fan). An additive `h_s +
//! slope*dist` limit cannot have both -- the failure the curved meshes showed.
//!
//! This is the single mechanism for the whole hierarchy: stage N's distributed
//! points are stage N+1's sources, so size flows smoothly from edges to surfaces
//! to the volume with one rule.

use crate::spatial::Octree;

type V3 = [f64; 3];

fn dist(a: V3, b: V3) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// A gradient-limited size field grown from point sources.
pub struct SizeField {
    sources: Vec<(V3, f64)>,
    grad: f64,
    maxh: f64,
    tree: Octree,
    /// Query radius beyond which no source can pull the size below `maxh`: a
    /// source of size `h_s` matters only within `h_s*ln(maxh/h_s)/ln(1+grad)`,
    /// maximised (over `h_s`) at `h_s = maxh/e`, giving `maxh/(e*ln(1+grad))`.
    r_query: f64,
}

impl SizeField {
    /// Builds the field. `grad` is the per-element growth ratio minus one (e.g.
    /// `0.3` -> adjacent elements within 1.3x); `maxh` caps the bulk size.
    pub fn new(sources: Vec<(V3, f64)>, grad: f64, maxh: f64) -> SizeField {
        let pos: Vec<V3> = sources.iter().map(|s| s.0).collect();
        let tree = Octree::build(&pos);
        let g = grad.max(1e-6);
        let r_query = 1.5 * maxh / (std::f64::consts::E * (1.0 + g).ln());
        SizeField { sources, grad: g, maxh, tree, r_query }
    }

    /// Number of sources (diagnostic).
    pub fn len(&self) -> usize {
        self.sources.len()
    }
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// The size at `x`: the gradient-limited minimum over sources, capped `maxh`.
    pub fn at(&self, x: V3) -> f64 {
        if self.sources.is_empty() {
            return self.maxh;
        }
        let eval = |idx: &[u32], best: &mut f64| {
            for &j in idx {
                let (p, h) = self.sources[j as usize];
                let v = h * (1.0 + self.grad).powf(dist(x, p) / h);
                if v < *best {
                    *best = v;
                }
            }
        };
        let mut best = self.maxh;
        let cand = self.tree.within_radius(x, self.r_query);
        if cand.is_empty() {
            // Far from every source: the nearest one still defines an upper bound,
            // but it is >= maxh by construction of r_query, so maxh stands. Guard
            // anyway with the single nearest in case r_query underestimates.
            if let Some(j) = self.tree.nearest(x) {
                eval(&[j], &mut best);
            }
        } else {
            eval(&cand, &mut best);
        }
        best.min(self.maxh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_source_grows_multiplicatively() {
        let f = SizeField::new(vec![([0.0, 0.0, 0.0], 0.01)], 0.3, 1.0);
        assert!((f.at([0.0, 0.0, 0.0]) - 0.01).abs() < 1e-9);
        // At distance = one element (0.01), size ~ 0.01 * 1.3 = 0.013.
        let h1 = f.at([0.01, 0.0, 0.0]);
        assert!((h1 - 0.013).abs() < 1e-3, "h(one element) = {h1}");
        // Far away: capped at maxh.
        assert!((f.at([10.0, 0.0, 0.0]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn min_over_sources_and_smoothness() {
        // A fine source and a coarse source; midpoint size is between, and the
        // field never jumps by more than the ratio over a small step.
        let (grad, minh) = (0.3, 0.02);
        let f = SizeField::new(vec![([0.0, 0.0, 0.0], minh), ([1.0, 0.0, 0.0], 0.2)], grad, 1.0);
        let n = 200;
        let ds = 1.0 / n as f64;
        // Over a step `ds`, the field grows by at most (1+grad)^(ds/h) <=
        // (1+grad)^(ds/minh): the multiplicative gradient bound.
        let bound = (1.0 + grad as f64).powf(ds / minh) + 1e-6;
        let mut prev = f.at([0.0, 0.0, 0.0]);
        let mut worst = 1.0f64;
        for i in 1..=n {
            let h = f.at([i as f64 / n as f64, 0.0, 0.0]);
            worst = worst.max((h / prev).max(prev / h));
            prev = h;
        }
        assert!(worst <= bound, "field grows faster than the gradient bound: {worst} > {bound}");
        // The minimum near the fine source must reflect it.
        assert!(f.at([0.05, 0.0, 0.0]) < 0.1, "fine source not honoured");
    }

    #[test]
    fn empty_field_is_maxh() {
        let f = SizeField::new(vec![], 0.3, 0.5);
        assert_eq!(f.at([1.0, 2.0, 3.0]), 0.5);
    }
}

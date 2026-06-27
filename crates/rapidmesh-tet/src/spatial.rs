//! A point-region octree over `[f64; 3]` for fast nearest-site and range
//! queries. The CVT mesher rebuilds this each Lloyd pass to find neighbors and
//! to seed the Delaunay walk; WP9 reuses it for octree-chunked parallel work.
//!
//! `build` takes a point slice and queries return indices into that slice.

// Leaf capacity + depth cap: see crate::constants.
use crate::constants::{OCTREE_LEAF_CAP as LEAF_CAP, OCTREE_MAX_DEPTH as MAX_DEPTH};
use rapidmesh_geom::vec3::{V3};

fn dist2(a: V3, b: V3) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

/// A density-adaptive, aspect-RATIO-preserving spatial insertion order: the
/// points in octree depth-first order. The octree subdivides into CUBIC octants,
/// so a flat or anisotropic point set is not stretched the way `morton_order`'s
/// per-axis normalisation stretches it (there the thin axis gets the full bit
/// range, so the Z-curve jumps across it). Visiting octants `0..8` is a Morton
/// order of the octants, but adaptive to the actual density -- inserting in this
/// order keeps an incremental Delaunay's point-location walks short on the
/// non-uniform inputs where the plain Morton order is poor.
pub fn octree_order(points: &[V3]) -> Vec<usize> {
    let mut out = Vec::with_capacity(points.len());
    if !points.is_empty() {
        octree_visit(&Octree::build(points).root, &mut out);
    }
    out
}

fn octree_visit(node: &Node, out: &mut Vec<usize>) {
    match node {
        Node::Leaf(idx) => out.extend(idx.iter().map(|&i| i as usize)),
        Node::Inner(children) => {
            for c in children.iter() {
                octree_visit(c, out);
            }
        }
    }
}

enum Node {
    Leaf(Vec<u32>),
    Inner(Box<[Node; 8]>),
}

pub struct Octree {
    pts: Vec<V3>,
    /// Cube root: center and half-extent (a cube comfortably enclosing all pts).
    center: V3,
    half: f64,
    root: Node,
}

impl Octree {
    /// Builds an octree over `points`. Empty input yields a queryable-but-empty
    /// tree (`nearest` returns `None`).
    pub fn build(points: &[V3]) -> Octree {
        let pts = points.to_vec();
        if pts.is_empty() {
            return Octree {
                pts,
                center: [0.0; 3],
                half: 1.0,
                root: Node::Leaf(Vec::new()),
            };
        }
        let mut lo = pts[0];
        let mut hi = pts[0];
        for p in &pts {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let center: V3 = std::array::from_fn(|k| 0.5 * (lo[k] + hi[k]));
        let mut half = 0.0_f64;
        for k in 0..3 {
            half = half.max(0.5 * (hi[k] - lo[k]));
        }
        // A non-degenerate, slightly padded cube so all points are strictly inside.
        half = (half * 1.0001).max(1e-12);
        let all: Vec<u32> = (0..pts.len() as u32).collect();
        let root = build_node(&pts, center, half, all, 0);
        Octree { pts, center, half, root }
    }

    /// Index of the point nearest to `q`, or `None` if the tree is empty.
    pub fn nearest(&self, q: V3) -> Option<u32> {
        if self.pts.is_empty() {
            return None;
        }
        let mut best = (f64::INFINITY, u32::MAX);
        nearest_in(&self.root, &self.pts, self.center, self.half, q, &mut best);
        if best.1 == u32::MAX {
            None
        } else {
            Some(best.1)
        }
    }

}

fn child_box(center: V3, half: f64, octant: usize) -> (V3, f64) {
    let h = 0.5 * half;
    let c: V3 = std::array::from_fn(|k| {
        let sign = if octant & (1 << k) != 0 { 1.0 } else { -1.0 };
        center[k] + sign * h
    });
    (c, h)
}

fn octant_of(center: V3, p: V3) -> usize {
    let mut o = 0;
    for k in 0..3 {
        if p[k] >= center[k] {
            o |= 1 << k;
        }
    }
    o
}

fn build_node(pts: &[V3], center: V3, half: f64, idx: Vec<u32>, depth: u32) -> Node {
    if idx.len() <= LEAF_CAP || depth >= MAX_DEPTH {
        return Node::Leaf(idx);
    }
    let mut buckets: [Vec<u32>; 8] = Default::default();
    for &i in &idx {
        buckets[octant_of(center, pts[i as usize])].push(i);
    }
    let children: [Node; 8] = std::array::from_fn(|o| {
        let (cc, ch) = child_box(center, half, o);
        build_node(pts, cc, ch, std::mem::take(&mut buckets[o]), depth + 1)
    });
    Node::Inner(Box::new(children))
}

/// Squared distance from `q` to the axis-aligned cube (center, half); 0 inside.
fn box_dist2(center: V3, half: f64, q: V3) -> f64 {
    let mut d2 = 0.0;
    for k in 0..3 {
        let lo = center[k] - half;
        let hi = center[k] + half;
        let e = if q[k] < lo {
            lo - q[k]
        } else if q[k] > hi {
            q[k] - hi
        } else {
            0.0
        };
        d2 += e * e;
    }
    d2
}

fn nearest_in(node: &Node, pts: &[V3], center: V3, half: f64, q: V3, best: &mut (f64, u32)) {
    match node {
        Node::Leaf(idx) => {
            for &i in idx {
                let d2 = dist2(pts[i as usize], q);
                if d2 < best.0 {
                    *best = (d2, i);
                }
            }
        }
        Node::Inner(children) => {
            // Visit the octant containing q first, then siblings by box distance,
            // pruning any whose box is farther than the current best. Fixed
            // 8-element stack array (no per-node heap allocation -- this runs
            // once per visited inner node, per nearest query, per Lloyd pass).
            let mut order: [(f64, usize); 8] = [(0.0, 0); 8];
            for (o, slot) in order.iter_mut().enumerate() {
                let (cc, ch) = child_box(center, half, o);
                *slot = (box_dist2(cc, ch, q), o);
            }
            // Total order (distance, then octant index) so equal-distance
            // octants keep ascending-index order -- identical tie-breaking to the
            // previous stable sort, hence bit-identical nearest results.
            order.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
            for (bd2, o) in order {
                if bd2 > best.0 {
                    break;
                }
                let (cc, ch) = child_box(center, half, o);
                nearest_in(&children[o], pts, cc, ch, q, best);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize) -> Vec<V3> {
        let mut v = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    v.push([i as f64 * 0.37, j as f64 * 0.61 - 2.0, k as f64 * 0.13 + 1.0]);
                }
            }
        }
        v
    }

    fn brute_nearest(pts: &[V3], q: V3) -> u32 {
        let mut best = (f64::INFINITY, 0u32);
        for (i, p) in pts.iter().enumerate() {
            let d2 = dist2(*p, q);
            if d2 < best.0 {
                best = (d2, i as u32);
            }
        }
        best.1
    }

    #[test]
    fn nearest_matches_brute_force() {
        let pts = grid(9);
        let tree = Octree::build(&pts);
        let queries = [
            [0.0, 0.0, 0.0],
            [1.234, -1.1, 1.7],
            [3.0, 0.5, 1.05],
            [-5.0, 5.0, 9.0],
            [2.0, -0.3, 1.2],
        ];
        for q in queries {
            let got = tree.nearest(q).unwrap();
            let want = brute_nearest(&pts, q);
            // Tie-robust: distances must match even if indices differ.
            assert!(
                (dist2(pts[got as usize], q) - dist2(pts[want as usize], q)).abs() < 1e-12,
                "nearest distance mismatch at {q:?}"
            );
        }
    }

    #[test]
    fn empty_tree_is_safe() {
        let tree = Octree::build(&[]);
        assert_eq!(tree.nearest([0.0, 0.0, 0.0]), None);
    }

    #[test]
    fn coincident_points_do_not_overflow_depth() {
        let pts = vec![[1.0, 1.0, 1.0]; 100];
        let tree = Octree::build(&pts);
        assert!(tree.nearest([1.0, 1.0, 1.0]).is_some());
    }
}

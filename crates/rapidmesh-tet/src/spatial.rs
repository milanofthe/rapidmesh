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

/// Spreads the low 21 bits of `x` to every third bit (3D Morton split).
fn split3(x: u64) -> u64 {
    let mut x = x & 0x1f_ffff;
    x = (x | x << 32) & 0x1f00_0000_00ff_ffff;
    x = (x | x << 16) & 0x1f00_00ff_0000_ffff;
    x = (x | x << 8) & 0x100f_00f0_0f00_f00f;
    x = (x | x << 4) & 0x10c3_0c30_c30c_30c3;
    x = (x | x << 2) & 0x1249_2492_4924_9249;
    x
}

/// A permutation of `0..points.len()` ordering the points along a Morton
/// (Z-order) curve. Inserting in this order keeps consecutive points spatially
/// close, so an incremental Delaunay's point-location walk stays short (the
/// classic BRIO/space-filling-curve speedup): near-linear construction instead
/// of the long walks a geometrically-unsorted insertion order causes.
pub fn morton_order(points: &[V3]) -> Vec<usize> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    let mut lo = points[0];
    let mut hi = points[0];
    for p in points {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let span: [f64; 3] = std::array::from_fn(|k| (hi[k] - lo[k]).max(1e-300));
    const MASK: u64 = (1 << 21) - 1;
    let max = MASK as f64;
    let code = |p: V3| -> u64 {
        let q: [u64; 3] =
            std::array::from_fn(|k| (((p[k] - lo[k]) / span[k] * max).round() as u64).min(MASK));
        split3(q[0]) | (split3(q[1]) << 1) | (split3(q[2]) << 2)
    };
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by_key(|&i| code(points[i]));
    idx
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

    /// All point indices within Euclidean distance `r` of `q`.
    pub fn within_radius(&self, q: V3, r: f64) -> Vec<u32> {
        let mut out = Vec::new();
        if self.pts.is_empty() {
            return out;
        }
        within_in(&self.root, &self.pts, self.center, self.half, q, r * r, &mut out);
        out
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

fn within_in(node: &Node, pts: &[V3], center: V3, half: f64, q: V3, r2: f64, out: &mut Vec<u32>) {
    if box_dist2(center, half, q) > r2 {
        return;
    }
    match node {
        Node::Leaf(idx) => {
            for &i in idx {
                if dist2(pts[i as usize], q) <= r2 {
                    out.push(i);
                }
            }
        }
        Node::Inner(children) => {
            for o in 0..8 {
                let (cc, ch) = child_box(center, half, o);
                within_in(&children[o], pts, cc, ch, q, r2, out);
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
    fn within_radius_matches_brute_force() {
        let pts = grid(8);
        let tree = Octree::build(&pts);
        let q = [1.5, -0.7, 1.4];
        let r = 1.0;
        let mut got = tree.within_radius(q, r);
        got.sort_unstable();
        let mut want: Vec<u32> = (0..pts.len() as u32)
            .filter(|&i| dist2(pts[i as usize], q) <= r * r)
            .collect();
        want.sort_unstable();
        assert_eq!(got, want);
    }

    #[test]
    fn empty_tree_is_safe() {
        let tree = Octree::build(&[]);
        assert_eq!(tree.nearest([0.0, 0.0, 0.0]), None);
        assert!(tree.within_radius([0.0, 0.0, 0.0], 10.0).is_empty());
    }

    #[test]
    fn coincident_points_do_not_overflow_depth() {
        let pts = vec![[1.0, 1.0, 1.0]; 100];
        let tree = Octree::build(&pts);
        assert!(tree.nearest([1.0, 1.0, 1.0]).is_some());
        assert_eq!(tree.within_radius([1.0, 1.0, 1.0], 0.1).len(), 100);
    }
}

//! A bounding-volume hierarchy over the boundary facets, for `O(log F)`
//! nearest-facet distance and graded distance-field queries (the sizing field's
//! wall term), replacing the `O(F)` brute scan that dominates high-facet meshes
//! (bunny/blob: the domain-octree build was 73% brute facet distance).
//!
//! A BVH (median split on facet centroids) is the right structure for triangle
//! nearest-distance, the way the point-octree ([`crate::spatial`] /
//! [`crate::domain`]) is right for sites: each facet sits in exactly one leaf,
//! internal nodes carry the subtree AABB and the finest target inside it, so
//! both queries prune by a node lower bound (branch and bound).

use rapidmesh_csg::Tri;

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Squared distance from point `p` to triangle `t` (closest-point clamp).
pub fn point_tri_dist2(p: V3, t: &Tri) -> f64 {
    let (a, b, c) = (t.v[0], t.v[1], t.v[2]);
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return dot(ap, ap);
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return dot(bp, bp);
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return dot(cp, cp);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let q: V3 = std::array::from_fn(|k| a[k] + v * ab[k]);
        return dot(sub(p, q), sub(p, q));
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let q: V3 = std::array::from_fn(|k| a[k] + w * ac[k]);
        return dot(sub(p, q), sub(p, q));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let q: V3 = std::array::from_fn(|k| b[k] + w * (c[k] - b[k]));
        return dot(sub(p, q), sub(p, q));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let q: V3 = std::array::from_fn(|k| a[k] + ab[k] * v + ac[k] * w);
    dot(sub(p, q), sub(p, q))
}

/// Squared distance from `p` to the axis-aligned box `[lo, hi]` (0 if inside).
fn box_dist2(lo: V3, hi: V3, p: V3) -> f64 {
    let mut d2 = 0.0;
    for k in 0..3 {
        let e = if p[k] < lo[k] {
            lo[k] - p[k]
        } else if p[k] > hi[k] {
            p[k] - hi[k]
        } else {
            0.0
        };
        d2 += e * e;
    }
    d2
}

struct Node {
    lo: V3,
    hi: V3,
    /// Finest facet target in this subtree (for the graded-min lower bound).
    min_target: f64,
    /// Child node indices (a flat-array BVH: the left subtree spans many slots,
    /// so the right child is NOT `left + 1`). Unused for a leaf (`count > 0`).
    left: u32,
    right: u32,
    start: u32,
    count: u32,
}

/// Maximum facets per BVH leaf.
const LEAF_MAX: usize = 4;

pub struct FacetBvh {
    tris: Vec<Tri>,
    targets: Vec<f64>,
    /// Facet indices grouped by leaf (a permutation of `0..tris.len()`).
    order: Vec<u32>,
    nodes: Vec<Node>,
}

impl FacetBvh {
    /// Builds a BVH over `(tri, target)` facets. Empty input is queryable
    /// (`nearest_dist` returns INFINITY).
    pub fn build(facets: &[(Tri, f64)]) -> FacetBvh {
        let tris: Vec<Tri> = facets.iter().map(|f| f.0).collect();
        let targets: Vec<f64> = facets.iter().map(|f| f.1).collect();
        let centroids: Vec<V3> = tris
            .iter()
            .map(|t| std::array::from_fn(|k| (t.v[0][k] + t.v[1][k] + t.v[2][k]) / 3.0))
            .collect();
        let mut order: Vec<u32> = (0..tris.len() as u32).collect();
        let mut nodes: Vec<Node> = Vec::new();
        if !tris.is_empty() {
            build_node(&tris, &targets, &centroids, &mut order, 0, tris.len(), &mut nodes);
        }
        FacetBvh { tris, targets, order, nodes }
    }

    pub fn is_empty(&self) -> bool {
        self.tris.is_empty()
    }

    /// Distance from `p` to the nearest facet (INFINITY if empty).
    pub fn nearest_dist(&self, p: V3) -> f64 {
        if self.nodes.is_empty() {
            return f64::INFINITY;
        }
        let mut best2 = f64::INFINITY;
        self.nearest_rec(0, p, &mut best2);
        best2.sqrt()
    }

    fn nearest_rec(&self, ni: usize, p: V3, best2: &mut f64) {
        let n = &self.nodes[ni];
        if box_dist2(n.lo, n.hi, p) >= *best2 {
            return;
        }
        if n.count > 0 {
            for &fi in &self.order[n.start as usize..(n.start + n.count) as usize] {
                let d2 = point_tri_dist2(p, &self.tris[fi as usize]);
                if d2 < *best2 {
                    *best2 = d2;
                }
            }
            return;
        }
        // Visit the nearer child first so the farther one prunes more often.
        let (l, r) = (n.left as usize, n.right as usize);
        let dl = box_dist2(self.nodes[l].lo, self.nodes[l].hi, p);
        let dr = box_dist2(self.nodes[r].lo, self.nodes[r].hi, p);
        if dl <= dr {
            self.nearest_rec(l, p, best2);
            self.nearest_rec(r, p, best2);
        } else {
            self.nearest_rec(r, p, best2);
            self.nearest_rec(l, p, best2);
        }
    }

    /// `min over facets ( target + grading * dist(p, facet) )`: the graded
    /// distance field that grows the sizing field from the fine wall targets.
    pub fn graded_min(&self, p: V3, grading: f64) -> f64 {
        if self.nodes.is_empty() {
            return f64::INFINITY;
        }
        let mut best = f64::INFINITY;
        self.graded_rec(0, p, grading, &mut best);
        best
    }

    fn graded_rec(&self, ni: usize, p: V3, grading: f64, best: &mut f64) {
        let n = &self.nodes[ni];
        // Lower bound for anything in this subtree: the finest target plus the
        // graded distance to the subtree box.
        let bound = n.min_target + grading * box_dist2(n.lo, n.hi, p).sqrt();
        if bound >= *best {
            return;
        }
        if n.count > 0 {
            for &fi in &self.order[n.start as usize..(n.start + n.count) as usize] {
                let v = self.targets[fi as usize]
                    + grading * point_tri_dist2(p, &self.tris[fi as usize]).sqrt();
                if v < *best {
                    *best = v;
                }
            }
            return;
        }
        let (l, r) = (n.left as usize, n.right as usize);
        let bl = self.nodes[l].min_target + grading * box_dist2(self.nodes[l].lo, self.nodes[l].hi, p).sqrt();
        let br = self.nodes[r].min_target + grading * box_dist2(self.nodes[r].lo, self.nodes[r].hi, p).sqrt();
        if bl <= br {
            self.graded_rec(l, p, grading, best);
            self.graded_rec(r, p, grading, best);
        } else {
            self.graded_rec(r, p, grading, best);
            self.graded_rec(l, p, grading, best);
        }
    }

    /// Facet indices whose AABB overlaps the infinite x-column `[y +/- m] x
    /// [z +/- m]` around `p` (x ignored): the candidate set for an axis-aligned
    /// +x parity ray-cast. `m` must cover the ray-cast's y,z jitter band, so the
    /// excluded facets cannot be crossed by any cast and parity stays exact.
    pub fn column_yz(&self, p: V3, m: f64, out: &mut Vec<u32>) {
        out.clear();
        if !self.nodes.is_empty() {
            self.column_rec(0, p, m, out);
        }
    }

    fn column_rec(&self, ni: usize, p: V3, m: f64, out: &mut Vec<u32>) {
        let n = &self.nodes[ni];
        if n.hi[1] < p[1] - m || n.lo[1] > p[1] + m || n.hi[2] < p[2] - m || n.lo[2] > p[2] + m {
            return;
        }
        if n.count > 0 {
            out.extend_from_slice(&self.order[n.start as usize..(n.start + n.count) as usize]);
            return;
        }
        self.column_rec(n.left as usize, p, m, out);
        self.column_rec(n.right as usize, p, m, out);
    }
}

/// Builds the node spanning `order[start..end]`, returns its node index. Splits
/// at the centroid median along the widest centroid axis.
fn build_node(
    tris: &[Tri],
    targets: &[f64],
    centroids: &[V3],
    order: &mut [u32],
    start: usize,
    end: usize,
    nodes: &mut Vec<Node>,
) -> u32 {
    // Subtree AABB and finest target.
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    let mut min_target = f64::MAX;
    for &fi in &order[start..end] {
        let t = &tris[fi as usize];
        for v in &t.v {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
        min_target = min_target.min(targets[fi as usize]);
    }
    let idx = nodes.len() as u32;
    let count = end - start;
    if count <= LEAF_MAX {
        nodes.push(Node {
            lo,
            hi,
            min_target,
            left: 0,
            right: 0,
            start: start as u32,
            count: count as u32,
        });
        return idx;
    }
    // Reserve this node's slot; children get appended and patched after.
    nodes.push(Node { lo, hi, min_target, left: 0, right: 0, start: 0, count: 0 });
    // Widest centroid axis.
    let mut clo = [f64::MAX; 3];
    let mut chi = [f64::MIN; 3];
    for &fi in &order[start..end] {
        for k in 0..3 {
            clo[k] = clo[k].min(centroids[fi as usize][k]);
            chi[k] = chi[k].max(centroids[fi as usize][k]);
        }
    }
    let axis = (0..3).max_by(|&a, &b| (chi[a] - clo[a]).partial_cmp(&(chi[b] - clo[b])).unwrap()).unwrap();
    let mid = start + count / 2;
    order[start..end]
        .select_nth_unstable_by(count / 2, |&a, &b| {
            centroids[a as usize][axis].partial_cmp(&centroids[b as usize][axis]).unwrap()
        });
    let left = build_node(tris, targets, centroids, order, start, mid, nodes);
    let right = build_node(tris, targets, centroids, order, mid, end, nodes);
    nodes[idx as usize].left = left;
    nodes[idx as usize].right = right;
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri(a: V3, b: V3, c: V3, target: f64) -> (Tri, f64) {
        (Tri::new(a, b, c), target)
    }

    fn brute_nearest(facets: &[(Tri, f64)], p: V3) -> f64 {
        facets.iter().map(|(t, _)| point_tri_dist2(p, t)).fold(f64::MAX, f64::min).sqrt()
    }

    fn brute_graded(facets: &[(Tri, f64)], p: V3, g: f64) -> f64 {
        facets.iter().map(|(t, tg)| tg + g * point_tri_dist2(p, t).sqrt()).fold(f64::MAX, f64::min)
    }

    fn box_facets() -> Vec<(Tri, f64)> {
        // Two triangles per face of the unit cube, target varying per face.
        let mut f = Vec::new();
        let c = [
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], 0.1),
            ([0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], 0.1),
            ([0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], 0.5),
            ([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], 0.5),
            ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], 0.3),
            ([0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0], 0.3),
        ];
        for (a, b, cc, t) in c {
            f.push(tri(a, b, cc, t));
        }
        f
    }

    #[test]
    fn nearest_matches_brute() {
        let f = box_facets();
        let bvh = FacetBvh::build(&f);
        for p in [[0.5, 0.5, 0.5], [0.1, 0.2, 0.9], [-1.0, 0.5, 0.5], [0.5, 0.5, 2.0]] {
            let got = bvh.nearest_dist(p);
            let want = brute_nearest(&f, p);
            assert!((got - want).abs() < 1e-12, "nearest at {p:?}: {got} vs {want}");
        }
    }

    #[test]
    fn graded_min_matches_brute() {
        let f = box_facets();
        let bvh = FacetBvh::build(&f);
        let g = 0.5;
        for p in [[0.5, 0.5, 0.5], [0.1, 0.2, 0.9], [0.5, 0.5, 0.05], [0.95, 0.5, 0.5]] {
            let got = bvh.graded_min(p, g);
            let want = brute_graded(&f, p, g);
            assert!((got - want).abs() < 1e-12, "graded at {p:?}: {got} vs {want}");
        }
    }

    #[test]
    fn deep_tree_matches_brute() {
        // Many facets force a multi-level tree, so the flat-array child indices
        // (right != left+1) are exercised: a grid of small triangles at varied
        // depths with varied targets.
        let mut f = Vec::new();
        for i in 0..7 {
            for j in 0..7 {
                let (x, y) = (i as f64 * 0.5, j as f64 * 0.5);
                let z = 0.1 * (i + j) as f64;
                let target = 0.05 + 0.02 * ((i * 7 + j) % 5) as f64;
                f.push(tri([x, y, z], [x + 0.4, y, z], [x, y + 0.4, z + 0.2], target));
            }
        }
        let bvh = FacetBvh::build(&f);
        for p in [
            [1.3, 1.7, 0.5],
            [-2.0, 0.5, 1.0],
            [3.1, 3.2, 0.0],
            [0.25, 0.25, 5.0],
            [1.0, 2.0, -1.0],
        ] {
            assert!(
                (bvh.nearest_dist(p) - brute_nearest(&f, p)).abs() < 1e-9,
                "nearest at {p:?}"
            );
            assert!(
                (bvh.graded_min(p, 0.5) - brute_graded(&f, p, 0.5)).abs() < 1e-9,
                "graded at {p:?}"
            );
        }
    }

    #[test]
    fn column_yz_is_a_superset() {
        let mut f = Vec::new();
        for i in 0..7 {
            for j in 0..7 {
                let (x, y) = (i as f64 * 0.5, j as f64 * 0.5);
                f.push(tri([x, y, 0.0], [x + 0.4, y, 0.3], [x, y + 0.4, 0.0], 0.1));
            }
        }
        let bvh = FacetBvh::build(&f);
        let m = 0.3;
        for p in [[1.3, 1.7, 0.1], [0.0, 3.0, 0.2], [3.0, 0.0, -0.1]] {
            let mut got = Vec::new();
            bvh.column_yz(p, m, &mut got);
            let got: std::collections::HashSet<u32> = got.into_iter().collect();
            // Every facet whose own AABB overlaps the y,z column must be returned.
            for (i, (t, _)) in f.iter().enumerate() {
                let (mut lo, mut hi) = (t.v[0], t.v[0]);
                for v in &t.v {
                    for k in 0..3 {
                        lo[k] = lo[k].min(v[k]);
                        hi[k] = hi[k].max(v[k]);
                    }
                }
                let overlaps = hi[1] >= p[1] - m && lo[1] <= p[1] + m && hi[2] >= p[2] - m && lo[2] <= p[2] + m;
                if overlaps {
                    assert!(got.contains(&(i as u32)), "facet {i} in column missing at {p:?}");
                }
            }
        }
    }

    #[test]
    fn empty_is_safe() {
        let bvh = FacetBvh::build(&[]);
        assert_eq!(bvh.nearest_dist([0.0; 3]), f64::INFINITY);
        assert!(bvh.is_empty());
    }
}

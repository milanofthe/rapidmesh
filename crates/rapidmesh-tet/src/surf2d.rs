//! 2D Delaunay + Lloyd relaxation for the surface (2D) stage of the hierarchy.
//!
//! Built on the EXACT robust predicates (`orient2d`/`incircle2d` from
//! rapidmesh-exact, the Shewchuk-style kernel): all triangulation decisions are
//! exact, so the relaxation is robust on degenerate (cocircular/collinear) grid
//! inputs. Only the centroid WEIGHTS are float (a non-decision quantity).
//!
//! Used by the surface stage: each planar patch is filled by scattering interior
//! points and relaxing them with the patch boundary (1D edge points) held fixed.
//! The triangulation here serves the relaxation; the conforming surface is read
//! back from the 3D mesh downstream.

use rapidmesh_exact::{incircle2d, orient2d, Axis, Point3, Sign};

type P2 = [f64; 2];

fn p3(p: P2) -> Point3 {
    Point3::explicit(p[0], p[1], 0.0)
}

/// Orientation of (a, b, c) in the xy plane (exact).
fn orient(a: P2, b: P2, c: P2) -> Sign {
    orient2d(&p3(a), &p3(b), &p3(c), Axis::Z).expect("explicit points are valid")
}

/// True iff `d` is strictly inside the circumcircle of CCW triangle (a, b, c)
/// (exact; cocircular -> false, a consistent choice yielding a valid mesh).
fn in_circumcircle(a: P2, b: P2, c: P2, d: P2) -> bool {
    incircle2d(&p3(a), &p3(b), &p3(c), &p3(d), Axis::Z) == Some(Sign::Positive)
}

/// The triangle reordered to CCW.
fn ccw(t: [usize; 3], pts: &[P2]) -> [usize; 3] {
    if orient(pts[t[0]], pts[t[1]], pts[t[2]]) == Sign::Negative {
        [t[0], t[2], t[1]]
    } else {
        t
    }
}

fn dist2(a: P2, b: P2) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

/// Bowyer-Watson 2D Delaunay; triangles as CCW index triples into `points`,
/// super-triangle removed. Exact predicates.
pub fn delaunay2(points: &[P2]) -> Vec<[usize; 3]> {
    let n = points.len();
    if n < 3 {
        return Vec::new();
    }
    let mut lo = points[0];
    let mut hi = points[0];
    for p in points {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let d = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-12);
    let mid = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
    let big = 1000.0 * d;
    let mut pts: Vec<P2> = points.to_vec();
    let (s0, s1, s2) = (n, n + 1, n + 2);
    pts.push([mid[0] - big, mid[1] - big]);
    pts.push([mid[0] + big, mid[1] - big]);
    pts.push([mid[0], mid[1] + big]);
    let mut tris: Vec<[usize; 3]> = vec![ccw([s0, s1, s2], &pts)];

    for i in 0..n {
        let p = pts[i];
        let mut bad: Vec<usize> = Vec::new();
        for (ti, t) in tris.iter().enumerate() {
            if in_circumcircle(pts[t[0]], pts[t[1]], pts[t[2]], p) {
                bad.push(ti);
            }
        }
        if bad.is_empty() {
            continue; // degenerate cocircular case: leave the mesh as is
        }
        // Cavity boundary = directed edges of bad triangles without a reverse.
        // Deterministic hashing: the boundary edge order sets the new-triangle
        // order, and the surface Lloyd then sums area-weighted centroids over
        // those triangles (a non-associative float sum), so a random order would
        // make the relaxed surface points (and the whole mesh) vary run-to-run.
        let mut count: rustc_hash::FxHashMap<(usize, usize), i32> =
            rustc_hash::FxHashMap::default();
        for &ti in &bad {
            let t = tris[ti];
            for e in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *count.entry(e).or_insert(0) += 1;
            }
        }
        let boundary: Vec<(usize, usize)> = count
            .keys()
            .copied()
            .filter(|&(a, b)| !count.contains_key(&(b, a)))
            .collect();
        bad.sort_unstable_by(|x, y| y.cmp(x));
        for ti in bad {
            tris.swap_remove(ti);
        }
        for (a, b) in boundary {
            tris.push(ccw([a, b, i], &pts));
        }
    }
    tris.into_iter().filter(|t| t.iter().all(|&v| v < n)).collect()
}

/// Fills a planar region with Lloyd-relaxed interior points at a GRADED local
/// `target` spacing (`target(q)` is the desired edge length at `q`). `step` is
/// the finest target on the patch, the grid step of the initial scatter; the
/// per-point separation is the LOCAL `0.5 * target`, so the density grades:
/// dense where `target` is small, sparse where it is large. `boundary` is the
/// set of FIXED boundary points (graded 1D edge points and corners); `inside`
/// decides patch membership (exact, supplied by the caller). Interior points are
/// scattered on a grid in `[lo, hi]`, kept inside and clear of the boundary by
/// the local radius, then moved toward the area-weighted centroid of their
/// incident triangles with a local separation guard (no collapse / sliver seed).
pub fn cvt_fill(
    boundary: &[P2],
    lo: P2,
    hi: P2,
    step: f64,
    target: impl Fn(P2) -> f64,
    iters: usize,
    inside: impl Fn(P2) -> bool,
) -> Vec<P2> {
    if !(step.is_finite() && step > 0.0) {
        return Vec::new();
    }
    let sep2 = |q: P2| (0.5 * target(q)).powi(2);
    let nb = boundary.len();
    let nx = (((hi[0] - lo[0]) / step).ceil() as usize).max(1);
    let ny = (((hi[1] - lo[1]) / step).ceil() as usize).max(1);
    // Greedy graded scatter: keep a grid node only if it clears the boundary and
    // every already-kept interior point by its OWN local radius.
    let mut interior: Vec<P2> = Vec::new();
    for i in 1..nx {
        for j in 1..ny {
            let q = [lo[0] + i as f64 * step, lo[1] + j as f64 * step];
            if !inside(q) {
                continue;
            }
            let r2 = sep2(q);
            if boundary.iter().all(|&b| dist2(q, b) >= r2)
                && interior.iter().all(|&p| dist2(q, p) >= r2)
            {
                interior.push(q);
            }
        }
    }

    for _ in 0..iters {
        if interior.is_empty() {
            break;
        }
        let mut all: Vec<P2> = boundary.to_vec();
        all.extend_from_slice(&interior);
        let tris = delaunay2(&all);
        let mut num = vec![[0.0f64; 2]; all.len()];
        let mut den = vec![0.0f64; all.len()];
        for t in &tris {
            let p = [all[t[0]], all[t[1]], all[t[2]]];
            // Float area as a relaxation WEIGHT (not a decision).
            let area = 0.5 * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]))
                .abs();
            let c = [
                (p[0][0] + p[1][0] + p[2][0]) / 3.0,
                (p[0][1] + p[1][1] + p[2][1]) / 3.0,
            ];
            for &v in t {
                num[v][0] += area * c[0];
                num[v][1] += area * c[1];
                den[v] += area;
            }
        }
        for k in 0..interior.len() {
            let v = nb + k;
            if den[v] == 0.0 {
                continue;
            }
            let tgt = [num[v][0] / den[v], num[v][1] / den[v]];
            if !inside(tgt) {
                continue;
            }
            let r2 = sep2(tgt);
            let clear = boundary.iter().all(|&b| dist2(tgt, b) >= r2)
                && interior.iter().enumerate().all(|(m, &q)| m == k || dist2(tgt, q) >= r2);
            if clear {
                interior[k] = tgt;
            }
        }
    }
    interior
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delaunay2_grid_triangle_count() {
        let mut pts = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                pts.push([i as f64, j as f64]);
            }
        }
        // 25 points, 16 on the hull -> 2*25 - 2 - 16 = 32 triangles.
        assert_eq!(delaunay2(&pts).len(), 32);
    }

    #[test]
    fn cvt_fill_square_well_separated() {
        // Unit square boundary at spacing 0.2.
        let m = 5;
        let mut boundary = Vec::new();
        for i in 0..m {
            boundary.push([i as f64 / m as f64, 0.0]);
            boundary.push([1.0, i as f64 / m as f64]);
            boundary.push([1.0 - i as f64 / m as f64, 1.0]);
            boundary.push([0.0, 1.0 - i as f64 / m as f64]);
        }
        let sq = |p: P2| p[0] > 0.0 && p[0] < 1.0 && p[1] > 0.0 && p[1] < 1.0;
        let interior = cvt_fill(&boundary, [0.0, 0.0], [1.0, 1.0], 0.2, |_| 0.2, 12, sq);
        assert!(!interior.is_empty());
        let mut all = boundary.clone();
        all.extend_from_slice(&interior);
        let mut min_sep2 = f64::MAX;
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                min_sep2 = min_sep2.min(dist2(all[i], all[j]));
            }
        }
        assert!(min_sep2.sqrt() >= 0.5 * 0.2, "points too close: {}", min_sep2.sqrt());
    }
}

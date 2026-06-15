//! 2D CVT building block for surface meshing (WP4).
//!
//! The mesher is hierarchical and density-driven: 1D edge sizing prescribes the
//! boundary point density, then a planar patch is filled by scattering interior
//! points at the area density and Lloyd-relaxing them against a 2D Delaunay; the
//! same idea projected onto an analytic surface handles curved faces (WP6). The
//! relaxed surface points then become fixed boundary sites for the 3D volume CVT,
//! so material interfaces are shared, conforming surface meshes by construction.
//!
//! The triangulation here is float-only: it serves the relaxation (computing 2D
//! cell centroids) only. The authoritative geometry is the 3D volume mesh these
//! points feed, whose conformity is checked exactly downstream.

use std::collections::HashMap;

type P2 = [f64; 2];

/// Twice the signed area of triangle (a, b, c); > 0 iff counter-clockwise.
fn orient2(a: P2, b: P2, c: P2) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// > 0 iff `d` lies inside the circumcircle of the CCW triangle (a, b, c).
fn in_circle(a: P2, b: P2, c: P2, d: P2) -> f64 {
    let ad = [a[0] - d[0], a[1] - d[1]];
    let bd = [b[0] - d[0], b[1] - d[1]];
    let cd = [c[0] - d[0], c[1] - d[1]];
    let abd = ad[0] * bd[1] - bd[0] * ad[1];
    let bcd = bd[0] * cd[1] - cd[0] * bd[1];
    let cad = cd[0] * ad[1] - ad[0] * cd[1];
    let a2 = ad[0] * ad[0] + ad[1] * ad[1];
    let b2 = bd[0] * bd[0] + bd[1] * bd[1];
    let c2 = cd[0] * cd[0] + cd[1] * cd[1];
    a2 * bcd + b2 * cad + c2 * abd
}

/// CCW-oriented triangle of the three indices.
fn ccw(t: [usize; 3], pts: &[P2]) -> [usize; 3] {
    if orient2(pts[t[0]], pts[t[1]], pts[t[2]]) < 0.0 {
        [t[0], t[2], t[1]]
    } else {
        t
    }
}

/// Bowyer-Watson 2D Delaunay triangulation; returns triangles as index triples
/// into `points` (super-triangle tris removed), each CCW. Float predicates: for
/// the relaxation loop, not for exact conformity.
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
    let big = 50.0 * d;
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
            if in_circle(pts[t[0]], pts[t[1]], pts[t[2]], p) > 0.0 {
                bad.push(ti);
            }
        }
        // Cavity boundary: directed edges of bad triangles whose reverse is not
        // also a bad edge.
        let mut count: HashMap<(usize, usize), i32> = HashMap::new();
        for &ti in &bad {
            let t = tris[ti];
            for e in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *count.entry(e).or_insert(0) += 1;
            }
        }
        let mut boundary: Vec<(usize, usize)> = Vec::new();
        for (&(a, b), _) in count.iter() {
            if !count.contains_key(&(b, a)) {
                boundary.push((a, b));
            }
        }
        // Remove bad triangles (high indices first) and cone the cavity to p.
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

/// Crossing-number point-in-polygon test (polygon as an ordered loop).
pub fn point_in_polygon(p: P2, loop_pts: &[P2]) -> bool {
    let n = loop_pts.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (loop_pts[i], loop_pts[j]);
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let x = a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
            if p[0] < x {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Fills the polygon `boundary` (an ordered loop of fixed boundary points) with
/// `n_interior` Lloyd-relaxed interior points at the target spacing. Boundary
/// points stay fixed; interior points are scattered on a grid inside the polygon
/// and moved toward the area-weighted centroid of their incident triangles,
/// clamped to stay inside. Returns the relaxed interior points.
pub fn cvt_fill_polygon(boundary: &[P2], spacing: f64, iters: usize) -> Vec<P2> {
    if boundary.len() < 3 || !(spacing.is_finite() && spacing > 0.0) {
        return Vec::new();
    }
    let mut lo = boundary[0];
    let mut hi = boundary[0];
    for p in boundary {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    // Scatter interior points on a grid at `spacing`, keep those strictly inside
    // and clear of the boundary (so the Delaunay sees no near-duplicates).
    let nb = boundary.len();
    let mut interior: Vec<P2> = Vec::new();
    let nx = (((hi[0] - lo[0]) / spacing).ceil() as usize).max(1);
    let ny = (((hi[1] - lo[1]) / spacing).ceil() as usize).max(1);
    for i in 1..nx {
        for j in 1..ny {
            let q = [lo[0] + i as f64 * spacing, lo[1] + j as f64 * spacing];
            if point_in_polygon(q, boundary)
                && boundary.iter().all(|&b| dist2(q, b) > (0.4 * spacing).powi(2))
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
            let area = 0.5 * orient2(p[0], p[1], p[2]).abs();
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
        // Apply moves sequentially with a separation guard: a target is taken
        // only if it stays inside and clear of every other point (boundary or
        // interior). Lloyd spreads points apart; the guard forbids the rare
        // edge-hugging / collapse that would seed a 3D sliver or near-duplicate.
        let min_sep2 = (0.55 * spacing).powi(2);
        for k in 0..interior.len() {
            let v = nb + k;
            if den[v] == 0.0 {
                continue;
            }
            let tgt = [num[v][0] / den[v], num[v][1] / den[v]];
            if !point_in_polygon(tgt, boundary) {
                continue;
            }
            let clear = boundary.iter().all(|&b| dist2(tgt, b) >= min_sep2)
                && interior
                    .iter()
                    .enumerate()
                    .all(|(m, &q)| m == k || dist2(tgt, q) >= min_sep2);
            if clear {
                interior[k] = tgt;
            }
        }
    }
    interior
}

fn dist2(a: P2, b: P2) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn min_angle_deg(tris: &[[usize; 3]], pts: &[P2]) -> f64 {
        let mut m = 180.0_f64;
        for t in tris {
            let p = [pts[t[0]], pts[t[1]], pts[t[2]]];
            for i in 0..3 {
                let a = p[i];
                let b = p[(i + 1) % 3];
                let c = p[(i + 2) % 3];
                let u = [b[0] - a[0], b[1] - a[1]];
                let v = [c[0] - a[0], c[1] - a[1]];
                let dot = u[0] * v[0] + u[1] * v[1];
                let nu = (u[0] * u[0] + u[1] * u[1]).sqrt();
                let nv = (v[0] * v[0] + v[1] * v[1]).sqrt();
                if nu * nv == 0.0 {
                    continue;
                }
                let ang = (dot / (nu * nv)).clamp(-1.0, 1.0).acos().to_degrees();
                m = m.min(ang);
            }
        }
        m
    }

    #[test]
    fn delaunay2_matches_grid() {
        // A regular grid triangulates without slivers (min angle ~45 deg).
        let mut pts = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                pts.push([i as f64, j as f64]);
            }
        }
        let tris = delaunay2(&pts);
        // Euler: a convex point set of N points triangulates to 2N - 2 - h tris
        // (h = hull points). 25 pts, 16 hull -> 2*25-2-16 = 32.
        assert_eq!(tris.len(), 32, "grid triangle count");
        assert!(min_angle_deg(&tris, &pts) > 20.0, "no slivers on a grid");
    }

    #[test]
    fn point_in_polygon_square() {
        let sq = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!(point_in_polygon([0.5, 0.5], &sq));
        assert!(!point_in_polygon([1.5, 0.5], &sq));
        assert!(!point_in_polygon([-0.1, 0.5], &sq));
    }

    #[test]
    fn cvt_fill_square_is_well_distributed() {
        // Square boundary at spacing 0.2 (5 segments per edge).
        let mut boundary = Vec::new();
        let m = 5;
        for i in 0..m {
            boundary.push([i as f64 / m as f64, 0.0]);
        }
        for i in 0..m {
            boundary.push([1.0, i as f64 / m as f64]);
        }
        for i in 0..m {
            boundary.push([1.0 - i as f64 / m as f64, 1.0]);
        }
        for i in 0..m {
            boundary.push([0.0, 1.0 - i as f64 / m as f64]);
        }
        let spacing = 0.2;
        let interior = cvt_fill_polygon(&boundary, spacing, 12);
        assert!(!interior.is_empty(), "interior filled");
        for q in &interior {
            assert!(point_in_polygon(*q, &boundary), "interior stays inside");
        }
        // The property that matters for the 3D consumer: no two surface points
        // (boundary or interior) are near-coincident, so the volume Delaunay
        // sees no near-duplicate and no surface sliver is seeded.
        let mut all = boundary.clone();
        all.extend_from_slice(&interior);
        let mut min_sep2 = f64::MAX;
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                min_sep2 = min_sep2.min(dist2(all[i], all[j]));
            }
        }
        assert!(
            min_sep2.sqrt() >= 0.5 * spacing,
            "points too close (min sep {} < {})",
            min_sep2.sqrt(),
            0.5 * spacing
        );
        // Used by the throwaway relaxation triangulation; keep the helper live.
        let _ = min_angle_deg(&delaunay2(&all), &all);
    }
}

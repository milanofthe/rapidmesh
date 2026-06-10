//! Incremental 3D Delaunay tetrahedralization (Bowyer-Watson) with exact
//! predicates.
//!
//! Points are inserted into a large enclosing super-tetrahedron; the cavity
//! of circumsphere-violating tets is carved out (exact insphere), repaired to
//! star-shapedness (exact orient3d, which handles the heavily cospherical /
//! on-face configurations of grid geometry), and re-filled by coning the new
//! point to the cavity boundary. Super-tets are dropped at the end.
//!
//! Correctness first: location is a linear scan and adjacency is rebuilt per
//! insertion. The HXT-style fast kernel can replace the internals later
//! without changing the contract.

use rapidmesh_exact::Sign;
use std::collections::HashMap;

/// A Delaunay tetrahedralization of a point set.
#[derive(Debug)]
pub struct DelaunayTets {
    /// The input points (super-tet corners excluded).
    pub points: Vec<[f64; 3]>,
    /// Positively oriented tets as point indices.
    pub tets: Vec<[usize; 4]>,
}

fn orient(pts: &[[f64; 3]], a: usize, b: usize, c: usize, d: usize) -> Sign {
    Sign::of_f64(geometry_predicates::orient3d(pts[a], pts[b], pts[c], pts[d]))
}

fn insphere(pts: &[[f64; 3]], t: [usize; 4], p: usize) -> Sign {
    Sign::of_f64(geometry_predicates::insphere(
        pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]], pts[p],
    ))
}

/// The four faces of a tet, each oriented so the opposite vertex is on the
/// positive side.
fn faces(t: [usize; 4]) -> [[usize; 3]; 4] {
    [
        [t[1], t[3], t[2]],
        [t[0], t[2], t[3]],
        [t[0], t[3], t[1]],
        [t[0], t[1], t[2]],
    ]
}

fn sorted3(f: [usize; 3]) -> [usize; 3] {
    let mut s = f;
    s.sort_unstable();
    s
}

/// Exact Delaunay tetrahedralization of `points` (degenerate inputs allowed:
/// duplicates are rejected by panic, fully coplanar input yields no tets).
pub fn tetrahedralize(points: &[[f64; 3]]) -> DelaunayTets {
    let n = points.len();
    // Working point array: inputs first, then the four super corners.
    let mut pts: Vec<[f64; 3]> = points.to_vec();
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for p in points {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let c: [f64; 3] = std::array::from_fn(|k| 0.5 * (lo[k] + hi[k]));
    let d = (0..3).map(|k| hi[k] - lo[k]).fold(1.0_f64, f64::max);
    let big = 64.0 * d;
    pts.push([c[0] - big, c[1] - big, c[2] - big]);
    pts.push([c[0] + 3.0 * big, c[1] - big, c[2] - big]);
    pts.push([c[0] - big, c[1] + 3.0 * big, c[2] - big]);
    pts.push([c[0] - big, c[1] - big, c[2] + 3.0 * big]);
    let s = [n, n + 1, n + 2, n + 3];
    let mut seed = [s[0], s[1], s[2], s[3]];
    if orient(&pts, seed[0], seed[1], seed[2], seed[3]) == Sign::Negative {
        seed.swap(2, 3);
    }
    assert_eq!(
        orient(&pts, seed[0], seed[1], seed[2], seed[3]),
        Sign::Positive
    );
    let mut tets: Vec<[usize; 4]> = vec![seed];

    for p in 0..n {
        // Locate a tet containing p (closed).
        let containing = tets
            .iter()
            .position(|&t| faces(t).iter().all(|&f| {
                orient(&pts, f[0], f[1], f[2], p) != Sign::Negative
            }))
            .expect("point must lie inside the super-tet");

        // Adjacency for this pass.
        let mut by_face: HashMap<[usize; 3], Vec<usize>> = HashMap::new();
        for (ti, &t) in tets.iter().enumerate() {
            for f in faces(t) {
                by_face.entry(sorted3(f)).or_default().push(ti);
            }
        }
        let neighbor = |ti: usize, f: [usize; 3]| -> Option<usize> {
            by_face[&sorted3(f)].iter().copied().find(|&o| o != ti)
        };

        // Cavity: strict circumsphere violations, grown by BFS.
        let mut in_cavity = vec![false; tets.len()];
        in_cavity[containing] = true;
        let mut stack = vec![containing];
        while let Some(ti) = stack.pop() {
            for f in faces(tets[ti]) {
                if let Some(o) = neighbor(ti, f) {
                    if !in_cavity[o] && insphere(&pts, tets[o], p) == Sign::Positive {
                        in_cavity[o] = true;
                        stack.push(o);
                    }
                }
            }
        }
        // Repair to star-shapedness: every cavity boundary face must see p
        // strictly on its positive side; otherwise absorb the neighbor
        // (handles p exactly on faces/edges and cospherical clusters).
        loop {
            let mut grew = false;
            for ti in 0..tets.len() {
                if !in_cavity[ti] {
                    continue;
                }
                for f in faces(tets[ti]) {
                    let o = neighbor(ti, f);
                    if o.is_some_and(|o| in_cavity[o]) {
                        continue;
                    }
                    if orient(&pts, f[0], f[1], f[2], p) != Sign::Positive {
                        let o = o.expect("cavity reached the super-tet hull");
                        in_cavity[o] = true;
                        grew = true;
                    }
                }
            }
            if !grew {
                break;
            }
        }

        // Re-fill: cone p to the cavity boundary.
        let mut new_tets: Vec<[usize; 4]> = Vec::new();
        for ti in 0..tets.len() {
            if !in_cavity[ti] {
                continue;
            }
            for f in faces(tets[ti]) {
                if neighbor(ti, f).is_some_and(|o| in_cavity[o]) {
                    continue;
                }
                debug_assert_eq!(orient(&pts, f[0], f[1], f[2], p), Sign::Positive);
                new_tets.push([f[0], f[1], f[2], p]);
            }
        }
        let mut kept: Vec<[usize; 4]> = tets
            .iter()
            .enumerate()
            .filter(|&(ti, _)| !in_cavity[ti])
            .map(|(_, &t)| t)
            .collect();
        kept.extend(new_tets);
        tets = kept;
    }

    // Drop everything touching the super corners.
    tets.retain(|t| t.iter().all(|&v| v < n));
    pts.truncate(n);
    for t in &tets {
        debug_assert_eq!(orient(&pts, t[0], t[1], t[2], t[3]), Sign::Positive);
    }
    DelaunayTets { points: pts, tets }
}

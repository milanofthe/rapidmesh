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

const NONE2: usize = usize::MAX;

/// Walks from triangle `start` to the (CCW) triangle containing `p`, stepping
/// across any edge that `p` lies strictly to the right of. Falls back to a
/// linear scan if the walk does not converge (degenerate connectivity).
fn locate2(start: usize, p: P2, tris: &[[usize; 3]], nbr: &[[usize; 3]], alive: &[bool], pts: &[P2]) -> usize {
    let mut t = start;
    for _ in 0..tris.len() * 2 + 16 {
        let tv = tris[t];
        let mut step = NONE2;
        for e in 0..3 {
            let (a, b) = (tv[e], tv[(e + 1) % 3]);
            // CCW triangle: its interior is left of each directed edge a->b, so
            // `p` strictly right (orient negative) means it lies across edge e.
            if orient(pts[a], pts[b], p) == Sign::Negative && nbr[t][e] != NONE2 {
                step = nbr[t][e];
                break;
            }
        }
        if step == NONE2 {
            return t;
        }
        t = step;
    }
    (0..tris.len())
        .find(|&t| {
            alive[t]
                && (0..3).all(|e| {
                    let (a, b) = (tris[t][e], tris[t][(e + 1) % 3]);
                    orient(pts[a], pts[b], p) != Sign::Negative
                })
        })
        .unwrap_or(start)
}

/// Links the directed p-edge `(u, v)` at edge slot `es` of triangle `slot` to
/// the neighbouring new triangle that owns the reverse edge `(v, u)`.
fn link_pedge(
    map: &mut rustc_hash::FxHashMap<(usize, usize), (usize, usize)>,
    nbr: &mut [[usize; 3]],
    slot: usize,
    es: usize,
    u: usize,
    v: usize,
) {
    if let Some((other, oes)) = map.remove(&(v, u)) {
        nbr[slot][es] = other;
        nbr[other][oes] = slot;
    } else {
        map.insert((u, v), (slot, es));
    }
}

/// Incremental 2D Delaunay (Bowyer-Watson) with triangle adjacency: each point
/// is located by a visibility walk (O(log n) amortized on the grid-ordered
/// scatter the caller feeds) and its cavity is grown by a flood-fill through
/// neighbour links, rather than scanning every triangle (the old O(n^2)).
/// Triangles are CCW index triples into `points`, super-triangle removed; exact
/// predicates throughout. The flood-fill order is deterministic (a LIFO over
/// fixed adjacency), so the caller's centroid sums stay reproducible.
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
    let mut nbr: Vec<[usize; 3]> = vec![[NONE2; 3]];
    let mut alive: Vec<bool> = vec![true];
    let mut free: Vec<usize> = Vec::new();
    let mut mark: Vec<u32> = vec![0];
    let mut epoch = 0u32;
    let mut last = 0usize;
    let mut edge_map: rustc_hash::FxHashMap<(usize, usize), (usize, usize)> =
        rustc_hash::FxHashMap::default();

    for i in 0..n {
        let p = pts[i];
        let start = locate2(last, p, &tris, &nbr, &alive, &pts);
        let tv = tris[start];
        if !in_circumcircle(pts[tv[0]], pts[tv[1]], pts[tv[2]], p) {
            continue; // cocircular: leave the mesh as is (consistent choice)
        }
        epoch += 1;
        mark[start] = epoch;
        let mut cavity = vec![start];
        let mut stack = vec![start];
        // Boundary edges (a, b, external triangle) found during the flood-fill.
        let mut boundary: Vec<(usize, usize, usize)> = Vec::new();
        while let Some(t) = stack.pop() {
            let tv = tris[t];
            for e in 0..3 {
                let nb = nbr[t][e];
                let bad = nb != NONE2 && {
                    let v = tris[nb];
                    in_circumcircle(pts[v[0]], pts[v[1]], pts[v[2]], p)
                };
                if bad {
                    if mark[nb] != epoch {
                        mark[nb] = epoch;
                        cavity.push(nb);
                        stack.push(nb);
                    }
                } else {
                    boundary.push((tv[e], tv[(e + 1) % 3], nb));
                }
            }
        }
        for &t in &cavity {
            alive[t] = false;
            free.push(t);
        }
        // Fan p to each boundary edge; link to the external triangle across that
        // edge and (via edge_map) to the adjacent new triangles along the p-edges.
        edge_map.clear();
        let mut last_new = start;
        for (a, b, x) in boundary {
            let slot = match free.pop() {
                Some(s) => {
                    tris[s] = [a, b, i];
                    nbr[s] = [NONE2; 3];
                    alive[s] = true;
                    s
                }
                None => {
                    tris.push([a, b, i]);
                    nbr.push([NONE2; 3]);
                    alive.push(true);
                    mark.push(0);
                    tris.len() - 1
                }
            };
            // edge 0 = (a,b) faces the external triangle x (which holds (b,a)).
            nbr[slot][0] = x;
            if x != NONE2 {
                for e in 0..3 {
                    if tris[x][e] == b && tris[x][(e + 1) % 3] == a {
                        nbr[x][e] = slot;
                    }
                }
            }
            // edge 1 = (b,i), edge 2 = (i,a): internal cavity edges.
            link_pedge(&mut edge_map, &mut nbr, slot, 1, b, i);
            link_pedge(&mut edge_map, &mut nbr, slot, 2, i, a);
            last_new = slot;
        }
        last = last_new;
    }
    tris.iter()
        .enumerate()
        .filter(|&(t, _)| alive[t] && tris[t].iter().all(|&v| v < n))
        .map(|(_, t)| *t)
        .collect()
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

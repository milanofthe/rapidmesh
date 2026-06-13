//! Exact constrained triangulation of a single facet.
//!
//! Given a facet (input triangle), a set of points and a set of constraint
//! segments on it (all exact, possibly implicit), produces a triangulation of
//! the facet whose vertex set is the input set plus all constraint-constraint
//! crossing points, and whose edge set contains every (sub-divided) constraint
//! segment. Everything runs on exact predicates over the facet's 2D
//! projection; no coordinate is ever rounded.
//!
//! Algorithm: classic incremental insertion (interior 1→3 split, on-edge
//! 2→4 split) followed by flip-based constraint edge recovery. Constraints
//! are pre-split at their mutual crossing points (constructed exactly from
//! constraint provenance) and at every vertex lying on them, so recovery only
//! ever handles segments with empty interiors — the regime where flip
//! recovery provably terminates. The triangulation is constrained, not
//! Delaunay; a CDT upgrade (indirect incircle) can be layered on later
//! without changing this module's contract.

use crate::constraint::Constraint;
use crate::tri::Tri;
use rapidmesh_exact::{
    cmp_along, collinear, incircle2d, lex_cmp, orient2d, strictly_between, Axis, Point3, Sign,
};

/// Result of triangulating a facet.
#[derive(Debug)]
pub struct FacetTriangulation {
    /// Vertex pool. Indices 0..3 are the facet corners.
    pub vertices: Vec<Point3>,
    /// Sub-triangles as vertex indices, all oriented like the facet
    /// (their 2D orientation in `axis` equals `orientation`).
    pub triangles: Vec<[usize; 3]>,
    /// The projection axis used.
    pub axis: Axis,
    /// The facet's 2D orientation in that projection.
    pub orientation: Sign,
}

impl FacetTriangulation {
    /// True if the (undirected) edge {u, v} is present.
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        self.triangles.iter().any(|t| {
            (0..3).any(|e| {
                let (a, b) = (t[e], t[(e + 1) % 3]);
                (a == u && b == v) || (a == v && b == u)
            })
        })
    }
}

/// Adds `p` to the pool unless an exactly coincident point exists; returns
/// its index.
fn add_vertex(pool: &mut Vec<Point3>, p: Point3) -> usize {
    if let Some(i) = pool.iter().position(|q| q.coincides(&p)) {
        return i;
    }
    pool.push(p);
    pool.len() - 1
}

/// True if `t` contains the directed edge x→y.
fn has_directed_edge(t: [usize; 3], x: usize, y: usize) -> bool {
    (0..3).any(|e| t[e] == x && t[(e + 1) % 3] == y)
}

/// The vertex of `t` opposite to its directed edge x→y.
fn opposite_vertex(t: [usize; 3], x: usize, y: usize) -> usize {
    for e in 0..3 {
        if t[e] == x && t[(e + 1) % 3] == y {
            return t[(e + 2) % 3];
        }
    }
    unreachable!("triangle {t:?} does not contain edge {x}->{y}");
}

/// Exact constrained triangulation of `facet` with the given points and
/// constraint segments. All points and constraint endpoints must lie on the
/// (closed) facet; constraint crossing points are constructed exactly from
/// constraint provenance.
pub fn triangulate_facet(
    facet: &Tri,
    points: &[Point3],
    constraints: &[Constraint],
) -> FacetTriangulation {
    let (axis, orientation) = facet.projection_axis();
    let o2d = |a: &Point3, b: &Point3, c: &Point3| -> Sign {
        orient2d(a, b, c, axis).expect("all triangulation points are valid")
    };

    let tri_trace = std::env::var_os("RAPIDMESH_TRI_TRACE").is_some();
    let t_pool = std::time::Instant::now();
    // ------------------------------------------------------ vertex pool
    let mut pool: Vec<Point3> = Vec::new();
    for i in 0..3 {
        add_vertex(&mut pool, facet.point(i));
    }
    for p in points {
        add_vertex(&mut pool, p.clone());
    }
    let mut constraint_ids: Vec<(usize, usize)> = Vec::new();
    for c in constraints {
        let ia = add_vertex(&mut pool, c.a.clone());
        let ib = add_vertex(&mut pool, c.b.clone());
        if ia != ib {
            constraint_ids.push((ia, ib));
        }
    }
    let d_pool = t_pool.elapsed();
    let t_presplit = std::time::Instant::now();

    // Pre-split: exact crossing points of strictly crossing constraint pairs.
    for (i, ci) in constraints.iter().enumerate() {
        for cj in constraints.iter().skip(i + 1) {
            let si_a = o2d(&ci.a, &ci.b, &cj.a);
            let si_b = o2d(&ci.a, &ci.b, &cj.b);
            if si_a.combine(si_b) != Sign::Negative {
                continue;
            }
            let sj_a = o2d(&cj.a, &cj.b, &ci.a);
            let sj_b = o2d(&cj.a, &cj.b, &ci.b);
            if sj_a.combine(sj_b) != Sign::Negative {
                continue;
            }
            let x = ci
                .line_intersection(cj, facet.v)
                .expect("strictly crossing constraints have intersecting lines");
            debug_assert!(x.is_valid());
            add_vertex(&mut pool, x);
        }
    }

    let d_presplit = t_presplit.elapsed();
    let t_insert = std::time::Instant::now();
    // ------------------------------------------------- point insertion
    let mut tris: Vec<[usize; 3]> = vec![[0, 1, 2]];
    for k in 3..pool.len() {
        insert_vertex(&mut tris, &pool, orientation, axis, k);
    }
    let d_insert = t_insert.elapsed();
    let t_recover = std::time::Instant::now();

    // -------------------------------------------- constraint recovery
    // Cached f64 positions for the segment bounding-box prefilter below.
    let approx: Vec<[f64; 3]> = pool.iter().map(|p| p.approx().expect("valid")).collect();
    let mut chain_edges: Vec<(usize, usize)> = Vec::new();
    for &(ia, ib) in &constraint_ids {
        // Padded segment bounding box (f64): a vertex strictly on the
        // segment lies within it (its approx is within an ulp of the exact
        // position; the pad is a million times that). On boolean scenes most
        // pool vertices belong to OTHER constraints far from this one, so the
        // cheap box test rejects them before the exact collinear/between
        // predicates run -- the difference between O(C*P) exact tests and a
        // handful per segment.
        let (pa, pb) = (approx[ia], approx[ib]);
        let seg_len = (0..3).map(|k| (pa[k] - pb[k]).powi(2)).sum::<f64>().sqrt();
        let pad = 1e-9 * seg_len.max(f64::MIN_POSITIVE);
        let slo: [f64; 3] = std::array::from_fn(|k| pa[k].min(pb[k]) - pad);
        let shi: [f64; 3] = std::array::from_fn(|k| pa[k].max(pb[k]) + pad);
        // All pool vertices strictly inside the segment split it into a chain.
        let mut on_seg: Vec<usize> = (0..pool.len())
            .filter(|&k| {
                k != ia
                    && k != ib
                    && (0..3).all(|d| approx[k][d] >= slo[d] && approx[k][d] <= shi[d])
                    && collinear(&pool[ia], &pool[ib], &pool[k]).expect("valid")
                    && strictly_between(&pool[ia], &pool[ib], &pool[k]).expect("valid")
            })
            .collect();
        on_seg.sort_by(|&p, &q| {
            match cmp_along(&pool[ia], &pool[ib], &pool[p], &pool[q]).expect("valid") {
                Sign::Negative => std::cmp::Ordering::Greater,
                Sign::Zero => std::cmp::Ordering::Equal,
                Sign::Positive => std::cmp::Ordering::Less,
            }
        });
        let mut prev = ia;
        for &k in on_seg.iter().chain(std::iter::once(&ib)) {
            chain_edges.push((prev, k));
            prev = k;
        }
    }
    for &(u, v) in &chain_edges {
        recover_edge(&mut tris, &pool, axis, u, v);
    }
    // Every chain edge must now be present (recovery of one constraint can
    // never flip away another: constraints are non-crossing after pre-split).
    for &(u, v) in &chain_edges {
        assert!(
            tris.iter().any(|&t| {
                has_directed_edge(t, u, v) || has_directed_edge(t, v, u)
            }),
            "constraint edge {u}-{v} missing after recovery"
        );
    }

    // Canonical (constrained Delaunay) pass: makes the triangulation a pure
    // function of the geometry, so coincident coplanar facets of different
    // inputs triangulate their overlap identically and can be matched
    // triangle-by-triangle downstream.
    let d_recover = t_recover.elapsed();
    let t_delaunay = std::time::Instant::now();
    let constrained: std::collections::HashSet<(usize, usize)> = chain_edges
        .iter()
        .map(|&(u, v)| (u.min(v), u.max(v)))
        .collect();
    delaunay_pass(&mut tris, &pool, axis, orientation, &constrained);
    if tri_trace {
        let total = t_pool.elapsed();
        if total.as_millis() > 50 {
            eprintln!(
                "tri facet: {} pts, {} constraints, {} tris in {:.1?} (pool {:.1?}, presplit {:.1?}, insert {:.1?}, recover {:.1?}, delaunay {:.1?})",
                pool.len(), constraints.len(), tris.len(), total,
                d_pool, d_presplit, d_insert, d_recover, t_delaunay.elapsed(),
            );
        }
    }

    FacetTriangulation {
        vertices: pool,
        triangles: tris,
        axis,
        orientation,
    }
}

/// Inserts pool vertex `k` into the triangulation (interior 1→3 split or
/// on-edge 2→4 split). Panics if the vertex lies outside the facet.
fn insert_vertex(
    tris: &mut Vec<[usize; 3]>,
    pool: &[Point3],
    orientation: Sign,
    axis: Axis,
    k: usize,
) {
    let p = &pool[k];
    let outside = orientation.flip();
    for ti in 0..tris.len() {
        let [i, j, l] = tris[ti];
        let si = orient2d(&pool[i], &pool[j], p, axis).expect("valid");
        if si == outside {
            continue;
        }
        let sj = orient2d(&pool[j], &pool[l], p, axis).expect("valid");
        if sj == outside {
            continue;
        }
        let sl = orient2d(&pool[l], &pool[i], p, axis).expect("valid");
        if sl == outside {
            continue;
        }
        match [si, sj, sl].iter().filter(|&&s| s == Sign::Zero).count() {
            0 => {
                // Strictly interior: 1→3.
                tris[ti] = [i, j, k];
                tris.push([j, l, k]);
                tris.push([l, i, k]);
                return;
            }
            1 => {
                // On the open edge whose test is zero.
                let (x, y) = if si == Sign::Zero {
                    (i, j)
                } else if sj == Sign::Zero {
                    (j, l)
                } else {
                    (l, i)
                };
                split_edge(tris, ti, x, y, k);
                return;
            }
            _ => unreachable!("vertex {k} coincides with a corner; dedup failed"),
        }
    }
    panic!("vertex {k} lies outside the facet");
}

/// Splits the directed edge x→y of triangle `ti` (and of its neighbor, if
/// any) at pool vertex `k`.
fn split_edge(tris: &mut Vec<[usize; 3]>, ti: usize, x: usize, y: usize, k: usize) {
    let neighbor = (0..tris.len()).find(|&tj| tj != ti && has_directed_edge(tris[tj], y, x));
    let c = opposite_vertex(tris[ti], x, y);
    tris[ti] = [x, k, c];
    tris.push([k, y, c]);
    if let Some(tj) = neighbor {
        let d = opposite_vertex(tris[tj], y, x);
        tris[tj] = [y, k, d];
        tris.push([k, x, d]);
    }
}

/// Flips non-constrained edges to the constrained Delaunay triangulation,
/// with a deterministic geometric tie-break for cocircular quads (prefer the
/// diagonal containing the lexicographically smallest of the four vertices).
/// The result is unique given the vertex set and constraints — the property
/// that makes coincident facets of different inputs match exactly.
fn delaunay_pass(
    tris: &mut [[usize; 3]],
    pool: &[Point3],
    axis: Axis,
    orientation: Sign,
    constrained: &std::collections::HashSet<(usize, usize)>,
) {
    let o2d = |a: usize, b: usize, c: usize| -> Sign {
        orient2d(&pool[a], &pool[b], &pool[c], axis).expect("valid")
    };
    let mut queue: std::collections::VecDeque<(usize, usize)> = tris
        .iter()
        .flat_map(|t| (0..3).map(move |e| (t[e], t[(e + 1) % 3])))
        .filter(|&(x, y)| x < y)
        .collect();
    let cap = 1000 + 64 * tris.len() * tris.len();
    let mut steps = 0usize;
    while let Some((x, y)) = queue.pop_front() {
        steps += 1;
        assert!(steps <= cap, "Delaunay pass did not converge");
        if constrained.contains(&(x.min(y), x.max(y))) {
            continue;
        }
        let left = (0..tris.len()).find(|&i| has_directed_edge(tris[i], x, y));
        let right = (0..tris.len()).find(|&i| has_directed_edge(tris[i], y, x));
        let (Some(li), Some(ri)) = (left, right) else {
            continue; // boundary edge or already flipped away
        };
        let c = opposite_vertex(tris[li], x, y);
        let d = opposite_vertex(tris[ri], y, x);
        // In-circle in the triangle's own handedness: (x, y, c) has the
        // facet orientation, so "d inside" carries that sign.
        let s = incircle2d(&pool[x], &pool[y], &pool[c], &pool[d], axis).expect("valid");
        let flip = if s == orientation {
            true
        } else if s == Sign::Zero {
            // Cocircular: prefer the diagonal owning the lex-smallest vertex.
            let min_of = |a: usize, b: usize| -> usize {
                match lex_cmp(&pool[a], &pool[b]).expect("valid") {
                    std::cmp::Ordering::Greater => b,
                    _ => a,
                }
            };
            let overall = min_of(min_of(x, y), min_of(c, d));
            overall == c || overall == d
        } else {
            false
        };
        if !flip {
            continue;
        }
        // Quad must be strictly convex (guards collinear/degenerate ties).
        if o2d(c, d, x).combine(o2d(c, d, y)) != Sign::Negative {
            continue;
        }
        tris[li] = [c, x, d];
        tris[ri] = [d, y, c];
        for &(p, q) in &[(c, x), (x, d), (d, y), (y, c)] {
            queue.push_back((p.min(q), p.max(q)));
        }
    }
}

/// Restores the edge {u, v} (whose open interior contains no vertices) by
/// flipping edges that cross it — Sloan-style FIFO processing.
///
/// All edges crossing the segment are collected once; an edge whose
/// surrounding quad is not strictly convex is deferred to the back of the
/// queue (some other flip will unlock it), and a flip's new diagonal is
/// re-enqueued only if it still crosses the segment. Always flipping the
/// first flippable edge found by rescanning would instead oscillate: a valid
/// flip's inverse is immediately valid again.
fn recover_edge(tris: &mut [[usize; 3]], pool: &[Point3], axis: Axis, u: usize, v: usize) {
    let o2d = |a: usize, b: usize, c: usize| -> Sign {
        orient2d(&pool[a], &pool[b], &pool[c], axis).expect("valid")
    };
    let crosses = |x: usize, y: usize| -> bool {
        o2d(u, v, x).combine(o2d(u, v, y)) == Sign::Negative
            && o2d(x, y, u).combine(o2d(x, y, v)) == Sign::Negative
    };

    let mut queue: std::collections::VecDeque<(usize, usize)> = std::collections::VecDeque::new();
    for t in tris.iter() {
        for e in 0..3 {
            let (x, y) = (t[e], t[(e + 1) % 3]);
            if x < y && crosses(x, y) {
                queue.push_back((x, y));
            }
        }
    }

    // Each deferred edge gets unlocked by some other flip; if a full round
    // of deferrals makes no progress, the input violated the contract.
    let mut deferred_in_a_row = 0;
    while let Some((x, y)) = queue.pop_front() {
        let left = (0..tris.len()).find(|&i| has_directed_edge(tris[i], x, y));
        let right = (0..tris.len()).find(|&i| has_directed_edge(tris[i], y, x));
        let (Some(li), Some(ri)) = (left, right) else {
            // The edge was flipped away while queued (as a new diagonal that
            // itself got flipped); nothing to do.
            continue;
        };
        let c = opposite_vertex(tris[li], x, y);
        let d = opposite_vertex(tris[ri], y, x);
        if o2d(c, d, x).combine(o2d(c, d, y)) != Sign::Negative {
            // Quad not strictly convex: defer.
            deferred_in_a_row += 1;
            assert!(
                deferred_in_a_row <= queue.len(),
                "edge recovery stalled for constraint {u}-{v}"
            );
            queue.push_back((x, y));
            continue;
        }
        deferred_in_a_row = 0;
        tris[li] = [c, x, d];
        tris[ri] = [d, y, c];
        if crosses(c.min(d), c.max(d)) {
            queue.push_back((c.min(d), c.max(d)));
        }
    }

    assert!(
        tris.iter()
            .any(|&t| has_directed_edge(t, u, v) || has_directed_edge(t, v, u)),
        "constraint edge {u}-{v} absent after crossing queue drained"
    );
}

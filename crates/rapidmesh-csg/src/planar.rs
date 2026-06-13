//! Conformal arrangement of planar facets.
//!
//! The triangle-soup arrangement ([`crate::arrange`]) tessellates every flat
//! face up front, so a curve piercing a flat face lands on intersection points
//! that sit a hair off the face's interior tessellation vertices -- the seam
//! micro-features (see docs/conformal-tessellation-plan.md). This module fixes
//! that at the source: a flat face is carried as a [`PlanarFacet`] (boundary
//! loops plus a helper triangulation that tiles it), intersections are computed
//! against the helper triangles, but the resulting sub-segments are MERGED
//! along their common line before they become constraints. The helper-internal
//! crossings (the near-twins) coincide on that line and merge away, so a face's
//! constraints meet exactly at the piercing surface's own vertices. The face is
//! then triangulated ONCE, from its boundary plus the merged constraints, so it
//! carries no artificial interior structure.
//!
//! Curved faces (barrels, spheres, tori) are passed as single-triangle planar
//! facets, so the same code path handles the whole scene; for them the merge is
//! a no-op (one helper triangle, one sub-segment per pair) and the result is
//! identical to the triangle-soup arrangement.

use crate::arrange::{adjacency_skip, build_bvh, clip_coplanar_edge, self_pairs, Aabb, Arrangement};
use crate::constraint::{Constraint, ConstraintLine};
use crate::facet::PlanarFacet;
use crate::tri::Tri;
use crate::tri_tri::{tri_tri_intersection, TriTriIsect};
use crate::triangulate::triangulate_seeded;
use rapidmesh_exact::{cmp_along, collinear, Point3, Sign};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

/// One input facet for the conformal arrangement: a planar boundary polygon
/// and a helper triangulation that exactly tiles it (used for intersection
/// finding only; its interior edges never become constraints).
#[derive(Debug, Clone)]
pub struct PlanarInput {
    /// Boundary loops (outer plus holes) of the facet.
    pub boundary: PlanarFacet,
    /// A valid triangulation of the facet (tiles it without gaps or overlaps).
    pub helpers: Vec<Tri>,
}

impl PlanarInput {
    /// A facet that is a single triangle (a curved-surface facet): boundary is
    /// the triangle, helper triangulation is itself.
    pub fn tri(t: Tri) -> PlanarInput {
        PlanarInput {
            boundary: PlanarFacet::new(t.v.to_vec()),
            helpers: vec![t],
        }
    }
}

/// A sub-segment cut into a facet by one member triangle of another facet,
/// carrying that member's plane for `PlaneCut` provenance.
#[derive(Clone)]
struct CutSeg {
    a: Point3,
    b: Point3,
    plane: [[f64; 3]; 3],
}

/// Conformal arrangement of planar facets. The result is indexed per facet
/// (NOT per triangle): `facets[k]` is the triangulation of `input[k]`.
pub fn arrange_facets(input: &[PlanarInput]) -> Arrangement {
    let n = input.len();

    // Flatten to member triangles tagged with their owning facet, for the
    // broad-phase and pairwise intersection.
    let mut members: Vec<Tri> = Vec::new();
    let mut member_facet: Vec<usize> = Vec::new();
    for (fi, f) in input.iter().enumerate() {
        for t in &f.helpers {
            members.push(*t);
            member_facet.push(fi);
        }
    }

    let boxes: Vec<Aabb> = members.iter().map(Aabb::of_tri).collect();
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    if !members.is_empty() {
        let mut idx: Vec<usize> = (0..members.len()).collect();
        let bvh = build_bvh(&mut idx, &boxes);
        self_pairs(&bvh, &boxes, &mut pairs);
    }

    // Per facet: sub-segments grouped by the cutting facet, plus touch points;
    // coplanar facet pairs are recorded for boundary-level handling.
    let mut cut: Vec<HashMap<usize, Vec<CutSeg>>> = vec![HashMap::new(); n];
    let mut points: Vec<Vec<Point3>> = vec![Vec::new(); n];
    let mut coplanar: HashSet<(usize, usize)> = HashSet::new();

    for (mi, mj) in pairs {
        let (fi, fj) = (member_facet[mi], member_facet[mj]);
        if fi == fj {
            // Two helper triangles of the SAME facet: coplanar, interior edges
            // are not features.
            continue;
        }
        if adjacency_skip(&members[mi], &members[mj]) {
            continue;
        }
        match tri_tri_intersection(&members[mi], &members[mj]) {
            TriTriIsect::Disjoint => {}
            TriTriIsect::Touching(p) => {
                points[fi].push(p.clone());
                points[fj].push(p);
            }
            TriTriIsect::Segment(a, b) => {
                cut[fi].entry(fj).or_default().push(CutSeg {
                    a: a.clone(),
                    b: b.clone(),
                    plane: members[mj].v,
                });
                cut[fj].entry(fi).or_default().push(CutSeg {
                    a,
                    b,
                    plane: members[mi].v,
                });
            }
            TriTriIsect::Coplanar => {
                coplanar.insert((fi.min(fj), fi.max(fj)));
            }
        }
    }

    // Coplanar facet pairs: clip each facet's BOUNDARY edges (not its helper
    // interior) against the other facet's helper triangles, merging the clipped
    // sub-segments along each edge. This is the polygon analog of the
    // triangle-soup coplanar clip and keeps the constraint set boundary-only.
    let mut cop: Vec<Vec<Constraint>> = vec![Vec::new(); n];
    for &(fi, fj) in &coplanar {
        for (target, source) in [(fi, fj), (fj, fi)] {
            for (u, v) in boundary_edges(&input[source].boundary) {
                let (segs, pts) = clip_edge_to_facet(&input[target].helpers, u, v);
                for (a, b) in segs {
                    cop[target].push(Constraint {
                        a,
                        b,
                        line: ConstraintLine::Edge(u, v),
                    });
                }
                points[target].extend(pts);
            }
        }
    }

    // Per facet: merge cut sub-segments into constraints, then triangulate the
    // facet once from its boundary plus all constraints.
    let mut out_facets = Vec::with_capacity(n);
    let mut out_constraints = Vec::with_capacity(n);
    for f in 0..n {
        let mut constraints: Vec<Constraint> = std::mem::take(&mut cop[f]);
        for segs in cut[f].values() {
            for group in collinear_groups(segs) {
                let raw: Vec<(Point3, Point3)> =
                    group.iter().map(|s| (s.a.clone(), s.b.clone())).collect();
                let merged = merge_on_line(&group[0].a, &group[0].b, &raw);
                for (a, b) in merged {
                    constraints.push(Constraint {
                        a,
                        b,
                        line: ConstraintLine::PlaneCut(group[0].plane),
                    });
                }
            }
        }

        let facet = &input[f].boundary;
        let (axis, orientation) = facet.projection_axis();
        let plane3 = noncollinear_triple(&facet.outer);
        let (seed_pool, seed_tris) = build_seed(facet, &input[f].helpers);
        let ft = triangulate_seeded(
            axis,
            orientation,
            plane3,
            seed_pool,
            seed_tris,
            &points[f],
            &constraints,
        );
        out_facets.push(ft);
        out_constraints.push(constraints);
    }

    Arrangement {
        facets: out_facets,
        constraints: out_constraints,
    }
}

/// Boundary edges of a facet (outer loop then each hole loop), as explicit
/// endpoint pairs.
fn boundary_edges(facet: &PlanarFacet) -> Vec<([f64; 3], [f64; 3])> {
    let mut out = Vec::new();
    for loop3 in std::iter::once(&facet.outer).chain(facet.holes.iter()) {
        let m = loop3.len();
        for i in 0..m {
            out.push((loop3[i], loop3[(i + 1) % m]));
        }
    }
    out
}

/// Clips the explicit coplanar edge `(u, v)` to a facet given by its helper
/// triangles, returning the merged in-facet sub-segments and any single-point
/// touches. All sub-segments lie on the line through `u`, `v`, so they merge
/// directly along it.
fn clip_edge_to_facet(
    helpers: &[Tri],
    u: [f64; 3],
    v: [f64; 3],
) -> (Vec<(Point3, Point3)>, Vec<Point3>) {
    let mut raw: Vec<(Point3, Point3)> = Vec::new();
    let mut pts: Vec<Point3> = Vec::new();
    for h in helpers {
        if let Some((lo, hi)) = clip_coplanar_edge(h, u, v) {
            if lo.coincides(&hi) {
                pts.push(lo);
            } else {
                raw.push((lo, hi));
            }
        }
    }
    let merged = merge_on_line(&Point3::Explicit(u), &Point3::Explicit(v), &raw);
    (merged, pts)
}

/// Partitions sub-segments into maximal collinear groups (all four endpoints of
/// any two members of a group lie on one line). A flat cutting facet yields one
/// group (all members share the cut line); a curved one yields a group per
/// member plane.
fn collinear_groups(segs: &[CutSeg]) -> Vec<Vec<CutSeg>> {
    let mut groups: Vec<Vec<CutSeg>> = Vec::new();
    'next: for s in segs {
        for g in groups.iter_mut() {
            let r = &g[0];
            if collinear(&r.a, &r.b, &s.a) == Some(true)
                && collinear(&r.a, &r.b, &s.b) == Some(true)
            {
                g.push(s.clone());
                continue 'next;
            }
        }
        groups.push(vec![s.clone()]);
    }
    groups
}

/// Merges segments that lie on the common line through `refa`, `refb` into
/// maximal intervals (touching or overlapping segments coalesce; gaps stay
/// separate). Endpoints are ordered exactly along the line via [`cmp_along`].
fn merge_on_line(
    refa: &Point3,
    refb: &Point3,
    segs: &[(Point3, Point3)],
) -> Vec<(Point3, Point3)> {
    if segs.is_empty() {
        return Vec::new();
    }
    // `x` strictly precedes `y` along refa->refb.
    let precedes = |x: &Point3, y: &Point3| cmp_along(refa, refb, x, y) == Some(Sign::Positive);
    // Orient every segment lo -> hi.
    let mut iv: Vec<(Point3, Point3)> = segs
        .iter()
        .map(|(a, b)| {
            if precedes(a, b) {
                (a.clone(), b.clone())
            } else {
                (b.clone(), a.clone())
            }
        })
        .collect();
    iv.sort_by(|p, q| match cmp_along(refa, refb, &p.0, &q.0) {
        Some(Sign::Positive) => Ordering::Less,
        Some(Sign::Negative) => Ordering::Greater,
        _ => Ordering::Equal,
    });
    let mut out: Vec<(Point3, Point3)> = Vec::new();
    let (mut lo, mut hi) = iv[0].clone();
    for (l, h) in iv.into_iter().skip(1) {
        if precedes(&hi, &l) {
            // Strict gap: emit the current interval and start a new one.
            out.push((lo.clone(), hi.clone()));
            lo = l;
            hi = h;
        } else {
            // Overlap or touch: extend hi to the farther endpoint.
            if precedes(&hi, &h) {
                hi = h;
            }
        }
    }
    out.push((lo, hi));
    out
}

/// Three non-collinear vertices of an outer loop, defining its plane (for
/// `PlaneCut` TPI provenance). Panics on a fully collinear loop.
fn noncollinear_triple(outer: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let n = outer.len();
    for i in 0..n {
        let (a, b, c) = (outer[i], outer[(i + 1) % n], outer[(i + 2) % n]);
        if collinear(
            &Point3::Explicit(a),
            &Point3::Explicit(b),
            &Point3::Explicit(c),
        ) == Some(false)
        {
            return [a, b, c];
        }
    }
    panic!("planar facet outer loop is fully collinear");
}

/// Builds a seed triangulation (vertex pool plus triangle indices) for the
/// final per-facet triangulation. A convex, hole-free facet is fanned from its
/// boundary (no interior vertex); otherwise the helper triangulation is used
/// (ear-clipped, boundary-only vertices, holes respected). Either way the seed
/// boundary loops are preserved by recovery and the interior is reshaped by the
/// Delaunay pass, so no artificial structure survives.
fn build_seed(facet: &PlanarFacet, helpers: &[Tri]) -> (Vec<Point3>, Vec<[usize; 3]>) {
    let tris: Vec<Tri> = if facet.is_convex() {
        facet.fan_tris()
    } else {
        helpers.to_vec()
    };
    let mut pool: Vec<Point3> = Vec::new();
    let index_of = |p: [f64; 3], pool: &mut Vec<Point3>| -> usize {
        if let Some(i) = pool.iter().position(|q| q.as_explicit() == Some(p)) {
            i
        } else {
            pool.push(Point3::Explicit(p));
            pool.len() - 1
        }
    };
    let seed_tris: Vec<[usize; 3]> = tris
        .iter()
        .map(|t| {
            [
                index_of(t.v[0], &mut pool),
                index_of(t.v[1], &mut pool),
                index_of(t.v[2], &mut pool),
            ]
        })
        .collect();
    (pool, seed_tris)
}

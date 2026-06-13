//! Arrangement of a triangle soup: every facet is subdivided by its exact
//! intersections with all other facets.
//!
//! Pair candidates come from a BVH over the facet bounding boxes. Each
//! intersecting pair contributes constraints to both facets (with line
//! provenance, see [`crate::constraint`]); coplanar pairs contribute the
//! other facet's edges clipped to this facet. Each facet is then
//! independently retriangulated — exact constructions and exact coincidence
//! guarantee that shared intersection vertices match across facets, which is
//! what downstream inside/outside classification relies on.

use crate::constraint::{Constraint, ConstraintLine};
use crate::tri::Tri;
use crate::tri_tri::{tri_tri_intersection, TriTriIsect};
use crate::triangulate::{triangulate_facet, FacetTriangulation};
use rapidmesh_exact::{cmp_along, collinear, within_closed, Point3, Sign};

// ------------------------------------------------------------------- BVH

#[derive(Debug, Clone, Copy)]
struct Aabb {
    min: [f64; 3],
    max: [f64; 3],
}

impl Aabb {
    fn of_tri(t: &Tri) -> Aabb {
        let mut min = t.v[0];
        let mut max = t.v[0];
        for v in &t.v[1..] {
            for k in 0..3 {
                min[k] = min[k].min(v[k]);
                max[k] = max[k].max(v[k]);
            }
        }
        Aabb { min, max }
    }

    fn union(&self, o: &Aabb) -> Aabb {
        Aabb {
            min: std::array::from_fn(|k| self.min[k].min(o.min[k])),
            max: std::array::from_fn(|k| self.max[k].max(o.max[k])),
        }
    }

    /// Closed-box overlap: touching boxes count (touching facets intersect).
    fn overlaps(&self, o: &Aabb) -> bool {
        (0..3).all(|k| self.min[k] <= o.max[k] && o.min[k] <= self.max[k])
    }

    fn centroid(&self, k: usize) -> f64 {
        0.5 * (self.min[k] + self.max[k])
    }
}

enum Bvh {
    Leaf { aabb: Aabb, items: Vec<usize> },
    Inner { aabb: Aabb, left: Box<Bvh>, right: Box<Bvh> },
}

impl Bvh {
    fn aabb(&self) -> &Aabb {
        match self {
            Bvh::Leaf { aabb, .. } | Bvh::Inner { aabb, .. } => aabb,
        }
    }
}

const BVH_LEAF_SIZE: usize = 4;

fn build_bvh(items: &mut [usize], boxes: &[Aabb]) -> Bvh {
    let aabb = items
        .iter()
        .map(|&i| boxes[i])
        .reduce(|a, b| a.union(&b))
        .expect("non-empty");
    if items.len() <= BVH_LEAF_SIZE {
        return Bvh::Leaf {
            aabb,
            items: items.to_vec(),
        };
    }
    // Split along the axis with the largest centroid spread.
    let axis = (0..3)
        .max_by(|&a, &b| {
            let spread = |k: usize| {
                let lo = items.iter().map(|&i| boxes[i].centroid(k)).fold(f64::MAX, f64::min);
                let hi = items.iter().map(|&i| boxes[i].centroid(k)).fold(f64::MIN, f64::max);
                hi - lo
            };
            spread(a).partial_cmp(&spread(b)).expect("finite")
        })
        .expect("three axes");
    items.sort_unstable_by(|&a, &b| {
        boxes[a]
            .centroid(axis)
            .partial_cmp(&boxes[b].centroid(axis))
            .expect("finite")
    });
    let mid = items.len() / 2;
    let (l, r) = items.split_at_mut(mid);
    Bvh::Inner {
        aabb,
        left: Box::new(build_bvh(l, boxes)),
        right: Box::new(build_bvh(r, boxes)),
    }
}

fn self_pairs(n: &Bvh, boxes: &[Aabb], out: &mut Vec<(usize, usize)>) {
    match n {
        Bvh::Leaf { items, .. } => {
            for (a, &i) in items.iter().enumerate() {
                for &j in &items[a + 1..] {
                    if boxes[i].overlaps(&boxes[j]) {
                        out.push((i.min(j), i.max(j)));
                    }
                }
            }
        }
        Bvh::Inner { left, right, .. } => {
            self_pairs(left, boxes, out);
            self_pairs(right, boxes, out);
            cross_pairs(left, right, boxes, out);
        }
    }
}

fn cross_pairs(a: &Bvh, b: &Bvh, boxes: &[Aabb], out: &mut Vec<(usize, usize)>) {
    if !a.aabb().overlaps(b.aabb()) {
        return;
    }
    match (a, b) {
        (Bvh::Leaf { items: ia, .. }, Bvh::Leaf { items: ib, .. }) => {
            for &i in ia {
                for &j in ib {
                    if boxes[i].overlaps(&boxes[j]) {
                        out.push((i.min(j), i.max(j)));
                    }
                }
            }
        }
        (Bvh::Inner { left, right, .. }, _) => {
            cross_pairs(left, b, boxes, out);
            cross_pairs(right, b, boxes, out);
        }
        (_, Bvh::Inner { left, right, .. }) => {
            cross_pairs(a, left, boxes, out);
            cross_pairs(a, right, boxes, out);
        }
    }
}

// -------------------------------------------------------- coplanar clip

/// Clips the explicit edge (u, v) of a triangle coplanar with `facet` to the
/// (closed, convex) facet. Returns the clipped sub-segment endpoints ordered
/// along u→v; they coincide for a single-point touch. `None` if the edge
/// misses the facet.
fn clip_coplanar_edge(facet: &Tri, u: [f64; 3], v: [f64; 3]) -> Option<(Point3, Point3)> {
    let (axis, orientation) = facet.projection_axis();
    let pu = Point3::Explicit(u);
    let pv = Point3::Explicit(v);
    let mut cands: Vec<Point3> = Vec::new();
    for p in [&pu, &pv] {
        if facet.contains_coplanar(p, axis, orientation) {
            cands.push(p.clone());
        }
    }
    for e in 0..3 {
        let (a, b) = (facet.v[e], facet.v[(e + 1) % 3]);
        let (pa, pb) = (Point3::Explicit(a), Point3::Explicit(b));
        // Proper line crossing with this facet edge, clamped to both
        // segments.
        if let Some(x) = Point3::lli_coplanar(u, v, a, b) {
            if within_closed(&pu, &pv, &x).expect("valid")
                && within_closed(&pa, &pb, &x).expect("valid")
            {
                cands.push(x);
            }
        }
        // Facet corner lying on the edge (covers collinear-overlap cases).
        if collinear(&pu, &pv, &pa).expect("valid")
            && within_closed(&pu, &pv, &pa).expect("valid")
        {
            cands.push(pa);
        }
    }
    // The facet is convex, so the clip is the extreme candidates along u→v.
    let mut iter = cands.into_iter();
    let first = iter.next()?;
    let (mut lo, mut hi) = (first.clone(), first);
    for c in iter {
        if cmp_along(&pu, &pv, &c, &lo).expect("valid") == Sign::Positive {
            lo = c.clone();
        }
        if cmp_along(&pu, &pv, &hi, &c).expect("valid") == Sign::Positive {
            hi = c;
        }
    }
    Some((lo, hi))
}

// ----------------------------------------------------------- arrangement

/// The arrangement of a triangle soup.
#[derive(Debug)]
pub struct Arrangement {
    /// Per input facet: its exact constrained triangulation.
    pub facets: Vec<FacetTriangulation>,
    /// Per input facet: the constraints that subdivided it (for downstream
    /// classification and inspection).
    pub constraints: Vec<Vec<Constraint>>,
}

/// Computes the arrangement of `tris`: each facet triangulated so that all
/// pairwise intersections appear as triangulation edges/vertices, exactly.
pub fn arrange(tris: &[Tri]) -> Arrangement {
    let boxes: Vec<Aabb> = tris.iter().map(Aabb::of_tri).collect();
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    if !tris.is_empty() {
        let mut idx: Vec<usize> = (0..tris.len()).collect();
        let bvh = build_bvh(&mut idx, &boxes);
        self_pairs(&bvh, &boxes, &mut pairs);
    }

    let trace = std::env::var_os("RAPIDMESH_TRACE").is_some();
    let t_pairs = std::time::Instant::now();
    let n_pairs = pairs.len();
    let mut points: Vec<Vec<Point3>> = vec![Vec::new(); tris.len()];
    let mut constraints: Vec<Vec<Constraint>> = vec![Vec::new(); tris.len()];
    let mut skipped = [0usize; 4];
    for (i, j) in pairs {
        // Fast paths for mesh-adjacent pairs, the dominant candidate kind in
        // clean closed surfaces (shared vertices are bit-identical there).
        // They contribute nothing to the arrangement: a shared edge of two
        // non-coplanar triangles is exactly their intersection and already a
        // boundary edge of both facets; a shared vertex with the remaining
        // vertices strictly on one side of the other plane is exactly a
        // touching point and already a facet corner. The signs come from the
        // adaptive exact predicate, so the skips are exact, and every other
        // configuration (coplanar, piercing) falls through to the full
        // intersection machinery.
        {
            let side = |t: &Tri, q: [f64; 3]| -> Sign {
                Sign::of_f64(geometry_predicates::orient3d(t.v[0], t.v[1], t.v[2], q))
            };
            let mut shared_j = [false; 3];
            let mut n_shared = 0;
            for (b, flag) in shared_j.iter_mut().enumerate() {
                if tris[i].v.iter().any(|&a| a == tris[j].v[b]) {
                    *flag = true;
                    n_shared += 1;
                }
            }
            let signs: [Sign; 3] = std::array::from_fn(|b| side(&tris[i], tris[j].v[b]));
            if signs == [Sign::Zero; 3] {
                // Exactly coplanar: tessellated flat regions produce many
                // such pairs whose intersection is at most a shared edge or
                // vertex. An exact 2D separating line among the six edges
                // proves it: one triangle's vertices weakly on one side, the
                // other's weakly on the other, and every on-line vertex is
                // either an endpoint of the line itself or shared between
                // the triangles (a non-shared on-line vertex could be a
                // T-configuration touching point, which the full machinery
                // must handle).
                let is_shared = |q: [f64; 3]| -> bool {
                    tris[i].v.contains(&q) && tris[j].v.contains(&q)
                };
                let (axis, _) = tris[i].projection_axis();
                let separated = (0..6).any(|e| {
                    let (t_edge, _t_other) = if e < 3 { (&tris[i], &tris[j]) } else { (&tris[j], &tris[i]) };
                    let (p, q) = (t_edge.v[e % 3], t_edge.v[(e + 1) % 3]);
                    let line_sign = |r: [f64; 3]| -> Option<Sign> {
                        rapidmesh_exact::orient2d(
                            &Point3::Explicit(p),
                            &Point3::Explicit(q),
                            &Point3::Explicit(r),
                            axis,
                        )
                        
                    };
                    let mut side_i = Sign::Zero;
                    let mut side_j = Sign::Zero;
                    for r in tris[i].v {
                        match line_sign(r) {
                            Some(Sign::Zero) => {
                                if !(r == p || r == q || is_shared(r)) {
                                    return false;
                                }
                            }
                            Some(sg) => {
                                if side_i != Sign::Zero && side_i != sg {
                                    return false;
                                }
                                side_i = sg;
                            }
                            None => return false,
                        }
                    }
                    for r in tris[j].v {
                        match line_sign(r) {
                            Some(Sign::Zero) => {
                                if !(r == p || r == q || is_shared(r)) {
                                    return false;
                                }
                            }
                            Some(sg) => {
                                if side_j != Sign::Zero && side_j != sg {
                                    return false;
                                }
                                side_j = sg;
                            }
                            None => return false,
                        }
                    }
                    side_i != Sign::Zero && side_j != Sign::Zero && side_i != side_j
                });
                if separated {
                    skipped[3] += 1;
                    continue;
                }
            } else if n_shared == 2 {
                let opp = (0..3).find(|&b| !shared_j[b]).expect("one non-shared");
                if signs[opp] != Sign::Zero {
                    skipped[2] += 1;
                    continue;
                }
            } else if n_shared == 1 {
                let mut others = (0..3).filter(|&b| !shared_j[b]);
                let (b1, b2) = (others.next().expect("two"), others.next().expect("two"));
                let (s1, s2) = (signs[b1], signs[b2]);
                if s1 != Sign::Zero && s1 == s2 {
                    skipped[1] += 1;
                    continue;
                }
            } else if n_shared == 0
                && (signs.iter().all(|&x| x == Sign::Positive)
                    || signs.iter().all(|&x| x == Sign::Negative))
            {
                // Strictly one side of the plane: disjoint.
                skipped[0] += 1;
                continue;
            }
        }
        match tri_tri_intersection(&tris[i], &tris[j]) {
            TriTriIsect::Disjoint => {}
            TriTriIsect::Touching(p) => {
                points[i].push(p.clone());
                points[j].push(p);
            }
            TriTriIsect::Segment(a, b) => {
                constraints[i].push(Constraint {
                    a: a.clone(),
                    b: b.clone(),
                    line: ConstraintLine::PlaneCut(tris[j].v),
                });
                constraints[j].push(Constraint {
                    a,
                    b,
                    line: ConstraintLine::PlaneCut(tris[i].v),
                });
            }
            TriTriIsect::Coplanar => {
                for (fi, fj) in [(i, j), (j, i)] {
                    let facet = &tris[fi];
                    let other = &tris[fj];
                    for e in 0..3 {
                        let (u, v) = (other.v[e], other.v[(e + 1) % 3]);
                        if let Some((lo, hi)) = clip_coplanar_edge(facet, u, v) {
                            if lo.coincides(&hi) {
                                points[fi].push(lo);
                            } else {
                                constraints[fi].push(Constraint {
                                    a: lo,
                                    b: hi,
                                    line: ConstraintLine::Edge(u, v),
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    if trace {
        eprintln!(
            "arrange: {n_pairs} pairs in {:.1?} (skipped edge {} vert {} oneside {})",
            t_pairs.elapsed(),
            skipped[2],
            skipped[1],
            skipped[0] + skipped[3]
        );
    }
    let t_tri = std::time::Instant::now();
    // Each facet retriangulates independently from its own intersection
    // points and constraints (read-only), and this dominates assembly on
    // boolean-heavy scenes; run it in parallel. collect() into an indexed
    // Vec keeps the result order identical to the serial map.
    use rayon::prelude::*;
    let facets = tris
        .par_iter()
        .enumerate()
        .map(|(i, t)| triangulate_facet(t, &points[i], &constraints[i]))
        .collect();
    if trace {
        eprintln!("arrange: triangulate {:.1?}", t_tri.elapsed());
    }
    Arrangement {
        facets,
        constraints,
    }
}

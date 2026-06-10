//! Exact point-vs-solid classification via parity ray casting.
//!
//! The representative point is implicit (a sub-triangle barycenter); the ray
//! is the segment from it to an explicit target far outside the solid.
//! Degenerate configurations (segment through an edge/vertex, target on a
//! plane) are detected exactly and resolved by retrying with a different
//! target — the set of bad targets is measure-zero, so a deterministic
//! pseudo-random target sequence escapes after a try or two.

use crate::tri::Tri;
use rapidmesh_exact::{orient2d, orient3d, Point3, Sign};

/// Where a point lies relative to a closed solid surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Strictly inside the solid.
    Inside,
    /// Strictly outside the solid.
    Outside,
    /// Exactly on the surface; `same_normal` compares the queried facet's
    /// orientation with the coincident solid facet's orientation.
    Boundary {
        /// True if the coincident facets face the same way.
        same_normal: bool,
    },
}

/// Does the segment (p, q) cross the interior of `tri`?
///
/// `None` means a degenerate configuration that requires a different target
/// `q` (segment through an edge/vertex of the triangle, or `q` on its plane).
/// Precondition: `p` does not lie on the solid's surface (checked by
/// [`on_solid_boundary`] beforehand), so `p` on this triangle's plane means
/// the segment only touches the plane at `p`, outside the triangle — no
/// crossing.
fn segment_crosses_triangle(p: &Point3, q: &Point3, tri: &Tri) -> Option<bool> {
    let (a, b, c) = (tri.point(0), tri.point(1), tri.point(2));
    let sp = orient3d(&a, &b, &c, p).expect("valid");
    if sp == Sign::Zero {
        return Some(false);
    }
    let sq = orient3d(&a, &b, &c, q).expect("valid");
    if sq == Sign::Zero {
        return None;
    }
    if sp == sq {
        return Some(false);
    }
    let s1 = orient3d(p, q, &a, &b).expect("valid");
    let s2 = orient3d(p, q, &b, &c).expect("valid");
    let s3 = orient3d(p, q, &c, &a).expect("valid");
    if s1 == Sign::Zero || s2 == Sign::Zero || s3 == Sign::Zero {
        return None;
    }
    Some(s1 == s2 && s2 == s3)
}

/// The solid facet whose (closed) area contains `p`, if `p` lies exactly on
/// the solid's surface.
pub fn on_solid_boundary(p: &Point3, solid: &[Tri]) -> Option<usize> {
    solid.iter().position(|t| {
        orient3d(&t.point(0), &t.point(1), &t.point(2), p) == Some(Sign::Zero) && {
            let (axis, orientation) = t.projection_axis();
            t.contains_coplanar(p, axis, orientation)
        }
    })
}

/// True if two coplanar triangles face the same way.
pub fn coplanar_same_normal(t1: &Tri, t2: &Tri) -> bool {
    let (axis, s1) = t1.projection_axis();
    let s2 = orient2d(&t2.point(0), &t2.point(1), &t2.point(2), axis)
        .expect("explicit points are always valid");
    debug_assert_ne!(s2, Sign::Zero, "triangles must be coplanar and non-degenerate");
    s1 == s2
}

/// Parity test: is `p` (not on the surface) inside the closed solid?
///
/// `bbox` is the solid's (or scene's) bounding box; targets are placed
/// outside it.
pub fn point_inside_solid(p: &Point3, solid: &[Tri], bbox: ([f64; 3], [f64; 3])) -> bool {
    let (lo, hi) = bbox;
    let diag = (0..3).map(|k| hi[k] - lo[k]).fold(1.0_f64, f64::max);
    'targets: for k in 0..32u64 {
        // Deterministic pseudo-random target outside the bounding box.
        let mut s = (k + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut coord = |d: usize| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let frac = (s >> 11) as f64 / (1u64 << 53) as f64;
            hi[d] + diag * (0.5 + frac)
        };
        let q = Point3::explicit(coord(0), coord(1), coord(2));
        let mut crossings = 0usize;
        for t in solid {
            match segment_crosses_triangle(p, &q, t) {
                None => continue 'targets,
                Some(true) => crossings += 1,
                Some(false) => {}
            }
        }
        return crossings % 2 == 1;
    }
    panic!("no generic ray target found in 32 attempts");
}

/// Full placement of `p` (interior representative of a facet of `own`)
/// relative to the closed solid `other`.
pub fn classify(p: &Point3, own_facet: &Tri, other: &[Tri], bbox: ([f64; 3], [f64; 3])) -> Placement {
    match on_solid_boundary(p, other) {
        Some(j) => Placement::Boundary {
            same_normal: coplanar_same_normal(own_facet, &other[j]),
        },
        None => {
            if point_inside_solid(p, other, bbox) {
                Placement::Inside
            } else {
                Placement::Outside
            }
        }
    }
}

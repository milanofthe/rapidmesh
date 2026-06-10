//! Staged-exact orientation predicates over explicit and implicit points.

use crate::expansion::Expansion;
use crate::geom::{det3, det4};
use crate::interval::Interval;
use crate::point::Point3;
use crate::{Axis, Sign};

/// Exact 3D orientation of four points, any of which may be implicit.
///
/// Sign convention: equals the sign of det [[a-d], [b-d], [c-d]] (rows), the
/// same convention as Shewchuk's `orient3d` — positive when `d` lies below the
/// plane through `a`, `b`, `c` oriented counterclockwise as seen from above
/// the plane.
///
/// Returns `None` if any implicit point is invalid (its defining primitives
/// do not intersect in a single point, exact w == 0).
///
/// Evaluation is staged: fast adaptive path for all-explicit inputs
/// (`geometry-predicates`), conservative interval filter for implicit inputs,
/// exact expansion arithmetic as the final word.
pub fn orient3d(a: &Point3, b: &Point3, c: &Point3, d: &Point3) -> Option<Sign> {
    // Fast adaptive path: all points explicit.
    if let (Some(pa), Some(pb), Some(pc), Some(pd)) = (
        a.as_explicit(),
        b.as_explicit(),
        c.as_explicit(),
        d.as_explicit(),
    ) {
        return Some(Sign::of_f64(geometry_predicates::orient3d(pa, pb, pc, pd)));
    }

    let pts = [a, b, c, d];

    // The 4x4 homogeneous determinant relates to the affine orientation by
    // det4 = (prod of w_i) * det3[[a-d],[b-d],[c-d]], so the orientation sign
    // is the det4 sign combined with each w sign.

    // Interval filter.
    'filter: {
        let homs: [[Interval; 4]; 4] = std::array::from_fn(|i| pts[i].hom::<Interval>());
        let Some(mut sign) = det4(&homs).sign() else {
            break 'filter;
        };
        for h in &homs {
            match h[3].sign() {
                // Strictly signed w: fold into the result.
                Some(Sign::Positive) => {}
                Some(Sign::Negative) => sign = sign.flip(),
                // w == 0 exactly or uncertain: let the exact stage decide
                // validity.
                _ => break 'filter,
            }
        }
        return Some(sign);
    }

    // Exact stage.
    let homs: [[Expansion; 4]; 4] = std::array::from_fn(|i| pts[i].hom::<Expansion>());
    let mut sign = det4(&homs).sign();
    for h in &homs {
        match h[3].sign() {
            Sign::Zero => return None,
            s => sign = sign.combine(s),
        }
    }
    Some(sign)
}

/// Exact 2D in-circle test in the axis-aligned projection that drops the
/// given axis. Points may be implicit.
///
/// Positive iff `d` lies strictly inside the circumcircle of the
/// counterclockwise triangle (a, b, c); the sign flips for clockwise
/// (a, b, c). Returns `None` if any point is invalid.
///
/// Homogeneous lifting: the classic row (x, y, x^2 + y^2, 1) scaled by w^2
/// becomes (X W, Y W, X^2 + Y^2, W^2), polynomial in the homogeneous
/// coordinates, and the scaling factors are strictly positive so the
/// determinant sign needs no w correction.
pub fn incircle2d(a: &Point3, b: &Point3, c: &Point3, d: &Point3, drop: Axis) -> Option<Sign> {
    // Fast adaptive path: all points explicit.
    if let (Some(pa), Some(pb), Some(pc), Some(pd)) = (
        a.as_explicit(),
        b.as_explicit(),
        c.as_explicit(),
        d.as_explicit(),
    ) {
        let proj = |p: [f64; 3]| match drop {
            Axis::X => [p[1], p[2]],
            Axis::Y => [p[2], p[0]],
            Axis::Z => [p[0], p[1]],
        };
        return Some(Sign::of_f64(geometry_predicates::incircle(
            proj(pa),
            proj(pb),
            proj(pc),
            proj(pd),
        )));
    }

    fn lifted_row<T: crate::ring::Ring>(h: &[T; 3]) -> [T; 4] {
        let (x, y, w) = (&h[0], &h[1], &h[2]);
        [
            x.mul(w),
            y.mul(w),
            x.mul(x).add(&y.mul(y)),
            w.mul(w),
        ]
    }

    let pts = [a, b, c, d];

    // Interval filter.
    {
        let rows: [[Interval; 4]; 4] =
            std::array::from_fn(|i| lifted_row(&pts[i].hom2::<Interval>(drop)));
        let ws_known = pts
            .iter()
            .all(|p| matches!(p.hom2::<Interval>(drop)[2].sign(), Some(s) if s != Sign::Zero));
        if ws_known {
            if let Some(sign) = det4(&rows).sign() {
                return Some(sign);
            }
        }
    }

    // Exact stage.
    for p in &pts {
        if p.hom2::<Expansion>(drop)[2].sign() == Sign::Zero {
            return None;
        }
    }
    let rows: [[Expansion; 4]; 4] =
        std::array::from_fn(|i| lifted_row(&pts[i].hom2::<Expansion>(drop)));
    Some(det4(&rows).sign())
}

/// Exact 2D orientation of three points in the axis-aligned projection that
/// drops the given axis. Points may be implicit.
///
/// Sign convention: in the projected coordinate pair (see
/// [`Point3::hom2`] for the cyclic pairing), positive when `a`, `b`, `c` are
/// counterclockwise — equivalently, the sign of the `drop` component of the
/// normal of triangle (a, b, c) in 3D.
///
/// Returns `None` if any implicit point is invalid (exact w == 0).
pub fn orient2d(a: &Point3, b: &Point3, c: &Point3, drop: Axis) -> Option<Sign> {
    // Fast adaptive path: all points explicit.
    if let (Some(pa), Some(pb), Some(pc)) = (a.as_explicit(), b.as_explicit(), c.as_explicit()) {
        let proj = |p: [f64; 3]| match drop {
            Axis::X => [p[1], p[2]],
            Axis::Y => [p[2], p[0]],
            Axis::Z => [p[0], p[1]],
        };
        return Some(Sign::of_f64(geometry_predicates::orient2d(
            proj(pa),
            proj(pb),
            proj(pc),
        )));
    }

    let pts = [a, b, c];

    // det3 of homogeneous rows = (prod of w_i) * det2[[a-c],[b-c]].

    // Interval filter.
    'filter: {
        let homs: [[Interval; 3]; 3] = std::array::from_fn(|i| pts[i].hom2::<Interval>(drop));
        let Some(mut sign) = det3(&homs).sign() else {
            break 'filter;
        };
        for h in &homs {
            match h[2].sign() {
                Some(Sign::Positive) => {}
                Some(Sign::Negative) => sign = sign.flip(),
                _ => break 'filter,
            }
        }
        return Some(sign);
    }

    // Exact stage.
    let homs: [[Expansion; 3]; 3] = std::array::from_fn(|i| pts[i].hom2::<Expansion>(drop));
    let mut sign = det3(&homs).sign();
    for h in &homs {
        match h[2].sign() {
            Sign::Zero => return None,
            s => sign = sign.combine(s),
        }
    }
    Some(sign)
}

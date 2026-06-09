//! Staged-exact orientation predicates over explicit and implicit points.

use crate::expansion::Expansion;
use crate::geom::det4;
use crate::interval::Interval;
use crate::point::Point3;
use crate::Sign;

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

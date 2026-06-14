//! Staged-exact orientation predicates over explicit and implicit points.

use crate::expansion::Expansion;
use crate::geom::{det3, det4, det5};
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
/// Affine interval coordinates for points whose homogeneous w is exactly 1
/// (explicit, Lnc, Pac); `None` for the projective kinds (Lpi, Tpi, Bary).
/// Lets the filters of [`orient3d`] and [`insphere3d`] use the plain affine
/// difference determinants instead of the homogeneous lifts: w-sign folding
/// disappears and the determinant shrinks by one dimension, several times
/// fewer interval operations on the dominant Lnc/Pac meshing path.
#[inline]
fn affine_interval(p: &Point3) -> Option<[Interval; 3]> {
    match p {
        Point3::Explicit(c) => Some(c.map(Interval::point)),
        Point3::Lnc { .. } | Point3::Pac { .. } => {
            let h = p.hom::<Interval>();
            Some([h[0], h[1], h[2]])
        }
        _ => None,
    }
}

pub fn orient3d(a: &Point3, b: &Point3, c: &Point3, d: &Point3) -> Option<Sign> {
    crate::stats::bump(&crate::stats::ORIENT3D_CALLS);
    // Fast adaptive path: all points explicit.
    if let (Some(pa), Some(pb), Some(pc), Some(pd)) = (
        a.as_explicit(),
        b.as_explicit(),
        c.as_explicit(),
        d.as_explicit(),
    ) {
        return Some(Sign::of_f64(geometry_predicates::orient3d(pa, pb, pc, pd)));
    }
    crate::stats::bump(&crate::stats::ORIENT3D_IMPLICIT);

    let pts = [a, b, c, d];

    // Affine-reduction fast path. orient3d is multilinear in each argument, so
    // if exactly one point is an affine Steiner point (Lnc) -- a convex
    // combination `(1-t) a + t b` of explicit parents with t in (0, 1) -- its
    // orientation is `(1-t) O_a + t O_b` where O_a, O_b are the orientations
    // with the parent substituted in. With both weights strictly positive, when
    // O_a and O_b share a sign (or are zero) the point shares it; each O_i is a
    // fully explicit predicate (the fast adaptive path). This resolves the
    // common "Steiner clearly on one side of the facet plane" case without the
    // implicit interval / expansion machinery; the straddling case (O_a, O_b
    // opposite) falls through to the exact stages below. Exact: the identity is
    // a real-number identity (the point IS `(1-t) a + t b` exactly) and the
    // per-parent signs are exact. (Lpi/Tpi/Bary and Pac -- whose `1-u-v` weight
    // is not as cheaply sign-certified in f64 -- take the path below.)
    {
        let mut implicit_idx: Option<usize> = None;
        let mut explicit = [[0.0f64; 3]; 4];
        let mut single = true;
        for (k, p) in pts.iter().enumerate() {
            match p.as_explicit() {
                Some(c) => explicit[k] = c,
                None if implicit_idx.is_none() => implicit_idx = Some(k),
                None => {
                    single = false;
                    break;
                }
            }
        }
        if single {
            if let Some(k) = implicit_idx {
                if let Some((parents, weights, 2)) = pts[k].affine_combo() {
                    if weights[0] > 0.0 && weights[1] > 0.0 {
                        let (mut pos, mut neg) = (false, false);
                        for parent in &parents[..2] {
                            let mut q = explicit;
                            q[k] = *parent;
                            match Sign::of_f64(geometry_predicates::orient3d(q[0], q[1], q[2], q[3]))
                            {
                                Sign::Positive => pos = true,
                                Sign::Negative => neg = true,
                                Sign::Zero => {}
                            }
                        }
                        if !(pos && neg) {
                            return Some(if pos {
                                Sign::Positive
                            } else if neg {
                                Sign::Negative
                            } else {
                                Sign::Zero
                            });
                        }
                    }
                }
            }
        }
    }

    // Affine interval filter (all w exactly 1: explicit, Lnc, Pac — the
    // dominant meshing path): the homogeneous det4 equals det3 of the rows
    // a-d, b-d, c-d, with no w corrections. When indecisive, the exact
    // stage decides directly (the projective filter sees the same widths).
    if let (Some(pa), Some(pb), Some(pc), Some(pd)) = (
        affine_interval(a),
        affine_interval(b),
        affine_interval(c),
        affine_interval(d),
    ) {
        let row = |p: &[Interval; 3]| -> [Interval; 3] {
            std::array::from_fn(|k| p[k].sub(pd[k]))
        };
        if let Some(sign) = det3(&[row(&pa), row(&pb), row(&pc)]).sign() {
            return Some(sign);
        }
    } else {
        // The 4x4 homogeneous determinant relates to the affine orientation
        // by det4 = (prod of w_i) * det3[[a-d],[b-d],[c-d]], so the
        // orientation sign is the det4 sign combined with each w sign.

        // Projective interval filter (some w != 1).
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
    }

    // Exact stage.
    crate::stats::bump(&crate::stats::ORIENT3D_EXACT);
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

/// Exact 3D in-sphere test, any point may be implicit.
///
/// Positive iff `e` lies strictly inside the circumsphere of the POSITIVELY
/// ORIENTED tetrahedron (a, b, c, d) (Shewchuk's `insphere` convention; the
/// sign flips for a negatively oriented tet). Returns `None` if any point is
/// invalid (exact w == 0).
///
/// Homogeneous lifting: the classic row (x, y, z, x^2 + y^2 + z^2, 1) scaled
/// by w^2 becomes (X W, Y W, Z W, X^2 + Y^2 + Z^2, W^2), polynomial in the
/// homogeneous coordinates; the scaling factors w^2 are strictly positive
/// for valid points, so the determinant sign needs no w correction.
pub fn insphere3d(
    a: &Point3,
    b: &Point3,
    c: &Point3,
    d: &Point3,
    e: &Point3,
) -> Option<Sign> {
    crate::stats::bump(&crate::stats::INSPHERE_CALLS);
    // Fast adaptive path: all points explicit.
    if let (Some(pa), Some(pb), Some(pc), Some(pd), Some(pe)) = (
        a.as_explicit(),
        b.as_explicit(),
        c.as_explicit(),
        d.as_explicit(),
        e.as_explicit(),
    ) {
        return Some(Sign::of_f64(geometry_predicates::insphere(
            pa, pb, pc, pd, pe,
        )));
    }
    crate::stats::bump(&crate::stats::INSPHERE_IMPLICIT);

    fn lifted_row<T: crate::ring::Ring>(h: &[T; 4]) -> [T; 5] {
        let (x, y, z, w) = (&h[0], &h[1], &h[2], &h[3]);
        [
            x.mul(w),
            y.mul(w),
            z.mul(w),
            x.mul(x).add(&y.mul(y)).add(&z.mul(z)),
            w.mul(w),
        ]
    }

    let pts = [a, b, c, d, e];

    // Affine interval filter (all w exactly 1): column operations reduce the
    // homogeneous 5x5 lift to Shewchuk's difference form, det4 of rows
    // (p - e, |p - e|^2) — several times fewer interval operations than the
    // projective det5. This is the dominant meshing path (explicit, Lnc and
    // Pac points all have w = 1); when indecisive it falls through to the
    // exact stage directly (the projective filter sees the same widths).
    if let (Some(pa), Some(pb), Some(pc), Some(pd), Some(pe)) = (
        affine_interval(a),
        affine_interval(b),
        affine_interval(c),
        affine_interval(d),
        affine_interval(e),
    ) {
        let row = |p: &[Interval; 3]| -> [Interval; 4] {
            let d: [Interval; 3] = std::array::from_fn(|k| p[k].sub(pe[k]));
            let lift = d[0].mul(d[0]).add(d[1].mul(d[1])).add(d[2].mul(d[2]));
            [d[0], d[1], d[2], lift]
        };
        if let Some(sign) = det4(&[row(&pa), row(&pb), row(&pc), row(&pd)]).sign() {
            return Some(sign);
        }
    } else {
        // Projective interval filter (some w != 1).
        let homs: [[Interval; 4]; 5] = std::array::from_fn(|i| pts[i].hom::<Interval>());
        let ws_known = homs
            .iter()
            .all(|h| matches!(h[3].sign(), Some(s) if s != Sign::Zero));
        if ws_known {
            let rows: [[Interval; 5]; 5] = std::array::from_fn(|i| lifted_row(&homs[i]));
            if let Some(sign) = det5(&rows).sign() {
                return Some(sign);
            }
        }
    }

    // Exact stage.
    crate::stats::bump(&crate::stats::INSPHERE_EXACT);
    let homs: [[Expansion; 4]; 5] = std::array::from_fn(|i| pts[i].hom::<Expansion>());
    for h in &homs {
        if h[3].sign() == Sign::Zero {
            return None;
        }
    }
    let rows: [[Expansion; 5]; 5] = std::array::from_fn(|i| lifted_row(&homs[i]));
    Some(det5(&rows).sign())
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

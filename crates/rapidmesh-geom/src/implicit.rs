//! Implicit (SDF) surface carrier: signed-distance expression trees, the
//! [`ImplicitSurface`] carrier queried by the mesher, and the surface-nets
//! tessellation that lets an implicit solid enter the pipeline.
//!
//! The design mirrors the STL import path (`import.rs`): an implicit solid is
//! tessellated once into a tagged [`Faceted`] proxy whose facets all carry ONE
//! [`SurfaceKind::Implicit`] carrier. The exact-CSG arrangement operates on
//! the proxy triangles like on any other input; refinement and optimization
//! then project onto the *analytic* field via gradient Newton, so the final
//! mesh converges to the true implicit surface — the same
//! discretize-then-remesh contract as `SurfaceKind::Discrete`, but with an
//! exact carrier instead of the frozen soup.
//!
//! Fields are signed-distance-*like*: primitives are exact SDFs, but smooth
//! booleans and non-uniform placements only bound the distance. Every
//! consumer therefore uses first-order normalized queries (`F/|∇F|`) rather
//! than trusting the raw field value as a metric distance.

use crate::faceted::{Faceted, SurfaceKind};
use rapidmesh_csg::Tri;
use crate::vec3::{add, cross, dot, len, normalize, scale, sub, V3};

// ======================================================================================
// SDF expression tree
// ======================================================================================

/// A signed-distance expression: negative inside, positive outside. A data
/// tree (not closures) so carriers stay `Debug`-printable, cloneable, and
/// composable under placement transforms.
#[derive(Debug, Clone)]
pub enum Sdf {
    /// Sphere around `center`.
    Sphere { center: V3, radius: f64 },
    /// Axis-aligned box around `center` with half extents `half` (exact SDF,
    /// including corner distance outside).
    Box { center: V3, half: V3 },
    /// Capped cylinder between the axis endpoints `a` and `b`.
    Cylinder { a: V3, b: V3, radius: f64 },
    /// Capsule (sphere-swept segment) between `a` and `b`.
    Capsule { a: V3, b: V3, radius: f64 },
    /// Torus around `center` with plane normal `axis`.
    Torus { center: V3, axis: V3, major: f64, minor: f64 },
    /// Half space: negative on the anti-`normal` side of `point`.
    HalfSpace { point: V3, normal: V3 },
    /// Boolean union, `min(a, b)`.
    Union(std::boxed::Box<Sdf>, std::boxed::Box<Sdf>),
    /// Boolean intersection, `max(a, b)`.
    Intersect(std::boxed::Box<Sdf>, std::boxed::Box<Sdf>),
    /// Boolean difference, `max(a, -b)`.
    Difference(std::boxed::Box<Sdf>, std::boxed::Box<Sdf>),
    /// Smooth (filleted) union with blend radius `k` (polynomial smooth-min).
    SmoothUnion { a: std::boxed::Box<Sdf>, b: std::boxed::Box<Sdf>, k: f64 },
    /// Smooth intersection with blend radius `k`.
    SmoothIntersect { a: std::boxed::Box<Sdf>, b: std::boxed::Box<Sdf>, k: f64 },
    /// Smooth difference with blend radius `k`.
    SmoothDifference { a: std::boxed::Box<Sdf>, b: std::boxed::Box<Sdf>, k: f64 },
    /// Offset surface: grows the solid by `d` (rounds convex edges — the
    /// fillet/plating/coating primitive).
    Offset { a: std::boxed::Box<Sdf>, d: f64 },
    /// Shell of half thickness `t` around the zero set, `|a| - t`.
    Shell { a: std::boxed::Box<Sdf>, t: f64 },
}

/// Polynomial smooth-min (quadratic, Inigo Quilez formulation): equals
/// `min(a, b)` outside the `k`-band around `a = b`, blends inside it.
#[inline]
fn smooth_min(a: f64, b: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

/// [`smooth_min`] with its exact gradient: with `h = max(k - |a-b|, 0)/k`,
/// `dm = (1 - h/2)·d(min) + (h/2)·d(max)` — continuous through `a = b`
/// (`h = 1` gives the even blend) and collapsing to the branch gradient
/// outside the band.
#[inline]
fn smooth_min_grad(fa: f64, ga: V3, fb: f64, gb: V3, k: f64) -> (f64, V3) {
    if k <= 0.0 {
        return if fa <= fb { (fa, ga) } else { (fb, gb) };
    }
    let h = (k - (fa - fb).abs()).max(0.0) / k;
    let m = fa.min(fb) - h * h * k * 0.25;
    if h == 0.0 {
        return if fa <= fb { (m, ga) } else { (m, gb) };
    }
    let (gmin, gmax) = if fa <= fb { (ga, gb) } else { (gb, ga) };
    let g: V3 = std::array::from_fn(|i| (1.0 - 0.5 * h) * gmin[i] + 0.5 * h * gmax[i]);
    (m, g)
}

/// Some unit vector perpendicular to the unit vector `a` (degenerate query
/// points on an axis need a well-defined arbitrary direction).
#[inline]
fn any_perp_of(a: V3) -> V3 {
    let t = if a[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    normalize(cross(a, t))
}

impl Sdf {
    /// Field value and gradient at `p` in one pass — analytic per node (chain
    /// rule through the tree; min/max pick the active branch, the smooth
    /// booleans blend the branch gradients with the exact partials of the
    /// quadratic smooth-min). Exact away from the measure-zero min/max kinks,
    /// where the one-sided branch gradient is returned (the correct choice for
    /// every projection/normal consumer).
    pub fn eval_grad(&self, p: V3) -> (f64, V3) {
        match self {
            Sdf::Sphere { center, radius } => {
                let d = sub(p, *center);
                let l = len(d);
                if l < 1e-300 {
                    (-radius, [1.0, 0.0, 0.0])
                } else {
                    (l - radius, scale(d, 1.0 / l))
                }
            }
            Sdf::Box { center, half } => {
                let d = sub(p, *center);
                let q: V3 = std::array::from_fn(|i| d[i].abs() - half[i]);
                let sgn: V3 = std::array::from_fn(|i| if d[i] >= 0.0 { 1.0 } else { -1.0 });
                let outside: V3 = std::array::from_fn(|i| q[i].max(0.0));
                let lo = len(outside);
                if lo > 0.0 {
                    // Outside: gradient points from the closest box point.
                    let g: V3 = std::array::from_fn(|i| sgn[i] * outside[i] / lo);
                    (lo, g)
                } else {
                    // Inside: the closest face is the largest (least negative)
                    // component; gradient is that face's outward normal.
                    let mut k = 0;
                    for i in 1..3 {
                        if q[i] > q[k] {
                            k = i;
                        }
                    }
                    let mut g = [0.0; 3];
                    g[k] = sgn[k];
                    (q[0].max(q[1]).max(q[2]), g)
                }
            }
            Sdf::Cylinder { a, b, radius } => {
                let ab = sub(*b, *a);
                let l = len(ab);
                let axis = scale(ab, 1.0 / l);
                let ap = sub(p, *a);
                let t = dot(ap, axis);
                let rad_vec = sub(ap, scale(axis, t));
                let rl = len(rad_vec);
                let radial = rl - radius;
                let axial = (-t).max(t - l);
                let g_rad: V3 = if rl > 1e-300 { scale(rad_vec, 1.0 / rl) } else { any_perp_of(axis) };
                let g_ax: V3 = if t < 0.5 * l { scale(axis, -1.0) } else { axis };
                let (qr, qa) = (radial.max(0.0), axial.max(0.0));
                if qr > 0.0 || qa > 0.0 {
                    // Outside: gradient of sqrt(qr² + qa²).
                    let no = (qr * qr + qa * qa).sqrt();
                    let g: V3 = std::array::from_fn(|i| (qr * g_rad[i] + qa * g_ax[i]) / no);
                    (radial.max(axial).min(0.0) + no, g)
                } else if radial > axial {
                    (radial, g_rad)
                } else {
                    (axial, g_ax)
                }
            }
            Sdf::Capsule { a, b, radius } => {
                let ab = sub(*b, *a);
                let t = (dot(sub(p, *a), ab) / dot(ab, ab)).clamp(0.0, 1.0);
                let d = sub(p, add(*a, scale(ab, t)));
                let l = len(d);
                if l < 1e-300 {
                    (-radius, any_perp_of(normalize(ab)))
                } else {
                    (l - radius, scale(d, 1.0 / l))
                }
            }
            Sdf::Torus { center, axis, major, minor } => {
                let n = normalize(*axis);
                let cp = sub(p, *center);
                let h = dot(cp, n);
                let rad_vec = sub(cp, scale(n, h));
                let rl = len(rad_vec);
                let d = rl - major;
                let q = (d * d + h * h).sqrt();
                if q < 1e-300 || rl < 1e-300 {
                    return (q - minor, n);
                }
                let g_r = scale(rad_vec, 1.0 / rl);
                let g: V3 = std::array::from_fn(|i| (d * g_r[i] + h * n[i]) / q);
                (q - minor, g)
            }
            Sdf::HalfSpace { point, normal } => {
                let n = normalize(*normal);
                (dot(sub(p, *point), n), n)
            }
            Sdf::Union(a, b) => {
                let (fa, ga) = a.eval_grad(p);
                let (fb, gb) = b.eval_grad(p);
                if fa <= fb { (fa, ga) } else { (fb, gb) }
            }
            Sdf::Intersect(a, b) => {
                let (fa, ga) = a.eval_grad(p);
                let (fb, gb) = b.eval_grad(p);
                if fa >= fb { (fa, ga) } else { (fb, gb) }
            }
            Sdf::Difference(a, b) => {
                let (fa, ga) = a.eval_grad(p);
                let (fb, gb) = b.eval_grad(p);
                if fa >= -fb { (fa, ga) } else { (-fb, scale(gb, -1.0)) }
            }
            Sdf::SmoothUnion { a, b, k } => {
                let (fa, ga) = a.eval_grad(p);
                let (fb, gb) = b.eval_grad(p);
                smooth_min_grad(fa, ga, fb, gb, *k)
            }
            Sdf::SmoothIntersect { a, b, k } => {
                let (fa, ga) = a.eval_grad(p);
                let (fb, gb) = b.eval_grad(p);
                let (f, g) = smooth_min_grad(-fa, scale(ga, -1.0), -fb, scale(gb, -1.0), *k);
                (-f, scale(g, -1.0))
            }
            Sdf::SmoothDifference { a, b, k } => {
                let (fa, ga) = a.eval_grad(p);
                let (fb, gb) = b.eval_grad(p);
                let (f, g) = smooth_min_grad(-fa, scale(ga, -1.0), fb, gb, *k);
                (-f, scale(g, -1.0))
            }
            Sdf::Offset { a, d } => {
                let (f, g) = a.eval_grad(p);
                (f - d, g)
            }
            Sdf::Shell { a, t } => {
                let (f, g) = a.eval_grad(p);
                if f >= 0.0 { (f - t, g) } else { (-f - t, scale(g, -1.0)) }
            }
        }
    }

    /// Field value at `p` (negative inside).
    pub fn eval(&self, p: V3) -> f64 {
        match self {
            Sdf::Sphere { center, radius } => len(sub(p, *center)) - radius,
            Sdf::Box { center, half } => {
                let q: V3 = std::array::from_fn(|i| (p[i] - center[i]).abs() - half[i]);
                let outside: V3 = std::array::from_fn(|i| q[i].max(0.0));
                let inside = q[0].max(q[1]).max(q[2]).min(0.0);
                len(outside) + inside
            }
            Sdf::Cylinder { a, b, radius } => {
                let ab = sub(*b, *a);
                let l = len(ab);
                let axis = scale(ab, 1.0 / l);
                let ap = sub(p, *a);
                let t = dot(ap, axis);
                let radial = len(sub(ap, scale(axis, t))) - radius;
                let axial = (-t).max(t - l);
                let (qr, qa) = (radial.max(0.0), axial.max(0.0));
                radial.max(axial).min(0.0) + (qr * qr + qa * qa).sqrt()
            }
            Sdf::Capsule { a, b, radius } => {
                let ab = sub(*b, *a);
                let t = (dot(sub(p, *a), ab) / dot(ab, ab)).clamp(0.0, 1.0);
                len(sub(p, add(*a, scale(ab, t)))) - radius
            }
            Sdf::Torus { center, axis, major, minor } => {
                let n = normalize(*axis);
                let cp = sub(p, *center);
                let h = dot(cp, n);
                let radial = len(sub(cp, scale(n, h)));
                let d = radial - major;
                (d * d + h * h).sqrt() - minor
            }
            Sdf::HalfSpace { point, normal } => dot(sub(p, *point), normalize(*normal)),
            Sdf::Union(a, b) => a.eval(p).min(b.eval(p)),
            Sdf::Intersect(a, b) => a.eval(p).max(b.eval(p)),
            Sdf::Difference(a, b) => a.eval(p).max(-b.eval(p)),
            Sdf::SmoothUnion { a, b, k } => smooth_min(a.eval(p), b.eval(p), *k),
            Sdf::SmoothIntersect { a, b, k } => -smooth_min(-a.eval(p), -b.eval(p), *k),
            Sdf::SmoothDifference { a, b, k } => -smooth_min(-a.eval(p), b.eval(p), *k),
            Sdf::Offset { a, d } => a.eval(p) - d,
            Sdf::Shell { a, t } => a.eval(p).abs() - t,
        }
    }
}

// ======================================================================================
// Interval arithmetic — the sound bound behind the crossing pruning
// ======================================================================================

/// Closed interval `[lo, hi]`. Bounds are conventionally rounded (no outward
/// ulp widening): the consumer prunes with strict sign tests, and a crossing
/// tangent at machine epsilon falls under the transversal-graze contract that
/// every carrier's crossing oracle already has at its tolerance.
#[derive(Debug, Clone, Copy)]
pub struct Iv {
    pub lo: f64,
    pub hi: f64,
}

impl Iv {
    #[inline]
    pub fn point(x: f64) -> Iv {
        Iv { lo: x, hi: x }
    }
    #[inline]
    pub fn hull(a: f64, b: f64) -> Iv {
        Iv { lo: a.min(b), hi: a.max(b) }
    }
    #[inline]
    fn add(self, o: Iv) -> Iv {
        Iv { lo: self.lo + o.lo, hi: self.hi + o.hi }
    }
    #[inline]
    fn sub(self, o: Iv) -> Iv {
        Iv { lo: self.lo - o.hi, hi: self.hi - o.lo }
    }
    #[inline]
    fn neg(self) -> Iv {
        Iv { lo: -self.hi, hi: -self.lo }
    }
    #[inline]
    fn add_scalar(self, s: f64) -> Iv {
        Iv { lo: self.lo + s, hi: self.hi + s }
    }
    #[inline]
    fn sub_scalar(self, s: f64) -> Iv {
        self.add_scalar(-s)
    }
    #[inline]
    fn mul_scalar(self, s: f64) -> Iv {
        if s >= 0.0 {
            Iv { lo: self.lo * s, hi: self.hi * s }
        } else {
            Iv { lo: self.hi * s, hi: self.lo * s }
        }
    }
    #[inline]
    fn abs(self) -> Iv {
        if self.lo >= 0.0 {
            self
        } else if self.hi <= 0.0 {
            self.neg()
        } else {
            Iv { lo: 0.0, hi: self.hi.max(-self.lo) }
        }
    }
    #[inline]
    fn square(self) -> Iv {
        let a = self.abs();
        Iv { lo: a.lo * a.lo, hi: a.hi * a.hi }
    }
    #[inline]
    fn sqrt(self) -> Iv {
        Iv { lo: self.lo.max(0.0).sqrt(), hi: self.hi.max(0.0).sqrt() }
    }
    #[inline]
    fn min_iv(self, o: Iv) -> Iv {
        Iv { lo: self.lo.min(o.lo), hi: self.hi.min(o.hi) }
    }
    #[inline]
    fn max_iv(self, o: Iv) -> Iv {
        Iv { lo: self.lo.max(o.lo), hi: self.hi.max(o.hi) }
    }
    #[inline]
    fn min_scalar(self, s: f64) -> Iv {
        Iv { lo: self.lo.min(s), hi: self.hi.min(s) }
    }
    #[inline]
    fn max_scalar(self, s: f64) -> Iv {
        Iv { lo: self.lo.max(s), hi: self.hi.max(s) }
    }
    #[inline]
    fn clamp01(self) -> Iv {
        Iv { lo: self.lo.clamp(0.0, 1.0), hi: self.hi.clamp(0.0, 1.0) }
    }
}

/// Interval length of an interval vector, `sqrt(Σ dᵢ²)`.
#[inline]
fn len_iv(d: [Iv; 3]) -> Iv {
    d[0].square().add(d[1].square()).add(d[2].square()).sqrt()
}

/// Interval dot with a constant vector.
#[inline]
fn dot_const(d: [Iv; 3], c: V3) -> Iv {
    d[0].mul_scalar(c[0]).add(d[1].mul_scalar(c[1])).add(d[2].mul_scalar(c[2]))
}

/// Interval smooth-min: monotone non-decreasing in both arguments, so the
/// endpoint images bound the range exactly.
#[inline]
fn smooth_min_iv(a: Iv, b: Iv, k: f64) -> Iv {
    Iv { lo: smooth_min(a.lo, b.lo, k), hi: smooth_min(a.hi, b.hi, k) }
}

impl Sdf {
    /// Sound range bound of the field over an axis-aligned box (natural
    /// interval extension: possibly loose under variable reuse, never wrong).
    /// `0 ∉ eval_interval(box)` PROVES the box is crossing-free — the pruning
    /// oracle of [`ImplicitSurface::segment_hits`].
    pub fn eval_interval(&self, p: &[Iv; 3]) -> Iv {
        match self {
            Sdf::Sphere { center, radius } => {
                let d: [Iv; 3] = std::array::from_fn(|i| p[i].sub_scalar(center[i]));
                len_iv(d).sub_scalar(*radius)
            }
            Sdf::Box { center, half } => {
                let q: [Iv; 3] =
                    std::array::from_fn(|i| p[i].sub_scalar(center[i]).abs().sub_scalar(half[i]));
                let outside: [Iv; 3] = std::array::from_fn(|i| q[i].max_scalar(0.0));
                let inside = q[0].max_iv(q[1]).max_iv(q[2]).min_scalar(0.0);
                len_iv(outside).add(inside)
            }
            Sdf::Cylinder { a, b, radius } => {
                let ab = sub(*b, *a);
                let l = len(ab);
                let axis = scale(ab, 1.0 / l);
                let ap: [Iv; 3] = std::array::from_fn(|i| p[i].sub_scalar(a[i]));
                let t = dot_const(ap, axis);
                let rad: [Iv; 3] = std::array::from_fn(|i| ap[i].sub(t.mul_scalar(axis[i])));
                let radial = len_iv(rad).sub_scalar(*radius);
                let axial = t.neg().max_iv(t.sub_scalar(l));
                let qr = radial.max_scalar(0.0);
                let qa = axial.max_scalar(0.0);
                radial
                    .max_iv(axial)
                    .min_scalar(0.0)
                    .add(qr.square().add(qa.square()).sqrt())
            }
            Sdf::Capsule { a, b, radius } => {
                let ab = sub(*b, *a);
                let ab2 = dot(ab, ab);
                let ap: [Iv; 3] = std::array::from_fn(|i| p[i].sub_scalar(a[i]));
                let t = dot_const(ap, ab).mul_scalar(1.0 / ab2).clamp01();
                let d: [Iv; 3] = std::array::from_fn(|i| ap[i].sub(t.mul_scalar(ab[i])));
                len_iv(d).sub_scalar(*radius)
            }
            Sdf::Torus { center, axis, major, minor } => {
                let n = normalize(*axis);
                let cp: [Iv; 3] = std::array::from_fn(|i| p[i].sub_scalar(center[i]));
                let h = dot_const(cp, n);
                let rad: [Iv; 3] = std::array::from_fn(|i| cp[i].sub(h.mul_scalar(n[i])));
                let d = len_iv(rad).sub_scalar(*major);
                d.square().add(h.square()).sqrt().sub_scalar(*minor)
            }
            Sdf::HalfSpace { point, normal } => {
                let n = normalize(*normal);
                let d: [Iv; 3] = std::array::from_fn(|i| p[i].sub_scalar(point[i]));
                dot_const(d, n)
            }
            Sdf::Union(a, b) => a.eval_interval(p).min_iv(b.eval_interval(p)),
            Sdf::Intersect(a, b) => a.eval_interval(p).max_iv(b.eval_interval(p)),
            Sdf::Difference(a, b) => a.eval_interval(p).max_iv(b.eval_interval(p).neg()),
            Sdf::SmoothUnion { a, b, k } => {
                smooth_min_iv(a.eval_interval(p), b.eval_interval(p), *k)
            }
            Sdf::SmoothIntersect { a, b, k } => {
                smooth_min_iv(a.eval_interval(p).neg(), b.eval_interval(p).neg(), *k).neg()
            }
            Sdf::SmoothDifference { a, b, k } => {
                smooth_min_iv(a.eval_interval(p).neg(), b.eval_interval(p), *k).neg()
            }
            Sdf::Offset { a, d } => a.eval_interval(p).sub_scalar(*d),
            Sdf::Shell { a, t } => a.eval_interval(p).abs().sub_scalar(*t),
        }
    }
}

/// Point on the segment `a→b` at parameter `t`.
#[inline]
fn lerp(a: V3, b: V3, t: f64) -> V3 {
    std::array::from_fn(|i| a[i] + t * (b[i] - a[i]))
}

// ======================================================================================
// The carrier
// ======================================================================================

/// An implicit surface as a meshing carrier: the zero set of an [`Sdf`] under
/// an optional placement affine, restricted to a local-frame bounding box.
///
/// Mirrors the query API of `DiscreteSurface`: closest-point projection with
/// outward normal, curvature radius for the sizing field, first-order signed
/// offset, and segment intersections for the crossing pulls.
#[derive(Debug)]
pub struct ImplicitSurface {
    sdf: Sdf,
    /// Placement: world = linear · local + offset (identity at construction;
    /// composed by [`Self::transformed`]).
    linear: [[f64; 3]; 3],
    offset: V3,
    /// Inverse of `linear` (placements are invertible by construction — the
    /// scene layer only produces rotations/uniform scales/translations).
    inv_linear: [[f64; 3]; 3],
    /// Local-frame axis-aligned bounds the zero set lives in (with margin).
    pub bbox: (V3, V3),
    /// Query length scale: the finite-difference step and the projection
    /// tolerance derive from the bbox diagonal.
    scale: f64,
}

fn mat_vec(m: &[[f64; 3]; 3], v: V3) -> V3 {
    std::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

fn mat_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn mat_inv(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    assert!(det.abs() > 1e-300, "implicit placement must be invertible");
    let inv_det = 1.0 / det;
    let cof = |r0: usize, r1: usize, c0: usize, c1: usize| {
        m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
    };
    [
        [cof(1, 2, 1, 2) * inv_det, -cof(0, 2, 1, 2) * inv_det, cof(0, 1, 1, 2) * inv_det],
        [-cof(1, 2, 0, 2) * inv_det, cof(0, 2, 0, 2) * inv_det, -cof(0, 1, 0, 2) * inv_det],
        [cof(1, 2, 0, 1) * inv_det, -cof(0, 2, 0, 1) * inv_det, cof(0, 1, 0, 1) * inv_det],
    ]
}

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

impl ImplicitSurface {
    /// New carrier in its own frame, bounded by `bbox` (must enclose the zero
    /// set with some margin; the tessellation samples only inside it).
    pub fn new(sdf: Sdf, bbox: (V3, V3)) -> ImplicitSurface {
        let scale = len(sub(bbox.1, bbox.0));
        assert!(scale > 0.0, "implicit bbox must be non-degenerate");
        ImplicitSurface {
            sdf,
            linear: IDENTITY,
            offset: [0.0; 3],
            inv_linear: IDENTITY,
            bbox,
            scale,
        }
    }

    /// Field value at a world point (negative inside).
    pub fn eval(&self, p: V3) -> f64 {
        let local = mat_vec(&self.inv_linear, sub(p, self.offset));
        self.sdf.eval(local)
    }

    /// Field value and world-frame gradient in one pass: the tree's analytic
    /// gradient, pulled back through the placement
    /// (`∇_world = inv_linearᵀ · ∇_local`).
    pub fn eval_grad(&self, p: V3) -> (f64, V3) {
        let local = mat_vec(&self.inv_linear, sub(p, self.offset));
        let (f, gl) = self.sdf.eval_grad(local);
        let gw: V3 = std::array::from_fn(|i| {
            self.inv_linear[0][i] * gl[0] + self.inv_linear[1][i] * gl[1] + self.inv_linear[2][i] * gl[2]
        });
        (f, gw)
    }

    /// World-frame gradient (analytic, see [`eval_grad`](Self::eval_grad)).
    pub fn grad(&self, p: V3) -> V3 {
        self.eval_grad(p).1
    }

    /// Central-difference gradient. The projection loop uses THIS, not the
    /// analytic gradient: distance fields of edged primitives (box, capped
    /// cylinder) carry measure-zero gradient JUMPS on their interior medial
    /// planes, where exact branch-picking makes gradient Newton hop between
    /// branches and stall — the 1e-7-scale FD average regularises the kink
    /// and keeps the iteration contracting (found the hard way: the offset
    /// box's edge band leaked a non-manifold edge under the exact gradient).
    /// Pointwise consumers (signed_offset, curvature probes, normals) keep
    /// the exact analytic gradient.
    fn grad_fd(&self, p: V3) -> V3 {
        let h = self.scale * 1e-7;
        std::array::from_fn(|i| {
            let mut a = p;
            let mut b = p;
            a[i] += h;
            b[i] -= h;
            (self.eval(a) - self.eval(b)) / (2.0 * h)
        })
    }

    /// First-order signed distance to the zero set, `F/|∇F|` (exact for pure
    /// primitive SDFs, first-order for blends/placements).
    pub fn signed_offset(&self, p: V3) -> f64 {
        let (f, g) = self.eval_grad(p);
        let gl = len(g);
        if gl < 1e-300 {
            return f;
        }
        f / gl
    }

    /// Closest point on the surface and the outward normal there — the
    /// projection contract of the meshing path. Gradient Newton on
    /// `F(x) = 0` with the analytic gradient: converges quadratically near
    /// the surface, which is where every mesher query originates (points
    /// drift off by one smoothing step). Terminates on the first-order
    /// surface distance `|F|/|∇F|`, not the raw field value.
    pub fn closest(&self, p: V3) -> (V3, V3) {
        let tol = self.scale * 1e-12;
        let mut x = p;
        let mut g = [0.0, 0.0, 1.0];
        for _ in 0..48 {
            let (f, gi) = (self.eval(x), self.grad_fd(x));
            let g2 = dot(gi, gi);
            if g2 < 1e-300 {
                break;
            }
            g = gi;
            let step = f / g2;
            let dx = scale(gi, -step);
            x = add(x, dx);
            if len(dx) < tol {
                break;
            }
        }
        (x, normalize(g))
    }

    /// Local curvature radius at (the projection of) `p`, from the turn of
    /// the FIELD normal across two tangent probes — no re-projection: near
    /// the surface the field normal equals the surface normal to first
    /// order, which is all the sizing bias consumes.
    pub fn curvature_radius(&self, p: V3) -> f64 {
        let (x, n) = self.closest(p);
        let h = self.scale * 1e-4;
        let t1 = {
            let cand = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
            normalize(cross(n, cand))
        };
        let t2 = cross(n, t1);
        let mut worst: f64 = f64::INFINITY;
        for t in [t1, t2] {
            let n_probe = normalize(self.eval_grad(add(x, scale(t, h))).1);
            let turn = len(sub(n_probe, n));
            if turn > 1e-14 {
                worst = worst.min(h / turn);
            }
        }
        worst
    }

    /// Parameters `t ∈ (0, 1)` where the segment `a→b` crosses the zero set,
    /// ascending — transversal crossings only, mirroring the discrete
    /// carrier's contract. Interval-guaranteed: the field is interval-
    /// evaluated over the segment's box hull, subintervals with `0 ∉ F` are
    /// PRUNED with proof (one interval evaluation instead of dozens of point
    /// samples), and only leaves that still bracket refine by bisection. A
    /// sub-resolution double crossing (both inside a leaf of width ~1e-9)
    /// is dropped like a tangential graze — the same semantics every other
    /// carrier's crossing oracle has at its tolerance.
    pub fn segment_hits(&self, a: V3, b: V3, out: &mut Vec<f64>) {
        if !(len(sub(b, a)) > 0.0) {
            return;
        }
        // A/B escape hatch: RAPIDMESH_IMPLICIT_SAMPLED=1 falls back to the
        // fixed-density sign-change sampler (no interval guarantees).
        static SAMPLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *SAMPLED.get_or_init(|| std::env::var_os("RAPIDMESH_IMPLICIT_SAMPLED").is_some()) {
            self.segment_hits_sampled(a, b, out);
            return;
        }
        let f_a = self.eval(a);
        let f_b = self.eval(b);
        self.seg_rec(a, b, 0.0, 1.0, f_a, f_b, 0, out);
        out.retain(|&t| t > 0.0 && t < 1.0);
        out.dedup_by(|x, y| (*x - *y).abs() < 1e-12);
    }

    /// The pre-interval crossing oracle: fixed-density sign-change sampling
    /// with bisection refinement (kept as the `RAPIDMESH_IMPLICIT_SAMPLED`
    /// A/B fallback; can miss features thinner than the sample spacing).
    fn segment_hits_sampled(&self, a: V3, b: V3, out: &mut Vec<f64>) {
        let seg = len(sub(b, a));
        let n = ((seg / self.scale) * 256.0).ceil().clamp(8.0, 256.0) as usize;
        let mut t_prev = 0.0;
        let mut f_prev = self.eval(a);
        for i in 1..=n {
            let t = i as f64 / n as f64;
            let f = self.eval(lerp(a, b, t));
            if f_prev != 0.0 && f != 0.0 && f_prev.signum() != f.signum() {
                let (mut lo, mut hi, mut flo) = (t_prev, t, f_prev);
                for _ in 0..40 {
                    let mid = 0.5 * (lo + hi);
                    let fm = self.eval(lerp(a, b, mid));
                    if fm == 0.0 {
                        lo = mid;
                        hi = mid;
                        break;
                    }
                    if flo * fm < 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                        flo = fm;
                    }
                }
                out.push(0.5 * (lo + hi));
            }
            t_prev = t;
            f_prev = f;
        }
        out.retain(|&t| t > 0.0 && t < 1.0);
        out.dedup_by(|x, y| (*x - *y).abs() < 1e-12);
    }

    /// Recursive interval subdivision for [`segment_hits`](Self::segment_hits).
    /// `f_lo`/`f_hi` are the endpoint field values (reused across levels so
    /// each level costs one midpoint evaluation plus one interval evaluation).
    #[allow(clippy::too_many_arguments)]
    fn seg_rec(&self, a: V3, b: V3, t0: f64, t1: f64, f0: f64, f1: f64, depth: u32, out: &mut Vec<f64>) {
        let p0 = lerp(a, b, t0);
        let p1 = lerp(a, b, t1);
        // Box hull of the subsegment in the local frame (interval affine map),
        // then one interval sweep of the tree: 0 outside the range proves no
        // crossing anywhere in the subsegment.
        let world: [Iv; 3] = std::array::from_fn(|i| Iv::hull(p0[i], p1[i]));
        let local: [Iv; 3] = std::array::from_fn(|i| {
            let mut acc = Iv::point(0.0);
            for j in 0..3 {
                acc = acc.add(world[j].sub_scalar(self.offset[j]).mul_scalar(self.inv_linear[i][j]));
            }
            acc
        });
        let range = self.sdf.eval_interval(&local);
        if range.lo > 0.0 || range.hi < 0.0 {
            return; // proven crossing-free
        }
        // Leaf: a confirmed transversal sign change refines by bisection as
        // soon as the subsegment is feature-separated (1e-4 of the segment —
        // the same order as the discrete arm's sign-probe DELTA); same-sign
        // subsegments that still bracket keep splitting down to the 1e-9
        // floor, where a remaining pair is sub-resolution and dropped like a
        // tangential graze.
        let sign_change = f0 != 0.0 && f1 != 0.0 && f0.signum() != f1.signum();
        if depth >= 36 || (t1 - t0) < 1e-9 || (sign_change && (t1 - t0) < 1e-4) {
            if sign_change {
                let (mut lo, mut hi, mut flo) = (t0, t1, f0);
                for _ in 0..40 {
                    let mid = 0.5 * (lo + hi);
                    let fm = self.eval(lerp(a, b, mid));
                    if fm == 0.0 {
                        lo = mid;
                        hi = mid;
                        break;
                    }
                    if flo * fm < 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                        flo = fm;
                    }
                }
                out.push(0.5 * (lo + hi));
            }
            return;
        }
        let mut tm = 0.5 * (t0 + t1);
        let mut fm = self.eval(lerp(a, b, tm));
        if fm == 0.0 {
            // A split point landing exactly on the zero set (crossings at
            // dyadic parameters do this) would blind both halves' sign
            // checks; nudge the split off the surface.
            tm += (t1 - t0) * 1e-3;
            fm = self.eval(lerp(a, b, tm));
        }
        // A transversal pair below the subdivision floor is invisible to the
        // endpoint signs; recursing until the interval either prunes or hits
        // the leaf floor preserves every provable crossing.
        self.seg_rec(a, b, t0, tm, f0, fm, depth + 1, out);
        self.seg_rec(a, b, tm, t1, fm, f1, depth + 1, out);
    }

    /// The carrier under a placement `world = linear · local + offset`,
    /// composed onto any existing placement. The SDF tree itself is untouched;
    /// only the frame maps.
    pub fn transformed(&self, linear: [[f64; 3]; 3], offset: V3) -> ImplicitSurface {
        let new_linear = mat_mul(&linear, &self.linear);
        let new_offset = add(mat_vec(&linear, self.offset), offset);
        // The query scale follows the placement (uniform scaling is the only
        // scale-changing placement the scene layer produces).
        let corners = [self.bbox.0, self.bbox.1];
        let w0 = add(mat_vec(&new_linear, corners[0]), new_offset);
        let w1 = add(mat_vec(&new_linear, corners[1]), new_offset);
        ImplicitSurface {
            sdf: self.sdf.clone(),
            linear: new_linear,
            offset: new_offset,
            inv_linear: mat_inv(&new_linear),
            bbox: self.bbox,
            scale: len(sub(w1, w0)).max(1e-300),
        }
    }
}

// ======================================================================================
// Surface-nets tessellation — the pipeline entry proxy
// ======================================================================================

/// Tessellate the zero set into a watertight triangle proxy by naive surface
/// nets over a `cells³` grid on the carrier's bbox: one vertex per
/// sign-changing cell (mean of its edge crossings), one quad (two triangles)
/// per sign-changing grid edge, wound outward by the field sign. The proxy is
/// closed and manifold as long as the zero set stays strictly inside the bbox
/// and features are resolved by the grid.
pub fn tessellate_surface_nets(surf: &ImplicitSurface, cells: usize) -> (Vec<V3>, Vec<[u32; 3]>) {
    let n = cells.max(4);
    let (lo, hi) = surf.bbox;
    let d: V3 = std::array::from_fn(|i| (hi[i] - lo[i]) / n as f64);
    let np = n + 1;
    let pt = |i: usize, j: usize, k: usize| -> V3 {
        [lo[0] + d[0] * i as f64, lo[1] + d[1] * j as f64, lo[2] + d[2] * k as f64]
    };

    // Sample the field at grid points (local frame == world frame here: the
    // proxy is built at construction, before any placement).
    let idx = |i: usize, j: usize, k: usize| (i * np + j) * np + k;
    let mut field = vec![0.0f64; np * np * np];
    for i in 0..np {
        for j in 0..np {
            for k in 0..np {
                field[idx(i, j, k)] = surf.eval(pt(i, j, k));
            }
        }
    }

    // One vertex per cell that has a sign change on any of its 12 edges: the
    // mean of the edge crossings (linear interpolation along each edge).
    let cell_id = |i: usize, j: usize, k: usize| (i * n + j) * n + k;
    let mut cell_vert = vec![u32::MAX; n * n * n];
    let mut verts: Vec<V3> = Vec::new();
    const CELL_EDGES: [([usize; 3], [usize; 3]); 12] = [
        ([0, 0, 0], [1, 0, 0]), ([0, 1, 0], [1, 1, 0]), ([0, 0, 1], [1, 0, 1]), ([0, 1, 1], [1, 1, 1]),
        ([0, 0, 0], [0, 1, 0]), ([1, 0, 0], [1, 1, 0]), ([0, 0, 1], [0, 1, 1]), ([1, 0, 1], [1, 1, 1]),
        ([0, 0, 0], [0, 0, 1]), ([1, 0, 0], [1, 0, 1]), ([0, 1, 0], [0, 1, 1]), ([1, 1, 0], [1, 1, 1]),
    ];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let mut sum = [0.0f64; 3];
                let mut cnt = 0usize;
                for (ea, eb) in CELL_EDGES {
                    let fa = field[idx(i + ea[0], j + ea[1], k + ea[2])];
                    let fb = field[idx(i + eb[0], j + eb[1], k + eb[2])];
                    if (fa < 0.0) != (fb < 0.0) {
                        let t = fa / (fa - fb);
                        let pa = pt(i + ea[0], j + ea[1], k + ea[2]);
                        let pb = pt(i + eb[0], j + eb[1], k + eb[2]);
                        for c in 0..3 {
                            sum[c] += pa[c] + t * (pb[c] - pa[c]);
                        }
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    cell_vert[cell_id(i, j, k)] = verts.len() as u32;
                    verts.push(std::array::from_fn(|c| sum[c] / cnt as f64));
                }
            }
        }
    }

    // One quad per sign-changing INTERIOR grid edge, spanned by the four
    // cells around the edge, wound so triangle normals point from inside
    // (field < 0) to outside.
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut quad = |v: [u32; 3+1], flip: bool| {
        let [a, b, c, dd] = v;
        if a == u32::MAX || b == u32::MAX || c == u32::MAX || dd == u32::MAX {
            return; // zero set touched the bbox shell — proxy stays open there
        }
        if flip {
            tris.push([a, c, b]);
            tris.push([a, dd, c]);
        } else {
            tris.push([a, b, c]);
            tris.push([a, c, dd]);
        }
    };
    for i in 0..np {
        for j in 0..np {
            for k in 0..np {
                let f0 = field[idx(i, j, k)];
                // x-edge (i,j,k)→(i+1,j,k): cells (i, j-1..j, k-1..k)
                if i < n && j > 0 && j < n && k > 0 && k < n {
                    let f1 = field[idx(i + 1, j, k)];
                    if (f0 < 0.0) != (f1 < 0.0) {
                        quad(
                            [
                                cell_vert[cell_id(i, j - 1, k - 1)],
                                cell_vert[cell_id(i, j, k - 1)],
                                cell_vert[cell_id(i, j, k)],
                                cell_vert[cell_id(i, j - 1, k)],
                            ],
                            f0 >= 0.0,
                        );
                    }
                }
                // y-edge (i,j,k)→(i,j+1,k): cells (i-1..i, j, k-1..k)
                if j < n && i > 0 && i < n && k > 0 && k < n {
                    let f1 = field[idx(i, j + 1, k)];
                    if (f0 < 0.0) != (f1 < 0.0) {
                        quad(
                            [
                                cell_vert[cell_id(i - 1, j, k - 1)],
                                cell_vert[cell_id(i - 1, j, k)],
                                cell_vert[cell_id(i, j, k)],
                                cell_vert[cell_id(i, j, k - 1)],
                            ],
                            f0 >= 0.0,
                        );
                    }
                }
                // z-edge (i,j,k)→(i,j,k+1): cells (i-1..i, j-1..j, k)
                if k < n && i > 0 && i < n && j > 0 && j < n {
                    let f1 = field[idx(i, j, k + 1)];
                    if (f0 < 0.0) != (f1 < 0.0) {
                        quad(
                            [
                                cell_vert[cell_id(i - 1, j - 1, k)],
                                cell_vert[cell_id(i, j - 1, k)],
                                cell_vert[cell_id(i, j, k)],
                                cell_vert[cell_id(i - 1, j, k)],
                            ],
                            f0 >= 0.0,
                        );
                    }
                }
            }
        }
    }
    (verts, tris)
}

/// Build a [`Faceted`] solid from an implicit surface: the surface-nets proxy
/// at `cells` resolution, every facet tagged with the ONE implicit carrier —
/// the exact mirror of the STL import path, with an analytic carrier.
pub fn implicit_solid(surf: ImplicitSurface, cells: usize) -> Result<Faceted, String> {
    let (verts, tris) = tessellate_surface_nets(&surf, cells);
    if tris.is_empty() {
        return Err("implicit solid: zero set not found inside the bbox".to_string());
    }
    let mut f = Faceted::new();
    let s = f.add_surface(SurfaceKind::Implicit(std::sync::Arc::new(surf)));
    for t in &tris {
        let tri = Tri::new(
            verts[t[0] as usize],
            verts[t[1] as usize],
            verts[t[2] as usize],
        );
        f.push_tri(tri, s);
    }
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(r: f64) -> ImplicitSurface {
        ImplicitSurface::new(
            Sdf::Sphere { center: [0.0; 3], radius: r },
            ([-r - 0.5; 3], [r + 0.5; 3]),
        )
    }

    #[test]
    fn closest_projects_onto_sphere() {
        let s = sphere(1.0);
        let (x, n) = s.closest([2.0, 0.3, -0.4]);
        assert!((len(x) - 1.0).abs() < 1e-9, "|x| = {}", len(x));
        let out = normalize(x);
        assert!(dot(n, out) > 0.999, "normal must point outward");
    }

    #[test]
    fn curvature_radius_of_sphere_is_radius() {
        let s = sphere(2.0);
        let r = s.curvature_radius([2.5, 0.0, 0.0]);
        assert!((r - 2.0).abs() < 0.05, "r = {r}");
    }

    #[test]
    fn segment_hits_finds_both_crossings() {
        let s = sphere(1.0);
        let mut hits = Vec::new();
        s.segment_hits([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], &mut hits);
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert!((hits[0] - 0.25).abs() < 1e-9 && (hits[1] - 0.75).abs() < 1e-9, "{hits:?}");
    }

    #[test]
    fn interval_segment_hits_catch_thin_shell() {
        // A shell of half thickness 1e-3 around the unit sphere, crossed on a
        // diameter: four transversal crossings, the inner pairs 2e-3 apart.
        // Uniform sampling at the pre-interval density (~1e-2 spacing) sat on
        // both sides of each pair and missed them; interval subdivision
        // proves every crossing-free stretch and isolates all four.
        let s = ImplicitSurface::new(
            Sdf::Shell {
                a: std::boxed::Box::new(Sdf::Sphere { center: [0.0; 3], radius: 1.0 }),
                t: 1e-3,
            },
            ([-1.5; 3], [1.5; 3]),
        );
        let mut hits = Vec::new();
        s.segment_hits([-1.3, 0.0, 0.0], [1.3, 0.0, 0.0], &mut hits);
        assert_eq!(hits.len(), 4, "{hits:?}");
        for (t, x) in [(0.11423, -1.001), (0.11577, -0.999), (0.88423, 0.999), (0.88577, 1.001)]
            .iter()
            .zip(hits.iter().map(|&t| -1.3 + t * 2.6))
            .map(|((t, _), x)| (*t, x))
        {
            let _ = t;
            assert!(
                (x.abs() - 1.0).abs() - 1e-3 < 1e-9,
                "crossing at x = {x} not on the shell"
            );
        }
    }

    #[test]
    fn analytic_gradient_matches_finite_differences() {
        // The analytic tree gradient against central differences, across every
        // node kind, at points off the kink sets.
        let tree = Sdf::SmoothDifference {
            a: std::boxed::Box::new(Sdf::SmoothUnion {
                a: std::boxed::Box::new(Sdf::Box { center: [0.0; 3], half: [0.8, 0.6, 0.5] }),
                b: std::boxed::Box::new(Sdf::Cylinder {
                    a: [0.0, 0.0, -1.0],
                    b: [0.2, 0.1, 1.0],
                    radius: 0.4,
                }),
                k: 0.3,
            }),
            b: std::boxed::Box::new(Sdf::Torus {
                center: [0.3, 0.0, 0.0],
                axis: [0.2, 1.0, 0.1],
                major: 0.9,
                minor: 0.25,
            }),
            k: 0.2,
        };
        let pts = [
            [1.3, 0.2, -0.4],
            [-0.9, 0.7, 0.6],
            [0.1, -1.2, 0.3],
            [0.5, 0.5, 1.1],
            [-1.1, -0.8, -0.9],
        ];
        for p in pts {
            let (_, g) = tree.eval_grad(p);
            let h = 1e-6;
            for i in 0..3 {
                let (mut a, mut b) = (p, p);
                a[i] += h;
                b[i] -= h;
                let fd = (tree.eval(a) - tree.eval(b)) / (2.0 * h);
                assert!(
                    (g[i] - fd).abs() < 1e-6,
                    "{p:?} component {i}: analytic {} vs FD {fd}",
                    g[i]
                );
            }
        }
    }

    #[test]
    fn interval_bounds_contain_point_samples() {
        // Soundness fuzz: every point evaluation inside a random box must lie
        // within the box's interval bound.
        let tree = Sdf::SmoothUnion {
            a: std::boxed::Box::new(Sdf::Sphere { center: [0.3, -0.2, 0.1], radius: 0.9 }),
            b: std::boxed::Box::new(Sdf::Capsule {
                a: [-0.8, 0.0, -0.5],
                b: [0.7, 0.4, 0.6],
                radius: 0.3,
            }),
            k: 0.25,
        };
        let mut state = 0x243F_6A88_85A3_08D3u64;
        let mut rnd = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..200 {
            let lo: V3 = std::array::from_fn(|_| rnd() * 3.0 - 1.5);
            let w: V3 = std::array::from_fn(|_| rnd() * 0.8);
            let bx: [Iv; 3] = std::array::from_fn(|i| Iv { lo: lo[i], hi: lo[i] + w[i] });
            let bound = tree.eval_interval(&bx);
            for _ in 0..8 {
                let p: V3 = std::array::from_fn(|i| lo[i] + rnd() * w[i]);
                let f = tree.eval(p);
                assert!(
                    f >= bound.lo - 1e-12 && f <= bound.hi + 1e-12,
                    "point {f} outside interval [{}, {}]",
                    bound.lo, bound.hi
                );
            }
        }
    }

    #[test]
    fn surface_nets_proxy_is_closed_and_oriented() {
        let s = sphere(1.0);
        let (verts, tris) = tessellate_surface_nets(&s, 24);
        assert!(!tris.is_empty());
        // Closed: every directed edge has its opposite exactly once.
        let mut edges = std::collections::HashMap::new();
        for t in &tris {
            for e in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *edges.entry(e).or_insert(0i32) += 1;
            }
        }
        for (&(a, b), &c) in &edges {
            assert_eq!(c, 1, "duplicate directed edge {a}-{b}");
            assert_eq!(edges.get(&(b, a)), Some(&1), "unmatched edge {a}-{b}");
        }
        // Oriented outward: signed volume positive and near the sphere volume.
        let mut vol6 = 0.0;
        for t in &tris {
            let (a, b, c) = (verts[t[0] as usize], verts[t[1] as usize], verts[t[2] as usize]);
            vol6 += dot(a, cross(b, c));
        }
        let vol = vol6 / 6.0;
        let exact = 4.0 / 3.0 * std::f64::consts::PI;
        assert!(
            (vol - exact).abs() < 0.05 * exact,
            "signed volume {vol} vs sphere {exact}"
        );
    }

    #[test]
    fn transformed_carrier_queries_in_world_frame() {
        let s = sphere(1.0);
        // Uniform scale by 2 and shift: the surface is |p - c| = 2 around c.
        let m = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let c = [5.0, -1.0, 3.0];
        let w = s.transformed(m, c);
        let (x, _) = w.closest(add(c, [3.5, 0.0, 0.0]));
        assert!((len(sub(x, c)) - 2.0).abs() < 1e-8, "{:?}", sub(x, c));
        assert!(w.signed_offset(add(c, [2.0, 0.0, 0.0])).abs() < 1e-6);
    }
}

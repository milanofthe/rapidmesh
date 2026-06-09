//! Explicit and implicit (lazily evaluated) 3D points.

use crate::expansion::Expansion;
use crate::geom::{lpi_hom, tpi_hom};
use crate::interval::Interval;
use crate::ring::Ring;
use crate::Sign;

/// A 3D point, either explicit or defined implicitly by the primitives whose
/// intersection it is.
///
/// Implicit points are never rounded to f64 coordinates inside predicates:
/// every predicate evaluates their homogeneous coordinates symbolically (as
/// polynomials in the defining inputs) at the precision the staged evaluation
/// requires. This is what makes cascaded constructions (intersection points of
/// intersection segments, Steiner points on recovered boundaries) exact.
#[derive(Debug, Clone, PartialEq)]
pub enum Point3 {
    /// An ordinary coordinate point.
    Explicit([f64; 3]),
    /// Line-plane intersection: line through `p`, `q`; plane through
    /// `r`, `s`, `t`.
    Lpi {
        /// First line point.
        p: [f64; 3],
        /// Second line point.
        q: [f64; 3],
        /// First plane point.
        r: [f64; 3],
        /// Second plane point.
        s: [f64; 3],
        /// Third plane point.
        t: [f64; 3],
    },
    /// Three-plane intersection, each plane through its three points.
    Tpi {
        /// The three defining planes, each as three points.
        planes: Box<[[[f64; 3]; 3]; 3]>,
    },
}

impl Point3 {
    /// An explicit point.
    pub fn explicit(x: f64, y: f64, z: f64) -> Point3 {
        Point3::Explicit([x, y, z])
    }

    /// The intersection of the line through `p`, `q` with the plane through
    /// `r`, `s`, `t`.
    pub fn lpi(p: [f64; 3], q: [f64; 3], r: [f64; 3], s: [f64; 3], t: [f64; 3]) -> Point3 {
        Point3::Lpi { p, q, r, s, t }
    }

    /// The intersection of three planes, each given by three points.
    pub fn tpi(plane0: [[f64; 3]; 3], plane1: [[f64; 3]; 3], plane2: [[f64; 3]; 3]) -> Point3 {
        Point3::Tpi {
            planes: Box::new([plane0, plane1, plane2]),
        }
    }

    /// The coordinates if this point is explicit.
    pub fn as_explicit(&self) -> Option<[f64; 3]> {
        match self {
            Point3::Explicit(c) => Some(*c),
            _ => None,
        }
    }

    /// Homogeneous coordinates (x, y, z, w) in the given ring. For explicit
    /// points w = 1.
    pub fn hom<T: Ring>(&self) -> [T; 4] {
        match self {
            Point3::Explicit(c) => [
                T::from_f64(c[0]),
                T::from_f64(c[1]),
                T::from_f64(c[2]),
                T::from_f64(1.0),
            ],
            Point3::Lpi { p, q, r, s, t } => lpi_hom(*p, *q, *r, *s, *t),
            Point3::Tpi { planes } => tpi_hom(planes),
        }
    }

    /// Exact sign of the homogeneous w coordinate. [`Sign::Zero`] means the
    /// defining primitives do not intersect in a single point (the point is
    /// invalid and must not be used in predicates).
    pub fn w_sign(&self) -> Sign {
        match self {
            Point3::Explicit(_) => Sign::Positive,
            _ => {
                // Interval filter first, exact fallback.
                if let Some(s) = self.hom::<Interval>()[3].sign() {
                    return s;
                }
                self.hom::<Expansion>()[3].sign()
            }
        }
    }

    /// True if the point is well defined (w != 0). Exact.
    pub fn is_valid(&self) -> bool {
        self.w_sign() != Sign::Zero
    }

    /// Approximate f64 coordinates (for output/visualization, never for
    /// predicates). `None` if the f64 evaluation of w underflows to zero.
    pub fn approx(&self) -> Option<[f64; 3]> {
        let h = self.hom::<Interval>();
        let mid = |iv: &Interval| 0.5 * (iv.lo() + iv.hi());
        let w = mid(&h[3]);
        if w == 0.0 {
            return None;
        }
        Some([mid(&h[0]) / w, mid(&h[1]) / w, mid(&h[2]) / w])
    }
}

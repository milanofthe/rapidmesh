//! Tensor-product rational B-spline (NURBS) surface.
//!
//! The missing geometry primitive for consuming general CAD/STEP geometry: a
//! trimmed NURBS surface is what a boolean of free-form bodies produces, and what
//! the B-rep layer must be able to carry as a first-class [`crate::SurfaceKind`]
//! sibling. This is the surface analogue of [`crate::nurbs::NurbsCurve`]: the same
//! clamped knot vectors and rational weights, in two parameter directions.
//!
//! Evaluation is the tensor product of the per-direction B-spline basis (Piegl &
//! Tiller A2.2) with the rational quotient on the homogeneous control net. The
//! inverse map (`(x,y,z) -> (u,v)`, a Newton projection) lands when the mesher
//! needs it; point evaluation is exact and complete here.

use crate::vec3::{V3};
/// A tensor-product rational B-spline surface `S(u,v)`.
#[derive(Debug, Clone, PartialEq)]
pub struct NurbsSurface {
    /// Polynomial degrees `(p, q)` in the `u` and `v` directions.
    pub degree: [usize; 2],
    /// Clamped, non-decreasing knot vectors `(U, V)`; `U.len() == n_u + p + 1`.
    pub knots: [Vec<f64>; 2],
    /// Control-point counts `(n_u, n_v)` per direction.
    pub n: [usize; 2],
    /// Control net, row-major: control `(i, j)` is `ctrl[i * n_v + j]`.
    pub ctrl: Vec<V3>,
    /// Rational weights, parallel to `ctrl` (all 1.0 = a plain B-spline).
    pub weights: Vec<f64>,
}

impl NurbsSurface {
    /// Builds a surface, validating the knot/control/weight sizes.
    pub fn new(
        degree: [usize; 2],
        knots: [Vec<f64>; 2],
        n: [usize; 2],
        ctrl: Vec<V3>,
        weights: Vec<f64>,
    ) -> NurbsSurface {
        assert!(degree[0] >= 1 && degree[1] >= 1, "degrees must be >= 1");
        assert!(n[0] >= degree[0] + 1 && n[1] >= degree[1] + 1, "need degree+1 controls per dir");
        assert_eq!(knots[0].len(), n[0] + degree[0] + 1, "U knot count must be n_u+p+1");
        assert_eq!(knots[1].len(), n[1] + degree[1] + 1, "V knot count must be n_v+q+1");
        assert_eq!(ctrl.len(), n[0] * n[1], "control net must be n_u*n_v");
        assert_eq!(weights.len(), ctrl.len(), "one weight per control point");
        assert!(weights.iter().all(|&w| w > 0.0), "weights must be positive");
        NurbsSurface { degree, knots, n, ctrl, weights }
    }

    /// Parameter domain `([u_min, u_max], [v_min, v_max])` (the clamped end knots).
    pub fn domain(&self) -> ([f64; 2], [f64; 2]) {
        let ud = [self.knots[0][self.degree[0]], self.knots[0][self.n[0]]];
        let vd = [self.knots[1][self.degree[1]], self.knots[1][self.n[1]]];
        (ud, vd)
    }

    /// Surface point `S(u, v)`.
    pub fn eval(&self, u: f64, v: f64) -> V3 {
        let (ud, vd) = self.domain();
        let u = u.clamp(ud[0], ud[1]);
        let v = v.clamp(vd[0], vd[1]);
        let (pu, pv) = (self.degree[0], self.degree[1]);
        let su = find_span(&self.knots[0], self.n[0] - 1, pu, u);
        let sv = find_span(&self.knots[1], self.n[1] - 1, pv, v);
        let bu = basis_funs(su, u, pu, &self.knots[0]);
        let bv = basis_funs(sv, v, pv, &self.knots[1]);
        let mut num = [0.0f64; 3];
        let mut den = 0.0f64;
        for i in 0..=pu {
            let ci = su - pu + i;
            for j in 0..=pv {
                let cj = sv - pv + j;
                let idx = ci * self.n[1] + cj;
                let w = self.weights[idx] * bu[i] * bv[j];
                den += w;
                for k in 0..3 {
                    num[k] += w * self.ctrl[idx][k];
                }
            }
        }
        std::array::from_fn(|k| num[k] / den)
    }
}

/// Knot span index containing `u` (Piegl & Tiller A2.1); `n` is the last control
/// index (`ctrl_count - 1`), `p` the degree.
fn find_span(knots: &[f64], n: usize, p: usize, u: f64) -> usize {
    if u >= knots[n + 1] {
        return n;
    }
    if u <= knots[p] {
        return p;
    }
    let (mut low, mut high) = (p, n + 1);
    let mut mid = (low + high) / 2;
    while u < knots[mid] || u >= knots[mid + 1] {
        if u < knots[mid] {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    mid
}

/// Nonzero basis functions `N_{span-p+j, p}(u)`, `j` in `0..=p` (Piegl & Tiller
/// A2.2).
fn basis_funs(span: usize, u: f64, p: usize, knots: &[f64]) -> Vec<f64> {
    let mut n = vec![0.0f64; p + 1];
    let mut left = vec![0.0f64; p + 1];
    let mut right = vec![0.0f64; p + 1];
    n[0] = 1.0;
    for j in 1..=p {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            let temp = n[r] / (right[r + 1] + left[j - r]);
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n[j] = saved;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bilinear (degree 1x1) patch over a 2x2 net: eval interpolates the corners
    /// and the center is the corner average.
    #[test]
    fn bilinear_patch_evaluates() {
        let ctrl = vec![
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ];
        let s = NurbsSurface::new(
            [1, 1],
            [vec![0.0, 0.0, 1.0, 1.0], vec![0.0, 0.0, 1.0, 1.0]],
            [2, 2],
            ctrl,
            vec![1.0; 4],
        );
        let (ud, vd) = s.domain();
        assert_eq!(ud, [0.0, 1.0]);
        assert_eq!(vd, [0.0, 1.0]);
        // corners
        assert_eq!(s.eval(0.0, 0.0), [0.0, 0.0, 0.0]);
        assert_eq!(s.eval(1.0, 1.0), [1.0, 1.0, 0.0]);
        // center = average of the four corners
        let c = s.eval(0.5, 0.5);
        assert!((c[0] - 0.5).abs() < 1e-12);
        assert!((c[1] - 0.5).abs() < 1e-12);
        assert!((c[2] - 0.5).abs() < 1e-12);
    }

    /// A degree-2 row stays planar in z when all control z are equal (partition of
    /// unity), confirming the rational tensor sum.
    #[test]
    fn quadratic_partition_of_unity() {
        let ctrl: Vec<V3> = (0..3)
            .flat_map(|i| (0..3).map(move |j| [i as f64, j as f64, 2.0]))
            .collect();
        let s = NurbsSurface::new(
            [2, 2],
            [vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
            [3, 3],
            ctrl,
            vec![1.0; 9],
        );
        for &(u, v) in &[(0.3, 0.7), (0.5, 0.5), (0.9, 0.1)] {
            assert!((s.eval(u, v)[2] - 2.0).abs() < 1e-12, "z must stay 2.0");
        }
    }
}

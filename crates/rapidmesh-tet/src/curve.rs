//! Stage 1 of the bottom-up sizing hierarchy: point distribution on a general
//! edge curve.
//!
//! A curve is parametrized by arc length `s in [0, length]`. Points are placed to
//! meet a chord (sagitta) error bound -- an element of length `h` on a curve of
//! radius `R` deviates by `eps ~ h^2/(8R)`, so bounding the relative deviation
//! `delta = eps/R` gives `h <= R*sqrt(8*delta)` -- under a SMOOTHNESS constraint:
//! the size field is gradient-limited so adjacent elements differ by at most a
//! fixed RATIO (`1 + grad`). The ratio (multiplicative) limit is what gives both a
//! NARROW transition (few points -- size coarsens geometrically away from a tight
//! feature) AND no abrupt density jump (every neighbour pair is within `1+grad`).
//! An additive `h0 + slope*dist` limit cannot have both: a gentle slope floods a
//! wide band with fine points, a steep slope jumps.
//!
//! Points are then placed at equal increments of the cumulative density
//! `C(s) = integral 1/h ds` -- the 1D centroidal-Voronoi (equal-error) optimum --
//! so no separate relaxation pass is needed. This module is geometry only: it does
//! not know about the mesh, surfaces, or the PLC; higher stages feed it curves and
//! consume its points.

use rapidmesh_geom::vec3::{V3, dist};
/// A general edge curve, parametrized by arc length `s in [0, length()]`.
pub trait Curve {
    /// Total arc length.
    fn length(&self) -> f64;
    /// Position at arc length `s` (clamped to `[0, length]`).
    fn point_at(&self, s: f64) -> V3;
    /// Local principal radius of curvature `R = 1/kappa` at `s` (`INFINITY` where
    /// the curve is straight). The input to the sagitta size bound.
    fn radius_at(&self, s: f64) -> f64;
}

/// A curve given as a polyline of on-curve sample points. Curvature is the
/// discrete osculating-circle radius (the circumradius of three consecutive
/// samples). Works for any edge; an analytic edge supplies a finely resampled,
/// exactly-on-curve polyline so the discrete radius matches the true one.
pub struct PolylineCurve {
    pts: Vec<V3>,
    cum: Vec<f64>, // cumulative arc length at each sample
}

impl PolylineCurve {
    /// Builds a polyline curve from samples (>= 2). Duplicate consecutive points
    /// are dropped so the arc-length table is strictly increasing.
    pub fn new(samples: &[V3]) -> Option<PolylineCurve> {
        let mut pts: Vec<V3> = Vec::with_capacity(samples.len());
        for &p in samples {
            if pts.last().map(|&q| dist(p, q) > 1e-15).unwrap_or(true) {
                pts.push(p);
            }
        }
        if pts.len() < 2 {
            return None;
        }
        let mut cum = vec![0.0f64; pts.len()];
        for i in 1..pts.len() {
            cum[i] = cum[i - 1] + dist(pts[i], pts[i - 1]);
        }
        Some(PolylineCurve { pts, cum })
    }

    /// Sample index `i` with `cum[i] <= s` (for interpolation).
    fn seg(&self, s: f64) -> usize {
        let s = s.clamp(0.0, self.cum[self.cum.len() - 1]);
        self.cum.partition_point(|&c| c < s).clamp(1, self.cum.len() - 1) - 1
    }
}

/// Circumradius of three points (osculating-circle radius); `INFINITY` if
/// collinear.
fn circumradius(a: V3, b: V3, c: V3) -> f64 {
    let (ab, bc, ca) = (dist(a, b), dist(b, c), dist(c, a));
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let cr = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let area2 = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
    if area2 <= 1e-300 {
        f64::INFINITY
    } else {
        ab * bc * ca / (2.0 * area2)
    }
}

impl Curve for PolylineCurve {
    fn length(&self) -> f64 {
        self.cum[self.cum.len() - 1]
    }

    fn point_at(&self, s: f64) -> V3 {
        let i = self.seg(s);
        let (s0, s1) = (self.cum[i], self.cum[i + 1]);
        let f = if s1 > s0 { (s.clamp(s0, s1) - s0) / (s1 - s0) } else { 0.0 };
        std::array::from_fn(|k| self.pts[i][k] + f * (self.pts[i + 1][k] - self.pts[i][k]))
    }

    fn radius_at(&self, s: f64) -> f64 {
        // Osculating radius at the sample nearest `s` (its two polyline neighbours).
        let i = self.seg(s);
        // Use the vertex closest to s as the apex of the triple.
        let apex = if i + 1 < self.pts.len()
            && (s - self.cum[i]) > (self.cum[i + 1] - s)
        {
            i + 1
        } else {
            i
        };
        if apex == 0 || apex + 1 >= self.pts.len() {
            return f64::INFINITY; // endpoints: no curvature defined
        }
        circumradius(self.pts[apex - 1], self.pts[apex], self.pts[apex + 1])
    }
}

/// Distributes points along `curve` and returns their arc-length parameters
/// (always including the endpoints `0` and `length`). The local target size is
/// `h(s) = min(maxh, radius(s)*sqrt(8*deflection))`, smoothed by a multiplicative
/// gradient limit so adjacent elements differ by at most the ratio `1 + grad`.
/// Points sit at equal increments of the cumulative density `integral 1/h`.
pub fn distribute(curve: &dyn Curve, deflection: f64, maxh: f64, grad: f64) -> Vec<f64> {
    let len = curve.length();
    if !(len > 0.0) || !(maxh > 0.0) {
        return vec![0.0];
    }
    let chord = (8.0 * deflection.max(1e-12)).sqrt();
    // Fine arc-length samples: enough to resolve the finest target. Cap the count.
    let m = ((len / (chord * maxh).max(maxh * 0.05)).ceil() as usize * 4).clamp(64, 8192);
    let ds = len / m as f64;
    let mut h = vec![0.0f64; m + 1];
    for i in 0..=m {
        let s = (i as f64) * ds;
        let r = curve.radius_at(s);
        h[i] = if r.is_finite() { (r * chord).min(maxh) } else { maxh }.max(1e-12);
    }
    // Multiplicative gradient limit: h cannot grow faster than the ratio (1+grad)
    // per element of its own length. Over a sub-element sample step `ds`, that is
    // h[i] <= h[i-1] * (1+grad)^(ds/h[i-1]). Forward then backward sweep makes the
    // field two-sided Lipschitz in log-space -> smooth, narrow, no jump.
    let g = grad.max(1e-6);
    for i in 1..=m {
        let cap = h[i - 1] * (1.0 + g).powf(ds / h[i - 1]);
        if h[i] > cap {
            h[i] = cap;
        }
    }
    for i in (0..m).rev() {
        let cap = h[i + 1] * (1.0 + g).powf(ds / h[i + 1]);
        if h[i] > cap {
            h[i] = cap;
        }
    }
    // Cumulative density C(s) = integral 1/h ds (trapezoid over the samples).
    let mut cum = vec![0.0f64; m + 1];
    for i in 1..=m {
        cum[i] = cum[i - 1] + 0.5 * (1.0 / h[i] + 1.0 / h[i - 1]) * ds;
    }
    let total = cum[m];
    let n = (total.round() as usize).max(1); // number of elements
    let mut out = Vec::with_capacity(n + 1);
    out.push(0.0);
    let mut j = 1usize;
    for k in 1..n {
        let target = k as f64 / n as f64 * total;
        while j <= m && cum[j] < target {
            j += 1;
        }
        let j = j.min(m);
        let seg = (cum[j] - cum[j - 1]).max(1e-30);
        let f = ((target - cum[j - 1]) / seg).clamp(0.0, 1.0);
        out.push(((j - 1) as f64 + f) * ds);
    }
    out.push(len);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn circle(r: f64, n: usize) -> PolylineCurve {
        let pts: Vec<V3> = (0..=n)
            .map(|i| {
                let t = 2.0 * PI * i as f64 / n as f64;
                [r * t.cos(), r * t.sin(), 0.0]
            })
            .collect();
        PolylineCurve::new(&pts).unwrap()
    }

    #[test]
    fn polyline_arc_length_and_radius() {
        let c = circle(2.0, 400);
        assert!((c.length() - 2.0 * PI * 2.0).abs() < 1e-2);
        // discrete radius ~ true radius
        let r = c.radius_at(c.length() * 0.3);
        assert!((r - 2.0).abs() < 0.05, "radius {r}");
    }

    #[test]
    fn circle_distribution_is_uniform_and_meets_bound() {
        let r = 2.0;
        let delta = 0.02;
        let c = circle(r, 2000);
        let s = distribute(&c, delta, 100.0, 0.3);
        // Expected element length ~ R*sqrt(8*delta); count ~ circumference / h.
        let h = r * (8.0 * delta).sqrt();
        let expect = (2.0 * PI * r / h).round() as usize;
        let n = s.len() - 1; // elements
        assert!(
            (n as i64 - expect as i64).abs() <= 2,
            "count {n} vs expected {expect}"
        );
        // Spacing is near-uniform on a circle (constant curvature).
        let mut spc: Vec<f64> = s.windows(2).map(|w| w[1] - w[0]).collect();
        spc.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ratio = spc[spc.len() - 1] / spc[0];
        assert!(ratio < 1.2, "circle spacing ratio {ratio}");
    }

    #[test]
    fn straight_edge_uses_maxh() {
        let pts: Vec<V3> = (0..=50).map(|i| [i as f64 / 50.0 * 10.0, 0.0, 0.0]).collect();
        let c = PolylineCurve::new(&pts).unwrap();
        let s = distribute(&c, 0.02, 1.0, 0.3);
        let n = s.len() - 1;
        assert_eq!(n, 10, "10 elements of maxh=1 on a length-10 line, got {n}");
    }

    #[test]
    fn high_curvature_spot_grades_smoothly() {
        // A curve: long straight arms with a tight semicircle bump in the middle,
        // so the field is fine at the bump and coarse on the arms. Verify the
        // OUTPUT spacing ratio between adjacent elements stays bounded (smooth).
        let mut pts: Vec<V3> = Vec::new();
        for i in 0..=100 {
            pts.push([-5.0 + i as f64 / 100.0 * 5.0, 0.0, 0.0]); // arm to origin
        }
        let rb = 0.1;
        for i in 1..100 {
            let t = PI * i as f64 / 100.0;
            pts.push([rb * t.sin(), rb * (1.0 - t.cos()), 0.0]); // semicircle bump
        }
        for i in 0..=100 {
            pts.push([i as f64 / 100.0 * 5.0, 0.0, 0.0]); // arm away
        }
        let c = PolylineCurve::new(&pts).unwrap();
        let s = distribute(&c, 0.02, 1.0, 0.3);
        let spc: Vec<f64> = s.windows(2).map(|w| w[1] - w[0]).collect();
        // No adjacent pair jumps by more than ~ (1+grad) plus a sampling margin.
        let mut worst = 1.0f64;
        for w in spc.windows(2) {
            worst = worst.max(w[1] / w[0]).max(w[0] / w[1]);
        }
        assert!(worst < 1.6, "adjacent spacing ratio {worst} not smooth");
        // And it must actually refine the bump (some element far below maxh).
        assert!(spc.iter().cloned().fold(f64::MAX, f64::min) < 0.1, "bump not refined");
    }
}

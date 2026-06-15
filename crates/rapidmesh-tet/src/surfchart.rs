//! Distance-faithful 2D charts of the analytic curved surfaces
//! (`rapidmesh_geom::SurfaceKind`), used to mesh a curved boundary group with
//! the planar CVT machinery (`surf2d`): project the group's boundary into the
//! chart, relax interior points there, triangulate, then lift every chart point
//! back EXACTLY onto the analytic surface (`to_xyz`). Because the lift is the
//! analytic surface point, the resulting surface vertices satisfy the same
//! on-carrier guarantee as the explicit `Site::on_surface` constraint.
//!
//! The charts are isometric where one exists (cylinder/cone unroll, both
//! developable) and azimuthal-equidistant for the sphere (distance from the
//! chart center is exact; circumferential distance stretches by `psi/sin psi`,
//! negligible for caps and bounded for sub-antipodal groups). A chart is a
//! bijection over any group that excludes its singular point (the sphere's
//! antipode, the seam where an unrolled angle wraps past 2pi); the caller
//! validates this with a round-trip check and falls back to per-facet planar
//! tiling when it fails (closed or wrapping groups).
//!
//! `curvature_radius` gives the local principal radius `R = 1/kappa`, the input
//! to the chord/volume-error sizing bias `h = sqrt(8*eps*R)`: a facet edge of
//! length `h` on a surface of radius `R` deviates from the true surface by a
//! sagitta `eps ~ h^2/(8R)`, so bounding the deviation bounds the enclosed
//! volume error.

use crate::project::{closest_on_surface, curve_footpoint};
use rapidmesh_geom::nurbs::NurbsCurve;
use rapidmesh_geom::SurfaceKind;
use std::sync::Arc;

type V3 = [f64; 3];
type P2 = [f64; 2];

/// A 2D chart of a curved surface for the curved CVT: a (near) distance-faithful
/// parametrization plus the surface constraint. The curved-Lloyd engine talks to
/// surfaces ONLY through this trait, so any representation (the analytic kinds,
/// later trimmed NURBS patches, discrete meshes) plugs in by implementing it:
/// `to_uv`/`to_xyz` give the chart the planar relaxation runs in, `project` is
/// the on-surface constraint, and `curvature_radius` feeds the sizing bias.
pub trait SurfaceChart {
    /// Projects an on- or near-surface point into the chart.
    fn to_uv(&self, p: V3) -> P2;
    /// Lifts a chart point onto the surface (lands exactly on it).
    fn to_xyz(&self, uv: P2) -> V3;
    /// Local principal radius of curvature `R = 1/kappa_max` at chart point
    /// `uv`, the input to the chord/volume-error sizing bias.
    fn curvature_radius(&self, uv: P2) -> f64;
    /// Closest point on the underlying surface (the carrier constraint), used to
    /// validate the chart is a bijection over a group.
    fn project(&self, p: V3) -> V3;
}

/// Builds a chart for the analytic `kind`, fixing the frame/branch from
/// representative on-surface points of the group. `None` for `Plane` (use the
/// planar path) or a degenerate group. The single factory the mesher calls; new
/// surface representations extend it with their own [`SurfaceChart`] impls.
pub fn build_chart(kind: &SurfaceKind, pts: &[V3]) -> Option<Box<dyn SurfaceChart>> {
    match kind {
        SurfaceKind::Extruded { .. } => {
            ExtrudedChart::new(kind).map(|c| Box::new(c) as Box<dyn SurfaceChart>)
        }
        _ => Chart::new(kind, pts).map(|c| Box::new(c) as Box<dyn SurfaceChart>),
    }
}

/// Chart of a linearly extruded profile curve: developable, so the chart is
/// EXACTLY isometric, `u = arc length along the profile`, `v = extrusion
/// height`. The profile is sampled once into an arc-length table for the
/// `t <-> s` conversions. A closed (wrapping) profile is not bijective here (the
/// seam) and is rejected upstream; an open profile strip charts cleanly.
pub struct ExtrudedChart {
    kind: SurfaceKind,
    profile: Arc<NurbsCurve>,
    base: V3,
    u: V3,
    v: V3,
    a: V3,
    /// Sorted parameter samples and their cumulative arc length from the start.
    ts: Vec<f64>,
    ss: Vec<f64>,
}

impl ExtrudedChart {
    fn new(kind: &SurfaceKind) -> Option<ExtrudedChart> {
        let SurfaceKind::Extruded { profile, base, udir, vdir, axis } = kind else {
            return None;
        };
        let (lo, hi) = profile.domain();
        let n = 256usize;
        let mut ts = Vec::with_capacity(n + 1);
        let mut ss = Vec::with_capacity(n + 1);
        let mut acc = 0.0;
        ts.push(lo);
        ss.push(0.0);
        let mut prev = lo;
        for i in 1..=n {
            let t = lo + (hi - lo) * i as f64 / n as f64;
            acc += profile.arc_length(prev, t, 2);
            ts.push(t);
            ss.push(acc);
            prev = t;
        }
        Some(ExtrudedChart {
            kind: kind.clone(),
            profile: profile.clone(),
            base: *base,
            u: normalize(*udir),
            v: normalize(*vdir),
            a: normalize(*axis),
            ts,
            ss,
        })
    }

    /// Forward map `t -> s` (arc length), by interpolation in the table.
    fn t_to_s(&self, t: f64) -> f64 {
        let (lo, hi) = (self.ts[0], self.ts[self.ts.len() - 1]);
        let t = t.clamp(lo, hi);
        let i = self.ts.partition_point(|&x| x < t).clamp(1, self.ts.len() - 1);
        let (t0, t1) = (self.ts[i - 1], self.ts[i]);
        let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
        self.ss[i - 1] + f * (self.ss[i] - self.ss[i - 1])
    }

    /// Inverse map `s -> t`, table interpolation refined by Newton on speed.
    fn s_to_t(&self, s: f64) -> f64 {
        let s = s.clamp(0.0, self.ss[self.ss.len() - 1]);
        let i = self.ss.partition_point(|&x| x < s).clamp(1, self.ss.len() - 1);
        let (s0, s1) = (self.ss[i - 1], self.ss[i]);
        let f = if s1 > s0 { (s - s0) / (s1 - s0) } else { 0.0 };
        let mut t = self.ts[i - 1] + f * (self.ts[i] - self.ts[i - 1]);
        let (lo, hi) = self.profile.domain();
        for _ in 0..3 {
            let (_, d1, _) = self.profile.ders2(t);
            let speed = (d1[0] * d1[0] + d1[1] * d1[1]).sqrt();
            if speed < 1e-15 {
                break;
            }
            t = (t + (s - self.t_to_s(t)) / speed).clamp(lo, hi);
        }
        t
    }
}

impl SurfaceChart for ExtrudedChart {
    fn to_uv(&self, p: V3) -> P2 {
        let h = dot(sub(p, self.base), self.a);
        let rel = sub(sub(p, self.base), scale(self.a, h));
        let q = [dot(rel, self.u), dot(rel, self.v)];
        let t = curve_footpoint(&self.profile, q);
        [self.t_to_s(t), h]
    }

    fn to_xyz(&self, uv: P2) -> V3 {
        let t = self.s_to_t(uv[0]);
        let c = self.profile.eval(t);
        add(
            add(self.base, scale(self.a, uv[1])),
            add(scale(self.u, c[0]), scale(self.v, c[1])),
        )
    }

    fn curvature_radius(&self, uv: P2) -> f64 {
        let t = self.s_to_t(uv[0]);
        let k = self.profile.curvature(t);
        if k > 1e-12 {
            1.0 / k
        } else {
            1e12
        }
    }

    fn project(&self, p: V3) -> V3 {
        closest_on_surface(&self.kind, p)
    }
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn normalize(a: V3) -> V3 {
    let l = norm(a);
    if l == 0.0 {
        a
    } else {
        scale(a, 1.0 / l)
    }
}
fn any_perp(a: V3) -> V3 {
    let t = if a[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    normalize(cross(a, t))
}

/// Circular mean angle of a sample set (atan2 of the summed unit vectors); 0 if
/// the samples cancel. Used to fix an unrolled chart's angular branch so the
/// group sits in the middle of `[-pi, pi)` and does not straddle the seam.
fn circular_mean(angles: &[f64]) -> f64 {
    let (mut sc, mut cc) = (0.0, 0.0);
    for &a in angles {
        sc += a.sin();
        cc += a.cos();
    }
    if sc == 0.0 && cc == 0.0 {
        0.0
    } else {
        sc.atan2(cc)
    }
}

/// Shift `theta` into the branch `[center - pi, center + pi)` so unrolled charts
/// stay continuous across the +/-pi seam.
fn unwrap(theta: f64, center: f64) -> f64 {
    let two_pi = std::f64::consts::TAU;
    let mut t = theta;
    while t - center >= std::f64::consts::PI {
        t -= two_pi;
    }
    while t - center < -std::f64::consts::PI {
        t += two_pi;
    }
    t
}

/// A 2D chart of one analytic curved surface, with a fixed frame.
#[derive(Clone, Debug)]
pub struct Chart {
    kind: SurfaceKind,
    inner: Inner,
}

#[derive(Clone, Debug)]
enum Inner {
    /// Azimuthal-equidistant: chart center direction `c`, tangent basis e1,e2.
    Sphere { center: V3, radius: f64, c: V3, e1: V3, e2: V3 },
    /// Isometric unroll: axis `a`, radial basis e1,e2, branch `theta0`.
    Cylinder { center: V3, a: V3, e1: V3, e2: V3, radius: f64, theta0: f64 },
    /// Unrolled sector: apex, axis `a`, radial basis, half-angle via `sin_a`.
    Cone { apex: V3, a: V3, e1: V3, e2: V3, tan: f64, sin_a: f64, theta0: f64 },
    /// Parametric (major angle `theta`, minor angle `phi`), near-isometric.
    Torus { center: V3, a: V3, e1: V3, e2: V3, major: f64, minor: f64, theta0: f64, phi0: f64 },
}

impl Chart {
    /// Builds the chart for `kind`, fixing the frame/branch from representative
    /// on-surface points of the group. Returns `None` for `Plane` (use the
    /// planar path) or a degenerate group.
    pub fn new(kind: &SurfaceKind, pts: &[V3]) -> Option<Chart> {
        if pts.is_empty() {
            return None;
        }
        let inner = match *kind {
            SurfaceKind::Plane => return None,
            // Extruded profiles use `ExtrudedChart` (see `build_chart`).
            SurfaceKind::Extruded { .. } => return None,
            SurfaceKind::Sphere { center, radius } => {
                let mut acc = [0.0; 3];
                for &p in pts {
                    acc = add(acc, normalize(sub(p, center)));
                }
                let c = normalize(acc);
                let c = if norm(c) == 0.0 { [0.0, 0.0, 1.0] } else { c };
                let e1 = any_perp(c);
                let e2 = cross(c, e1);
                Inner::Sphere { center, radius, c, e1, e2 }
            }
            SurfaceKind::Cylinder { center, axis, radius } => {
                let a = normalize(axis);
                let e1 = any_perp(a);
                let e2 = cross(a, e1);
                let angles: Vec<f64> = pts
                    .iter()
                    .map(|&p| {
                        let w = sub(p, center);
                        let r = sub(w, scale(a, dot(w, a)));
                        dot(r, e2).atan2(dot(r, e1))
                    })
                    .collect();
                Inner::Cylinder { center, a, e1, e2, radius, theta0: circular_mean(&angles) }
            }
            SurfaceKind::Cone { apex, axis, tan_half_angle } => {
                let a = normalize(axis);
                let e1 = any_perp(a);
                let e2 = cross(a, e1);
                let sin_a = tan_half_angle / (1.0 + tan_half_angle * tan_half_angle).sqrt();
                let angles: Vec<f64> = pts
                    .iter()
                    .map(|&p| {
                        let w = sub(p, apex);
                        let r = sub(w, scale(a, dot(w, a)));
                        dot(r, e2).atan2(dot(r, e1))
                    })
                    .collect();
                Inner::Cone { apex, a, e1, e2, tan: tan_half_angle, sin_a, theta0: circular_mean(&angles) }
            }
            SurfaceKind::Torus { center, axis, major_radius, minor_radius } => {
                let a = normalize(axis);
                let e1 = any_perp(a);
                let e2 = cross(a, e1);
                let mut th = Vec::with_capacity(pts.len());
                let mut ph = Vec::with_capacity(pts.len());
                for &p in pts {
                    let w = sub(p, center);
                    let z = dot(w, a);
                    let planar = sub(w, scale(a, z));
                    let rho = norm(planar);
                    th.push(dot(planar, e2).atan2(dot(planar, e1)));
                    ph.push(z.atan2(rho - major_radius));
                }
                Inner::Torus {
                    center,
                    a,
                    e1,
                    e2,
                    major: major_radius,
                    minor: minor_radius,
                    theta0: circular_mean(&th),
                    phi0: circular_mean(&ph),
                }
            }
        };
        Some(Chart { kind: kind.clone(), inner })
    }
}

impl SurfaceChart for Chart {
    fn project(&self, p: V3) -> V3 {
        closest_on_surface(&self.kind, p)
    }

    fn to_uv(&self, p: V3) -> P2 {
        match &self.inner {
            Inner::Sphere { center, radius, c, e1, e2 } => {
                let d = normalize(sub(p, *center));
                let cosang = dot(d, *c).clamp(-1.0, 1.0);
                let ang = cosang.acos();
                let t = sub(d, scale(*c, cosang));
                let tl = norm(t);
                if tl < 1e-15 {
                    return [0.0, 0.0];
                }
                let dirt = scale(t, 1.0 / tl);
                let arc = radius * ang;
                [arc * dot(dirt, *e1), arc * dot(dirt, *e2)]
            }
            Inner::Cylinder { center, a, e1, e2, radius, theta0 } => {
                let w = sub(p, *center);
                let z = dot(w, *a);
                let r = sub(w, scale(*a, z));
                let theta = unwrap(dot(r, *e2).atan2(dot(r, *e1)), *theta0);
                [radius * theta, z]
            }
            Inner::Cone { apex, a, e1, e2, tan, sin_a, theta0 } => {
                let w = sub(p, *apex);
                let h_ax = dot(w, *a).max(0.0);
                let r = sub(w, scale(*a, h_ax));
                let theta = unwrap(dot(r, *e2).atan2(dot(r, *e1)), *theta0);
                let slant = h_ax * (1.0 + tan * tan).sqrt();
                let phi = theta * sin_a;
                [slant * phi.cos(), slant * phi.sin()]
            }
            Inner::Torus { center, a, e1, e2, major, minor, theta0, phi0 } => {
                let w = sub(p, *center);
                let z = dot(w, *a);
                let planar = sub(w, scale(*a, z));
                let rho = norm(planar);
                let theta = unwrap(dot(planar, *e2).atan2(dot(planar, *e1)), *theta0);
                let phi = unwrap(z.atan2(rho - major), *phi0);
                [major * theta, minor * phi]
            }
        }
    }

    fn to_xyz(&self, uv: P2) -> V3 {
        match &self.inner {
            Inner::Sphere { center, radius, c, e1, e2 } => {
                let rho = (uv[0] * uv[0] + uv[1] * uv[1]).sqrt();
                let ang = rho / radius;
                if rho < 1e-15 {
                    return add(*center, scale(*c, *radius));
                }
                let dirt = add(scale(*e1, uv[0] / rho), scale(*e2, uv[1] / rho));
                let d = add(scale(*c, ang.cos()), scale(dirt, ang.sin()));
                add(*center, scale(d, *radius))
            }
            Inner::Cylinder { center, a, e1, e2, radius, .. } => {
                let theta = uv[0] / radius;
                let dir = add(scale(*e1, theta.cos()), scale(*e2, theta.sin()));
                add(add(*center, scale(*a, uv[1])), scale(dir, *radius))
            }
            Inner::Cone { apex, a, e1, e2, tan, sin_a, .. } => {
                let slant = (uv[0] * uv[0] + uv[1] * uv[1]).sqrt();
                let phi = uv[1].atan2(uv[0]);
                let theta = phi / sin_a;
                let h_ax = slant / (1.0 + tan * tan).sqrt();
                let r = h_ax * tan;
                let dir = add(scale(*e1, theta.cos()), scale(*e2, theta.sin()));
                add(add(*apex, scale(*a, h_ax)), scale(dir, r))
            }
            Inner::Torus { center, a, e1, e2, major, minor, .. } => {
                let theta = uv[0] / major;
                let phi = uv[1] / minor;
                let pdir = add(scale(*e1, theta.cos()), scale(*e2, theta.sin()));
                let tube_center = add(*center, scale(pdir, *major));
                let off = add(scale(pdir, phi.cos() * minor), scale(*a, phi.sin() * minor));
                add(tube_center, off)
            }
        }
    }

    fn curvature_radius(&self, uv: P2) -> f64 {
        match &self.inner {
            Inner::Sphere { radius, .. } => *radius,
            Inner::Cylinder { radius, .. } => *radius,
            Inner::Cone { tan, .. } => {
                // Perpendicular distance to the axis at this slant position; the
                // cone is flat along the generator, so this is the tightest radius.
                let slant = (uv[0] * uv[0] + uv[1] * uv[1]).sqrt();
                let h_ax = slant / (1.0 + tan * tan).sqrt();
                (h_ax * tan).max(1e-12)
            }
            Inner::Torus { minor, .. } => *minor,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surfchart::SurfaceChart;

    fn dist(a: V3, b: V3) -> f64 {
        norm(sub(a, b))
    }

    /// A chart round-trips every group point: `to_xyz(to_uv(p)) == p` (the point
    /// is already on the surface), and chart distance from the center matches
    /// surface (geodesic) distance for the sphere.
    #[test]
    fn sphere_chart_roundtrips_and_is_equidistant() {
        let k = SurfaceKind::Sphere { center: [1.0, 2.0, 3.0], radius: 2.0 };
        // a cap of points around the +x direction
        let mut pts = Vec::new();
        for i in 0..6 {
            for j in 0..6 {
                let th = 0.6 * (i as f64 / 5.0 - 0.5);
                let ph = 0.6 * (j as f64 / 5.0 - 0.5);
                let d = normalize([th.cos() * ph.cos(), th.sin(), ph.sin()]);
                pts.push(add([1.0, 2.0, 3.0], scale(d, 2.0)));
            }
        }
        let chart = Chart::new(&k, &pts).unwrap();
        for &p in &pts {
            let q = chart.to_xyz(chart.to_uv(p));
            assert!(dist(p, q) < 1e-9, "roundtrip {p:?} -> {q:?}");
        }
        // Equidistance: chart radius of a point equals its arc length from the
        // chart center direction (R * angular distance).
        let center = [1.0, 2.0, 3.0];
        let p = add(center, scale(normalize([1.0, 0.3, 0.0]), 2.0));
        let uv = chart.to_uv(p);
        let chart_r = (uv[0] * uv[0] + uv[1] * uv[1]).sqrt();
        let dir_c = normalize(sub(chart.to_xyz([0.0, 0.0]), center));
        let arc = 2.0 * dot(normalize(sub(p, center)), dir_c).clamp(-1.0, 1.0).acos();
        assert!((chart_r - arc).abs() < 1e-9, "equidistant: chart_r {chart_r} vs arc {arc}");
    }

    #[test]
    fn cylinder_chart_roundtrips() {
        let k = SurfaceKind::Cylinder { center: [0.0, 0.0, 0.0], axis: [0.0, 0.0, 1.0], radius: 1.5 };
        let mut pts = Vec::new();
        for i in 0..6 {
            for j in 0..6 {
                let th = 1.2 * (i as f64 / 5.0 - 0.5);
                let z = j as f64 / 5.0 * 3.0;
                pts.push([1.5 * th.cos(), 1.5 * th.sin(), z]);
            }
        }
        let chart = Chart::new(&k, &pts).unwrap();
        for &p in &pts {
            let q = chart.to_xyz(chart.to_uv(p));
            assert!(dist(p, q) < 1e-9, "roundtrip {p:?} -> {q:?}");
        }
    }

    #[test]
    fn cone_chart_roundtrips() {
        let k = SurfaceKind::Cone { apex: [0.0, 0.0, 0.0], axis: [0.0, 0.0, 1.0], tan_half_angle: 0.5 };
        let mut pts = Vec::new();
        for i in 0..6 {
            for j in 1..6 {
                let th = 1.0 * (i as f64 / 5.0 - 0.5);
                let h = j as f64 / 5.0 * 2.0;
                let r = h * 0.5;
                pts.push([r * th.cos(), r * th.sin(), h]);
            }
        }
        let chart = Chart::new(&k, &pts).unwrap();
        for &p in &pts {
            let q = chart.to_xyz(chart.to_uv(p));
            assert!(dist(p, q) < 1e-9, "roundtrip {p:?} -> {q:?}");
        }
    }

    #[test]
    fn torus_chart_roundtrips() {
        let k = SurfaceKind::Torus {
            center: [0.0, 0.0, 0.0],
            axis: [0.0, 0.0, 1.0],
            major_radius: 3.0,
            minor_radius: 1.0,
        };
        let mut pts = Vec::new();
        for i in 0..6 {
            for j in 0..6 {
                let th = 1.0 * (i as f64 / 5.0 - 0.5);
                let ph = 1.0 * (j as f64 / 5.0 - 0.5);
                let pdir = [th.cos(), th.sin(), 0.0];
                let tc = scale(pdir, 3.0);
                let off = add(scale(pdir, ph.cos()), [0.0, 0.0, ph.sin()]);
                pts.push(add(tc, off));
            }
        }
        let chart = Chart::new(&k, &pts).unwrap();
        for &p in &pts {
            let q = chart.to_xyz(chart.to_uv(p));
            assert!(dist(p, q) < 1e-9, "roundtrip {p:?} -> {q:?}");
        }
    }
}

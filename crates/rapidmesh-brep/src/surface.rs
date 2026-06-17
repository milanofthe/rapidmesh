//! The unified surface interface.
//!
//! Every B-rep face references one [`Surface`]. It is a self-contained enum --
//! each variant carries the full parameters it needs to evaluate and invert,
//! including the frame a plane lacks in [`SurfaceKind`] -- with a single interface
//! (`eval_uv` / `project_uv` / `normal` / `curvature_radius` / `exact_plane`). The
//! builder and the mesher branch ONCE here; everything downstream is
//! representation-agnostic. Analytic primitives keep exact carriers and cheap,
//! robust closed-form maps; a NURBS face is one more variant behind the same
//! interface. The exact CSG stays a separate, untouched layer.

use rapidmesh_geom::nurbs::NurbsCurve;
use rapidmesh_geom::{NurbsSurface, SurfaceKind};
use std::sync::Arc;

type V3 = [f64; 3];
type P2 = [f64; 2];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add3(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn norm(a: V3) -> V3 {
    let l = dot(a, a).sqrt();
    if l > 0.0 {
        scale(a, 1.0 / l)
    } else {
        a
    }
}
/// An arbitrary unit vector perpendicular to `a` (a reference for `theta = 0`).
fn perp(a: V3) -> V3 {
    let t = if a[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    norm(cross(a, t))
}

/// A trimmed surface's underlying geometry, with one parameter-map interface.
#[derive(Debug, Clone)]
pub enum Surface {
    /// Plane with orthonormal frame `(o; u, v)` and `normal = u x v`.
    Plane { o: V3, u: V3, v: V3, normal: V3 },
    /// Cylinder: `(theta, h)`, `theta` from `x` about `axis`, `h` along `axis`.
    Cylinder { center: V3, axis: V3, x: V3, radius: f64 },
    /// Sphere: `(theta, phi)`, `phi` the latitude about `z`.
    Sphere { center: V3, x: V3, z: V3, radius: f64 },
    /// Cone: `(theta, s)`, `s` the signed distance from `apex` along `axis`.
    Cone { apex: V3, axis: V3, x: V3, half_angle: f64 },
    /// Torus: `(theta_major, phi_minor)`.
    Torus { center: V3, axis: V3, x: V3, major: f64, minor: f64 },
    /// Extruded profile: `(t, h)`, `profile(t)` in the `(u, v)` plane, `h` along `axis`.
    Extruded { base: V3, u: V3, v: V3, axis: V3, profile: Arc<NurbsCurve> },
    /// A trimmed NURBS surface, mapped by its own `(u, v)`.
    Nurbs(Arc<NurbsSurface>),
}

impl Surface {
    /// Builds the surface from a CSG [`SurfaceKind`]. `frame_pts` are the face's
    /// ordered boundary points, needed ONLY to fit a plane's frame (every other
    /// kind is self-contained from its parameters).
    pub fn from_kind(kind: &SurfaceKind, frame_pts: &[V3]) -> Surface {
        match kind {
            SurfaceKind::Plane => {
                let (o, u, v) = fit_plane(frame_pts);
                Surface::Plane { o, u, v, normal: norm(cross(u, v)) }
            }
            SurfaceKind::Cylinder { center, axis, radius } => {
                let a = norm(*axis);
                Surface::Cylinder { center: *center, axis: a, x: perp(a), radius: *radius }
            }
            SurfaceKind::Sphere { center, radius } => {
                let z = [0.0, 0.0, 1.0];
                Surface::Sphere { center: *center, x: perp(z), z, radius: *radius }
            }
            SurfaceKind::Cone { apex, axis, tan_half_angle } => {
                let a = norm(*axis);
                Surface::Cone { apex: *apex, axis: a, x: perp(a), half_angle: tan_half_angle.atan() }
            }
            SurfaceKind::Torus { center, axis, major_radius, minor_radius } => {
                let a = norm(*axis);
                Surface::Torus {
                    center: *center,
                    axis: a,
                    x: perp(a),
                    major: *major_radius,
                    minor: *minor_radius,
                }
            }
            SurfaceKind::Extruded { profile, base, udir, vdir, axis } => Surface::Extruded {
                base: *base,
                u: norm(*udir),
                v: norm(*vdir),
                axis: norm(*axis),
                profile: profile.clone(),
            },
        }
    }

    /// Parameter point `(u, v)` -> 3D.
    pub fn eval_uv(&self, p: P2) -> V3 {
        match self {
            Surface::Plane { o, u, v, .. } => add3(*o, add3(scale(*u, p[0]), scale(*v, p[1]))),
            Surface::Cylinder { center, axis, x, radius } => {
                let y = cross(*axis, *x);
                let r = add3(scale(*x, p[0].cos()), scale(y, p[0].sin()));
                add3(add3(*center, scale(*axis, p[1])), scale(r, *radius))
            }
            Surface::Sphere { center, x, z, radius } => {
                let y = cross(*z, *x);
                let eq = add3(scale(*x, p[0].cos()), scale(y, p[0].sin()));
                let dir = add3(scale(eq, p[1].cos()), scale(*z, p[1].sin()));
                add3(*center, scale(dir, *radius))
            }
            Surface::Cone { apex, axis, x, half_angle } => {
                let y = cross(*axis, *x);
                let rho = p[1] * half_angle.tan();
                let r = add3(scale(*x, p[0].cos()), scale(y, p[0].sin()));
                add3(add3(*apex, scale(*axis, p[1])), scale(r, rho))
            }
            Surface::Torus { center, axis, x, major, minor } => {
                let y = cross(*axis, *x);
                let dir = add3(scale(*x, p[0].cos()), scale(y, p[0].sin()));
                let ring = scale(dir, major + minor * p[1].cos());
                add3(add3(*center, ring), scale(*axis, minor * p[1].sin()))
            }
            Surface::Extruded { base, u, v, axis, profile } => {
                let c = profile.eval(p[0]);
                add3(add3(*base, scale(*axis, p[1])), add3(scale(*u, c[0]), scale(*v, c[1])))
            }
            Surface::Nurbs(s) => s.eval(p[0], p[1]),
        }
    }

    /// 3D point -> parameter `(u, v)` (closest point on the surface).
    pub fn project_uv(&self, p: V3) -> P2 {
        match self {
            Surface::Plane { o, u, v, .. } => [dot(sub(p, *o), *u), dot(sub(p, *o), *v)],
            Surface::Cylinder { center, axis, x, .. } => {
                let y = cross(*axis, *x);
                let d = sub(p, *center);
                let h = dot(d, *axis);
                let rd = sub(d, scale(*axis, h));
                [dot(rd, y).atan2(dot(rd, *x)), h]
            }
            Surface::Sphere { center, x, z, .. } => {
                let y = cross(*z, *x);
                let d = norm(sub(p, *center));
                [dot(d, y).atan2(dot(d, *x)), dot(d, *z).clamp(-1.0, 1.0).asin()]
            }
            Surface::Cone { apex, axis, x, .. } => {
                let y = cross(*axis, *x);
                let d = sub(p, *apex);
                let s = dot(d, *axis);
                let rd = sub(d, scale(*axis, s));
                [dot(rd, y).atan2(dot(rd, *x)), s]
            }
            Surface::Torus { center, axis, x, major, .. } => {
                let y = cross(*axis, *x);
                let d = sub(p, *center);
                let zc = dot(d, *axis);
                let pd = sub(d, scale(*axis, zc));
                let theta = dot(pd, y).atan2(dot(pd, *x));
                let rho = dot(pd, pd).sqrt() - major;
                [theta, zc.atan2(rho)]
            }
            Surface::Extruded { base, u, v, axis, profile } => {
                let rel = sub(p, *base);
                [profile_footpoint(profile, [dot(rel, *u), dot(rel, *v)]), dot(rel, *axis)]
            }
            Surface::Nurbs(s) => nurbs_footpoint(s, p),
        }
    }

    /// Outward unit normal at parameter `(u, v)`.
    pub fn normal(&self, p: P2) -> V3 {
        match self {
            Surface::Plane { normal, .. } => *normal,
            Surface::Cylinder { axis, x, .. } => {
                let y = cross(*axis, *x);
                add3(scale(*x, p[0].cos()), scale(y, p[0].sin()))
            }
            Surface::Sphere { center, .. } => norm(sub(self.eval_uv(p), *center)),
            Surface::Cone { axis, x, half_angle, .. } => {
                let y = cross(*axis, *x);
                let radial = add3(scale(*x, p[0].cos()), scale(y, p[0].sin()));
                norm(sub(scale(radial, half_angle.cos()), scale(*axis, half_angle.sin())))
            }
            Surface::Torus { center, axis, x, major, .. } => {
                let y = cross(*axis, *x);
                let dir = add3(scale(*x, p[0].cos()), scale(y, p[0].sin()));
                let tube_center = add3(*center, scale(dir, *major));
                norm(sub(self.eval_uv(p), tube_center))
            }
            Surface::Extruded { u, v, axis, profile, .. } => {
                let (_, c1, _) = profile.ders2(p[0]);
                let tangent = add3(scale(*u, c1[0]), scale(*v, c1[1]));
                norm(cross(tangent, *axis))
            }
            Surface::Nurbs(s) => {
                // central difference of eval (analytic derivatives land later)
                let (ud, vd) = s.domain();
                let du = (ud[1] - ud[0]) * 1e-5;
                let dv = (vd[1] - vd[0]) * 1e-5;
                let su = sub(s.eval(p[0] + du, p[1]), s.eval(p[0] - du, p[1]));
                let sv = sub(s.eval(p[0], p[1] + dv), s.eval(p[0], p[1] - dv));
                norm(cross(su, sv))
            }
        }
    }

    /// Smallest principal radius of curvature at `(u, v)` (`INFINITY` where flat),
    /// the input to the sagitta size bound.
    pub fn curvature_radius(&self, p: P2) -> f64 {
        match self {
            Surface::Plane { .. } => f64::INFINITY,
            Surface::Cylinder { radius, .. } => *radius,
            Surface::Sphere { radius, .. } => *radius,
            Surface::Cone { half_angle, .. } => (p[1] * half_angle.tan()).abs().max(1e-12),
            Surface::Torus { minor, .. } => *minor,
            Surface::Extruded { profile, .. } => {
                let k = profile.curvature(p[0]);
                if k > 1e-12 {
                    1.0 / k
                } else {
                    f64::INFINITY
                }
            }
            Surface::Nurbs(_) => f64::INFINITY, // analytic curvature lands later
        }
    }

    /// The exact carrier plane `(origin, unit normal)` if this surface is planar
    /// -- the hook the mesher uses to keep planar faces on an exact `Point3`
    /// carrier (exact-volume conformity). Curved surfaces have none.
    pub fn exact_plane(&self) -> Option<(V3, V3)> {
        match self {
            Surface::Plane { o, normal, .. } => Some((*o, *normal)),
            _ => None,
        }
    }

    /// The plane's full orthonormal frame `(origin, u, v, normal)` if planar -- the
    /// input to a [`crate`]-side `PlaneChart` so the unified chart-driven path runs
    /// in the SAME in-plane coordinates the surface's own `project_uv` uses.
    pub fn plane_frame(&self) -> Option<(V3, V3, V3, V3)> {
        match self {
            Surface::Plane { o, u, v, normal } => Some((*o, *u, *v, *normal)),
            _ => None,
        }
    }
}

/// Fits an orthonormal plane frame to points: centroid origin, Newell normal, an
/// in-plane `u` from the first significant boundary direction, `v = n x u`.
fn fit_plane(pts: &[V3]) -> (V3, V3, V3) {
    if pts.is_empty() {
        return ([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    }
    let n = pts.len() as f64;
    let o: V3 = std::array::from_fn(|k| pts.iter().map(|p| p[k]).sum::<f64>() / n);
    let mut nrm = [0.0f64; 3];
    for i in 0..pts.len() {
        let a = pts[i];
        let b = pts[(i + 1) % pts.len()];
        nrm[0] += (a[1] - b[1]) * (a[2] + b[2]);
        nrm[1] += (a[2] - b[2]) * (a[0] + b[0]);
        nrm[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    let nrm = norm(nrm);
    let mut u = [1.0, 0.0, 0.0];
    for p in pts {
        let d = sub(*p, o);
        let prp: V3 = std::array::from_fn(|k| d[k] - nrm[k] * dot(d, nrm));
        if dot(prp, prp) > 1e-20 {
            u = norm(prp);
            break;
        }
    }
    (o, u, norm(cross(nrm, u)))
}

/// Parameter on `profile` nearest the 2D point `c` (dense sample).
pub fn profile_footpoint(profile: &NurbsCurve, c: P2) -> f64 {
    let (t0, t1) = profile.domain();
    let n = 512usize;
    let (mut bt, mut bd) = (t0, f64::INFINITY);
    for i in 0..=n {
        let t = t0 + (t1 - t0) * i as f64 / n as f64;
        let q = profile.eval(t);
        let d = (q[0] - c[0]).powi(2) + (q[1] - c[1]).powi(2);
        if d < bd {
            bd = d;
            bt = t;
        }
    }
    bt
}

/// `(u, v)` on a NURBS surface nearest `p` (coarse grid; Newton refine lands later).
fn nurbs_footpoint(surf: &NurbsSurface, p: V3) -> P2 {
    let (ud, vd) = surf.domain();
    let n = 32usize;
    let (mut buv, mut bd) = ([ud[0], vd[0]], f64::INFINITY);
    for i in 0..=n {
        let u = ud[0] + (ud[1] - ud[0]) * i as f64 / n as f64;
        for j in 0..=n {
            let v = vd[0] + (vd[1] - vd[0]) * j as f64 / n as f64;
            let d = dot(sub(surf.eval(u, v), p), sub(surf.eval(u, v), p));
            if d < bd {
                bd = d;
                buv = [u, v];
            }
        }
    }
    buv
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn close(a: V3, b: V3) -> bool {
        sub(a, b).iter().all(|&x| x.abs() < 1e-9)
    }

    #[test]
    fn plane_roundtrip() {
        let s = Surface::from_kind(
            &SurfaceKind::Plane,
            &[[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [2.0, 3.0, 1.0], [0.0, 3.0, 1.0]],
        );
        let p = [1.3, 2.1, 1.0];
        assert!(close(s.eval_uv(s.project_uv(p)), p), "plane eval/project roundtrip");
        assert!(s.exact_plane().is_some());
        assert!(s.curvature_radius([0.0, 0.0]).is_infinite());
    }

    #[test]
    fn cylinder_roundtrip_and_radius() {
        let s = Surface::from_kind(
            &SurfaceKind::Cylinder { center: [0.0, 0.0, 0.0], axis: [0.0, 0.0, 1.0], radius: 2.0 },
            &[],
        );
        // a point exactly on the barrel
        let p = s.eval_uv([0.7, 1.5]);
        assert!((dot([p[0], p[1], 0.0], [p[0], p[1], 0.0]).sqrt() - 2.0).abs() < 1e-9);
        let uv = s.project_uv(p);
        assert!(close(s.eval_uv(uv), p), "cylinder roundtrip");
        assert_eq!(s.curvature_radius([0.0, 0.0]), 2.0);
    }

    #[test]
    fn sphere_roundtrip() {
        let s = Surface::from_kind(
            &SurfaceKind::Sphere { center: [1.0, 0.0, 0.0], radius: 3.0 },
            &[],
        );
        for &(t, f) in &[(0.5, 0.4), (-1.2, -0.8), (PI * 0.5, 0.0)] {
            let p = s.eval_uv([t, f]);
            assert!((sub(p, [1.0, 0.0, 0.0]).iter().map(|x| x * x).sum::<f64>().sqrt() - 3.0).abs() < 1e-9);
            assert!(close(s.eval_uv(s.project_uv(p)), p), "sphere roundtrip");
        }
        assert_eq!(s.curvature_radius([0.0, 0.0]), 3.0);
    }
}

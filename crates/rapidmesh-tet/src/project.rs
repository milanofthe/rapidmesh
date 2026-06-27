//! Closest-point projection onto the analytic surfaces a PLC facet may carry
//! (`rapidmesh_geom::SurfaceKind`). The CVT mesher uses these to keep boundary
//! sites on their true surface while smoothing (WP4/WP6).
//!
//! `SurfaceKind::Plane` has no intrinsic plane data (the facet triangle IS the
//! plane), so planar projection is a separate entry, `closest_on_plane`, which
//! takes the plane explicitly. The curved kinds project analytically by snapping
//! the point onto the surface at its current axial/angular position; the result
//! lies exactly on the analytic surface (geometric accuracy for curved boundaries
//! is a tolerance property, matching the curved-scene fixtures).

use rapidmesh_geom::nurbs::NurbsCurve;
use rapidmesh_geom::vec3::{V3, sub, add, scale, dot, cross, len as norm, normalize};
use rapidmesh_geom::SurfaceKind;

/// Nearest parameter on a 2D curve to point `q` (the footpoint): a coarse scan
/// for a basin, then Newton on `g(t) = (C(t) - q) . C'(t) = 0`.
pub fn curve_footpoint(curve: &NurbsCurve, q: [f64; 2]) -> f64 {
    let (lo, hi) = curve.domain();
    let n = 64usize;
    let mut best_t = lo;
    let mut best_d2 = f64::INFINITY;
    for i in 0..=n {
        let t = lo + (hi - lo) * i as f64 / n as f64;
        let c = curve.eval(t);
        let d2 = (c[0] - q[0]).powi(2) + (c[1] - q[1]).powi(2);
        if d2 < best_d2 {
            best_d2 = d2;
            best_t = t;
        }
    }
    let mut t = best_t;
    for _ in 0..24 {
        let (c, d1, d2) = curve.ders2(t);
        let r = [c[0] - q[0], c[1] - q[1]];
        let g = r[0] * d1[0] + r[1] * d1[1];
        let gp = d1[0] * d1[0] + d1[1] * d1[1] + r[0] * d2[0] + r[1] * d2[1];
        if gp.abs() < 1e-15 {
            break;
        }
        let step = g / gp;
        t = (t - step).clamp(lo, hi);
        if step.abs() < 1e-13 {
            break;
        }
    }
    t
}

/// Some unit vector perpendicular to `a` (for degenerate on-axis cases).
fn any_perp(a: V3) -> V3 {
    let t = if a[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    normalize(cross(a, t))
}

/// Closest point on the plane through `origin` with (not necessarily unit)
/// `normal`.
pub fn closest_on_plane(p: V3, origin: V3, normal: V3) -> V3 {
    let n = normalize(normal);
    let d = dot(sub(p, origin), n);
    sub(p, scale(n, d))
}

/// Closest point on the analytic surface. `Plane` returns `p` unchanged (use
/// `closest_on_plane` with the facet plane instead). Degenerate placements
/// (point on the axis of a cylinder/cone, on the torus axis) fall back to a
/// well-defined arbitrary perpendicular direction.
pub fn closest_on_surface(kind: &SurfaceKind, p: V3) -> V3 {
    match *kind {
        SurfaceKind::Plane => p,
        SurfaceKind::Sphere { center, radius } => {
            let r = sub(p, center);
            let l = norm(r);
            if l == 0.0 {
                add(center, [radius, 0.0, 0.0])
            } else {
                add(center, scale(r, radius / l))
            }
        }
        SurfaceKind::Cylinder { center, axis, radius } => {
            let a = normalize(axis);
            let t = dot(sub(p, center), a);
            let foot = add(center, scale(a, t));
            let radial = sub(p, foot);
            let l = norm(radial);
            let dir = if l == 0.0 { any_perp(a) } else { scale(radial, 1.0 / l) };
            add(foot, scale(dir, radius))
        }
        SurfaceKind::Cone { apex, axis, tan_half_angle } => {
            // Snap radially at the current axial position: at axial distance h
            // (>= 0) the cone radius is h*tan(alpha). Lands exactly on the cone.
            let a = normalize(axis);
            let h = dot(sub(p, apex), a).max(0.0);
            let axis_pt = add(apex, scale(a, h));
            let radial = sub(p, axis_pt);
            let l = norm(radial);
            let dir = if l == 0.0 { any_perp(a) } else { scale(radial, 1.0 / l) };
            add(axis_pt, scale(dir, h * tan_half_angle))
        }
        SurfaceKind::Torus { center, axis, major_radius, minor_radius } => {
            let a = normalize(axis);
            let z = dot(sub(p, center), a);
            let planar = sub(sub(p, center), scale(a, z));
            let rho = norm(planar);
            let pdir = if rho == 0.0 { any_perp(a) } else { scale(planar, 1.0 / rho) };
            // Nearest point on the tube-center (major) circle.
            let tube_center = add(center, scale(pdir, major_radius));
            let r = sub(p, tube_center);
            let l = norm(r);
            let dir = if l == 0.0 { pdir } else { scale(r, 1.0 / l) };
            add(tube_center, scale(dir, minor_radius))
        }
        SurfaceKind::Extruded { ref profile, base, udir, vdir, axis } => {
            // Keep the axial coordinate; snap the in-plane part to the curve
            // footpoint. Lands exactly on the analytic extruded surface.
            let (u, v, a) = (normalize(udir), normalize(vdir), normalize(axis));
            let h = dot(sub(p, base), a);
            let rel = sub(sub(p, base), scale(a, h));
            let q = [dot(rel, u), dot(rel, v)];
            let t = curve_footpoint(profile, q);
            let c = profile.eval(t);
            add(add(base, scale(a, h)), add(scale(u, c[0]), scale(v, c[1])))
        }
    }
}

/// Local tightest principal radius of curvature `R = 1/kappa_max` of the
/// analytic surface at (or nearest) `p`. Drives the curvature/volume-error
/// sizing bias so the VOLUME refines near tightly curved boundaries. `Plane`
/// is flat (infinite radius).
pub fn surface_curvature_radius(kind: &SurfaceKind, p: V3) -> f64 {
    match *kind {
        SurfaceKind::Plane => f64::INFINITY,
        SurfaceKind::Sphere { radius, .. } => radius,
        SurfaceKind::Cylinder { radius, .. } => radius,
        SurfaceKind::Cone { apex, axis, tan_half_angle } => {
            // The tightest radius is the local cross-section radius (the cone is
            // flat along the generator): perpendicular distance to the axis.
            let a = normalize(axis);
            let h = dot(sub(p, apex), a).max(0.0);
            (h * tan_half_angle).max(1e-12)
        }
        SurfaceKind::Torus { minor_radius, .. } => minor_radius,
        SurfaceKind::Extruded { ref profile, base, udir, vdir, axis } => {
            let (u, v, a) = (normalize(udir), normalize(vdir), normalize(axis));
            let h = dot(sub(p, base), a);
            let rel = sub(sub(p, base), scale(a, h));
            let q = [dot(rel, u), dot(rel, v)];
            let t = curve_footpoint(profile, q);
            let k = profile.curvature(t);
            if k > 1e-12 {
                1.0 / k
            } else {
                f64::INFINITY
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on_surface(kind: &SurfaceKind, q: V3, tol: f64) -> bool {
        norm(sub(q, closest_on_surface(kind, q))) < tol
    }

    #[test]
    fn plane_projection_lands_and_is_idempotent() {
        let o = [1.0, 2.0, 3.0];
        let n = [0.0, 0.0, 2.0];
        let p = [5.0, -1.0, 9.0];
        let q = closest_on_plane(p, o, n);
        assert!((q[2] - 3.0).abs() < 1e-12, "on plane z=3");
        let q2 = closest_on_plane(q, o, n);
        assert!(norm(sub(q, q2)) < 1e-12, "idempotent");
    }

    #[test]
    fn sphere_projection() {
        let k = SurfaceKind::Sphere { center: [0.0, 0.0, 0.0], radius: 2.0 };
        let q = closest_on_surface(&k, [3.0, 4.0, 0.0]);
        assert!((norm(q) - 2.0).abs() < 1e-12);
        assert!(on_surface(&k, q, 1e-12));
        assert!(norm(sub(q, closest_on_surface(&k, q))) < 1e-12, "idempotent");
    }

    #[test]
    fn cylinder_projection() {
        let k = SurfaceKind::Cylinder { center: [0.0, 0.0, 0.0], axis: [0.0, 0.0, 5.0], radius: 1.0 };
        let q = closest_on_surface(&k, [3.0, 0.0, 7.0]);
        // radial distance to z-axis is the radius, axial position preserved.
        assert!(((q[0] * q[0] + q[1] * q[1]).sqrt() - 1.0).abs() < 1e-12);
        assert!((q[2] - 7.0).abs() < 1e-12);
        assert!(norm(sub(q, closest_on_surface(&k, q))) < 1e-12, "idempotent");
    }

    #[test]
    fn cone_projection() {
        // 45 degree cone from origin along +z.
        let k = SurfaceKind::Cone { apex: [0.0, 0.0, 0.0], axis: [0.0, 0.0, 1.0], tan_half_angle: 1.0 };
        let q = closest_on_surface(&k, [3.0, 0.0, 2.0]);
        let rho = (q[0] * q[0] + q[1] * q[1]).sqrt();
        assert!((rho - q[2]).abs() < 1e-12, "on 45deg cone: radius == axial");
        assert!(norm(sub(q, closest_on_surface(&k, q))) < 1e-9, "idempotent");
    }

    #[test]
    fn torus_projection() {
        let k = SurfaceKind::Torus {
            center: [0.0, 0.0, 0.0],
            axis: [0.0, 0.0, 1.0],
            major_radius: 3.0,
            minor_radius: 1.0,
        };
        let q = closest_on_surface(&k, [5.0, 0.0, 0.5]);
        // distance from the tube center circle (radius 3 in z=0 plane) equals 1.
        let z = q[2];
        let rho = (q[0] * q[0] + q[1] * q[1]).sqrt();
        let d = ((rho - 3.0) * (rho - 3.0) + z * z).sqrt();
        assert!((d - 1.0).abs() < 1e-12, "on tube radius 1");
        assert!(norm(sub(q, closest_on_surface(&k, q))) < 1e-9, "idempotent");
    }
}

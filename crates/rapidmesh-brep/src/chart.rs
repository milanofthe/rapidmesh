//! Per-face parameter map (chart): `(u,v) <-> 3D`, for building PCurves and for
//! meshing/trimming a face in its parameter space.
//!
//! This is the holistic home of the surface parametrisation -- the geometry
//! layer, not the mesher. A `Chart` is built for one face from its [`Surface`]
//! plus, for a plane (which carries no frame of its own), the face's boundary
//! points. Analytic kinds use their own parameters; a NURBS face maps by its
//! `(u,v)` directly.

use crate::Surface;
use rapidmesh_geom::nurbs::NurbsCurve;
use rapidmesh_geom::{NurbsSurface, SurfaceKind};
use std::sync::Arc;

type V3 = [f64; 3];
type P2 = [f64; 2];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: V3) -> V3 {
    let l = dot(a, a).sqrt();
    if l > 0.0 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        a
    }
}

/// A face parameter map.
pub enum Chart {
    /// A plane with an orthonormal frame `(o; u, v)` fitted to the face.
    Plane { o: V3, u: V3, v: V3 },
    /// An extruded profile: `(t, h)` -> `base + a*h + u*profile(t).x + v*profile(t).y`.
    Extruded { base: V3, u: V3, v: V3, a: V3, profile: Arc<NurbsCurve> },
    /// A NURBS surface, mapped by its own `(u, v)`.
    Nurbs { surf: Arc<NurbsSurface> },
}

impl Chart {
    /// Builds the chart for a face. `boundary` are ordered face-boundary points,
    /// used only to fit a plane's frame (other kinds are self-contained).
    pub fn build(surface: &Surface, boundary: &[V3]) -> Chart {
        match surface {
            Surface::Nurbs(s) => Chart::Nurbs { surf: s.clone() },
            Surface::Analytic(SurfaceKind::Extruded { profile, base, udir, vdir, axis }) => {
                Chart::Extruded {
                    base: *base,
                    u: norm(*udir),
                    v: norm(*vdir),
                    a: norm(*axis),
                    profile: profile.clone(),
                }
            }
            // Plane and (for now) every other analytic kind fit a plane to the
            // face boundary; the curved analytic charts (cylinder/sphere/cone/
            // torus) land when the mesher is wired and surfchart is retired.
            Surface::Analytic(_) => {
                let (o, u, v) = fit_plane(boundary);
                Chart::Plane { o, u, v }
            }
        }
    }

    /// 3D point -> `(u, v)`.
    pub fn to_uv(&self, p: V3) -> P2 {
        match self {
            Chart::Plane { o, u, v } => [dot(sub(p, *o), *u), dot(sub(p, *o), *v)],
            Chart::Extruded { base, u, v, a, profile } => {
                let rel = sub(p, *base);
                [profile_footpoint(profile, [dot(rel, *u), dot(rel, *v)]), dot(rel, *a)]
            }
            Chart::Nurbs { surf } => nurbs_footpoint(surf, p),
        }
    }

    /// `(u, v)` -> 3D point.
    pub fn to_xyz(&self, uv: P2) -> V3 {
        match self {
            Chart::Plane { o, u, v } => std::array::from_fn(|k| o[k] + u[k] * uv[0] + v[k] * uv[1]),
            Chart::Extruded { base, u, v, a, profile } => {
                let c = profile.eval(uv[0]);
                std::array::from_fn(|k| base[k] + a[k] * uv[1] + u[k] * c[0] + v[k] * c[1])
            }
            Chart::Nurbs { surf } => surf.eval(uv[0], uv[1]),
        }
    }
}

/// Fits an orthonormal plane frame to points: centroid origin, Newell normal, and
/// an in-plane `u` from the first significant edge.
fn fit_plane(pts: &[V3]) -> (V3, V3, V3) {
    let n = pts.len().max(1) as f64;
    let o: V3 = std::array::from_fn(|k| pts.iter().map(|p| p[k]).sum::<f64>() / n);
    // Newell normal (robust to non-planarity / vertex order).
    let mut nrm = [0.0f64; 3];
    for i in 0..pts.len() {
        let a = pts[i];
        let b = pts[(i + 1) % pts.len()];
        nrm[0] += (a[1] - b[1]) * (a[2] + b[2]);
        nrm[1] += (a[2] - b[2]) * (a[0] + b[0]);
        nrm[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    let nrm = norm(nrm);
    // u = first boundary direction made perpendicular to the normal.
    let mut u = [1.0, 0.0, 0.0];
    for p in pts {
        let d = sub(*p, o);
        let perp: V3 = std::array::from_fn(|k| d[k] - nrm[k] * dot(d, nrm));
        if dot(perp, perp) > 1e-20 {
            u = norm(perp);
            break;
        }
    }
    let v = norm(cross(nrm, u));
    (o, u, v)
}

/// Parameter on `profile` nearest the 2D point `c` (dense sample).
pub fn profile_footpoint(profile: &NurbsCurve, c: P2) -> f64 {
    let (t0, t1) = profile.domain();
    let n = 512usize;
    let (mut bt, mut bd) = (t0, f64::INFINITY);
    for i in 0..=n {
        let t = t0 + (t1 - t0) * i as f64 / n as f64;
        let p = profile.eval(t);
        let d = (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2);
        if d < bd {
            bd = d;
            bt = t;
        }
    }
    bt
}

/// `(u, v)` on a NURBS surface nearest `p` (coarse grid sample; a Newton refine
/// lands when needed).
fn nurbs_footpoint(surf: &NurbsSurface, p: V3) -> P2 {
    let (ud, vd) = surf.domain();
    let n = 32usize;
    let (mut buv, mut bd) = ([ud[0], vd[0]], f64::INFINITY);
    for i in 0..=n {
        let u = ud[0] + (ud[1] - ud[0]) * i as f64 / n as f64;
        for j in 0..=n {
            let v = vd[0] + (vd[1] - vd[0]) * j as f64 / n as f64;
            let q = surf.eval(u, v);
            let d = dot(sub(q, p), sub(q, p));
            if d < bd {
                bd = d;
                buv = [u, v];
            }
        }
    }
    buv
}

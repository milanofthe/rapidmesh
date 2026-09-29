//! The geometry of a STEP file: placements, curves and surfaces, each with
//! its evaluation and the parameter of a point on it.

use crate::part21::{Exchange, Value};
use rapidmesh_geom::{NurbsSurface, SurfaceKind};
use std::f64::consts::TAU;
use std::sync::Arc;

pub type P3 = [f64; 3];

pub(crate) fn add(a: P3, b: P3) -> P3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub(crate) fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub(crate) fn scale(a: P3, s: f64) -> P3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub(crate) fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub(crate) fn cross(a: P3, b: P3) -> P3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub(crate) fn norm(a: P3) -> f64 {
    dot(a, a).sqrt()
}
pub(crate) fn unit(a: P3) -> P3 {
    let l = norm(a);
    if l > 0.0 {
        scale(a, 1.0 / l)
    } else {
        a
    }
}
pub(crate) fn dist(a: P3, b: P3) -> f64 {
    norm(sub(a, b))
}

/// A right-handed orthonormal frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub o: P3,
    pub x: P3,
    pub y: P3,
    pub z: P3,
}

impl Frame {
    /// The frame at `o` with axis `z` and `x` toward `x_hint` (any
    /// perpendicular direction where there is none).
    pub fn new(o: P3, z: P3, x_hint: Option<P3>) -> Frame {
        let z = unit(z);
        let hint = x_hint.unwrap_or(if z[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        });
        let x = unit(sub(hint, scale(z, dot(hint, z))));
        Frame {
            o,
            x,
            y: cross(z, x),
            z,
        }
    }

    fn at(&self, a: f64, b: f64, c: f64) -> P3 {
        add(
            self.o,
            add(scale(self.x, a), add(scale(self.y, b), scale(self.z, c))),
        )
    }

    fn local(&self, p: P3) -> P3 {
        let d = sub(p, self.o);
        [dot(d, self.x), dot(d, self.y), dot(d, self.z)]
    }
}

/// A rational B-spline curve in space, with its knots expanded.
#[derive(Clone, Debug)]
pub struct Spline {
    pub degree: usize,
    pub knots: Vec<f64>,
    pub ctrl: Vec<P3>,
    pub weights: Vec<f64>,
}

impl Spline {
    pub fn domain(&self) -> (f64, f64) {
        (
            self.knots[self.degree],
            self.knots[self.knots.len() - self.degree - 1],
        )
    }

    /// De Boor's algorithm in homogeneous coordinates.
    pub fn eval(&self, t: f64) -> P3 {
        let p = self.degree;
        let (lo, hi) = self.domain();
        let t = t.clamp(lo, hi);
        let n = self.ctrl.len();
        let mut k = p;
        while k + 1 < n && self.knots[k + 1] <= t {
            k += 1;
        }
        let mut d: Vec<[f64; 4]> = (0..=p)
            .map(|j| {
                let i = k + j - p;
                let w = self.weights[i];
                let c = self.ctrl[i];
                [c[0] * w, c[1] * w, c[2] * w, w]
            })
            .collect();
        for r in 1..=p {
            for j in (r..=p).rev() {
                let i = k + j - p;
                let den = self.knots[i + p + 1 - r] - self.knots[i];
                let a = if den > 0.0 {
                    (t - self.knots[i]) / den
                } else {
                    0.0
                };
                for c in 0..4 {
                    d[j][c] = (1.0 - a) * d[j - 1][c] + a * d[j][c];
                }
            }
        }
        let h = d[p];
        [h[0] / h[3], h[1] / h[3], h[2] / h[3]]
    }
}

/// A curve of the file.
#[derive(Clone, Debug)]
pub enum Curve {
    Line { p: P3, d: P3 },
    Circle { f: Frame, r: f64 },
    Ellipse { f: Frame, a: f64, b: f64 },
    Spline(Spline),
}

impl Curve {
    pub fn eval(&self, t: f64) -> P3 {
        match self {
            Curve::Line { p, d } => add(*p, scale(*d, t)),
            Curve::Circle { f, r } => f.at(r * t.cos(), r * t.sin(), 0.0),
            Curve::Ellipse { f, a, b } => f.at(a * t.cos(), b * t.sin(), 0.0),
            Curve::Spline(s) => s.eval(t),
        }
    }

    /// The period of a closed curve: 2 pi for the conics, the domain of a
    /// B-spline whose ends coincide.
    pub fn period(&self) -> Option<f64> {
        match self {
            Curve::Circle { .. } | Curve::Ellipse { .. } => Some(TAU),
            Curve::Line { .. } => None,
            Curve::Spline(s) => {
                let (lo, hi) = s.domain();
                let size = s
                    .ctrl
                    .iter()
                    .fold(0.0f64, |m, c| m.max(norm(sub(*c, s.ctrl[0]))))
                    .max(1e-300);
                (dist(s.eval(lo), s.eval(hi)) <= 1e-9 * size).then_some(hi - lo)
            }
        }
    }

    /// The parameter of the point of the curve nearest `q`.
    pub fn param(&self, q: P3) -> f64 {
        match self {
            Curve::Line { p, d } => dot(sub(q, *p), *d) / dot(*d, *d),
            Curve::Circle { f, .. } => {
                let l = f.local(q);
                l[1].atan2(l[0])
            }
            Curve::Ellipse { f, a, b } => {
                let l = f.local(q);
                // The angle of the point, refined by Newton on the distance.
                let mut t = (l[1] / b).atan2(l[0] / a);
                for _ in 0..20 {
                    let (c, s) = (t.cos(), t.sin());
                    let (x, y) = (a * c - l[0], b * s - l[1]);
                    let g = -x * a * s + y * b * c;
                    let h = a * a * s * s + b * b * c * c - x * a * c - y * b * s;
                    if h.abs() < 1e-300 {
                        break;
                    }
                    let step = g / h;
                    t -= step;
                    if step.abs() < 1e-14 {
                        break;
                    }
                }
                t
            }
            Curve::Spline(s) => {
                let (lo, hi) = s.domain();
                // The best of a sampling, refined by golden section.
                let n = 16 * s.ctrl.len().max(4);
                let mut best = (lo, f64::INFINITY);
                for i in 0..=n {
                    let t = lo + (hi - lo) * i as f64 / n as f64;
                    let d = dist(s.eval(t), q);
                    if d < best.1 {
                        best = (t, d);
                    }
                }
                let h = (hi - lo) / n as f64;
                let (mut a, mut b) = ((best.0 - h).max(lo), (best.0 + h).min(hi));
                let g = (5f64.sqrt() - 1.0) / 2.0;
                for _ in 0..80 {
                    let (c, d) = (b - g * (b - a), a + g * (b - a));
                    if dist(s.eval(c), q) < dist(s.eval(d), q) {
                        b = d;
                    } else {
                        a = c;
                    }
                }
                0.5 * (a + b)
            }
        }
    }
}

/// A surface of the file.
#[derive(Clone, Debug)]
pub enum Surface {
    Plane(Frame),
    /// `(theta, height)`
    Cylinder(Frame, f64),
    /// `(theta, height)`, the radius `r + height tan(semi)`.
    Cone(Frame, f64, f64),
    /// `(theta, latitude)`
    Sphere(Frame, f64),
    /// `(theta, phi)`: major and minor radius.
    Torus(Frame, f64, f64),
    Spline(Arc<NurbsSurface>),
}

impl Surface {
    pub fn eval(&self, uv: [f64; 2]) -> P3 {
        let [u, v] = uv;
        match self {
            Surface::Plane(f) => f.at(u, v, 0.0),
            Surface::Cylinder(f, r) => f.at(r * u.cos(), r * u.sin(), v),
            Surface::Cone(f, r, semi) => {
                let rr = r + v * semi.tan();
                f.at(rr * u.cos(), rr * u.sin(), v)
            }
            Surface::Sphere(f, r) => {
                f.at(r * v.cos() * u.cos(), r * v.cos() * u.sin(), r * v.sin())
            }
            Surface::Torus(f, big, small) => {
                let rr = big + small * v.cos();
                f.at(rr * u.cos(), rr * u.sin(), small * v.sin())
            }
            Surface::Spline(s) => s.eval(u, v),
        }
    }

    /// The parameters of the point of the surface nearest `p`, the angles
    /// in `(-pi, pi]`.
    pub fn param(&self, p: P3) -> [f64; 2] {
        match self {
            Surface::Plane(f) => {
                let l = f.local(p);
                [l[0], l[1]]
            }
            Surface::Cylinder(f, _) => {
                let l = f.local(p);
                [l[1].atan2(l[0]), l[2]]
            }
            Surface::Cone(f, r, semi) => {
                let l = f.local(p);
                // The nearest point on the meridian line through (r, 0)
                // with slope tan(semi).
                let rho = (l[0] * l[0] + l[1] * l[1]).sqrt();
                let (s, c) = semi.sin_cos();
                let along = (rho - r) * s + l[2] * c;
                [l[1].atan2(l[0]), along * c]
            }
            Surface::Sphere(f, _) => {
                let l = f.local(p);
                let rho = (l[0] * l[0] + l[1] * l[1]).sqrt();
                [l[1].atan2(l[0]), l[2].atan2(rho)]
            }
            Surface::Torus(f, big, _) => {
                let l = f.local(p);
                let rho = (l[0] * l[0] + l[1] * l[1]).sqrt();
                [l[1].atan2(l[0]), l[2].atan2(rho - big)]
            }
            Surface::Spline(s) => s.closest_param(p),
        }
    }

    /// The period of each parameter that wraps around: 2 pi for the
    /// angles, the domain of a B-spline direction whose two ends coincide.
    pub fn periods(&self) -> [Option<f64>; 2] {
        match self {
            Surface::Plane(_) => [None, None],
            Surface::Cylinder(..) | Surface::Cone(..) | Surface::Sphere(..) => [Some(TAU), None],
            Surface::Torus(..) => [Some(TAU), Some(TAU)],
            Surface::Spline(s) => {
                let (du, dv) = s.domain();
                let size = s
                    .ctrl
                    .iter()
                    .fold(0.0f64, |m, c| m.max(norm(sub(*c, s.ctrl[0]))))
                    .max(1e-300);
                let closed = |k: usize| {
                    (0..=8).all(|i| {
                        let t = i as f64 / 8.0;
                        let (a, b) = if k == 0 {
                            let v = dv[0] + t * (dv[1] - dv[0]);
                            (s.eval(du[0], v), s.eval(du[1], v))
                        } else {
                            let u = du[0] + t * (du[1] - du[0]);
                            (s.eval(u, dv[0]), s.eval(u, dv[1]))
                        };
                        dist(a, b) <= 1e-9 * size
                    })
                };
                [
                    closed(0).then_some(du[1] - du[0]),
                    closed(1).then_some(dv[1] - dv[0]),
                ]
            }
        }
    }

    /// The lengths of the derivatives at `uv` (for spacing in parameters).
    pub fn stretch(&self, uv: [f64; 2]) -> [f64; 2] {
        let h = 1e-6;
        let p = self.eval(uv);
        [
            dist(self.eval([uv[0] + h, uv[1]]), p) / h,
            dist(self.eval([uv[0], uv[1] + h]), p) / h,
        ]
        .map(|x| x.max(1e-12))
    }

    /// The smallest radius of curvature (infinite for a plane).
    pub fn radius(&self) -> f64 {
        match self {
            Surface::Plane(_) => f64::INFINITY,
            Surface::Cylinder(_, r) | Surface::Sphere(_, r) => *r,
            Surface::Cone(_, r, _) => r.abs().max(1e-9),
            Surface::Torus(_, _, small) => *small,
            Surface::Spline(_) => f64::INFINITY,
        }
    }

    /// The smallest radius of curvature at `uv` (a B-spline's own there,
    /// the constant one of the others).
    pub fn radius_at(&self, uv: [f64; 2]) -> f64 {
        match self {
            Surface::Spline(s) => s
                .principal_curvatures(uv[0], uv[1])
                .map_or(f64::INFINITY, |k| {
                    1.0 / k[0].abs().max(k[1].abs()).max(1e-300)
                }),
            _ => self.radius(),
        }
    }

    /// The carrier for the model.
    pub fn kind(&self) -> SurfaceKind {
        match self {
            Surface::Plane(_) => SurfaceKind::Plane,
            Surface::Cylinder(f, r) => SurfaceKind::Cylinder {
                center: f.o,
                axis: f.z,
                radius: *r,
            },
            Surface::Cone(f, r, semi) => SurfaceKind::Cone {
                apex: add(f.o, scale(f.z, -r / semi.tan())),
                axis: f.z,
                tan_half_angle: semi.tan(),
            },
            Surface::Sphere(f, r) => SurfaceKind::Sphere {
                center: f.o,
                radius: *r,
            },
            Surface::Torus(f, big, small) => SurfaceKind::Torus {
                center: f.o,
                axis: f.z,
                major_radius: *big,
                minor_radius: *small,
            },
            Surface::Spline(s) => SurfaceKind::Nurbs(s.clone()),
        }
    }
}

/// Reads the geometry of instances of `x`.
pub struct Decoder<'a> {
    pub x: &'a Exchange,
}

fn arg<'v>(args: &'v [Value], i: usize, what: &str, id: u32) -> Result<&'v Value, String> {
    args.get(i)
        .ok_or_else(|| format!("#{id} {what}: missing parameter {i}"))
}

fn num(v: &Value, id: u32) -> Result<f64, String> {
    v.as_f64()
        .ok_or_else(|| format!("#{id}: expected a number"))
}

fn refr(v: &Value, id: u32) -> Result<u32, String> {
    v.as_ref()
        .ok_or_else(|| format!("#{id}: expected a reference"))
}

impl Decoder<'_> {
    fn simple(&self, id: u32) -> Result<(&str, &[Value]), String> {
        let rs = self
            .x
            .instances
            .get(&id)
            .ok_or_else(|| format!("#{id} does not exist"))?;
        match rs.as_slice() {
            [r] => Ok((&r.name, &r.args)),
            _ => Ok(("", &[])),
        }
    }

    fn triple(&self, id: u32, name: &str) -> Result<P3, String> {
        let r = self
            .x
            .record(id, name)
            .ok_or_else(|| format!("#{id} is no {name}"))?;
        let l = arg(&r.args, 1, name, id)?
            .as_list()
            .ok_or_else(|| format!("#{id}: coordinates are no list"))?;
        let c: Vec<f64> = l.iter().map(|v| num(v, id)).collect::<Result<_, _>>()?;
        Ok([
            c.first().copied().unwrap_or(0.0),
            c.get(1).copied().unwrap_or(0.0),
            c.get(2).copied().unwrap_or(0.0),
        ])
    }

    pub fn point(&self, id: u32) -> Result<P3, String> {
        self.triple(id, "CARTESIAN_POINT")
    }

    pub fn direction(&self, id: u32) -> Result<P3, String> {
        self.triple(id, "DIRECTION")
    }

    /// AXIS2_PLACEMENT_3D
    pub fn placement(&self, id: u32) -> Result<Frame, String> {
        let (name, a) = self.simple(id)?;
        if name != "AXIS2_PLACEMENT_3D" {
            return Err(format!("#{id} is no AXIS2_PLACEMENT_3D but {name}"));
        }
        let o = self.point(refr(arg(a, 1, name, id)?, id)?)?;
        let z = match a.get(2).and_then(Value::as_ref) {
            Some(r) => self.direction(r)?,
            None => [0.0, 0.0, 1.0],
        };
        let x = match a.get(3).and_then(Value::as_ref) {
            Some(r) => Some(self.direction(r)?),
            None => None,
        };
        Ok(Frame::new(o, z, x))
    }

    /// The 3D curve of an edge's geometry: through SURFACE_CURVE and
    /// SEAM_CURVE to their curve, and TRIMMED_CURVE to its basis.
    pub fn curve(&self, id: u32) -> Result<Curve, String> {
        if let Some(r) = self.x.record(id, "B_SPLINE_CURVE") {
            return self.spline_curve(id, r.args.as_slice());
        }
        let (name, a) = self.simple(id)?;
        match name {
            "SURFACE_CURVE" | "SEAM_CURVE" | "INTERSECTION_CURVE" => {
                self.curve(refr(arg(a, 1, name, id)?, id)?)
            }
            "TRIMMED_CURVE" => self.curve(refr(arg(a, 1, name, id)?, id)?),
            "LINE" => {
                let p = self.point(refr(arg(a, 1, name, id)?, id)?)?;
                let v = refr(arg(a, 2, name, id)?, id)?;
                let vr = self
                    .x
                    .record(v, "VECTOR")
                    .ok_or_else(|| format!("#{v} is no VECTOR"))?;
                let d = self.direction(refr(arg(&vr.args, 1, "VECTOR", v)?, v)?)?;
                let m = num(arg(&vr.args, 2, "VECTOR", v)?, v)?;
                Ok(Curve::Line {
                    p,
                    d: scale(unit(d), m),
                })
            }
            "CIRCLE" => Ok(Curve::Circle {
                f: self.placement(refr(arg(a, 1, name, id)?, id)?)?,
                r: num(arg(a, 2, name, id)?, id)?,
            }),
            "ELLIPSE" => Ok(Curve::Ellipse {
                f: self.placement(refr(arg(a, 1, name, id)?, id)?)?,
                a: num(arg(a, 2, name, id)?, id)?,
                b: num(arg(a, 3, name, id)?, id)?,
            }),
            "B_SPLINE_CURVE_WITH_KNOTS" => self.spline_curve(id, a),
            other => Err(format!("#{id}: curve {other:?} is not read")),
        }
    }

    /// A B-spline curve, simple (B_SPLINE_CURVE_WITH_KNOTS) or complex
    /// (with RATIONAL_B_SPLINE_CURVE).
    fn spline_curve(&self, id: u32, base: &[Value]) -> Result<Curve, String> {
        // The simple form carries the curve's and the knots' parameters in
        // one record: name, degree, points, form, closed, self-intersect,
        // multiplicities, knots, spec.
        let simple = self.x.kind(id) == Some("B_SPLINE_CURVE_WITH_KNOTS");
        let (degree, ctrl_ids) = if simple {
            (base.get(1), base.get(2))
        } else {
            (base.first(), base.get(1))
        };
        let degree = degree
            .and_then(Value::as_int)
            .ok_or(format!("#{id}: degree"))? as usize;
        let ctrl: Vec<P3> = ctrl_ids
            .and_then(Value::as_list)
            .ok_or(format!("#{id}: control points"))?
            .iter()
            .map(|v| self.point(refr(v, id)?))
            .collect::<Result<_, _>>()?;
        let knots_args: Vec<Value> = if simple {
            base[6..].to_vec()
        } else {
            self.x
                .record(id, "B_SPLINE_CURVE_WITH_KNOTS")
                .ok_or(format!("#{id}: no knots"))?
                .args
                .clone()
        };
        let knots = expand(&knots_args[0], &knots_args[1], id)?;
        let weights = match self.x.record(id, "RATIONAL_B_SPLINE_CURVE") {
            Some(r) => r.args[0]
                .as_list()
                .ok_or(format!("#{id}: weights"))?
                .iter()
                .map(|v| num(v, id))
                .collect::<Result<_, _>>()?,
            None => vec![1.0; ctrl.len()],
        };
        if knots.len() != ctrl.len() + degree + 1 {
            return Err(format!(
                "#{id}: {} knots for {} points of degree {degree}",
                knots.len(),
                ctrl.len()
            ));
        }
        Ok(Curve::Spline(Spline {
            degree,
            knots,
            ctrl,
            weights,
        }))
    }

    pub fn surface(&self, id: u32) -> Result<Surface, String> {
        if self.x.record(id, "B_SPLINE_SURFACE").is_some()
            || self.x.kind(id) == Some("B_SPLINE_SURFACE_WITH_KNOTS")
        {
            return self.spline_surface(id);
        }
        let (name, a) = self.simple(id)?;
        let frame = || self.placement(refr(arg(a, 1, name, id)?, id)?);
        match name {
            "PLANE" => Ok(Surface::Plane(frame()?)),
            "CYLINDRICAL_SURFACE" => {
                Ok(Surface::Cylinder(frame()?, num(arg(a, 2, name, id)?, id)?))
            }
            "CONICAL_SURFACE" => Ok(Surface::Cone(
                frame()?,
                num(arg(a, 2, name, id)?, id)?,
                num(arg(a, 3, name, id)?, id)?,
            )),
            "SPHERICAL_SURFACE" => Ok(Surface::Sphere(frame()?, num(arg(a, 2, name, id)?, id)?)),
            "TOROIDAL_SURFACE" => Ok(Surface::Torus(
                frame()?,
                num(arg(a, 2, name, id)?, id)?,
                num(arg(a, 3, name, id)?, id)?,
            )),
            other => Err(format!("#{id}: surface {other:?} is not read")),
        }
    }

    fn spline_surface(&self, id: u32) -> Result<Surface, String> {
        let simple = self.x.kind(id) == Some("B_SPLINE_SURFACE_WITH_KNOTS");
        let (base, off) = if simple {
            (&self.x.instances[&id][0].args, 1)
        } else {
            (
                &self
                    .x
                    .record(id, "B_SPLINE_SURFACE")
                    .ok_or(format!("#{id}: no B_SPLINE_SURFACE"))?
                    .args,
                0,
            )
        };
        let du = base[off].as_int().ok_or(format!("#{id}: u degree"))? as usize;
        let dv = base[off + 1].as_int().ok_or(format!("#{id}: v degree"))? as usize;
        let rows = base[off + 2]
            .as_list()
            .ok_or(format!("#{id}: control net"))?;
        let mut ctrl = Vec::new();
        let mut nv = 0;
        for row in rows {
            let row = row.as_list().ok_or(format!("#{id}: control row"))?;
            nv = row.len();
            for v in row {
                ctrl.push(self.point(refr(v, id)?)?);
            }
        }
        let nu = rows.len();
        let k: Vec<Value> = if simple {
            base[off + 6..].to_vec()
        } else {
            self.x
                .record(id, "B_SPLINE_SURFACE_WITH_KNOTS")
                .ok_or(format!("#{id}: no knots"))?
                .args
                .clone()
        };
        let ku = expand(&k[0], &k[2], id)?;
        let kv = expand(&k[1], &k[3], id)?;
        let weights: Vec<f64> = match self.x.record(id, "RATIONAL_B_SPLINE_SURFACE") {
            Some(r) => r.args[0]
                .as_list()
                .ok_or(format!("#{id}: weights"))?
                .iter()
                .flat_map(|row| row.as_list().unwrap_or(&[]).iter())
                .map(|v| num(v, id))
                .collect::<Result<_, _>>()?,
            None => vec![1.0; ctrl.len()],
        };
        NurbsSurface::try_new([du, dv], [ku, kv], [nu, nv], ctrl, weights)
            .map(|s| Surface::Spline(Arc::new(s)))
            .map_err(|e| format!("#{id}: {e}"))
    }
}

/// Knots from their distinct values and multiplicities.
fn expand(mults: &Value, values: &Value, id: u32) -> Result<Vec<f64>, String> {
    let m = mults.as_list().ok_or(format!("#{id}: multiplicities"))?;
    let k = values.as_list().ok_or(format!("#{id}: knots"))?;
    let mut out = Vec::new();
    for (m, k) in m.iter().zip(k) {
        let (m, k) = (
            m.as_int().ok_or(format!("#{id}: multiplicity"))?,
            num(k, id)?,
        );
        out.extend(std::iter::repeat_n(k, m.max(0) as usize));
    }
    Ok(out)
}

/// `x` shifted by whole periods into `(ref - period/2, ref + period/2]`.
pub fn near(x: f64, reference: f64, period: f64) -> f64 {
    x - period * ((x - reference + 0.5 * period) / period).floor()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surfaces_invert_their_parameters() {
        let f = Frame::new([1.0, 2.0, 3.0], [0.3, -0.2, 1.0], Some([1.0, 0.0, 0.0]));
        for s in [
            Surface::Plane(f),
            Surface::Cylinder(f, 2.0),
            Surface::Cone(f, 2.0, 0.4),
            Surface::Sphere(f, 2.0),
            Surface::Torus(f, 3.0, 1.0),
        ] {
            for uv in [[0.3, 0.2], [-2.0, 0.5], [2.5, -0.7]] {
                let back = s.param(s.eval(uv));
                assert!(
                    dist(s.eval(back), s.eval(uv)) < 1e-9,
                    "{s:?} {uv:?} {back:?}"
                );
            }
        }
    }

    #[test]
    fn a_spline_with_unit_weights_is_polynomial() {
        // A quadratic Bezier arc.
        let s = Spline {
            degree: 2,
            knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ctrl: vec![[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0]],
            weights: vec![1.0; 3],
        };
        let m = s.eval(0.5);
        assert!(dist(m, [1.0, 1.0, 0.0]) < 1e-12);
        let c = Curve::Spline(s);
        assert!((c.param([1.0, 1.2, 0.0]) - 0.5).abs() < 1e-6);
    }
}

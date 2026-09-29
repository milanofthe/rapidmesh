//! The bodies of a STEP file as faceted solids for the model: every edge
//! sampled once and shared by the faces on it, so each body is closed by
//! construction; every face triangulated in its surface's parameters, the
//! facets carrying the face's surface as their carrier.

use crate::geometry::{dist, near, Curve, Decoder, Surface, P3};
use crate::part21::{Exchange, Value};
use rapidmesh_csg::Tri;
use rapidmesh_exact::{Axis, Point3, Sign};
use rapidmesh_geom::{polygon_orientation, triangulate_polygon, Faceted};
use rustc_hash::FxHashMap;
use std::f64::consts::TAU;

/// A body of the file.
pub struct Body {
    /// Its product's name, else the solid's.
    pub name: String,
    pub solid: Faceted,
}

/// How finely the carriers are faceted.
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    /// The largest distance of a facet from its surface, relative to the
    /// model's size.
    pub chord: f64,
    /// The fewest segments on a full circle.
    pub min_segments: usize,
}

impl Default for Tolerance {
    fn default() -> Tolerance {
        Tolerance {
            chord: 1e-3,
            min_segments: 12,
        }
    }
}

fn refs(v: Option<&Value>) -> Vec<u32> {
    v.and_then(Value::as_list)
        .map(|l| l.iter().filter_map(Value::as_ref).collect())
        .unwrap_or_default()
}

struct Tess<'a> {
    x: &'a Exchange,
    d: Decoder<'a>,
    /// The largest distance of a facet from its surface.
    chord: f64,
    min_segments: usize,
    /// Samples of each edge, from its first vertex to its second.
    edges: FxHashMap<u32, Vec<P3>>,
    /// The vertices of the edges the current body met.
    corners: Vec<P3>,
}

impl Tess<'_> {
    fn args(&self, id: u32, name: &str) -> Result<&[Value], String> {
        self.x
            .record(id, name)
            .map(|r| r.args.as_slice())
            .ok_or_else(|| format!("#{id} is no {name}"))
    }

    fn vertex(&self, id: u32) -> Result<P3, String> {
        let a = self.args(id, "VERTEX_POINT")?;
        self.d.point(
            a.get(1)
                .and_then(Value::as_ref)
                .ok_or(format!("#{id}: no point"))?,
        )
    }

    /// The samples of EDGE_CURVE `id`, from its first vertex to its second.
    fn edge(&mut self, id: u32) -> Result<Vec<P3>, String> {
        if let Some(s) = self.edges.get(&id) {
            let s = s.clone();
            self.corners.push(s[0]);
            self.corners.push(s[s.len() - 1]);
            return Ok(s);
        }
        let a = self.args(id, "EDGE_CURVE")?;
        let r = |i: usize| {
            a.get(i)
                .and_then(Value::as_ref)
                .ok_or(format!("#{id}: parameter {i}"))
        };
        let (p0, p1) = (self.vertex(r(1)?)?, self.vertex(r(2)?)?);
        let curve = self.d.curve(r(3)?)?;
        let forward = a.get(4).and_then(Value::as_bool).unwrap_or(true);
        let closed = r(1)? == r(2)?;
        let mut s = self.sample(&curve, p0, p1, forward, closed);
        s[0] = p0;
        let last = s.len() - 1;
        s[last] = p1;
        self.edges.insert(id, s.clone());
        self.corners.push(p0);
        self.corners.push(p1);
        Ok(s)
    }

    /// Points along `curve` from `p0` to `p1` (the whole curve when
    /// `closed`), running with its parameter or, not `forward`, against it.
    fn sample(&self, curve: &Curve, p0: P3, p1: P3, forward: bool, closed: bool) -> Vec<P3> {
        let t0 = curve.param(p0);
        let mut t1 = curve.param(p1);
        if let Some(period) = curve.period() {
            let sign = if forward { 1.0 } else { -1.0 };
            // Onward from t0 by less than a period, a whole one if closed.
            let mut delta = sign * (t1 - t0);
            delta -= period * (delta / period).floor();
            if closed || delta < 1e-12 {
                delta = period;
            }
            t1 = t0 + sign * delta;
        }
        let n = match curve {
            Curve::Line { .. } => 1,
            Curve::Circle { r, .. } => self.segments(*r, (t1 - t0).abs()),
            Curve::Ellipse { a, b, .. } => self
                .segments(a.min(*b).max(1e-12), (t1 - t0).abs())
                .max(self.segments(a.max(*b), (t1 - t0).abs())),
            Curve::Spline(_) => {
                // Halved where the curve strays from the chord by more than
                // the tolerance, from one span per control point.
                let mut ts: Vec<f64> = vec![t0, t1];
                let first = 2 * match curve {
                    Curve::Spline(s) => s.ctrl.len(),
                    _ => 1,
                };
                ts = (0..=first)
                    .map(|i| t0 + (t1 - t0) * i as f64 / first as f64)
                    .collect();
                let mut i = 0;
                while i + 1 < ts.len() && ts.len() < 4000 {
                    let (a, b) = (ts[i], ts[i + 1]);
                    let (pa, pb, pm) = (curve.eval(a), curve.eval(b), curve.eval(0.5 * (a + b)));
                    let mid = [
                        0.5 * (pa[0] + pb[0]),
                        0.5 * (pa[1] + pb[1]),
                        0.5 * (pa[2] + pb[2]),
                    ];
                    if dist(pm, mid) > self.chord {
                        ts.insert(i + 1, 0.5 * (a + b));
                    } else {
                        i += 1;
                    }
                }
                return ts.into_iter().map(|t| curve.eval(t)).collect();
            }
        };
        (0..=n)
            .map(|i| curve.eval(t0 + (t1 - t0) * i as f64 / n as f64))
            .collect()
    }

    /// Segments on an arc of `angle` at radius `r` within the chord.
    fn segments(&self, r: f64, angle: f64) -> usize {
        let step = 2.0 * (1.0 - (self.chord / r).min(1.0)).acos();
        let full = (TAU / step.max(1e-6)).ceil().max(self.min_segments as f64);
        ((full * angle / TAU).ceil() as usize).max(1)
    }

    /// The boundary of a face bound as a closed ring of points (the loop's
    /// edges in turn, each without its last point).
    fn ring(&mut self, bound: u32) -> Result<Vec<P3>, String> {
        let (loop_id, same) = {
            let a = self
                .x
                .record(bound, "FACE_OUTER_BOUND")
                .or_else(|| self.x.record(bound, "FACE_BOUND"))
                .ok_or(format!("#{bound} is no face bound"))?;
            (
                a.args
                    .get(1)
                    .and_then(Value::as_ref)
                    .ok_or(format!("#{bound}: loop"))?,
                a.args.get(2).and_then(Value::as_bool).unwrap_or(true),
            )
        };
        let mut ring = Vec::new();
        if let Some(v) = self.x.record(loop_id, "VERTEX_LOOP") {
            let v = v
                .args
                .get(1)
                .and_then(Value::as_ref)
                .ok_or(format!("#{loop_id}: vertex"))?;
            ring.push(self.vertex(v)?);
            return Ok(ring);
        }
        let edges = refs(self.args(loop_id, "EDGE_LOOP")?.get(1));
        for oe in edges {
            let a = self.args(oe, "ORIENTED_EDGE")?;
            let e = a
                .get(3)
                .and_then(Value::as_ref)
                .ok_or(format!("#{oe}: edge"))?;
            let along = a.get(4).and_then(Value::as_bool).unwrap_or(true);
            let mut s = self.edge(e)?;
            if !along {
                s.reverse();
            }
            ring.extend_from_slice(&s[..s.len() - 1]);
        }
        if !same {
            ring.reverse();
        }
        Ok(ring)
    }

    /// The facets of ADVANCED_FACE `id` into `out`, carried by `surface`.
    fn face(&mut self, id: u32, out: &mut Faceted) -> Result<(), String> {
        let a = self.args(id, "ADVANCED_FACE")?.to_vec();
        let surf = self.d.surface(
            a.get(2)
                .and_then(Value::as_ref)
                .ok_or(format!("#{id}: surface"))?,
        )?;
        let same_sense = a.get(3).and_then(Value::as_bool).unwrap_or(true);
        let mut rings: Vec<Vec<P3>> = Vec::new();
        for b in refs(a.get(1)) {
            let r = self.ring(b)?;
            if r.len() >= 3 {
                rings.push(r);
            }
        }
        if rings.is_empty() {
            return Err(format!("#{id}: no bounds"));
        }
        // The rings in parameters, periodic ones unwrapped along each ring
        // and every ring moved into the turn of the first.
        let periods = surf.periods();
        let mut uv: Vec<Vec<[f64; 2]>> = rings
            .iter()
            .map(|r| {
                let mut prev: Option<[f64; 2]> = None;
                r.iter()
                    .map(|&p| {
                        let mut q = surf.param(p);
                        if let Some(pr) = prev {
                            for k in 0..2 {
                                if let Some(period) = periods[k] {
                                    q[k] = near(q[k], pr[k], period);
                                }
                            }
                        }
                        prev = Some(q);
                        q
                    })
                    .collect()
            })
            .collect();
        let mean = |r: &[[f64; 2]], k: usize| r.iter().map(|q| q[k]).sum::<f64>() / r.len() as f64;
        let outer = uv
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| area(a).abs().total_cmp(&area(b).abs()))
            .map(|(i, _)| i)
            .unwrap_or(0);
        for k in 0..2 {
            let Some(period) = periods[k] else {
                continue;
            };
            let m0 = mean(&uv[outer], k);
            for (i, r) in uv.iter_mut().enumerate() {
                if i != outer {
                    let shift = near(mean(r, k), m0, period) - mean(r, k);
                    r.iter_mut().for_each(|q| q[k] += shift);
                }
            }
        }
        // The points: every ring's, then the inner ones.
        let mut pts3: Vec<P3> = rings.iter().flatten().copied().collect();
        let mut pts2: Vec<[f64; 2]> = uv.iter().flatten().copied().collect();
        let n_boundary = pts3.len();
        let holes: Vec<Vec<[f64; 2]>> = uv
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != outer)
            .map(|(_, r)| r.clone())
            .collect();
        let seed = triangulate_polygon(&uv[outer], &holes);
        if seed.is_empty() {
            return Err(format!("#{id}: its bounds do not triangulate"));
        }
        let inner = self.inner_points(&surf, &uv[outer], &holes);
        let index: FxHashMap<[u64; 2], usize> = pts2
            .iter()
            .enumerate()
            .map(|(i, q)| (q.map(f64::to_bits), i))
            .collect();
        let seed_tris: Vec<[usize; 3]> = seed
            .iter()
            .filter_map(|t| {
                let i = t.map(|q| index.get(&q.map(f64::to_bits)).copied());
                Some([i[0]?, i[1]?, i[2]?])
            })
            .collect();
        let tris: Vec<[usize; 3]> = if inner.is_empty() {
            seed_tris
        } else {
            let pool: Vec<Point3> = pts2
                .iter()
                .map(|q| Point3::explicit(q[0], q[1], 0.0))
                .collect();
            let points: Vec<Point3> = inner
                .iter()
                .map(|q| Point3::explicit(q[0], q[1], 0.0))
                .collect();
            let t = rapidmesh_csg::triangulate_seeded(
                Axis::Z,
                Sign::Positive,
                [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                pool,
                seed_tris,
                &points,
                &[],
                false,
            )
            .map_err(|e| format!("#{id}: {e}"))?;
            for v in &t.vertices[n_boundary..] {
                let q = v.approx().ok_or(format!("#{id}: an inner point"))?;
                pts2.push([q[0], q[1]]);
                pts3.push(surf.eval([q[0], q[1]]));
            }
            t.triangles
        };
        let s = out.add_surface(surf.kind());
        for t in tris {
            let [a, b, c] = t.map(|i| pts3[i]);
            // Counterclockwise in the parameters is the surface's normal;
            // the face's is it or its opposite.
            let ccw = polygon_orientation(&t.map(|i| pts2[i])) == Sign::Positive;
            let tri = if ccw == same_sense {
                Tri::new(a, b, c)
            } else {
                Tri::new(a, c, b)
            };
            out.push_tri(tri, s);
        }
        Ok(())
    }

    /// Points inside the parameter polygon `outer` less `holes`, spaced so
    /// the facets between them stay within the chord of the surface.
    fn inner_points(
        &self,
        surf: &Surface,
        outer: &[[f64; 2]],
        holes: &[Vec<[f64; 2]>],
    ) -> Vec<[f64; 2]> {
        if matches!(surf, Surface::Plane(_)) {
            return Vec::new();
        }
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for q in outer {
            for k in 0..2 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
        // The largest stretch and the smallest radius over the face's box.
        let (mut st, mut r) = ([0.0f64; 2], surf.radius());
        for i in 0..=4 {
            for j in 0..=4 {
                let q = [
                    lo[0] + (hi[0] - lo[0]) * i as f64 / 4.0,
                    lo[1] + (hi[1] - lo[1]) * j as f64 / 4.0,
                ];
                let s = surf.stretch(q);
                st = [st[0].max(s[0]), st[1].max(s[1])];
                r = r.min(surf.radius_at(q));
            }
        }
        // The size a facet may have on this surface.
        let span = ((hi[0] - lo[0]) * st[0]).max((hi[1] - lo[1]) * st[1]);
        let h = if r.is_finite() {
            (8.0 * self.chord * r).sqrt()
        } else {
            span / 12.0
        }
        .min(span / 4.0)
        .max(span / 200.0);
        let steps = [h / st[0], h / st[1]];
        let n = [
            ((hi[0] - lo[0]) / steps[0]).ceil().max(1.0) as usize,
            ((hi[1] - lo[1]) / steps[1]).ceil().max(1.0) as usize,
        ];
        let mut out = Vec::new();
        for i in 1..n[0] {
            for j in 1..n[1] {
                let q = [
                    lo[0] + (hi[0] - lo[0]) * i as f64 / n[0] as f64,
                    lo[1] + (hi[1] - lo[1]) * j as f64 / n[1] as f64,
                ];
                let inside = contains(outer, q) && !holes.iter().any(|h| contains(h, q));
                // Clear of the boundary by a third of a step.
                let clear = std::iter::once(outer)
                    .chain(holes.iter().map(|h| h.as_slice()))
                    .all(|ring| seg_dist(ring, q, steps) > 0.33);
                if inside && clear {
                    out.push(q);
                }
            }
        }
        out
    }
}

/// Twice the signed area of a ring.
fn area(r: &[[f64; 2]]) -> f64 {
    (0..r.len())
        .map(|i| {
            let (a, b) = (r[i], r[(i + 1) % r.len()]);
            a[0] * b[1] - a[1] * b[0]
        })
        .sum()
}

/// Even-odd point in polygon.
fn contains(r: &[[f64; 2]], q: [f64; 2]) -> bool {
    let mut inside = false;
    for i in 0..r.len() {
        let (a, b) = (r[i], r[(i + 1) % r.len()]);
        if (a[1] > q[1]) != (b[1] > q[1]) {
            let x = a[0] + (q[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
            if q[0] < x {
                inside = !inside;
            }
        }
    }
    inside
}

/// The distance from `q` to the ring in steps (each direction scaled by
/// its step).
fn seg_dist(r: &[[f64; 2]], q: [f64; 2], steps: [f64; 2]) -> f64 {
    let s = |p: [f64; 2]| [p[0] / steps[0], p[1] / steps[1]];
    let q = s(q);
    (0..r.len())
        .map(|i| {
            let (a, b) = (s(r[i]), s(r[(i + 1) % r.len()]));
            let d = [b[0] - a[0], b[1] - a[1]];
            let l = d[0] * d[0] + d[1] * d[1];
            let t = if l > 0.0 {
                (((q[0] - a[0]) * d[0] + (q[1] - a[1]) * d[1]) / l).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let p = [a[0] + t * d[0] - q[0], a[1] + t * d[1] - q[1]];
            (p[0] * p[0] + p[1] * p[1]).sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

/// The bodies of `x`: one per MANIFOLD_SOLID_BREP, closed and oriented
/// outward.
pub fn bodies(x: &Exchange, tol: Tolerance) -> Result<Vec<Body>, String> {
    let d = Decoder { x };
    // The model's size, from its points.
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for id in x.all("VERTEX_POINT") {
        if let Some(p) = x
            .record(id, "VERTEX_POINT")
            .and_then(|r| r.args.get(1).and_then(Value::as_ref))
            .and_then(|p| d.point(p).ok())
        {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    let size = dist(lo, hi).max(1e-12);
    let mut t = Tess {
        x,
        d: Decoder { x },
        chord: tol.chord * size,
        min_segments: tol.min_segments,
        edges: FxHashMap::default(),
        corners: Vec::new(),
    };
    let mut out = Vec::new();
    for solid in x.all("MANIFOLD_SOLID_BREP") {
        let a = t.args(solid, "MANIFOLD_SOLID_BREP")?;
        let name = a.first().and_then(Value::as_str).unwrap_or("").to_string();
        let shell = a
            .get(1)
            .and_then(Value::as_ref)
            .ok_or(format!("#{solid}: shell"))?;
        let faces = refs(t.args(shell, "CLOSED_SHELL")?.get(1));
        let mut f = Faceted::new();
        t.corners.clear();
        for face in faces {
            t.face(face, &mut f)?;
        }
        // The body's vertices are its corners: its edges end there.
        let mut corners = std::mem::take(&mut t.corners);
        corners.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        corners.dedup();
        f.corners = corners;
        out.push(Body { name, solid: f });
    }
    Ok(out)
}

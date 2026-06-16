//! Bridge from a [`rapidmesh_brep::Brep`] to the bottom-up mesher stages.
//!
//! The B-rep is the geometry source; this module turns its analytic edges and
//! trimmed faces into the inputs the existing stages consume -- stage 1 here
//! (an edge's analytic curve for `curve::distribute`), the face and volume
//! stages follow. The B-rep only changes WHERE the surface points come from; the
//! volume Lloyd, region classification and restricted-Delaunay extraction are
//! unchanged.

use crate::conform::MeshParams;
use crate::curve::{distribute, Curve, PolylineCurve};
use crate::site::Site;
use crate::sizefield::SizeField;
use crate::surf2d::cvt_fill;
use rapidmesh_brep::{Brep, Curve as BCurve, Edge as BEdge};
use rapidmesh_geom::nurbs::NurbsCurve;
use rapidmesh_geom::{FaceTag, TaggedPlc};
use std::sync::Arc;

type V3 = [f64; 3];
type P2 = [f64; 2];

/// Surface 2D Lloyd passes (mirrors `cvt::SURF_LLOYD_ITERS`).
const SURF_LLOYD_ITERS: usize = 4;

fn dist(a: V3, b: V3) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}
fn dist2(a: P2, b: P2) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Arc-length-parametrised analytic curve for a [`BCurve::Profile`] edge: the 2D
/// profile lifted to 3D on its extrusion frame at height `z`, over the parameter
/// range `[t0, t1]` the edge covers. Curvature comes from the exact profile, so
/// the distribution is tessellation-independent (the airfoil outline).
struct ProfileCurve {
    profile: Arc<NurbsCurve>,
    base: V3,
    u: V3,
    v: V3,
    axis: V3,
    z: f64,
    ts: Vec<f64>,
    ss: Vec<f64>,
}

impl ProfileCurve {
    fn new(profile: Arc<NurbsCurve>, base: V3, u: V3, v: V3, axis: V3, t: [f64; 2], z: f64) -> Option<ProfileCurve> {
        let (lo, hi) = (t[0].min(t[1]), t[0].max(t[1]));
        if !(hi > lo) {
            return None;
        }
        let n = 256usize;
        let (mut ts, mut ss) = (vec![lo], vec![0.0f64]);
        let (mut prev, mut acc) = (lo, 0.0);
        for i in 1..=n {
            let tt = lo + (hi - lo) * i as f64 / n as f64;
            acc += profile.arc_length(prev, tt, 2);
            ts.push(tt);
            ss.push(acc);
            prev = tt;
        }
        Some(ProfileCurve { profile, base, u, v, axis, z, ts, ss })
    }
    fn s_to_t(&self, s: f64) -> f64 {
        let s = s.clamp(0.0, self.ss[self.ss.len() - 1]);
        let i = self.ss.partition_point(|&x| x < s).clamp(1, self.ss.len() - 1);
        let (s0, s1) = (self.ss[i - 1], self.ss[i]);
        let f = if s1 > s0 { (s - s0) / (s1 - s0) } else { 0.0 };
        self.ts[i - 1] + f * (self.ts[i] - self.ts[i - 1])
    }
    fn at3(&self, t: f64) -> V3 {
        let c = self.profile.eval(t);
        add(add(self.base, scale(self.axis, self.z)), add(scale(self.u, c[0]), scale(self.v, c[1])))
    }
}

impl Curve for ProfileCurve {
    fn length(&self) -> f64 {
        self.ss[self.ss.len() - 1]
    }
    fn point_at(&self, s: f64) -> V3 {
        self.at3(self.s_to_t(s))
    }
    fn radius_at(&self, s: f64) -> f64 {
        let k = self.profile.curvature(self.s_to_t(s));
        if k > 1e-12 {
            1.0 / k
        } else {
            f64::INFINITY
        }
    }
}

/// A circular arc (or full circle) parametrised by arc length. Radius is constant,
/// so the sagitta sizing places uniform points; the arc range is taken from the
/// edge's chain endpoints (a closed rim spans the full `2*pi`).
struct CircleCurve {
    center: V3,
    x: V3,
    y: V3,
    radius: f64,
    a0: f64,
    span: f64,
}

impl CircleCurve {
    fn new(center: V3, axis: V3, x: V3, radius: f64, chain: &[V3]) -> Option<CircleCurve> {
        if !(radius > 0.0) || chain.len() < 2 {
            return None;
        }
        let y = cross(axis, x);
        let ang = |p: V3| {
            let d = sub(p, center);
            dot(d, y).atan2(dot(d, x))
        };
        // Total signed swept angle = sum of per-segment increments (each in
        // (-pi, pi]); robust for an arc (partial) and a closed rim (sums to +-2*pi).
        let pi = std::f64::consts::PI;
        let wrap = |a: f64| (a + pi).rem_euclid(2.0 * pi) - pi;
        let a0 = ang(chain[0]);
        let mut span = 0.0;
        for w in chain.windows(2) {
            span += wrap(ang(w[1]) - ang(w[0]));
        }
        if span.abs() < 1e-9 {
            return None;
        }
        Some(CircleCurve { center, x, y, radius, a0, span })
    }
}

impl Curve for CircleCurve {
    fn length(&self) -> f64 {
        self.radius * self.span.abs()
    }
    fn point_at(&self, s: f64) -> V3 {
        let f = (s / self.length()).clamp(0.0, 1.0);
        let t = self.a0 + self.span * f;
        let (st, ct) = t.sin_cos();
        std::array::from_fn(|k| self.center[k] + self.radius * (ct * self.x[k] + st * self.y[k]))
    }
    fn radius_at(&self, _s: f64) -> f64 {
        self.radius
    }
}

/// The analytic curve to distribute points on for a B-rep edge: the exact profile
/// or circle where recovered, else the faceted chain polyline (a straight `Line`
/// is exactly a 2-point polyline, so it reduces to uniform spacing).
pub fn edge_curve(edge: &BEdge) -> Option<Box<dyn Curve>> {
    match &edge.curve {
        BCurve::Profile { profile, base, u, v, axis, t, z } => {
            ProfileCurve::new(profile.clone(), *base, *u, *v, *axis, *t, *z)
                .map(|c| Box::new(c) as Box<dyn Curve>)
        }
        BCurve::Circle { center, axis, radius, x } => {
            CircleCurve::new(*center, *axis, *x, *radius, &edge.chain)
                .map(|c| Box::new(c) as Box<dyn Curve>)
        }
        _ => PolylineCurve::new(&edge.chain).map(|c| Box::new(c) as Box<dyn Curve>),
    }
}

/// The surface points produced from a B-rep, in the exact form the volume stage
/// of `cvt::mesh` consumes.
pub struct SurfaceSites {
    /// Corner + edge points (pinned), then per-face interior points.
    pub sites: Vec<Site>,
    /// `sites.len()` -- all of `sites` are surface points.
    pub n_surf: usize,
    /// Per site: the B-rep face it tiles, or `u32::MAX` for a shared corner/edge
    /// point (which defers its output tag to an interior face point).
    pub point_tile: Vec<u32>,
    /// Per B-rep face: `(originating plc surface index, face tag)` for output.
    pub tiles: Vec<(u32, FaceTag)>,
    /// Number of pinned corner points (the `plc_points` count).
    pub plc_points: usize,
}

/// The user size cap for a B-rep face (face tag, then surface owner, then region).
fn face_cap(face: &rapidmesh_brep::Face, params: &MeshParams) -> f64 {
    let ft = face.face_tag.0;
    if let Some(&(_, h)) = params.face_maxh.iter().find(|(t, _)| *t == ft) {
        return h.min(params.maxh);
    }
    if let Some(&(_, h)) = params.surface_maxh.iter().find(|(o, _)| *o == face.owner) {
        return h.min(params.maxh);
    }
    let mut h = params.maxh;
    for r in face.regions {
        if r.0 != 0 {
            if let Some(&(_, rh)) = params.region_maxh.iter().find(|(rr, _)| *rr == r.0) {
                h = h.min(rh);
            }
        }
    }
    h
}

/// Even-odd point-in-region test over a face's loop segments (in (u,v)). Holes are
/// handled by the parity rule (all loops contribute their segments).
fn in_loops(uv: P2, segs: &[(P2, P2)]) -> bool {
    let mut c = false;
    for &(a, b) in segs {
        if (a[1] > uv[1]) != (b[1] > uv[1]) {
            let xint = a[0] + (uv[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
            if uv[0] < xint {
                c = !c;
            }
        }
    }
    c
}

/// Builds the surface points from a B-rep: stage 1 distributes on every edge curve
/// (shared, so adjacent faces agree), stage 2 meshes each trimmed face in its
/// (u,v) parameter space, trimmed by the loop PCurves, and lifts the points onto
/// the surface (an exact planar carrier where the face is a plane). Closed faces
/// with no loops (a full sphere) are skipped here -- periodic param meshing is a
/// later step; until then they stay on the faceted path.
pub fn surface_sites(brep: &Brep, plc: &TaggedPlc, params: &MeshParams) -> SurfaceSites {
    let grad = if params.grading > 0.0 { params.grading } else { 0.5 };
    let mut sites: Vec<Site> = Vec::new();
    let mut point_tile: Vec<u32> = Vec::new();

    // ---- stage 1a: corners (pinned) -------------------------------------
    for v in &brep.vertices {
        sites.push(Site::vertex(v.pos));
        point_tile.push(u32::MAX);
    }
    let plc_points = sites.len();

    // ---- stage 1b: edge points (shared across adjacent faces) -----------
    let mut edge_sources: Vec<(V3, f64)> = Vec::new();
    for edge in &brep.edges {
        let cap = edge
            .coedges
            .iter()
            .map(|&c| face_cap(brep.face(brep.coedge(c).face), params))
            .fold(params.maxh, f64::min);
        let Some(curve) = edge_curve(edge) else { continue };
        let ps = distribute(&*curve, params.surface_deflection, cap, grad);
        let pts3: Vec<V3> = ps.iter().map(|&s| curve.point_at(s)).collect();
        for k in 0..pts3.len() {
            let mut sp = cap;
            if k > 0 {
                sp = sp.min(dist(pts3[k], pts3[k - 1]));
            }
            if k + 1 < pts3.len() {
                sp = sp.min(dist(pts3[k], pts3[k + 1]));
            }
            edge_sources.push((pts3[k], sp));
        }
        // interior points only; endpoints are the (already pinned) corners
        for &p in pts3.iter().take(pts3.len().saturating_sub(1)).skip(1) {
            sites.push(Site::vertex(p));
            point_tile.push(u32::MAX);
        }
    }
    for &(p, h) in &params.size_points {
        edge_sources.push((p, h));
    }
    let surf_min_h = edge_sources.iter().map(|s| s.1).fold(params.maxh, f64::min).max(1e-9);
    let surf_field = SizeField::new(edge_sources, grad, params.maxh);

    let tiles: Vec<(u32, FaceTag)> =
        brep.faces.iter().map(|f| (f.plc_surface, f.face_tag)).collect();

    // ---- stage 2: per face, trimmed (u,v) Lloyd, lifted -----------------
    let chord = (8.0 * params.surface_deflection).sqrt();
    for (fid, face) in brep.faces.iter().enumerate() {
        if face.loops.is_empty() {
            continue; // closed face: faceted fallback (periodic param meshing later)
        }
        let surf = brep.surface(face.surface);
        let mut bnd: Vec<P2> = Vec::new();
        let mut segs: Vec<(P2, P2)> = Vec::new();
        for lp in &face.loops {
            let mut loop_uv: Vec<P2> = Vec::new();
            for &cid in &lp.coedges {
                for &p in &brep.coedge(cid).pcurve.uv {
                    if loop_uv.last().map(|&q| dist2(q, p) > 1e-18).unwrap_or(true) {
                        loop_uv.push(p);
                    }
                }
            }
            for w in 0..loop_uv.len() {
                segs.push((loop_uv[w], loop_uv[(w + 1) % loop_uv.len()]));
            }
            bnd.extend_from_slice(&loop_uv);
        }
        if bnd.len() < 3 {
            continue;
        }
        let (mut lo, mut hi) = (bnd[0], bnd[0]);
        for &p in &bnd {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let fcap = face_cap(face, params);
        // finest step: the edge spacing, refined by the face's own curvature
        // (its interior is not seen by the boundary edges).
        let min_r = bnd.iter().map(|&uv| surf.curvature_radius(uv)).fold(f64::INFINITY, f64::min);
        let step = if min_r.is_finite() { surf_min_h.min(min_r * chord) } else { surf_min_h }.max(1e-9);
        let target = |uv: P2| {
            let xyz = surf.eval_uv(uv);
            let hc = surf.curvature_radius(uv) * chord;
            surf_field.at(xyz).min(hc).min(fcap)
        };
        let inside = |uv: P2| in_loops(uv, &segs);
        for uv in cvt_fill(&bnd, lo, hi, step, target, SURF_LLOYD_ITERS, inside, params.density_weighted) {
            let xyz = surf.eval_uv(uv);
            let site = match surf.exact_plane() {
                Some((o, n)) => Site::on_plane(o, n, xyz),
                None => Site::on_surface(plc.surfaces[face.plc_surface as usize].clone(), xyz),
            };
            sites.push(site);
            point_tile.push(fid as u32);
        }
    }

    let n_surf = sites.len();
    SurfaceSites { sites, n_surf, point_tile, tiles, plc_points }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapidmesh_brep::build::from_plc;
    use rapidmesh_geom::{extrude_spline_profile, naca0012_profile, solid_box, Scene};

    #[test]
    fn box_edges_are_uniform_lines() {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 1.0, 1.0]));
        let b = from_plc(&scene.assemble());
        // a length-4 edge with maxh 1 -> 4 uniform segments
        let long = b
            .edges
            .iter()
            .find(|e| {
                let c = edge_curve(e).unwrap();
                (c.length() - 4.0).abs() < 1e-9
            })
            .expect("a length-4 edge");
        let c = edge_curve(long).unwrap();
        let s = distribute(&*c, 0.02, 1.0, 0.3);
        assert_eq!(s.len() - 1, 4, "4 uniform segments of maxh=1");
    }

    #[test]
    fn airfoil_profile_edge_is_curvature_graded() {
        let profile = naca0012_profile(1.0, 40);
        let solid = extrude_spline_profile(
            profile,
            80,
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5],
        );
        let mut scene = Scene::new();
        scene.add_solid(solid);
        let b = from_plc(&scene.assemble());
        let prof = b
            .edges
            .iter()
            .find(|e| matches!(e.curve, BCurve::Profile { .. }))
            .expect("a profile edge");
        let c = edge_curve(prof).unwrap();
        let s = distribute(&*c, 0.01, 0.2, 0.3);
        let spc: Vec<f64> = s.windows(2).map(|w| w[1] - w[0]).collect();
        let (mn, mx) = spc.iter().fold((f64::MAX, 0.0f64), |(a, b), &x| (a.min(x), b.max(x)));
        // curvature grading: the nose spacing is much finer than the flat tail
        assert!(mx / mn > 2.0, "profile distribution should grade (ratio {})", mx / mn);
    }

    #[test]
    fn box_surface_sites_lie_on_exact_planes() {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]));
        let plc = scene.assemble();
        let b = from_plc(&plc);
        let params = MeshParams { maxh: 0.5, ..Default::default() };
        let ss = surface_sites(&b, &plc, &params);

        assert_eq!(ss.plc_points, 8, "8 box corners pinned");
        assert!(ss.n_surf > 8, "edge + face interior points added");
        // every face-interior point lies exactly on its face's plane
        let mut n_face_pts = 0;
        for (i, &tile) in ss.point_tile.iter().enumerate() {
            if tile == u32::MAX {
                continue;
            }
            n_face_pts += 1;
            let surf = b.surface(b.faces[tile as usize].surface);
            let (o, n) = surf.exact_plane().expect("box face is planar");
            let p = ss.sites[i].pos();
            let off = (p[0] - o[0]) * n[0] + (p[1] - o[1]) * n[1] + (p[2] - o[2]) * n[2];
            assert!(off.abs() < 1e-9, "face point off its plane by {off}");
        }
        assert!(n_face_pts > 0, "faces produced interior points");
    }
}

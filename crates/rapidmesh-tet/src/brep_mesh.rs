//! Bridge from a [`rapidmesh_brep::Brep`] to the bottom-up mesher stages.
//!
//! The B-rep is the geometry source; this module turns its analytic edges and
//! trimmed faces into the inputs the existing stages consume -- stage 1 here
//! (an edge's analytic curve for `curve::distribute`), the face and volume
//! stages follow. The B-rep only changes WHERE the surface points come from; the
//! volume Lloyd, region classification and restricted-Delaunay extraction are
//! unchanged.

use crate::curve::{Curve, PolylineCurve};
use rapidmesh_brep::{Curve as BCurve, Edge as BEdge};
use rapidmesh_geom::nurbs::NurbsCurve;
use std::sync::Arc;

type V3 = [f64; 3];

fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
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

/// The analytic curve to distribute points on for a B-rep edge: the exact profile
/// where recovered, else the faceted chain polyline (a straight `Line` is exactly
/// a 2-point polyline, so it reduces to uniform spacing).
pub fn edge_curve(edge: &BEdge) -> Option<Box<dyn Curve>> {
    match &edge.curve {
        BCurve::Profile { profile, base, u, v, axis, t, z } => {
            ProfileCurve::new(profile.clone(), *base, *u, *v, *axis, *t, *z)
                .map(|c| Box::new(c) as Box<dyn Curve>)
        }
        _ => PolylineCurve::new(&edge.chain).map(|c| Box::new(c) as Box<dyn Curve>),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::distribute;
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
}

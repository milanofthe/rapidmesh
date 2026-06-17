//! Region / surface / edge topology with stable ids and per-entity geometry,
//! extracted from the [`Brep`]. This is the read model the hierarchical sizing
//! API (`g.region(...).surf(...).edge(...).maxh/.tol`) navigates: ids are indices
//! into the brep (stable for one assembly), and each entity carries the geometry
//! a selector needs (a face centroid/normal, an edge midpoint/length) plus the
//! incidence (region -> faces -> edges) so a scope can walk down the hierarchy.

use crate::{Brep, Curve};
use rapidmesh_geom::TaggedPlc;

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: V3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Analytic curve kind of an edge, as a small code (for the Python selector to
/// distinguish straight from curved edges without the full enum).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    Line = 0,
    Circle = 1,
    Profile = 2,
    Intersection = 3,
    Polyline = 4,
}

/// One face of the boundary, with its sizing-relevant geometry and incidence.
#[derive(Debug, Clone)]
pub struct FaceTopo {
    /// Area-weighted centroid of the face's PLC facets.
    pub centroid: V3,
    /// Area-weighted unit normal (front side, `regions[0]`).
    pub normal: V3,
    /// Total facet area.
    pub area: f64,
    /// Materials on the front (`+normal`) and back sides.
    pub regions: [u32; 2],
    /// Face tag (ports, PEC), 0 if untagged.
    pub face_tag: u32,
    /// Index into `plc.surfaces` (the analytic surface).
    pub surface: u32,
    /// Scene-solid owner.
    pub owner: u32,
    /// Edge ids on this face's boundary loops (sorted, deduplicated).
    pub edges: Vec<u32>,
}

/// One edge of the boundary, with its sizing-relevant geometry and incidence.
#[derive(Debug, Clone)]
pub struct EdgeTopo {
    /// Chain endpoints (the two corners).
    pub p0: V3,
    pub p1: V3,
    /// Midpoint of the chain (a stable point selector target).
    pub midpoint: V3,
    /// Arc length of the chain polyline.
    pub length: f64,
    /// Analytic curve kind.
    pub kind: EdgeKind,
    /// Face ids meeting along this edge (the radial cycle), sorted, deduplicated.
    pub faces: Vec<u32>,
}

/// The region / face / edge read model. Face and edge ids are indices into
/// `faces` / `edges`; region ids are the material tags.
#[derive(Debug, Clone, Default)]
pub struct Topology {
    /// Distinct meshed region tags (`> 0`), ascending.
    pub regions: Vec<u32>,
    pub faces: Vec<FaceTopo>,
    pub edges: Vec<EdgeTopo>,
}

impl Topology {
    /// Face ids bounding region `tag` (the region on either side of the face).
    pub fn region_faces(&self, tag: u32) -> Vec<u32> {
        self.faces
            .iter()
            .enumerate()
            .filter(|(_, f)| f.regions[0] == tag || f.regions[1] == tag)
            .map(|(i, _)| i as u32)
            .collect()
    }
}

/// Builds the [`Topology`] from the assembled `plc` (for facet geometry) and its
/// `brep` (for topology + analytic curves).
pub fn extract_topology(plc: &TaggedPlc, brep: &Brep) -> Topology {
    let vtx = |i: u32| plc.vertices[i as usize];

    // Faces: geometry from the PLC facets, edges from the loops.
    let mut faces: Vec<FaceTopo> = Vec::with_capacity(brep.faces.len());
    for f in &brep.faces {
        let (mut area, mut cen, mut nrm) = (0.0f64, [0.0; 3], [0.0; 3]);
        for &ti in &f.facets {
            let t = plc.triangles[ti as usize];
            let (a, b, c) = (vtx(t[0]), vtx(t[1]), vtx(t[2]));
            let n = cross(sub(b, a), sub(c, a));
            let ar = 0.5 * norm(n);
            area += ar;
            let g = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0];
            for k in 0..3 {
                cen[k] += ar * g[k];
                nrm[k] += 0.5 * n[k]; // area-weighted (|n| = 2*area)
            }
        }
        if area > 0.0 {
            for k in 0..3 {
                cen[k] /= area;
            }
        }
        let nl = norm(nrm).max(1e-30);
        let normal = [nrm[0] / nl, nrm[1] / nl, nrm[2] / nl];
        // Edges from the boundary loops.
        let mut edges: Vec<u32> = Vec::new();
        for lp in &f.loops {
            for &ce in &lp.coedges {
                edges.push(brep.coedge(ce).edge.0);
            }
        }
        edges.sort_unstable();
        edges.dedup();
        faces.push(FaceTopo {
            centroid: cen,
            normal,
            area,
            regions: [f.regions[0].0, f.regions[1].0],
            face_tag: f.face_tag.0,
            surface: f.plc_surface,
            owner: f.owner,
            edges,
        });
    }

    // Edges: endpoints / length / kind from the chain + curve, faces from coedges.
    let mut edges: Vec<EdgeTopo> = Vec::with_capacity(brep.edges.len());
    for e in &brep.edges {
        let p0 = *e.chain.first().unwrap_or(&[0.0; 3]);
        let p1 = *e.chain.last().unwrap_or(&[0.0; 3]);
        let mut length = 0.0;
        for w in e.chain.windows(2) {
            length += norm(sub(w[1], w[0]));
        }
        // Midpoint at half the arc length along the chain.
        let mut mid = p0;
        let mut acc = 0.0;
        for w in e.chain.windows(2) {
            let seg = norm(sub(w[1], w[0]));
            if acc + seg >= 0.5 * length && seg > 0.0 {
                let t = (0.5 * length - acc) / seg;
                mid = [w[0][0] + t * (w[1][0] - w[0][0]), w[0][1] + t * (w[1][1] - w[0][1]), w[0][2] + t * (w[1][2] - w[0][2])];
                break;
            }
            acc += seg;
        }
        let kind = match e.curve {
            Curve::Line { .. } => EdgeKind::Line,
            Curve::Circle { .. } => EdgeKind::Circle,
            Curve::Profile { .. } => EdgeKind::Profile,
            Curve::Intersection { .. } => EdgeKind::Intersection,
            Curve::Polyline => EdgeKind::Polyline,
        };
        let mut faces_of: Vec<u32> = e.coedges.iter().map(|&c| brep.coedge(c).face.0).collect();
        faces_of.sort_unstable();
        faces_of.dedup();
        edges.push(EdgeTopo { p0, p1, midpoint: mid, length, kind, faces: faces_of });
    }

    // Regions: distinct meshed tags (> 0).
    let mut regions: Vec<u32> = faces
        .iter()
        .flat_map(|f| f.regions.into_iter())
        .filter(|&r| r != 0)
        .collect();
    regions.sort_unstable();
    regions.dedup();

    Topology { regions, faces, edges }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapidmesh_geom::{solid_box, Scene};

    #[test]
    fn box_topology_is_one_region_six_faces_twelve_edges() {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]));
        let plc = scene.assemble();
        let brep = crate::build::from_plc(&plc);
        let topo = extract_topology(&plc, &brep);

        assert_eq!(topo.regions, vec![1], "one meshed region");
        assert_eq!(topo.faces.len(), 6, "six box faces");
        assert_eq!(topo.edges.len(), 12, "twelve box edges");
        // Every face is a quad: four boundary edges.
        for (i, f) in topo.faces.iter().enumerate() {
            assert_eq!(f.edges.len(), 4, "face {i} should have 4 edges");
            assert!([6.0, 8.0, 12.0].iter().any(|&a| (f.area - a).abs() < 1e-9), "box face area {} unexpected", f.area);
            assert!(f.regions.contains(&1) && f.regions.contains(&0), "outer wall separates region 1 from void 0");
        }
        // Every edge is straight (a box) and shared by exactly two faces.
        for (i, e) in topo.edges.iter().enumerate() {
            assert_eq!(e.kind, EdgeKind::Line, "box edge {i} is straight");
            assert_eq!(e.faces.len(), 2, "box edge {i} shared by two faces");
            assert!([2.0, 3.0, 4.0].iter().any(|&l| (e.length - l).abs() < 1e-9), "box edge length {} unexpected", e.length);
        }
        // region 1 is bounded by all six faces.
        assert_eq!(topo.region_faces(1).len(), 6);
    }
}

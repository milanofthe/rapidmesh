//! Initial site seeding and the sizing field for the CVT mesher.
//!
//! Seeding lays down a BCC (body-centered cubic) background lattice over the
//! domain bounding box: two interpenetrating cubic grids whose Delaunay dual is
//! the well-known BCC tetrahedralization with dihedral angles 60/90/120 deg, the
//! lattice Labelle-Shewchuk isosurface stuffing builds on for its provable angle
//! bounds. The Lloyd/CVT pass (WP3+) relaxes these toward a centroidal layout.
//!
//! `SizingField` turns the `MeshParams` sizing handles into a local target edge
//! length `h(x)`, Lipschitz by the grading constant. WP2 honors the regional cap
//! and point sources (the uniform seeding spacing); per-face/surface caps and
//! graded (variable-spacing) seeding layer on in WP7.

use crate::conform::MeshParams;
use rapidmesh_geom::vec3::{V3, dist};

/// Evaluates the target edge length `h(x)` from the mesh parameters.
pub struct SizingField {
    maxh: f64,
    grading: f64,
    region_maxh: Vec<(u32, f64)>,
    size_points: Vec<(V3, f64)>,
}

impl SizingField {
    pub fn new(params: &MeshParams) -> SizingField {
        SizingField {
            maxh: params.maxh,
            grading: params.grading,
            region_maxh: params.region_maxh.clone(),
            size_points: params.size_points.clone(),
        }
    }

    /// Regional cap: the per-region override if present, else the global `maxh`.
    /// Region 0 (background, not meshed) has no finite cap.
    pub fn region_cap(&self, region: u32) -> f64 {
        if region == 0 {
            return f64::INFINITY;
        }
        self.region_maxh
            .iter()
            .find(|(r, _)| *r == region)
            .map(|&(_, h)| h)
            .unwrap_or(self.maxh)
    }

    /// Target edge length at `pos` inside `region`: the regional cap tightened
    /// by the Lipschitz envelope of every point source (`sh + grading * dist`).
    /// Lipschitz in `pos` with constant `grading`.
    pub fn target_at(&self, pos: V3, region: u32) -> f64 {
        let mut h = self.region_cap(region);
        for (sp, sh) in &self.size_points {
            h = h.min(sh + self.grading * dist(pos, *sp));
        }
        h
    }

    /// The finest finite regional cap across the given regions (the uniform
    /// seeding spacing for WP3; INFINITY if none is finite).
    pub fn finest_cap(&self, regions: &[u32]) -> f64 {
        let mut h = f64::INFINITY;
        for &r in regions {
            h = h.min(self.region_cap(r));
        }
        for (_, sh) in &self.size_points {
            h = h.min(*sh);
        }
        h
    }
}

/// BCC lattice over `[lo, hi]` with cell size `h`: a cubic grid of corner nodes
/// (multiples of `h` from `lo`, inclusive of the high corner) plus a body-
/// centered grid offset by `h/2`. Every returned point lies within `[lo, hi]`.
/// `h` must be finite and positive.
pub fn bcc_lattice(lo: V3, hi: V3, h: f64) -> Vec<V3> {
    assert!(h.is_finite() && h > 0.0, "bcc_lattice needs a finite positive h");
    let n: [usize; 3] = std::array::from_fn(|k| (((hi[k] - lo[k]) / h).ceil() as usize).max(1));
    let mut pts = Vec::new();
    // Corner sub-lattice (inclusive high corner via <=).
    for i in 0..=n[0] {
        for j in 0..=n[1] {
            for k in 0..=n[2] {
                let p = [
                    (lo[0] + i as f64 * h).min(hi[0]),
                    (lo[1] + j as f64 * h).min(hi[1]),
                    (lo[2] + k as f64 * h).min(hi[2]),
                ];
                pts.push(p);
            }
        }
    }
    // Body-centered sub-lattice (cell centers; strictly interior to the grid).
    for i in 0..n[0] {
        for j in 0..n[1] {
            for k in 0..n[2] {
                let p = [
                    lo[0] + (i as f64 + 0.5) * h,
                    lo[1] + (j as f64 + 0.5) * h,
                    lo[2] + (k as f64 + 0.5) * h,
                ];
                if p[0] <= hi[0] && p[1] <= hi[1] && p[2] <= hi[2] {
                    pts.push(p);
                }
            }
        }
    }
    pts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(maxh: f64, grading: f64, region_maxh: Vec<(u32, f64)>, size_points: Vec<(V3, f64)>) -> MeshParams {
        MeshParams {
            maxh,
            region_maxh,
            radius_edge_bound: 2.0,
            max_points: 100_000,
            grading,
            face_maxh: Vec::new(),
            surface_maxh: Vec::new(),
            size_points,
            density_weighted: false,
            tol_edge: 1e-2,
            tol_surf: 1e-2,
            maxh_edge: f64::INFINITY,
            maxh_surf: f64::INFINITY,
            maxh_vol: f64::INFINITY,
            edge_maxh: Vec::new(),
            edge_tol: Vec::new(),
            surf_maxh: Vec::new(),
            surf_tol: Vec::new(),
            min_h_surf: 0.0,
            min_h_vol: 0.0,
            surf_min_angle: 0.0,
        }
    }

    #[test]
    fn region_cap_falls_back_to_maxh() {
        let f = SizingField::new(&params(0.5, 0.5, vec![(2, 0.1)], vec![]));
        assert_eq!(f.region_cap(0), f64::INFINITY);
        assert_eq!(f.region_cap(1), 0.5);
        assert_eq!(f.region_cap(2), 0.1);
    }

    #[test]
    fn sizing_is_lipschitz_in_grading() {
        let g = 0.5;
        let f = SizingField::new(&params(1.0, g, vec![], vec![([0.0, 0.0, 0.0], 0.05)]));
        // h shrinks to ~0.05 at the source and grows with grading away from it.
        assert!((f.target_at([0.0, 0.0, 0.0], 1) - 0.05).abs() < 1e-12);
        let a = [0.3, 0.0, 0.0];
        let b = [0.7, 0.0, 0.0];
        let ha = f.target_at(a, 1);
        let hb = f.target_at(b, 1);
        assert!((ha - hb).abs() <= g * dist(a, b) + 1e-12, "Lipschitz bound");
        assert!(ha < hb, "closer to the source is finer");
    }

    #[test]
    fn bcc_count_and_bounds() {
        let lo = [0.0, 0.0, 0.0];
        let hi = [1.0, 1.0, 1.0];
        let pts = bcc_lattice(lo, hi, 0.25); // n = 4 per axis
        // 5^3 corner nodes + 4^3 body centers.
        assert_eq!(pts.len(), 5 * 5 * 5 + 4 * 4 * 4);
        for p in &pts {
            for k in 0..3 {
                assert!(p[k] >= lo[k] - 1e-12 && p[k] <= hi[k] + 1e-12, "within box");
            }
        }
    }

    #[test]
    fn bcc_density_tracks_h() {
        // Halving h multiplies the point count by ~8.
        let coarse = bcc_lattice([0.0; 3], [1.0; 3], 0.2).len();
        let fine = bcc_lattice([0.0; 3], [1.0; 3], 0.1).len();
        let ratio = fine as f64 / coarse as f64;
        assert!(ratio > 6.0 && ratio < 9.0, "density ~8x, got {ratio}");
    }
}

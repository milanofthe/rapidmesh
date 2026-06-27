//! The sizing field for the CVT mesher: turns the `MeshParams` sizing handles
//! into a local target edge length `h(x)`, Lipschitz by the grading constant.
//! WP2 honors the regional cap and point sources; per-face/surface caps and
//! graded seeding layer on in the surface/volume stages.

use crate::conform::MeshParams;
use rapidmesh_geom::vec3::V3;

/// Evaluates the target edge length `h(x)` from the mesh parameters.
pub struct SizingField {
    maxh: f64,
    region_maxh: Vec<(u32, f64)>,
    size_points: Vec<(V3, f64)>,
}

impl SizingField {
    pub fn new(params: &MeshParams) -> SizingField {
        SizingField {
            maxh: params.maxh,
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

    /// The finest finite regional cap across the given regions (the uniform
    /// seeding spacing; INFINITY if none is finite).
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
            cap_edge: f64::INFINITY,
            cap_surf: f64::INFINITY,
            cap_vol: f64::INFINITY,
            edge_maxh: Vec::new(),
            edge_tol: Vec::new(),
            surf_maxh: Vec::new(),
            surf_tol: Vec::new(),
            min_h_surf: 0.0,
            min_h_vol: 0.0,
            surf_min_angle: 0.0,
            surf_target_count: 0,
        }
    }

    #[test]
    fn region_cap_falls_back_to_maxh() {
        let f = SizingField::new(&params(0.5, 0.5, vec![(2, 0.1)], vec![]));
        assert_eq!(f.region_cap(0), f64::INFINITY);
        assert_eq!(f.region_cap(1), 0.5);
        assert_eq!(f.region_cap(2), 0.1);
    }
}

//! Edge-length calibration instrument: bare WR-90 box at two sizing
//! targets, reporting tet count and the edge-length distribution relative
//! to maxh. The goal is a mean near 1.0 (maxh is a target, not a bound to
//! duck under) while holding the 1.5 h max-edge contract and quality.
//!
//!     cargo run --release -p rapidmesh --example measure_density

use rapidmesh_geom::{solid_box, Scene};
use rapidmesh_tet::{mesh_plc_with, optimize, quality_stats, MeshParams, OptimizeParams};
use std::collections::HashSet;

fn main() {
    let (a, b, l) = (22.86e-3, 10.16e-3, 30.0e-3);
    for maxh in [3.0e-3, 2.0e-3] {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [a, b, l]));
        let plc = scene.assemble();
        let params = MeshParams { maxh, max_points: 500_000, ..MeshParams::default() };
        let mut mesh = mesh_plc_with(&plc, &params);
        optimize(&mut mesh, &OptimizeParams { maxh, ..OptimizeParams::default() });
        let mut edges: HashSet<(usize, usize)> = HashSet::new();
        for t in &mesh.tets {
            for i in 0..4 {
                for j in i + 1..4 {
                    edges.insert((t[i].min(t[j]), t[i].max(t[j])));
                }
            }
        }
        let mut ls: Vec<f64> = edges
            .iter()
            .map(|&(p, q)| {
                (0..3)
                    .map(|k| (mesh.points[p][k] - mesh.points[q][k]).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    / maxh
            })
            .collect();
        ls.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
        let mean: f64 = ls.iter().sum::<f64>() / ls.len() as f64;
        let pct = |p: f64| ls[((ls.len() - 1) as f64 * p) as usize];
        let q = quality_stats(&mesh);
        println!(
            "maxh {:>4.1}mm: {:>6} tets  edge/h mean {:.3} p10 {:.3} p50 {:.3} p90 {:.3} max {:.3}  min-dih {:>5.2}  r/e {:.2}",
            maxh * 1e3,
            mesh.tets.len(),
            mean,
            pct(0.10),
            pct(0.50),
            pct(0.90),
            ls.last().unwrap(),
            q.min_dihedral_deg,
            q.max_radius_edge,
        );
    }
}

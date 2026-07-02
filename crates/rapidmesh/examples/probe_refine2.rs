//! mesh_refine + optimize: end-quality probe with h-convergence.
use rapidmesh_geom::{cylinder, solid_box, Scene};
use rapidmesh_tet::{mesh_refine, optimize, quality_stats, MeshParams, OptimizeParams};

fn volume(m: &rapidmesh_tet::TetMesh) -> f64 {
    let mut vol = 0.0;
    for t in &m.tets {
        let p: [[f64; 3]; 4] = std::array::from_fn(|k| m.points[t[k]]);
        let d = |i: usize, k: usize| p[i][k] - p[0][k];
        vol += (d(1, 0) * (d(2, 1) * d(3, 2) - d(2, 2) * d(3, 1))
            - d(1, 1) * (d(2, 0) * d(3, 2) - d(2, 2) * d(3, 0))
            + d(1, 2) * (d(2, 0) * d(3, 1) - d(2, 1) * d(3, 0))).abs() / 6.0;
    }
    vol
}

fn main() {
    let want = 16.0 - std::f64::consts::PI * 0.75 * 0.75; // analytic
    for maxh in [0.4, 0.2] {
        let t0 = std::time::Instant::now();
        let mut scene = Scene::new();
        scene.add_solid(solid_box([-2.0, -2.0, 0.0], [2.0, 2.0, 1.0]));
        scene.add_void(cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.75, 24));
        let plc = scene.assemble();
        let mut m = mesh_refine(&plc, &MeshParams { maxh, ..Default::default() });
        let q0 = quality_stats(&m);
        optimize(&mut m, &OptimizeParams { maxh, ..Default::default() });
        let q = quality_stats(&m);
        let v = volume(&m);
        println!("h={maxh}: tets {} -> min dihedral {:.2} -> {:.2} deg, slivers {} -> {}, re {:.1} -> {:.1}",
            m.tets.len(), q0.min_dihedral_deg, q.min_dihedral_deg, q0.n_slivers, q.n_slivers,
            q0.max_radius_edge, q.max_radius_edge);
        println!("   volume rel err {:.3e} (vs analytic), time {:.2?}", (v - want).abs() / want, t0.elapsed());
    }
}

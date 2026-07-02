//! Issue #1 repro: two face-adjacent boxes forming a tee, flush over a
//! sub-rectangle, at mm scale. Usage: repro_tee [maxh] (default 1e-3).
use rapidmesh_geom::{solid_box, Scene};
use rapidmesh_tet::{mesh_plc_with, quality_stats, MeshParams};

fn main() {
    let maxh: f64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1e-3);
    let mm = 1e-3;
    let mut scene = Scene::new();
    // Main arm and a stub flush on its y=4mm face over a sub-rectangle.
    scene.add_solid(solid_box([0.0, 0.0, 0.0], [10.0 * mm, 4.0 * mm, 2.0 * mm]));
    scene.add_solid(solid_box([3.0 * mm, 4.0 * mm, 0.0], [7.0 * mm, 8.0 * mm, 2.0 * mm]));
    let plc = scene.assemble();
    println!("plc: {} verts, {} tris", plc.vertices.len(), plc.triangles.len());
    let params = MeshParams { maxh, ..Default::default() };
    let m = mesh_plc_with(&plc, &params);
    let q = quality_stats(&m);
    println!("tets {}  min dihedral {:.3} deg  slivers {}", q.n_tets, q.min_dihedral_deg, q.n_slivers);
    // Volume check: exact tee volume = 10*4*2 + 4*4*2 = 112 mm^3.
    let mut vol = 0.0;
    for t in &m.tets {
        let p: [[f64; 3]; 4] = std::array::from_fn(|k| m.points[t[k]]);
        let d = |i: usize, k: usize| p[i][k] - p[0][k];
        vol += (d(1, 0) * (d(2, 1) * d(3, 2) - d(2, 2) * d(3, 1))
            - d(1, 1) * (d(2, 0) * d(3, 2) - d(2, 2) * d(3, 0))
            + d(1, 2) * (d(2, 0) * d(3, 1) - d(2, 1) * d(3, 0))).abs() / 6.0;
    }
    let want = 112.0 * mm * mm * mm;
    println!("volume {:.6e} vs exact {:.6e} (rel err {:.2e})", vol, want, (vol - want).abs() / want);
}

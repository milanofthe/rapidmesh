//! Probe: mesh_refine vs mesh_cdt on simple geometries.
use rapidmesh_geom::{cylinder, solid_box, Scene};
use rapidmesh_tet::{mesh_refine, quality_stats, MeshParams};

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

/// Watertight: every boundary face (used once among region!=0 tets... ) simpler:
/// count facets used exactly once -> they must tile the outer surface; and
/// check non-manifold edges of the region-boundary faces.
fn report(label: &str, m: &rapidmesh_tet::TetMesh, want_vol: f64) {
    let q = quality_stats(m);
    println!("== {label}: tets {} points {} faces {}", m.tets.len(), m.points.len(), m.faces.len());
    println!("   min dihedral {:.2} deg, slivers {}, max radius-edge {:.2}", q.min_dihedral_deg, q.n_slivers, q.max_radius_edge);
    let v = volume(m);
    println!("   volume {:.6e} vs {:.6e} (rel {:.2e})", v, want_vol, (v - want_vol).abs() / want_vol);
    // boundary face edge manifoldness: each edge of the tagged faces should
    // bound exactly 2 faces (closed surface) unless on a feature/junction
    let mut ecount: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
    for f in &m.faces {
        for k in 0..3 {
            let (a, b) = (f.tri[k], f.tri[(k + 1) % 3]);
            *ecount.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    let odd = ecount.values().filter(|&&c| c % 2 == 1).count();
    println!("   boundary-face edges: {} total, {} odd-incidence", ecount.len(), odd);
}

fn main() {
    let t0 = std::time::Instant::now();
    let mut scene = Scene::new();
    scene.add_solid(solid_box([0.0, 0.0, 0.0], [2.0, 1.0, 1.0]));
    let plc = scene.assemble();
    let m = mesh_refine(&plc, &MeshParams { maxh: 0.3, ..Default::default() });
    report("box maxh=0.3", &m, 2.0);
    println!("   time {:.2?}", t0.elapsed());

    let t0 = std::time::Instant::now();
    let mm = 1e-3;
    let mut scene = Scene::new();
    scene.add_solid(solid_box([0.0, 0.0, 0.0], [10.0 * mm, 4.0 * mm, 2.0 * mm]));
    scene.add_solid(solid_box([3.0 * mm, 4.0 * mm, 0.0], [7.0 * mm, 8.0 * mm, 2.0 * mm]));
    let plc = scene.assemble();
    let m = mesh_refine(&plc, &MeshParams { maxh: 1.0 * mm, ..Default::default() });
    report("tee maxh=1mm", &m, 112.0 * mm * mm * mm);
    println!("   time {:.2?}", t0.elapsed());

    let t0 = std::time::Instant::now();
    let mut scene = Scene::new();
    scene.add_solid(solid_box([-2.0, -2.0, 0.0], [2.0, 2.0, 1.0]));
    scene.add_void(cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.75, 24));
    let plc = scene.assemble();
    let m = mesh_refine(&plc, &MeshParams { maxh: 0.4, ..Default::default() });
    // block minus 24-gon prism: 16 - area_24gon(r=0.75)*1; analytic cylinder vol used loosely
    let a24 = 0.5 * 24.0 * 0.75 * 0.75 * (2.0 * std::f64::consts::PI / 24.0).sin();
    report("drilled block maxh=0.4", &m, 16.0 - a24);
    println!("   time {:.2?}", t0.elapsed());
}

//! Repro for recovery failures on the torus (concave inner side).
//!
//!     cargo run --release -p rapidmesh --example repro_torus -- 16 8

use rapidmesh_geom::{torus, Scene};
use rapidmesh_tet::{mesh_plc_with, quality_stats, MeshParams};

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("segments"))
        .collect();
    let (sj, sn) = (
        args.first().copied().unwrap_or(16),
        args.get(1).copied().unwrap_or(8),
    );
    let mut scene = Scene::new();
    scene.add_solid(torus([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 2.0, 0.5, sj, sn));
    let plc = scene.assemble();
    let params = MeshParams {
        maxh: 0.4,
        ..MeshParams::default()
    };
    let t0 = std::time::Instant::now();
    let mesh = mesh_plc_with(&plc, &params);
    let q = quality_stats(&mesh);
    println!(
        "torus {sj}x{sn}: {} tets {} pts min-dih {:.2} r/e {:.2} {:?}",
        mesh.tets.len(),
        mesh.points.len(),
        q.min_dihedral_deg,
        q.max_radius_edge,
        t0.elapsed()
    );
}

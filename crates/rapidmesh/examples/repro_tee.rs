//! Repro for the T-junction recovery spiral: an H-plane waveguide tee tiled
//! from two face-adjacent (partially flush) air boxes -- the crossbar's bottom
//! face is coincident with the stem's top face over the hub sub-rectangle.
//! The CSG arrangement handles it, but the face/segment recovery alternation
//! can fail to converge at mm scale (the rapidfem power-divider example dodges
//! this by modeling the tee as one prism).
//!
//!     RAPIDMESH_RECOVER_TRACE=1 cargo run --release -p rapidmesh --example repro_tee

use rapidmesh_geom::{solid_box, Scene};
use rapidmesh_tet::{mesh_plc_with, MeshParams};

fn main() {
    let args: Vec<f64> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let s = args.first().copied().unwrap_or(1e-3);
    let maxh = args.get(1).copied().unwrap_or(4e-3);
    let (w, h, l) = (20.0 * s, 10.0 * s, 40.0 * s);
    let mut scene = Scene::new();
    // crossbar: full x span, y in [-w/2, w/2]
    scene.add_solid(solid_box([-(w / 2.0 + l), -w / 2.0, -h / 2.0], [w / 2.0 + l, w / 2.0, h / 2.0]));
    // stem: below the crossbar, top face flush over x in [-w/2, w/2]
    scene.add_solid(solid_box([-w / 2.0, -w / 2.0 - l, -h / 2.0], [w / 2.0, -w / 2.0, h / 2.0]));

    let plc = scene.assemble();
    let params = MeshParams {
        maxh,
        region_maxh: vec![],
        radius_edge_bound: 2.0,
        max_points: 500_000,
        grading: 0.5,
        face_maxh: vec![],
        surface_maxh: vec![],
        size_points: vec![],
    };
    eprintln!("meshing flush tee (mm scale)...");
    let mesh = mesh_plc_with(&plc, &params);
    eprintln!("done: {} tets", mesh.tets.len());
}

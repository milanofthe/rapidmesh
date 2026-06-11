//! Repro: face_maxh on sheets coincident with a region interface starves
//! the interface patch tiling (stagnant POSITIVE deficit, real holes).
//!
//!     cargo run --release -p rapidmesh --example repro_trace_sizing

use rapidmesh_geom::{sheet_rect, solid_box, FaceTag, Scene};
use rapidmesh_tet::{mesh_plc_with, quality_stats, MeshParams};

fn main() {
    let mm = 1e-3;
    let mil = 0.0254 * mm;
    let lengths: Vec<f64> = [400.0, 660.0, 660.0, 660.0, 660.0, 660.0, 400.0]
        .iter()
        .map(|x| x * mil)
        .collect();
    let widths: Vec<f64> = [50.0, 128.0, 8.0, 224.0, 8.0, 128.0, 50.0]
        .iter()
        .map(|x| x * mil)
        .collect();
    let sub_h = 62.0 * mil;
    let air_h = 15.0 * mm;
    let pad_y = 12.0 * mm;
    let c0 = 299_792_458.0_f64;
    let maxh = c0 / 8.0e9 / 12.0;
    let total_l: f64 = lengths.iter().sum();
    let sub_w = widths.iter().cloned().fold(0.0, f64::max) + 2.0 * pad_y;
    let x_lo = -total_l / 2.0;

    let mut scene = Scene::new();
    scene.add_solid(solid_box(
        [x_lo, -sub_w / 2.0, 0.0],
        [total_l / 2.0, sub_w / 2.0, air_h + sub_h],
    ));
    scene.add_solid(solid_box(
        [x_lo, -sub_w / 2.0, 0.0],
        [total_l / 2.0, sub_w / 2.0, sub_h],
    ));
    let mut x = x_lo;
    for (l, w) in lengths.iter().zip(&widths) {
        scene.add_sheet(
            sheet_rect([x, -w / 2.0, sub_h], [*l, 0.0, 0.0], [0.0, *w, 0.0]),
            FaceTag(10),
        );
        x += l;
    }
    let plc = scene.assemble();
    let params = MeshParams {
        maxh,
        face_maxh: vec![(10, 0.4 * mm)],
        ..MeshParams::default()
    };
    let t0 = std::time::Instant::now();
    let mesh = mesh_plc_with(&plc, &params);
    let q = quality_stats(&mesh);
    println!(
        "{} tets  {} pts  min-dih {:5.2}  r/e {:8.2}  abandoned {:?}  {:?}",
        mesh.tets.len(),
        mesh.points.len(),
        q.min_dihedral_deg,
        q.max_radius_edge,
        mesh.abandoned_patches,
        t0.elapsed()
    );
}

//! Repro for queue-refinement overshoot on the patch-antenna example
//! geometry (thin substrate, two tagged sheets with rim creases).
//!
//!     cargo run --release -p rapidmesh --example repro_patch_antenna

use rapidmesh_geom::{sheet_rect, solid_box, FaceTag, Scene};
use rapidmesh_tet::{mesh_plc_with, quality_stats, MeshParams};

fn main() {
    let mm = 1e-3;
    let c0 = 299_792_458.0_f64;
    let f = 2.8e9_f64;
    let lambda_maxh = |er: f64| c0 / (f * 10.0 * er.sqrt());

    let (sub_w, sub_l, sub_h, er_sub) = (60.0 * mm, 60.0 * mm, 1.6 * mm, 4.4);
    let (patch_w, patch_l) = (38.0 * mm, 29.0 * mm);
    let (pad_xy, pad_z) = (25.0 * mm, 30.0 * mm);
    let total_w = sub_w + 2.0 * pad_xy;
    let total_l = sub_l + 2.0 * pad_xy;

    let mut scene = Scene::new();
    scene.add_solid(solid_box(
        [-total_w / 2.0, -total_l / 2.0, 0.0],
        [total_w / 2.0, total_l / 2.0, sub_h + pad_z],
    ));
    let sub = scene.add_solid(solid_box(
        [-sub_w / 2.0, -sub_l / 2.0, 0.0],
        [sub_w / 2.0, sub_l / 2.0, sub_h],
    ));
    scene.add_sheet(
        sheet_rect(
            [-patch_w / 2.0, -patch_l / 2.0, sub_h],
            [patch_w, 0.0, 0.0],
            [0.0, patch_l, 0.0],
        ),
        FaceTag(7),
    );
    scene.add_sheet(
        sheet_rect(
            [-sub_w / 2.0, -sub_l / 2.0, 0.0],
            [sub_w, 0.0, 0.0],
            [0.0, sub_l, 0.0],
        ),
        FaceTag(7),
    );
    let plc = scene.assemble();
    let grading: f64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(0.5);
    let params = MeshParams {
        maxh: lambda_maxh(1.0),
        region_maxh: vec![(sub.0, lambda_maxh(er_sub) / 2.0)],
        grading,
        ..MeshParams::default()
    };
    let t0 = std::time::Instant::now();
    let mesh = mesh_plc_with(&plc, &params);
    let q = quality_stats(&mesh);
    println!(
        "{} tets  {} pts  min-dih {:5.2}  r/e {:8.2}  {:?}",
        mesh.tets.len(),
        mesh.points.len(),
        q.min_dihedral_deg,
        q.max_radius_edge,
        t0.elapsed()
    );
    // Per-region density check: mean longest edge per tet.
    let mut by_region: std::collections::BTreeMap<u32, (usize, f64)> =
        std::collections::BTreeMap::new();
    for (t, r) in mesh.tets.iter().zip(&mesh.tet_regions) {
        let mut lmax: f64 = 0.0;
        for i in 0..4 {
            for j in i + 1..4 {
                let d: f64 = (0..3)
                    .map(|k| (mesh.points[t[i]][k] - mesh.points[t[j]][k]).powi(2))
                    .sum();
                lmax = lmax.max(d.sqrt());
            }
        }
        let e = by_region.entry(r.0).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += lmax;
    }
    for (r, (cnt, sum)) in by_region {
        println!("  region {r}: {cnt:7} tets, mean lmax {:.4} mm", 1e3 * sum / cnt as f64);
    }
}

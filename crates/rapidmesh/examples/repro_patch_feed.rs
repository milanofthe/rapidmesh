//! Repro for the patch-antenna crease-recovery runaway: a vertical feed
//! sheet inside the substrate whose TOP EDGE lies on the patch sheet's rim
//! (and inside the substrate-top interface plane). Missing-crease counts
//! double per round with halving lengths (encroachment ping-pong).
//!
//!     cargo run --release -p rapidmesh --example repro_patch_feed [stage]

use rapidmesh_geom::{sheet_rect, solid_box, FaceTag, Scene};
use rapidmesh_tet::{mesh_plc_with, quality_stats, MeshParams};

fn main() {
    let mm = 1e-3;
    let (sub_w, sub_l, sub_h) = (60.0 * mm, 60.0 * mm, 1.6 * mm);
    let (patch_w, patch_l) = (38.0 * mm, 29.0 * mm);
    let feed_w = 1.5 * mm;
    let feed_y = -patch_l / 2.0;
    let pad_xy = 25.0 * mm;
    let pad_z = 60.0 * mm;
    let c0 = 299_792_458.0_f64;
    // exactly rapidfem's lambda_maxh expression (single division by the
    // product; the two-division variant differs by an ulp and dodges the
    // tessellation-lottery instance this repro chases)
    let maxh = c0 / (2.8e9_f64 * 12.0);

    let stage = std::env::args().nth(1).unwrap_or_else(|| "full".into());
    let pml_t = 20.0 * mm;
    let (x_out, y_out) = (sub_w / 2.0 + pad_xy, sub_l / 2.0 + pad_xy);
    let air_top = sub_h + pad_z;
    let mut scene = Scene::new();
    let mut region_maxh: Vec<(u32, f64)> = Vec::new();
    if stage != "nosub" {
        // big air box
        scene.add_solid(solid_box(
            [-sub_w / 2.0 - pad_xy, -sub_l / 2.0 - pad_xy, 0.0],
            [sub_w / 2.0 + pad_xy, sub_l / 2.0 + pad_xy, sub_h + pad_z],
        ));
    }
    if stage == "pml" || stage == "fullpml" {
        // the five PML slabs of the example, coarse like pml_air maxh=2*MAXH
        let slabs = [
            ([x_out, -y_out - pml_t, 0.0], [x_out + pml_t, y_out + pml_t, air_top]),
            ([-x_out - pml_t, -y_out - pml_t, 0.0], [-x_out, y_out + pml_t, air_top]),
            ([-x_out, y_out, 0.0], [x_out, y_out + pml_t, air_top]),
            ([-x_out, -y_out - pml_t, 0.0], [x_out, -y_out, air_top]),
            ([-x_out - pml_t, -y_out - pml_t, air_top],
             [x_out + pml_t, y_out + pml_t, air_top + pml_t]),
        ];
        for (lo, hi) in slabs {
            let r = scene.add_solid(solid_box(lo, hi));
            region_maxh.push((r.0, 2.0 * maxh));
        }
    }
    let sub = scene.add_solid(solid_box(
        [-sub_w / 2.0, -sub_l / 2.0, 0.0],
        [sub_w / 2.0, sub_l / 2.0, sub_h],
    ));
    if stage != "nopatch" {
        scene.add_sheet(
            sheet_rect(
                [-patch_w / 2.0, -patch_l / 2.0, sub_h],
                [patch_w, 0.0, 0.0],
                [0.0, patch_l, 0.0],
            ),
            FaceTag(7),
        );
    }
    if stage != "nofeed" {
        scene.add_sheet(
            sheet_rect(
                [-feed_w / 2.0, feed_y, 0.0],
                [feed_w, 0.0, 0.0],
                [0.0, 0.0, sub_h],
            ),
            FaceTag(8),
        );
    }
    // substrate sizing like the example: min(material 1.5*sub_h, auto thickness/1.5)
    region_maxh.push((sub.0, (sub_h / 1.5).min(1.5 * sub_h)));
    let plc = scene.assemble();
    let params = MeshParams {
        maxh,
        region_maxh,
        max_points: 2_000_000,
        ..MeshParams::default()
    };
    let t0 = std::time::Instant::now();
    let mesh = mesh_plc_with(&plc, &params);
    let q = quality_stats(&mesh);
    println!(
        "stage {stage}: {} tets  min-dih {:5.2}  abandoned {:?}  {:?}",
        mesh.tets.len(),
        q.min_dihedral_deg,
        mesh.abandoned_patches,
        t0.elapsed()
    );
}

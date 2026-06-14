//! Profiling harness for the face-recovery (gift-wrap) hotspot: meshes a
//! fine-sized microstrip (substrate + air + PEC trace sheet, the seam-dense
//! case) and prints the mesh-stage timings + gift-wrap / predicate stats.
//! Runs via cargo (no Python pyd), so the gift-wrap fix can be A/B'd with
//! `git stash`.
//!
//!     cargo run --release -p rapidmesh --example profile_gw

use rapidmesh_geom::{cylinder, sheet_rect, solid_box, FaceTag, Scene};
use rapidmesh_tet::{mesh_plc_with, MeshParams};

fn report(name: &str, scene: &Scene, params: MeshParams) {
    rapidmesh_exact::log::clear();
    let plc = scene.assemble();
    let t0 = std::time::Instant::now();
    let mesh = mesh_plc_with(&plc, &params);
    let elapsed = t0.elapsed();
    let (timings, stats, _) = rapidmesh_exact::log::take();
    println!("{name}: {} tets, {} pts in {:?}", mesh.tets.len(), mesh.points.len(), elapsed);
    for (k, v) in &timings {
        if k.starts_with("mesh.") {
            println!("  {k:<16} {:.3}s", v);
        }
    }
    for (k, v) in &stats {
        if k.contains("gift_wrap") || k.contains("predicates") || k.contains("facets")
            || k.contains("rounds")
        {
            println!("  {k:<28} {v}");
        }
    }
}

fn main() {
    let mm = 1e-3;

    // coax impedance step: outer dielectric + two stepped inner-conductor
    // voids with finer wall sizing (the heaviest showcase scene, ~19 s).
    {
        let (ri1, ri2, ro, l1, l2) = (1.5 * mm, 0.99 * mm, 3.45 * mm, 15.0 * mm, 15.0 * mm);
        let mut scene = Scene::new();
        scene.add_solid(cylinder([0.0, 0.0, 0.0], [0.0, 0.0, l1 + l2], ro, 24));
        scene.add_void(cylinder([0.0, 0.0, 0.0], [0.0, 0.0, l1], ri1, 24));
        scene.add_void(cylinder([0.0, 0.0, l1], [0.0, 0.0, l2], ri2, 24));
        report(
            "coax_step",
            &scene,
            MeshParams {
                maxh: 1.4 * mm,
                region_maxh: vec![],
                radius_edge_bound: 2.0,
                max_points: 500_000,
                grading: 0.35,
                face_maxh: vec![],
                surface_maxh: vec![(1, 0.45 * mm), (2, 0.32 * mm)],
                size_points: vec![],
            },
        );
    }

    // microstrip: substrate + air + PEC trace sheet (seam-dense interface).
    {
        let (sub_w, line_l, sub_h, air_h, line_w) =
            (20.0 * mm, 30.0 * mm, 0.508 * mm, 10.0 * mm, 1.13 * mm);
        let mut scene = Scene::new();
        scene.add_solid(solid_box([-sub_w / 2.0, 0.0, 0.0], [sub_w / 2.0, line_l, air_h + sub_h]));
        let subst = scene.add_solid(solid_box([-sub_w / 2.0, 0.0, 0.0], [sub_w / 2.0, line_l, sub_h]));
        scene.add_sheet(
            sheet_rect([-line_w / 2.0, 0.0, sub_h], [line_w, 0.0, 0.0], [0.0, line_l, 0.0]),
            FaceTag(7),
        );
        report(
            "microstrip",
            &scene,
            MeshParams {
                maxh: 4.0 * mm,
                region_maxh: vec![(subst.0, 0.7 * mm)],
                radius_edge_bound: 2.0,
                max_points: 500_000,
                grading: 0.35,
                face_maxh: vec![(7, 0.4 * mm)],
                surface_maxh: vec![],
                size_points: vec![],
            },
        );
    }
}

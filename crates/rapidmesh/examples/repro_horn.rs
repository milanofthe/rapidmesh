//! Repro for the horn-antenna loft "giving up on patch tiling (-2.6% area
//! uncovered)" warning: a NEGATIVE deficit means the tiling scan counts MORE
//! projected area than the patch owns, which repair points can never fix, so
//! the stagnation guard abandons an intact patch.
//!
//!     cargo run --release -p rapidmesh --example repro_horn

use rapidmesh_geom::{loft, solid_box, Scene};
use rapidmesh_tet::{mesh_plc_with, quality_stats, MeshParams};

fn main() {
    let mm = 1e-3;
    let c0 = 299_792_458.0_f64;
    let maxh = c0 / 11.0e9 / 8.0; // horn walls, lambda/8 at 11 GHz
    let maxh_air = c0 / 11.0e9 / 3.0;

    let (wga, wgb) = (22.86 * mm, 10.16 * mm);
    let l_feed = 15.0 * mm;
    let l_horn = 50.0 * mm;
    let (wh, hh) = (30.0 * mm, 22.0 * mm);
    let (lpad_beam, lpad_side) = (88.0 * mm, 30.0 * mm);
    let pml_t = 15.0 * mm;
    let (x0, x1) = (-l_feed, l_horn + lpad_beam);
    let (y0, y1) = (-wh / 2.0 - lpad_side, wh / 2.0 + lpad_side);
    let (z0, z1) = (-hh / 2.0 - lpad_side, hh / 2.0 + lpad_side);

    let mut scene = Scene::new();
    scene.add_solid(solid_box([x0, y0, z0], [x1, y1, z1]));
    let pml = scene.add_solid(solid_box([x1, y0, z0], [x1 + pml_t, y1, z1]));
    let feed = scene.add_solid(solid_box(
        [-l_feed, -wga / 2.0, -wgb / 2.0],
        [0.0, wga / 2.0, wgb / 2.0],
    ));
    let horn = scene.add_solid(loft(
        &[
            [0.0, -wga / 2.0, -wgb / 2.0],
            [0.0, wga / 2.0, -wgb / 2.0],
            [0.0, wga / 2.0, wgb / 2.0],
            [0.0, -wga / 2.0, wgb / 2.0],
        ],
        &[
            [l_horn, -wh / 2.0, -hh / 2.0],
            [l_horn, wh / 2.0, -hh / 2.0],
            [l_horn, wh / 2.0, hh / 2.0],
            [l_horn, -wh / 2.0, hh / 2.0],
        ],
    ));
    let plc = scene.assemble();
    let params = MeshParams {
        maxh: maxh_air,
        region_maxh: vec![
            (pml.0, 2.0 * maxh),
            (feed.0, wgb / 3.0),
            (horn.0, maxh),
        ],
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
}

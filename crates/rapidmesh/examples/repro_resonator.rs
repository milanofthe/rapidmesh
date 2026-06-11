//! Repro harness for the cylinder-tessellation recovery lottery: meshes the
//! dielectric_resonator example geometry for a range of `segments` values and
//! reports recovery health (abandoned patches print to stderr).
//!
//!     cargo run --release -p rapidmesh --example repro_resonator -- 24

use rapidmesh_geom::{cylinder, solid_box, Scene};
use rapidmesh_tet::{mesh_plc_with, quality_stats, MeshParams};

fn main() {
    let mm = 1e-3;
    let inch = 25.4 * mm;
    let c0 = 299_792_458.0_f64;
    let f = 3.0e9_f64;
    let lambda_maxh = |er: f64| c0 / (f * 10.0 * er.sqrt());

    let w = 2.0 * inch;
    let s = 2.03 * inch;
    let (d_sup, l_sup, er_sup) = (0.56 * inch, 0.80 * inch, 10.0);
    let (d_res, l_res, er_res) = (1.176 * inch, 0.481 * inch, 34.0);

    let args: Vec<String> = std::env::args().skip(1).collect();
    let segs: Vec<usize> = if args.is_empty() {
        vec![20, 24, 28, 32]
    } else {
        args.iter().map(|a| a.parse().expect("segments")).collect()
    };

    for &segments in &segs {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([-w / 2.0, -w / 2.0, 0.0], [w / 2.0, w / 2.0, s]));
        let sup = scene.add_solid(cylinder(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, l_sup],
            d_sup / 2.0,
            segments,
        ));
        let res = scene.add_solid(cylinder(
            [0.0, 0.0, l_sup],
            [0.0, 0.0, l_res],
            d_res / 2.0,
            segments,
        ));
        let plc = scene.assemble();
        let params = MeshParams {
            maxh: lambda_maxh(1.0),
            region_maxh: vec![(sup.0, lambda_maxh(er_sup)), (res.0, lambda_maxh(er_res))],
            ..MeshParams::default()
        };
        let t0 = std::time::Instant::now();
        let mesh = mesh_plc_with(&plc, &params);
        let q = quality_stats(&mesh);
        println!(
            "segments={segments:3}  {:6} tets  {:6} pts  min-dih {:5.2}  r/e {:8.2}  {:?}",
            mesh.tets.len(),
            mesh.points.len(),
            q.min_dihedral_deg,
            q.max_radius_edge,
            t0.elapsed()
        );
    }
}

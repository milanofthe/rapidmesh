//! CVT mesher benchmark: meshes a few representative geometries at several
//! target sizes and prints the per-stage timing breakdown (surface / seed /
//! lloyd / classify), point and tet counts, and worst dihedral, so we can see
//! where the time goes and how it scales before optimizing.
//!
//!     cargo run --release -p rapidmesh --example bench

use rapidmesh_geom::{cylinder, solid_box, Scene, TaggedPlc};
use rapidmesh_tet::{mesh_plc_with, MeshParams};
use std::time::Instant;

fn val(v: &[(String, f64)], key: &str) -> f64 {
    v.iter().find(|(k, _)| k == key).map(|(_, x)| *x).unwrap_or(0.0)
}

fn run(name: &str, plc: &TaggedPlc, maxh: f64) {
    // Clear any pending log records.
    let _ = rapidmesh_exact::log::take();
    let params = MeshParams { maxh, ..MeshParams::default() };
    let t = Instant::now();
    let _mesh = mesh_plc_with(plc, &params);
    let total = t.elapsed().as_secs_f64() * 1e3;
    let (timings, stats, _) = rapidmesh_exact::log::take();
    let ms = |k: &str| val(&timings, k) * 1e3;
    println!(
        "{:<14} {:>5.3} | {:>7} pts {:>8} tets | tot {:>8.1} | surf {:>7.1} seed {:>6.1} lloyd {:>8.1} cls {:>7.1} | mindih {:>5.1}",
        name,
        maxh,
        val(&stats, "mesh.points") as usize,
        val(&stats, "mesh.tets") as usize,
        total,
        ms("mesh.surface"),
        ms("mesh.seed"),
        ms("mesh.lloyd"),
        ms("mesh.classify"),
        val(&stats, "mesh.min_dihedral_deg"),
    );
}

fn box_plc(s: f64) -> TaggedPlc {
    let mut scene = Scene::new();
    scene.add_solid(solid_box([0.0, 0.0, 0.0], [s, s, s]));
    scene.assemble()
}

fn nested_plc() -> TaggedPlc {
    let mut scene = Scene::new();
    scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
    scene.add_solid(solid_box([1.0, 1.0, 1.0], [3.0, 3.0, 2.0]));
    scene.assemble()
}

fn cylinder_plc(seg: usize) -> TaggedPlc {
    let mut scene = Scene::new();
    scene.add_solid(cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], 1.5, seg));
    scene.assemble()
}

fn main() {
    println!("=== CVT mesher benchmark (times in ms) ===");
    let cube = box_plc(4.0);
    for &h in &[1.0, 0.6, 0.4, 0.3] {
        run("box", &cube, h);
    }
    let nested = nested_plc();
    for &h in &[1.0, 0.6, 0.4] {
        run("air+diel", &nested, h);
    }
    let cyl = cylinder_plc(24);
    for &h in &[1.0, 0.6, 0.4] {
        run("cylinder", &cyl, h);
    }
}

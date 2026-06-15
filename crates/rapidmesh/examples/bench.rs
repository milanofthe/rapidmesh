//! CVT mesher benchmark: meshes representative geometries over a sequence of
//! shrinking target sizes and prints the per-stage timing breakdown (domain
//! octree / surface / seed / lloyd / classify), point and tet counts, worst
//! dihedral, and the LOG-LOG SCALING EXPONENT of each stage vs tet count, so we
//! can see which stage dominates and how each scales BEFORE optimizing.
//!
//!     cargo run --release -p rapidmesh --example bench
//!
//! Exponent reading: ~1.0 is linear in tets (ideal), ~1.33 is N^(4/3) (e.g. a
//! per-boundary-tet O(F) raycast), ~1.67 is N^(5/3) (e.g. O(leaves*facets)
//! build), ~2.0 is quadratic (e.g. an all-pairs / all-triangle scan).

use rapidmesh_geom::{cylinder, solid_box, Scene, TaggedPlc};
use rapidmesh_tet::{mesh_plc_with, MeshParams};
use std::time::Instant;

/// The stages we track, in print order. `total` is wall-clock of the whole call.
const STAGES: [&str; 6] =
    ["mesh.domain", "mesh.surface", "mesh.seed", "mesh.lloyd", "mesh.classify", "total"];

fn val(v: &[(String, f64)], key: &str) -> f64 {
    v.iter().find(|(k, _)| k == key).map(|(_, x)| *x).unwrap_or(0.0)
}

struct Sample {
    n_tets: f64,
    /// Milliseconds per stage, parallel to `STAGES`.
    ms: [f64; 6],
}

fn run(name: &str, plc: &TaggedPlc, maxh: f64) -> Sample {
    let _ = rapidmesh_exact::log::take(); // clear pending records
    let params = MeshParams { maxh, ..MeshParams::default() };
    let t = Instant::now();
    let _mesh = mesh_plc_with(plc, &params);
    let total = t.elapsed().as_secs_f64() * 1e3;
    let (timings, stats, _) = rapidmesh_exact::log::take();
    let mut ms = [0.0f64; 6];
    for (i, s) in STAGES.iter().enumerate() {
        ms[i] = if *s == "total" { total } else { val(&timings, s) * 1e3 };
    }
    let n_tets = val(&stats, "mesh.tets");
    println!(
        "{:<10} h{:>5.3} | {:>7} pts {:>8} tets | dom {:>6.1} surf {:>7.1} seed {:>6.1} \
         lloyd {:>8.1} cls {:>6.1} | tot {:>8.1} | mindih {:>4.1}",
        name,
        maxh,
        val(&stats, "mesh.points") as usize,
        n_tets as usize,
        ms[0],
        ms[1],
        ms[2],
        ms[3],
        ms[4],
        ms[5],
        val(&stats, "mesh.min_dihedral_deg"),
    );
    Sample { n_tets, ms }
}

/// Least-squares slope of ln(ms) vs ln(n_tets) across the samples: the scaling
/// exponent. Stages with negligible time on any sample are skipped (NaN).
fn exponent(samples: &[Sample], stage: usize) -> f64 {
    let pts: Vec<(f64, f64)> = samples
        .iter()
        .filter(|s| s.n_tets > 0.0 && s.ms[stage] > 0.05)
        .map(|s| (s.n_tets.ln(), s.ms[stage].ln()))
        .collect();
    if pts.len() < 2 {
        return f64::NAN;
    }
    let n = pts.len() as f64;
    let sx: f64 = pts.iter().map(|p| p.0).sum();
    let sy: f64 = pts.iter().map(|p| p.1).sum();
    let sxx: f64 = pts.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = pts.iter().map(|p| p.0 * p.1).sum();
    (n * sxy - sx * sy) / (n * sxx - sx * sx)
}

fn sweep(name: &str, plc: &TaggedPlc, hs: &[f64]) {
    let samples: Vec<Sample> = hs.iter().map(|&h| run(name, plc, h)).collect();
    print!("{name:<10} exponents (time ~ tets^k):");
    for (i, s) in STAGES.iter().enumerate() {
        let label = s.strip_prefix("mesh.").unwrap_or(s);
        print!(" {label} {:>4.2}", exponent(&samples, i));
    }
    println!("\n");
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
    println!("=== CVT mesher benchmark (times in ms) ===\n");
    // Shrinking h ~ factor 1.4 per step: each step ~2.7x more tets, ~5 doublings
    // total, enough to fit a clean scaling exponent.
    sweep("box", &box_plc(4.0), &[0.7, 0.5, 0.36, 0.26, 0.18, 0.13]);
    sweep("air+diel", &nested_plc(), &[0.7, 0.5, 0.36, 0.26, 0.18]);
    sweep("cylinder", &cylinder_plc(24), &[0.7, 0.5, 0.36, 0.26, 0.18]);
}

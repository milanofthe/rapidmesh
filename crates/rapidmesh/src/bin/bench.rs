//! Benchmark harness: meshes the comparison scenes and any surface models in
//! bench/models/ (STL/OBJ, fetched by bench/fetch_models.py), with per-stage
//! timing and quality statistics. Writes bench/results.json, viewer JSONs for
//! the surface models, and PLC JSONs (bench/plc/) so the tetgen reference
//! script meshes identical inputs.

use rapidmesh::export::{write_manifest, write_mesh_json, write_plc_json};
use rapidmesh::scenes::comparison_scenes;
use rapidmesh_geom::{import_obj, import_stl, min_height_ratio, validate_closed, Scene, TaggedPlc};
use rapidmesh_tet::{mesh_plc_with, optimize, quality_stats, MeshParams, OptimizeParams};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Viewer JSONs above this tet count are skipped (the canvas viewer is not
/// built for multi-hundred-MB payloads).
const VIEWER_TET_LIMIT: usize = 300_000;

/// Models whose smallest facet height (relative to the bounding-box
/// diagonal) is below this need mesh repair (near-degenerate facets defeat
/// exact conforming meshing) and are skipped.
const MIN_FACET_HEIGHT_REL: f64 = 5e-4;

#[derive(Serialize)]
struct BenchRecord {
    name: String,
    kind: String,
    n_input_facets: usize,
    n_plc_facets: usize,
    n_points: usize,
    n_tets: usize,
    min_dihedral_deg: f64,
    max_radius_edge: f64,
    assemble_ms: u128,
    mesh_ms: u128,
    optimize_ms: u128,
    total_ms: u128,
}

struct OutDirs {
    meshes: PathBuf,
    plc: PathBuf,
}

fn run_one(
    name: &str,
    kind: &str,
    n_input_facets: usize,
    plc: &TaggedPlc,
    params: &MeshParams,
    assemble_ms: u128,
    dirs: &OutDirs,
) -> BenchRecord {
    let (out_dir, plc_dir): (&Path, &Path) = (&dirs.meshes, &dirs.plc);
    println!("  assembled: {} facets, {} ms", plc.triangles.len(), assemble_ms);
    let t1 = Instant::now();
    let mut mesh = mesh_plc_with(plc, params);
    let mesh_ms = t1.elapsed().as_millis();
    println!("  meshed: {} tets, {} ms", mesh.tets.len(), mesh_ms);
    let t2 = Instant::now();
    optimize(&mut mesh, &OptimizeParams::default());
    let optimize_ms = t2.elapsed().as_millis();
    println!("  optimized: {} ms", optimize_ms);
    let total_ms = assemble_ms + mesh_ms + optimize_ms;
    let q = quality_stats(&mesh);
    if mesh.tets.len() <= VIEWER_TET_LIMIT {
        write_mesh_json(out_dir, "rapidmesh", name, &mesh, &q, total_ms);
    } else {
        println!("  (viewer json skipped, {} tets)", mesh.tets.len());
    }
    write_plc_json(plc_dir, name, plc, params, &mesh);
    BenchRecord {
        name: name.to_string(),
        kind: kind.to_string(),
        n_input_facets,
        n_plc_facets: plc.triangles.len(),
        n_points: mesh.points.len(),
        n_tets: mesh.tets.len(),
        min_dihedral_deg: q.min_dihedral_deg,
        max_radius_edge: q.max_radius_edge,
        assemble_ms,
        mesh_ms,
        optimize_ms,
        total_ms,
    }
}

fn main() {
    let dirs = OutDirs {
        meshes: PathBuf::from("viewer/public/meshes"),
        plc: PathBuf::from("bench/plc"),
    };
    std::fs::create_dir_all(&dirs.meshes).expect("mkdir meshes");
    std::fs::create_dir_all(&dirs.plc).expect("mkdir plc");
    let mut records: Vec<BenchRecord> = Vec::new();

    // ------------------------------------------------------ CSG scenes
    for bs in comparison_scenes() {
        println!("scene {} ...", bs.name);
        let n_input: usize = 0; // primitives, no imported facets
        let t0 = Instant::now();
        let plc = bs.scene.assemble();
        let assemble_ms = t0.elapsed().as_millis();
        let params = MeshParams {
            maxh: bs.maxh,
            region_maxh: bs.region_maxh.clone(),
            radius_edge_bound: 2.0,
            max_points: 200_000,
        };
        records.push(run_one(
            bs.name,
            "scene",
            n_input,
            &plc,
            &params,
            assemble_ms,
            &dirs,
        ));
    }

    // -------------------------------------------------- surface models
    let models_dir = PathBuf::from("bench/models");
    let mut model_files: Vec<PathBuf> = std::fs::read_dir(&models_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    matches!(
                        p.extension().and_then(|s| s.to_str()),
                        Some("stl") | Some("obj")
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    // Smallest models first: scaling becomes visible early and a stuck big
    // model does not block all results.
    model_files.sort_by_key(|p| {
        (std::fs::metadata(p).map(|m| m.len()).unwrap_or(u64::MAX), p.clone())
    });
    if model_files.is_empty() {
        println!("no surface models in bench/models (run bench/fetch_models.py)");
    }
    for path in model_files {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        println!("model {name} ...");
        let t0 = Instant::now();
        let imported = match path.extension().and_then(|s| s.to_str()) {
            Some("stl") => import_stl(&path),
            _ => import_obj(&path),
        };
        let faceted = match imported {
            Ok(f) => f,
            Err(e) => {
                println!("  SKIP (import failed: {e})");
                continue;
            }
        };
        if let Err(e) = validate_closed(&faceted) {
            println!("  SKIP ({e})");
            continue;
        }
        let hr = min_height_ratio(&faceted);
        if hr < MIN_FACET_HEIGHT_REL {
            println!("  SKIP (needs mesh repair: min facet height {hr:.1e} of bbox diagonal)");
            continue;
        }
        let n_input = faceted.tris.len();
        let mut scene = Scene::new();
        scene.add_solid(faceted);
        let plc = scene.assemble();
        let assemble_ms = t0.elapsed().as_millis();
        let params = MeshParams {
            maxh: f64::INFINITY,
            region_maxh: Vec::new(),
            radius_edge_bound: 2.0,
            max_points: 500_000,
        };
        records.push(run_one(
            &name,
            "model",
            n_input,
            &plc,
            &params,
            assemble_ms,
            &dirs,
        ));
    }

    write_manifest(&dirs.meshes);

    // ----------------------------------------------------------- report
    println!();
    println!(
        "{:<20} {:>9} {:>8} {:>8} {:>8} {:>7} {:>9} {:>8} {:>7} {:>8}",
        "geometry", "in-faces", "points", "tets", "min-dih", "max-re", "assemble", "mesh", "opt", "total"
    );
    for r in &records {
        println!(
            "{:<20} {:>9} {:>8} {:>8} {:>8.1} {:>7.2} {:>8}ms {:>6}ms {:>5}ms {:>6}ms",
            r.name,
            r.n_input_facets,
            r.n_points,
            r.n_tets,
            r.min_dihedral_deg,
            r.max_radius_edge,
            r.assemble_ms,
            r.mesh_ms,
            r.optimize_ms,
            r.total_ms
        );
    }
    std::fs::create_dir_all("bench").expect("mkdir bench");
    std::fs::write(
        "bench/results.json",
        serde_json::to_string_pretty(&records).expect("serialize"),
    )
    .expect("write results");
    println!("\n-> bench/results.json");
}

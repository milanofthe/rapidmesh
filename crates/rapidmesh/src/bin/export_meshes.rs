//! Exports rapidmesh meshes of the comparison geometries as JSON for the
//! standalone viewer (viewer/public/meshes/), plus the assembled PLCs with
//! region seeds (bench/plc/) for the tetgen reference script. The gmsh/tetgen
//! reference exporters write the viewer schema with their own mesher prefix.

use rapidmesh::export::{write_manifest, write_mesh_json, write_plc_json};
use rapidmesh::scenes::comparison_scenes;
use rapidmesh_tet::{mesh_plc_with, optimize, quality_stats, MeshParams, OptimizeParams, TetMesh};
use std::time::Instant;

fn main() {
    let out_dir = std::path::PathBuf::from("viewer/public/meshes");
    let plc_dir = std::path::PathBuf::from("bench/plc");
    std::fs::create_dir_all(&out_dir).expect("mkdir meshes");
    std::fs::create_dir_all(&plc_dir).expect("mkdir plc");

    for bs in comparison_scenes() {
        let t0 = Instant::now();
        let plc = bs.scene.assemble();
        let params = MeshParams {
            maxh: bs.maxh,
            region_maxh: bs.region_maxh.clone(),
            radius_edge_bound: 2.0,
            max_points: 200_000,
            grading: 0.5,
            face_maxh: Vec::new(),
            surface_maxh: Vec::new(),
            size_points: Vec::new(),
            density_weighted: false,
            tol_edge: 1e-2,
            tol_surf: 1e-2,
            cap_edge: f64::INFINITY,
            cap_surf: f64::INFINITY,
            cap_vol: f64::INFINITY,
            edge_maxh: Vec::new(),
            edge_tol: Vec::new(),
            surf_maxh: Vec::new(),
            surf_tol: Vec::new(),
            ..Default::default()
        };
        let mut mesh: TetMesh = mesh_plc_with(&plc, &params);
        let opt = OptimizeParams {
            maxh: params.maxh,
            region_maxh: params.region_maxh.clone(),
            ..OptimizeParams::default()
        };
        optimize(&mut mesh, &opt);
        let millis = t0.elapsed().as_millis();
        let q = quality_stats(&mesh);
        println!(
            "{}: {} tets, min dihedral {:.1} deg, radius-edge {:.2}, {} ms",
            bs.name, q.n_tets, q.min_dihedral_deg, q.max_radius_edge, millis
        );
        write_mesh_json(&out_dir, "rapidmesh", bs.name, &mesh, &q, millis);
        write_plc_json(&plc_dir, bs.name, &plc, &params, &mesh);
    }

    write_manifest(&out_dir);
}

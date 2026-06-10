//! Exports rapidmesh meshes of the comparison geometries as JSON for the
//! standalone viewer (viewer/public/meshes/). The gmsh/tetgen reference
//! exporters write the same schema with their own mesher prefix.

use rapidmesh_geom::{cylinder, extrude_polygon, sheet_rect, solid_box, sphere, FaceTag, Scene};
use rapidmesh_tet::{mesh_plc_with, optimize, quality_stats, MeshParams, OptimizeParams, TetMesh};
use serde::Serialize;
use std::time::Instant;

#[derive(Serialize)]
struct FaceJson {
    tri: [usize; 3],
    tag: u32,
    regions: [u32; 2],
}

#[derive(Serialize)]
struct StatsJson {
    n_points: usize,
    n_tets: usize,
    min_dihedral_deg: f64,
    max_radius_edge: f64,
    max_edge: f64,
    millis: u128,
}

#[derive(Serialize)]
struct MeshJson {
    name: String,
    mesher: String,
    points: Vec<[f64; 3]>,
    tets: Vec<[usize; 4]>,
    tet_regions: Vec<u32>,
    faces: Vec<FaceJson>,
    stats: StatsJson,
}

fn export(name: &str, scene: &Scene, maxh: f64, out_dir: &std::path::Path) {
    export_with(name, scene, maxh, Vec::new(), out_dir)
}

fn export_with(
    name: &str,
    scene: &Scene,
    maxh: f64,
    region_maxh: Vec<(u32, f64)>,
    out_dir: &std::path::Path,
) {
    let t0 = Instant::now();
    let plc = scene.assemble();
    let params = MeshParams {
        maxh,
        region_maxh,
        radius_edge_bound: 2.0,
        max_points: 200_000,
    };
    let mut mesh: TetMesh = mesh_plc_with(&plc, &params);
    optimize(&mut mesh, &OptimizeParams::default());
    let millis = t0.elapsed().as_millis();
    let q = quality_stats(&mesh);
    println!(
        "{name}: {} tets, min dihedral {:.1} deg, radius-edge {:.2}, {} ms",
        q.n_tets, q.min_dihedral_deg, q.max_radius_edge, millis
    );
    let json = MeshJson {
        name: name.to_string(),
        mesher: "rapidmesh".to_string(),
        points: mesh.points.clone(),
        tets: mesh.tets.clone(),
        tet_regions: mesh.tet_regions.iter().map(|r| r.0).collect(),
        faces: mesh
            .faces
            .iter()
            .map(|f| FaceJson {
                tri: f.tri,
                tag: f.face_tag.0,
                regions: [f.regions[0].0, f.regions[1].0],
            })
            .collect(),
        stats: StatsJson {
            n_points: mesh.points.len(),
            n_tets: mesh.tets.len(),
            min_dihedral_deg: q.min_dihedral_deg,
            max_radius_edge: q.max_radius_edge,
            max_edge: q.max_edge,
            millis,
        },
    };
    let path = out_dir.join(format!("rapidmesh_{name}.json"));
    std::fs::write(&path, serde_json::to_string(&json).expect("serialize")).expect("write");
    println!("  -> {}", path.display());
}

fn main() {
    let out_dir = std::path::PathBuf::from("viewer/public/meshes");
    std::fs::create_dir_all(&out_dir).expect("mkdir");
    let mut names: Vec<&str> = Vec::new();

    // 1. The EM test scene: air box + dielectric block + PEC patch + sheet.
    {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
        let diel = scene.add_solid(solid_box([1.0, 1.0, 1.0], [3.0, 3.0, 2.0]));
        scene.add_sheet(
            sheet_rect([1.5, 1.5, 2.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            FaceTag(7),
        );
        scene.add_sheet(
            sheet_rect([0.5, 0.5, 3.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            FaceTag(7),
        );
        export_with("em_scene", &scene, 0.9, vec![(diel.0, 0.45)], &out_dir);
        names.push("em_scene");
    }

    // 2. Cylindrical via through a dielectric block.
    {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([-2.0, -2.0, 0.0], [2.0, 2.0, 1.0]));
        let via = scene.add_solid(cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.75, 12));
        export_with("via", &scene, 0.6, vec![(via.0, 0.3)], &out_dir);
        names.push("via");
    }

    // 3. Microstrip-like: substrate + air, PEC strip on the interface.
    {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [6.0, 3.0, 3.0]));
        let subst = scene.add_solid(solid_box([0.0, 0.0, 0.0], [6.0, 3.0, 0.5]));
        scene.add_sheet(
            sheet_rect([0.0, 1.25, 0.5], [6.0, 0.0, 0.0], [0.0, 0.5, 0.0]),
            FaceTag(7),
        );
        export_with("microstrip", &scene, 0.8, vec![(subst.0, 0.35)], &out_dir);
        names.push("microstrip");
    }

    // 4. Sphere in a box (curved geometry).
    {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]));
        let ball = scene.add_solid(sphere([0.0, 0.0, 0.0], 1.0, 16, 8));
        export_with("sphere", &scene, 0.8, vec![(ball.0, 0.4)], &out_dir);
        names.push("sphere");
    }

    // 5. L-shaped extruded resonator cavity inside air.
    {
        let l_shape = vec![
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ];
        let mut scene = Scene::new();
        scene.add_solid(solid_box([-1.0, -1.0, -1.0], [4.0, 3.0, 2.0]));
        scene.add_solid(extrude_polygon(
            &l_shape,
            &[],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ));
        export("l_prism", &scene, 0.5, &out_dir);
        names.push("l_prism");
    }

    // 6. Density transition: fine dielectric inside coarse air.
    {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
        let diel = scene.add_solid(solid_box([1.0, 1.0, 1.0], [3.0, 3.0, 2.0]));
        export_with("density_transition", &scene, 1.4, vec![(diel.0, 0.45)], &out_dir);
        names.push("density_transition");
    }

    std::fs::write(
        out_dir.join("manifest.json"),
        serde_json::to_string(&names).expect("serialize"),
    )
    .expect("write manifest");
    println!("manifest: {names:?}");
}

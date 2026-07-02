//! Refinement-core acceptance sweep over the known-problem geometries, with
//! viewer-JSON export for the headless renders (`bench/render_probe.py`): a
//! NORMAL view and a DEBUG view with located defect markers per geometry.
use rapidmesh_geom::{cylinder, frustum, icosphere, solid_box, torus, Scene};
use rapidmesh_tet::diagnostics::diagnose;
use rapidmesh_tet::{mesh_refine, optimize, quality_stats, MeshParams, OptimizeParams, TetMesh};
use std::path::Path;

/// Writes `probe_<name>.json` in the comparison-viewer schema (the same dict
/// `Mesh.to_viewer_dict` produces on the Python side), including the located
/// defects the DEBUG render overlays.
fn write_probe_json(dir: &Path, name: &str, mesh: &TetMesh, millis: u128) {
    use serde_json::json;
    let q = quality_stats(mesh);
    let dg = diagnose(mesh);
    let kind_str = |k: rapidmesh_tet::diagnostics::DefectKind| -> &'static str {
        use rapidmesh_tet::diagnostics::DefectKind::*;
        match k {
            Sliver => "sliver",
            NonManifoldEdge => "nonmanifold_edge",
            Straddler => "straddler",
        }
    };
    let doc = json!({
        "name": name,
        "mesher": "rapidmesh",
        "points": mesh.points,
        "tets": mesh.tets,
        "tet_regions": mesh.tet_regions.iter().map(|r| r.0).collect::<Vec<_>>(),
        "faces": mesh.faces.iter().map(|f| json!({
            "tri": f.tri,
            "tag": f.face_tag.0,
            "regions": [f.regions[0].0, f.regions[1].0],
            "surface": f.surface,
        })).collect::<Vec<_>>(),
        "surface_owners": mesh.surface_owners.iter()
            .map(|&o| if o == u32::MAX { -1i64 } else { o as i64 })
            .collect::<Vec<_>>(),
        "solids": [],
        "tag_labels": {},
        "edges": mesh.feature_edges(),
        "stats": {
            "n_points": mesh.points.len(),
            "n_tets": q.n_tets,
            "min_dihedral_deg": q.min_dihedral_deg,
            "max_radius_edge": q.max_radius_edge,
            "max_edge": q.max_edge,
            "millis": millis as u64,
        },
        "defects": dg.defects.iter().map(|d| json!({
            "kind": kind_str(d.kind),
            "pos": d.pos,
            "value": d.value,
        })).collect::<Vec<_>>(),
    });
    std::fs::create_dir_all(dir).expect("mkdir probe meshes");
    std::fs::write(dir.join(format!("probe_{name}.json")), doc.to_string())
        .expect("write probe json");
}

fn run(label: &str, plc: &rapidmesh_geom::TaggedPlc, maxh: f64, out: &Path) {
    let t0 = std::time::Instant::now();
    let mut m = mesh_refine(plc, &MeshParams { maxh, ..Default::default() });
    optimize(&mut m, &OptimizeParams { maxh, ..Default::default() });
    let millis = t0.elapsed().as_millis();
    let q = quality_stats(&m);
    // watertight-ish check: odd-incidence edges of tagged faces (3+ radial at
    // junctions is legit, so report but do not judge here)
    let mut ecount: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for f in &m.faces {
        for k in 0..3 {
            let (a, b) = (f.tri[k], f.tri[(k + 1) % 3]);
            *ecount.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    let odd = ecount.values().filter(|&&c| c % 2 == 1).count();
    println!(
        "{label:28} tets {:6}  min-dih {:6.2}  slivers {:4}  re {:5.1}  odd-e {:4}  {:.2?}",
        m.tets.len(),
        q.min_dihedral_deg,
        q.n_slivers,
        q.max_radius_edge,
        odd,
        t0.elapsed()
    );
    let name: String = label
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    write_probe_json(out, &name, &m, millis);
}

fn main() {
    let out = Path::new("viewer/public/meshes");

    // 1. via: cylinder through a dielectric block (ignored test #51 geometry)
    let mut s = Scene::new();
    s.add_solid(solid_box([-2.0, -2.0, 0.0], [2.0, 2.0, 1.0]));
    s.add_solid(cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.75, 12));
    run("via in block", &s.assemble(), 0.4, out);

    // 2. em scene: air box + dielectric box (multi-region, WP5 test)
    let mut s = Scene::new();
    s.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
    s.add_solid(solid_box([1.0, 1.0, 1.0], [3.0, 3.0, 2.0]));
    run("em scene (nested boxes)", &s.assemble(), 0.8, out);

    // 3. tee (issue #1)
    let mm = 1e-3;
    let mut s = Scene::new();
    s.add_solid(solid_box([0.0, 0.0, 0.0], [10.0 * mm, 4.0 * mm, 2.0 * mm]));
    s.add_solid(solid_box([3.0 * mm, 4.0 * mm, 0.0], [7.0 * mm, 8.0 * mm, 2.0 * mm]));
    run("tee (flush T-junction)", &s.assemble(), 1.0 * mm, out);

    // 4. box minus sphere
    let mut s = Scene::new();
    s.add_solid(solid_box([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]));
    s.add_void(icosphere([0.0, 0.0, 0.0], 1.2, 3));
    run("box minus sphere", &s.assemble(), 0.5, out);

    // 5. crossed cylinders (drilled)
    let mut s = Scene::new();
    s.add_solid(cylinder([-2.0, 0.0, 0.0], [4.0, 0.0, 0.0], 0.8, 24));
    s.add_void(cylinder([0.0, -2.0, 0.0], [0.0, 4.0, 0.0], 0.4, 24));
    run("cross cylinders", &s.assemble(), 0.25, out);

    // 6. cone (apex sliver class)
    let mut s = Scene::new();
    s.add_solid(frustum([0.0, 0.0, 0.0], [0.0, 0.0, 1.5], 0.8, 0.001, 24));
    run("cone", &s.assemble(), 0.3, out);

    // 7. torus
    let mut s = Scene::new();
    s.add_solid(torus([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.2, 0.4, 24, 12));
    run("torus", &s.assemble(), 0.3, out);

    // 8. union of box and sphere (overlapping)
    let mut s = Scene::new();
    s.add_solid(solid_box([-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]));
    s.add_solid(icosphere([1.5, 0.0, 0.0], 1.0, 3));
    run("box + sphere union", &s.assemble(), 0.4, out);
}

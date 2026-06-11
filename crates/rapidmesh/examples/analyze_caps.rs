//! Classifies the worst tets of an optimized surface-model mesh: how many
//! vertices sit on the surface, where the short edge lies (interior, same
//! patch, across patches), ring sizes. Drives the edge-collapse design.
//!
//!     cargo run --release -p rapidmesh --example analyze_caps -- bench/models/fandisk.stl

use rapidmesh_geom::{import_obj, import_stl, Scene};
use rapidmesh_tet::{mesh_plc_with, optimize, MeshParams, OptimizeParams};
use std::collections::{HashMap, HashSet};

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let path = std::path::PathBuf::from(path);
    let faceted = match path.extension().and_then(|s| s.to_str()) {
        Some("stl") => import_stl(&path).expect("import"),
        _ => import_obj(&path).expect("import"),
    };
    let mut scene = Scene::new();
    scene.add_solid(faceted);
    let plc = scene.assemble();
    let params = MeshParams {
        maxh: f64::INFINITY,
        region_maxh: Vec::new(),
        radius_edge_bound: 2.0,
        max_points: 500_000,
        grading: 0.5,
            face_maxh: Vec::new(),
            surface_maxh: Vec::new(),
            size_points: Vec::new(),
    };
    let mut mesh = mesh_plc_with(&plc, &params);
    let degree_stats = |mesh: &rapidmesh_tet::TetMesh, label: &str| {
        let mut deg = vec![0usize; mesh.points.len()];
        for t in &mesh.tets {
            for &v in t {
                deg[v] += 1;
            }
        }
        let mut ds: Vec<usize> = deg.iter().copied().filter(|&d| d > 0).collect();
        ds.sort_unstable();
        let hi = deg.iter().filter(|&&d| d > 60).count();
        println!(
            "{label}: max degree {}, p99 {}, vertices >60: {hi}",
            ds[ds.len() - 1],
            ds[ds.len() * 99 / 100]
        );
    };
    degree_stats(&mesh, "vor optimize");
    optimize(&mut mesh, &OptimizeParams::default());
    degree_stats(&mesh, "nach optimize");

    // Surface bookkeeping.
    let mut vert_patches: HashMap<usize, HashSet<u32>> = HashMap::new();
    let mut surf_edges: HashSet<(usize, usize)> = HashSet::new();
    for f in &mesh.faces {
        for &v in &f.tri {
            vert_patches.entry(v).or_default().insert(f.patch);
        }
        for e in 0..3 {
            let (a, b) = (f.tri[e], f.tri[(e + 1) % 3]);
            surf_edges.insert((a.min(b), a.max(b)));
        }
    }

    let dihedral_min = |p: [[f64; 3]; 4]| -> f64 {
        let mut mind = 180.0f64;
        for i in 0..4 {
            for j in i + 1..4 {
                let (k, l): (usize, usize) = {
                    let o: Vec<usize> = (0..4).filter(|&x| x != i && x != j).collect();
                    (o[0], o[1])
                };
                let e: [f64; 3] = std::array::from_fn(|m| p[j][m] - p[i][m]);
                let t2: f64 = e.iter().map(|x| x * x).sum();
                if t2 == 0.0 {
                    return 0.0;
                }
                let perp = |c: [f64; 3]| -> [f64; 3] {
                    let w: [f64; 3] = std::array::from_fn(|m| c[m] - p[i][m]);
                    let s: f64 = (0..3).map(|m| w[m] * e[m]).sum::<f64>() / t2;
                    std::array::from_fn(|m| w[m] - s * e[m])
                };
                let (u, v) = (perp(p[k]), perp(p[l]));
                let nu: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
                let nv: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if nu * nv == 0.0 {
                    return 0.0;
                }
                let cos = ((0..3).map(|m| u[m] * v[m]).sum::<f64>() / (nu * nv)).clamp(-1.0, 1.0);
                mind = mind.min(cos.acos().to_degrees());
            }
        }
        mind
    };

    let mut bad: Vec<(f64, usize)> = Vec::new();
    for (ti, t) in mesh.tets.iter().enumerate() {
        let p: [[f64; 3]; 4] = std::array::from_fn(|k| mesh.points[t[k]]);
        let m = dihedral_min(p);
        if m < 10.0 {
            bad.push((m, ti));
        }
    }
    bad.sort_by(|x, y| x.0.total_cmp(&y.0));
    println!("{}: {} tets unter 10 Grad", path.display(), bad.len());

    // Collapsibility per bad tet: does ANY of its 6 edges admit a collapse
    // b -> a under the PLC-preserving rules? b must be a STEINER vertex
    // (index >= the PLC vertex count); interior b always, surface b only
    // along its own constraint (same patch in-plane, or along its crease
    // toward a crease neighbor).
    let n_plc = plc.vertices.len();
    let mut classes: HashMap<&'static str, usize> = HashMap::new();
    for &(_, ti) in &bad {
        let t = mesh.tets[ti];
        let mut best: &'static str = "unfixable (all input/pinned)";
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    continue;
                }
                let (a, b) = (t[i], t[j]); // collapse b onto a
                if b < n_plc {
                    continue; // input vertices are the PLC: never removed
                }
                match vert_patches.get(&b) {
                    None => {
                        best = "interior steiner edge";
                    }
                    Some(pb) => {
                        let Some(pa) = vert_patches.get(&a) else { continue };
                        let shared = pa.intersection(pb).count();
                        if !surf_edges.contains(&(a.min(b), a.max(b))) {
                            continue; // surface b moves only along the surface
                        }
                        if shared >= pb.len() {
                            // every patch of b also contains a: the collapse
                            // keeps b's whole surface star inside its own
                            // constraint set (in-plane or along the crease)
                            if best == "unfixable (all input/pinned)" {
                                best = "surface steiner edge";
                            }
                        }
                    }
                }
            }
        }
        *classes.entry(best).or_default() += 1;
    }
    let mut cv: Vec<(&str, usize)> = classes.into_iter().collect();
    cv.sort_by_key(|e| std::cmp::Reverse(e.1));
    for (k, c) in cv {
        println!("  {c:4}  {k}");
    }
    let n_steiner = (n_plc..mesh.points.len()).count();
    println!("points: {} ({} plc + {n_steiner} steiner)", mesh.points.len(), n_plc);
}

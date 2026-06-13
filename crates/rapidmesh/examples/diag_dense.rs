//! Diagnoses dense zones: how short are the constraint (PLC) edges that the
//! CSG arrangement hands the mesher, versus the size target, versus the
//! final mesh? Tells us whether the over-refinement is baked into the input
//! (seam micro-edges) or produced by refinement.
//!
//!     cargo run --release -p rapidmesh --example diag_dense

use rapidmesh_geom::{cylinder, solid_box, Scene};
use rapidmesh_tet::{mesh_plc_with, optimize, quality_stats, MeshParams, OptimizeParams};

fn edge_stats(label: &str, verts: &dyn Fn(usize) -> [f64; 3], edges: &[(usize, usize)], target: f64) {
    let mut lens: Vec<f64> = edges
        .iter()
        .map(|&(a, b)| {
            let (pa, pb) = (verts(a), verts(b));
            (0..3).map(|k| (pa[k] - pb[k]).powi(2)).sum::<f64>().sqrt()
        })
        .filter(|&l| l > 0.0)
        .collect();
    lens.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = lens.len();
    let pct = |p: f64| lens[((n as f64 * p) as usize).min(n - 1)];
    let below = |t: f64| lens.iter().filter(|&&l| l < t).count();
    println!(
        "{label}: {n} edges  min {:.2e}  p1 {:.4}  p50 {:.4}  | target {target}  \
         < t/4: {}  < t/10: {}",
        lens[0],
        pct(0.01),
        pct(0.5),
        below(target / 4.0),
        below(target / 10.0),
    );
}

fn main() {
    let target = 0.15;
    let mut scene = Scene::new();
    scene.add_solid(solid_box([-1.5, -1.0, 0.0], [1.5, 1.0, 0.5]));
    // four counterbores: wide shallow recess over a narrow through bore
    for (cx, cy) in [(-1.0, -0.55), (-1.0, 0.55), (1.0, -0.55), (1.0, 0.55)] {
        scene.add_void(cylinder([cx, cy, 0.28], [0.0, 0.0, 0.3], 0.26, 18));
        scene.add_void(cylinder([cx, cy, -0.05], [0.0, 0.0, 0.6], 0.13, 14));
    }

    let plc = scene.assemble();
    // unique undirected PLC edges
    let mut pe: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    for t in &plc.triangles {
        for e in 0..3 {
            let (a, b) = (t[e] as usize, t[(e + 1) % 3] as usize);
            pe.insert((a.min(b), a.max(b)));
        }
    }
    let pe: Vec<(usize, usize)> = pe.into_iter().collect();
    let pv = plc.vertices.clone();
    println!("PLC: {} verts, {} tris", pv.len(), plc.triangles.len());
    edge_stats("PLC", &|i| pv[i], &pe, target);

    for (tag, re) in [("quality re=2", 2.0_f64), ("sizing-only re=inf", f64::INFINITY)] {
        let params = MeshParams {
            maxh: target,
            radius_edge_bound: re,
            max_points: 500_000,
            grading: 0.5,
            ..MeshParams::default()
        };
        let mut mesh = mesh_plc_with(&plc, &params);
        optimize(
            &mut mesh,
            &OptimizeParams {
                maxh: target,
                ..OptimizeParams::default()
            },
        );
        let q = quality_stats(&mesh);
        let mut me: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        for t in &mesh.tets {
            for i in 0..4 {
                for j in i + 1..4 {
                    let (a, b) = (t[i], t[j]);
                    me.insert((a.min(b), a.max(b)));
                }
            }
        }
        let me: Vec<(usize, usize)> = me.into_iter().collect();
        let mv = mesh.points.clone();
        println!(
            "MESH [{tag}]: {} tets, {} pts ({} PLC), min-dih {:.2}, r/e {:.2}",
            mesh.tets.len(),
            mesh.points.len(),
            mesh.plc_points,
            q.min_dihedral_deg,
            q.max_radius_edge,
        );
        edge_stats(&format!("MESH [{tag}]"), &|i| mv[i], &me, target);
        // Classify the short edges by endpoint provenance.
        let np = mesh.plc_points;
        let mut pp = 0usize;
        let mut ps = 0usize;
        let mut ss = 0usize;
        for &(a, b) in &me {
            let l: f64 = (0..3).map(|k| (mv[a][k] - mv[b][k]).powi(2)).sum::<f64>().sqrt();
            if l >= target / 4.0 {
                continue;
            }
            match ((a < np), (b < np)) {
                (true, true) => pp += 1,
                (false, false) => ss += 1,
                _ => ps += 1,
            }
        }
        println!("  short (< t/4) endpoints: PLC-PLC {pp}, PLC-Steiner {ps}, Steiner-Steiner {ss}");
    }
}

//! Refinement-core acceptance sweep over the known-problem geometries.
use rapidmesh_geom::{frustum, cylinder, icosphere, solid_box, torus, Scene};
use rapidmesh_tet::{mesh_refine, optimize, quality_stats, MeshParams, OptimizeParams};

fn run(label: &str, plc: &rapidmesh_geom::TaggedPlc, maxh: f64) {
    let t0 = std::time::Instant::now();
    let mut m = mesh_refine(plc, &MeshParams { maxh, ..Default::default() });
    optimize(&mut m, &OptimizeParams { maxh, ..Default::default() });
    let q = quality_stats(&m);
    // watertight-ish check: odd-incidence edges of tagged faces (3+ radial at
    // junctions is legit, so report but do not judge here)
    let mut ecount: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
    for f in &m.faces {
        for k in 0..3 {
            let (a, b) = (f.tri[k], f.tri[(k + 1) % 3]);
            *ecount.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    let odd = ecount.values().filter(|&&c| c % 2 == 1).count();
    println!("{label:28} tets {:6}  min-dih {:6.2}  slivers {:4}  re {:5.1}  odd-e {:4}  {:.2?}",
        m.tets.len(), q.min_dihedral_deg, q.n_slivers, q.max_radius_edge, odd, t0.elapsed());
}

fn main() {
    // 1. via: cylinder through a dielectric block (ignored test #51 geometry)
    let mut s = Scene::new();
    s.add_solid(solid_box([-2.0, -2.0, 0.0], [2.0, 2.0, 1.0]));
    s.add_solid(cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.75, 12));
    run("via in block", &s.assemble(), 0.4);

    // 2. em scene: air box + dielectric box (multi-region, WP5 test)
    let mut s = Scene::new();
    s.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
    s.add_solid(solid_box([1.0, 1.0, 1.0], [3.0, 3.0, 2.0]));
    run("em scene (nested boxes)", &s.assemble(), 0.8);

    // 3. tee (issue #1)
    let mm = 1e-3;
    let mut s = Scene::new();
    s.add_solid(solid_box([0.0, 0.0, 0.0], [10.0 * mm, 4.0 * mm, 2.0 * mm]));
    s.add_solid(solid_box([3.0 * mm, 4.0 * mm, 0.0], [7.0 * mm, 8.0 * mm, 2.0 * mm]));
    run("tee (flush T-junction)", &s.assemble(), 1.0 * mm);

    // 4. box minus sphere
    let mut s = Scene::new();
    s.add_solid(solid_box([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]));
    s.add_void(icosphere([0.0, 0.0, 0.0], 1.2, 3));
    run("box minus sphere", &s.assemble(), 0.5);

    // 5. crossed cylinders (drilled)
    let mut s = Scene::new();
    s.add_solid(cylinder([-2.0, 0.0, 0.0], [4.0, 0.0, 0.0], 0.8, 24));
    s.add_void(cylinder([0.0, -2.0, 0.0], [0.0, 4.0, 0.0], 0.4, 24));
    run("cross cylinders", &s.assemble(), 0.25);

    // 6. cone (apex sliver class)
    let mut s = Scene::new();
    s.add_solid(frustum([0.0, 0.0, 0.0], [0.0, 0.0, 1.5], 0.8, 0.001, 24));
    run("cone", &s.assemble(), 0.3);

    // 7. torus
    let mut s = Scene::new();
    s.add_solid(torus([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.2, 0.4, 24, 12));
    run("torus", &s.assemble(), 0.3);

    // 8. union of box and sphere (overlapping)
    let mut s = Scene::new();
    s.add_solid(solid_box([-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]));
    s.add_solid(icosphere([1.5, 0.0, 0.0], 1.0, 3));
    run("box + sphere union", &s.assemble(), 0.4);
}

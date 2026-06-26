//! Grading benchmark: mesh the unit square under a radial sizing field (fine at
//! the centre, coarse at the rim) with `surf2d::mesh_polygon`, and dump the mesh
//! as JSON for the gmsh comparison (see scratchpad/grading_compare.py).

use rapidmesh_tet::surf2d::mesh_polygon;

type P2 = [f64; 2];

fn main() {
    let h_min = 0.012;
    let h_max = 0.09;
    let grade = 0.12;
    let c: P2 = [0.5, 0.5];
    let field = |p: P2| {
        let d = ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2)).sqrt();
        (h_min + grade * d).min(h_max)
    };

    // Square outline, walked edge by edge with steps of the LOCAL field size so
    // the boundary spacing already grades (no long constraint edges).
    let corners: [P2; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut outline: Vec<P2> = Vec::new();
    for k in 0..4 {
        let a = corners[k];
        let b = corners[(k + 1) % 4];
        let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        let mut t = 0.0;
        while t < len - 1e-9 {
            let p = [a[0] + (b[0] - a[0]) * t / len, a[1] + (b[1] - a[1]) * t / len];
            outline.push(p);
            t += field(p).max(h_min * 0.5);
        }
    }

    let loops = vec![outline];
    // rapidmesh's best quality: more CVT relaxation, full Ruppert.
    let (pts, tris) = mesh_polygon(&loops, field, h_min, 25.0, 10, 60);

    print!("{{\"points\":[");
    for (i, p) in pts.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("[{},{}]", p[0], p[1]);
    }
    print!("],\"tris\":[");
    for (i, t) in tris.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("[{},{},{}]", t[0], t[1], t[2]);
    }
    println!("]}}");
}

//! Anatomy of pre-optimize slivers on cross cylinders.
use rapidmesh_geom::{cylinder, Scene};
use rapidmesh_tet::{mesh_refine, optimize, MeshParams, OptimizeParams};

fn main() {
    let mut s = Scene::new();
    s.add_solid(cylinder([-2.0, 0.0, 0.0], [4.0, 0.0, 0.0], 0.8, 24));
    s.add_void(cylinder([0.0, -2.0, 0.0], [0.0, 4.0, 0.0], 0.4, 24));
    let plc = s.assemble();
    let mut m = mesh_refine(&plc, &MeshParams { maxh: 0.25, ..Default::default() });
    optimize(&mut m, &OptimizeParams { maxh: 0.25, ..Default::default() });
    println!("tets {} points {}", m.tets.len(), m.points.len());
    // per-vertex: on big cyl / small cyl / neither (dist to analytic)
    let on_big = |p: [f64; 3]| ((p[1] * p[1] + p[2] * p[2]).sqrt() - 0.8).abs() < 1e-6;
    let on_small = |p: [f64; 3]| ((p[0] * p[0] + p[2] * p[2]).sqrt() - 0.4).abs() < 1e-6;
    let on_cap = |p: [f64; 3]| (p[0].abs() - 2.0).abs() < 1e-6;
    let mut class = [0usize; 6]; // #surface-verts 0..4, [5]=all4 same carrier
    let mut n_sliver = 0;
    let mut vol_min = f64::INFINITY;
    for t in &m.tets {
        let p: [[f64; 3]; 4] = std::array::from_fn(|k| m.points[t[k]]);
        let md = rapidmesh_tet::diagnostics::tet_min_dihedral(p);
        if md >= 10.0 { continue; }
        n_sliver += 1;
        let ns = p.iter().filter(|&&q| on_big(q) || on_small(q) || on_cap(q)).count();
        class[ns] += 1;
        if ns == 4 {
            let all_big = p.iter().all(|&q| on_big(q));
            let all_small = p.iter().all(|&q| on_small(q));
            let all_cap = p.iter().all(|&q| on_cap(q));
            if all_big || all_small || all_cap { class[5] += 1; }
        }
        let d = |i: usize, k: usize| p[i][k] - p[0][k];
        let v = (d(1,0)*(d(2,1)*d(3,2)-d(2,2)*d(3,1)) - d(1,1)*(d(2,0)*d(3,2)-d(2,2)*d(3,0)) + d(1,2)*(d(2,0)*d(3,1)-d(2,1)*d(3,0))).abs()/6.0;
        vol_min = vol_min.min(v);
    }
    println!("slivers {} by #on-surface-verts: {:?} (class[5]=all-4-on-ONE-carrier)", n_sliver, class);
    println!("min sliver volume {:.3e}", vol_min);
    // how many sliver tets have a CONSTRAINED face (a tagged boundary face)?
    let mut cf: std::collections::HashSet<[usize; 3]> = std::collections::HashSet::new();
    for f in &m.faces {
        let mut s = f.tri;
        s.sort_unstable();
        cf.insert(s);
    }
    let mut n_wedged = 0;
    let mut n_faceted = 0;
    for t in &m.tets {
        let p: [[f64; 3]; 4] = std::array::from_fn(|k| m.points[t[k]]);
        if rapidmesh_tet::diagnostics::tet_min_dihedral(p) >= 10.0 { continue; }
        let mut nf = 0;
        for fl in [[0,1,2],[0,1,3],[0,2,3],[1,2,3]] {
            let mut s = [t[fl[0]], t[fl[1]], t[fl[2]]];
            s.sort_unstable();
            if cf.contains(&s) { nf += 1; }
        }
        if nf >= 2 { n_wedged += 1; }
        if nf >= 1 { n_faceted += 1; }
    }
    println!("post-optimize slivers with >=1 constrained face: {n_faceted}, wedged (>=2): {n_wedged}");
}

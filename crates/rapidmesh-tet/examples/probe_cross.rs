use rapidmesh_geom::{cylinder, Scene};
use rapidmesh_tet::{mesh_refine, optimize, quality_stats, MeshParams, OptimizeParams};
fn main() {
    let mut s = Scene::new();
    s.add_solid(cylinder([-2.0, 0.0, 0.0], [4.0, 0.0, 0.0], 0.8, 24));
    s.add_void(cylinder([0.0, -2.0, 0.0], [0.0, 4.0, 0.0], 0.4, 24));
    let plc = s.assemble();
    let mut m = mesh_refine(&plc, &MeshParams { maxh: 0.25, ..Default::default() });
    let q0 = quality_stats(&m);
    optimize(&mut m, &OptimizeParams { maxh: 0.25, ..Default::default() });
    let qa = quality_stats(&m);
    optimize(&mut m, &OptimizeParams { maxh: 0.25, ..Default::default() });
    let qb = quality_stats(&m);
    optimize(&mut m, &OptimizeParams { maxh: 0.25, ..Default::default() });
    let q = quality_stats(&m);
    println!("multi-opt: {:.3} ({}) -> {:.3} ({}) -> {:.3} ({})", qa.min_dihedral_deg, qa.n_slivers, qb.min_dihedral_deg, qb.n_slivers, q.min_dihedral_deg, q.n_slivers);
    println!("tets {} points {}", m.tets.len(), m.points.len());
    println!("min-dih {:.3} -> {:.3} deg, slivers {} -> {}, re {:.1} -> {:.1}",
        q0.min_dihedral_deg, q.min_dihedral_deg, q0.n_slivers, q.n_slivers, q0.max_radius_edge, q.max_radius_edge);
    println!("worst at {:?} region {}", q.worst_location, q.worst_region);
    // min tet edge location
    let mut min_e = f64::INFINITY;
    let mut at = [0.0; 3];
    for t in &m.tets {
        for a in 0..4 {
            for b in (a + 1)..4 {
                let (p, q) = (m.points[t[a]], m.points[t[b]]);
                let d = ((p[0]-q[0]).powi(2)+(p[1]-q[1]).powi(2)+(p[2]-q[2]).powi(2)).sqrt();
                if d < min_e { min_e = d; at = p; }
            }
        }
    }
    println!("min tet edge {:.3e} at {:?}", min_e, at);
}

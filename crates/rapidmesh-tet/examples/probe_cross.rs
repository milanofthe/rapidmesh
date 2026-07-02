use rapidmesh_geom::{cylinder, Scene};
use rapidmesh_tet::{mesh_refine, MeshParams};
fn main() {
    let mut s = Scene::new();
    s.add_solid(cylinder([-2.0, 0.0, 0.0], [4.0, 0.0, 0.0], 0.8, 24));
    s.add_void(cylinder([0.0, -2.0, 0.0], [0.0, 4.0, 0.0], 0.4, 24));
    let plc = s.assemble();
    let m = mesh_refine(&plc, &MeshParams { maxh: 0.25, ..Default::default() });
    println!("tets {} points {}", m.tets.len(), m.points.len());
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

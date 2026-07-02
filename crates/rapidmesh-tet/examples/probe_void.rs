use rapidmesh_geom::{solid_box, Scene};
use rapidmesh_tet::{mesh_plc, MeshParams, mesh_refine};

fn vol(m: &rapidmesh_tet::TetMesh) -> f64 {
    let mut v = 0.0;
    for t in &m.tets {
        let p: [[f64; 3]; 4] = std::array::from_fn(|k| m.points[t[k]]);
        let d = |i: usize, k: usize| p[i][k] - p[0][k];
        v += (d(1,0)*(d(2,1)*d(3,2)-d(2,2)*d(3,1)) - d(1,1)*(d(2,0)*d(3,2)-d(2,2)*d(3,0)) + d(1,2)*(d(2,0)*d(3,1)-d(2,1)*d(3,0))).abs()/6.0;
    }
    v
}

fn main() {
    let mut scene = Scene::new();
    scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 2.0]));
    scene.add_void(solid_box([1.0, 1.0, 0.5], [3.0, 3.0, 1.5]));
    let plc = scene.assemble();
    let m = mesh_plc(&plc);
    println!("mesh_plc (inf): tets {} vol {:.9} want 28", m.tets.len(), vol(&m));
    let m = mesh_refine(&plc, &MeshParams { maxh: 0.5, ..Default::default() });
    println!("maxh 0.5: tets {} vol {:.9} want 28", m.tets.len(), vol(&m));
}

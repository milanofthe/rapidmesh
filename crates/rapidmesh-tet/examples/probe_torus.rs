use rapidmesh_geom::{torus, Scene};
use rapidmesh_tet::{mesh_refine, MeshParams};
fn main() {
    let mut s = Scene::new();
    s.add_solid(torus([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.2, 0.4, 24, 12));
    let plc = s.assemble();
    println!("plc: {} verts {} tris {} surfaces", plc.vertices.len(), plc.triangles.len(), plc.surfaces.len());
    let b = rapidmesh_brep::build::from_plc(&plc);
    println!("brep: {} verts {} edges {} faces", b.vertices.len(), b.edges.len(), b.faces.len());
    for (i, f) in b.faces.iter().enumerate() {
        println!("  face {i}: loops {} facets {} kind {:?}", f.loops.len(), f.facets.len(), plc.surfaces[f.plc_surface as usize]);
    }
    let m = mesh_refine(&plc, &MeshParams { maxh: 0.3, ..Default::default() });
    println!("tets {} points {}", m.tets.len(), m.points.len());
}

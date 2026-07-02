//! Where is the hole boundary? Radial distance of barrel-face vertices.
use rapidmesh_geom::{cylinder, solid_box, Scene, SurfaceKind};
use rapidmesh_tet::{mesh_refine, MeshParams};

fn main() {
    let mut scene = Scene::new();
    scene.add_solid(solid_box([-2.0, -2.0, 0.0], [2.0, 2.0, 1.0]));
    scene.add_void(cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.75, 24));
    let plc = scene.assemble();
    let m = mesh_refine(&plc, &MeshParams { maxh: 0.4, ..Default::default() });
    // faces tagged with the cylinder surface
    let cyl_sid: Vec<usize> = m.surfaces.iter().enumerate()
        .filter(|(_, s)| matches!(s, SurfaceKind::Cylinder { .. }))
        .map(|(i, _)| i).collect();
    println!("cylinder surface ids: {:?}", cyl_sid);
    let mut rmin = f64::INFINITY;
    let mut rmax = 0.0f64;
    let mut n = 0;
    let mut hist = [0usize; 10]; // radius bins 0.70..0.90
    for f in &m.faces {
        if !cyl_sid.contains(&(f.surface as usize)) { continue; }
        n += 1;
        for &v in &f.tri {
            let p = m.points[v];
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            rmin = rmin.min(r);
            rmax = rmax.max(r);
            let bin = (((r - 0.70) / 0.02).floor() as i64).clamp(0, 9) as usize;
            hist[bin] += 1;
        }
    }
    println!("barrel faces {n}: vertex radius [{rmin:.4}, {rmax:.4}]");
    println!("radius hist (0.70+0.02k): {:?}", hist);
    // total area of barrel faces vs analytic 2*pi*0.75*1
    let mut area = 0.0;
    for f in &m.faces {
        if !cyl_sid.contains(&(f.surface as usize)) { continue; }
        let (a, b, c) = (m.points[f.tri[0]], m.points[f.tri[1]], m.points[f.tri[2]]);
        let u = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
        let v = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
        let cr = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]];
        area += 0.5 * (cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]).sqrt();
    }
    println!("barrel area {area:.4} vs analytic {:.4}", 2.0 * std::f64::consts::PI * 0.75);
    // region-0-dropped volume inside the block bbox but outside hole: scan all tets? we only have kept tets.
    // instead: total mesh volume + hole faces → implied hole volume
}

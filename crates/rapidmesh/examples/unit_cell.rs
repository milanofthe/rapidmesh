//! A periodic unit cell of a frequency selective surface, meshed from Rust
//! alone: an air box over a substrate with a square PEC patch, the x and y
//! sides paired periodically, the ports named, written as gmsh MSH and VTU.
//!
//! cargo run --release -p rapidmesh --example unit_cell [out_dir]

use rapidmesh::shapes::{Cuboid, Sheet};
use rapidmesh::{FaceFilter, Geometry, MeshOptions, Scope};

fn main() -> rapidmesh::Result<()> {
    let out = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let (a, h_sub, h_air) = (10.0, 1.6, 12.0);

    let mut g = Geometry::new(Some(1.5));
    let air = g.add(Cuboid::new([a, a, h_sub + h_air]))?;
    let sub = g.add_solid(Cuboid::new([a, a, h_sub]), Some(0.8), false)?;
    g.label_solid(air, "air");
    g.label_solid(sub, "substrate");
    g.add_sheet(&Sheet::xy(6.0, 6.0, [2.0, 2.0, h_sub]), 1, Some(0.5))?;
    g.label_tag(1, "patch");

    let side = |n: [f64; 3]| Scope::surf(Some(FaceFilter::normal(n)));
    let sx = g.periodic(&side([-1.0, 0.0, 0.0]), &side([1.0, 0.0, 0.0]), None)?;
    let sy = g.periodic(&side([0.0, -1.0, 0.0]), &side([0.0, 1.0, 0.0]), None)?;
    println!("periodic shifts {sx:?} {sy:?}");
    // A normal selects every face facing that way, the patch and the
    // substrate top included; the ports are the faces nearest a point.
    let near = |p: [f64; 3]| Scope::surf(Some(FaceFilter::near(p)));
    g.name(&near([a / 2.0, a / 2.0, h_sub + h_air]), "port_top")?;
    g.name(&near([a / 2.0, a / 2.0, 0.0]), "port_bottom")?;

    let mesh = g.mesh(&MeshOptions::default())?;
    println!("{}", mesh.report());

    let sets = mesh.sets();
    for (name, cells) in &sets.cells {
        println!("cells {name}: {}", cells.len());
    }
    for (name, faces) in &sets.faces {
        println!("faces {name}: {}", faces.len());
    }
    println!("periodic point pairs: {}", mesh.periodic_points.len());
    let d = mesh.diagnostics();
    println!(
        "watertight {}, {} defects",
        d.mesh.watertight,
        d.defects().count()
    );

    let dir = std::path::Path::new(&out);
    mesh.write_msh(dir.join("unit_cell.msh"), rapidmesh::Order::Linear)?;
    mesh.write_vtu(dir.join("unit_cell.vtu"), rapidmesh::Order::Linear)?;
    println!("wrote {}", dir.join("unit_cell.msh").display());
    Ok(())
}

//! A mesh large enough to profile: a dielectric sphere in a box, about a
//! million tets at the default size, or the parts of a STEP file.
//!
//! cargo run --release -p rapidmesh --example profile [maxh] [part.step]
//! samply record target/release/examples/profile

use rapidmesh::shapes::{Cuboid, Sphere};
use rapidmesh::{Geometry, MeshOptions};

fn main() -> rapidmesh::Result<()> {
    let mut args = std::env::args().skip(1);
    let h: f64 = args.next().map_or(0.04, |s| s.parse().expect("maxh"));
    let mut g = Geometry::new(Some(h));
    match args.next() {
        Some(step) => {
            g.import_step(&step, None)?;
        }
        None => {
            g.add(Cuboid::new([2.0, 2.0, 2.0]))?;
            g.add_solid(Sphere::new(0.5).at([1.0, 1.0, 1.0]), Some(h / 2.0), false)?;
        }
    }
    let mesh = g.mesh(&MeshOptions::default())?;
    println!("{}", mesh.report());
    Ok(())
}

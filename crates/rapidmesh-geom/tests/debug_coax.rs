//! Regression: stacked coaxial cylinders at millimetre scale. Constructed
//! arrangement points lying exactly on the shared cap plane must SNAP onto
//! it bit-exactly (correctly rounded Point3::approx), or the plane shatters
//! into micro-patches whose crease recovery never terminates.

use rapidmesh_geom::{cylinder, Scene};

#[test]
fn coax_cap_plane_snaps_exactly() {
    let mm = 1e-3;
    let mut scene = Scene::new();
    scene.add_solid(cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 15.0 * mm], 1.5 * mm, 24));
    scene.add_solid(cylinder([0.0, 0.0, 15.0 * mm], [0.0, 0.0, 15.0 * mm], 0.99 * mm, 24));
    let plc = scene.assemble();
    let off: Vec<_> = plc
        .vertices
        .iter()
        .enumerate()
        .filter(|(_, v)| (v[2] - 0.015).abs() < 1e-6 && v[2] != 0.015)
        .collect();
    assert!(
        off.is_empty(),
        "{} vertices snapped off the z = 0.015 cap plane: {off:?}",
        off.len()
    );
}

//! Builder tests: TaggedPlc -> Brep on the canonical shapes.

use rapidmesh_brep::{build::from_plc, Curve, Surface};
use rapidmesh_geom::{
    extrude_spline_profile, icosphere, naca0012_profile, solid_box, Scene, SurfaceKind,
};

#[test]
fn box_has_6_faces_12_edges_8_corners() {
    let mut scene = Scene::new();
    scene.add_solid(solid_box([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]));
    let plc = scene.assemble();
    let b = from_plc(&plc);

    assert_eq!(b.faces.len(), 6, "box has 6 faces");
    assert_eq!(b.edges.len(), 12, "box has 12 edges");
    assert_eq!(b.vertices.len(), 8, "box has 8 corners");

    // every face is planar with a single outer loop of 4 edges
    for f in &b.faces {
        assert!(matches!(b.surface(f.surface), Surface::Analytic(SurfaceKind::Plane)));
        assert_eq!(f.loops.len(), 1, "a box face has one loop");
        assert_eq!(f.loops[0].edges.len(), 4, "a box face loop has 4 edges");
        // one side is a region, the other background (0)
        let (a, c) = (f.regions[0].0, f.regions[1].0);
        assert!((a == 0) ^ (c == 0), "box wall separates region from background");
    }
    // every edge is a straight line, shared by exactly two faces
    for e in &b.edges {
        assert!(matches!(e.curve, Curve::Line { .. }), "box edge is a Line");
        assert_eq!(e.faces.len(), 2, "box edge is shared by two faces");
    }
}

#[test]
fn sphere_is_one_closed_face_no_edges() {
    let mut scene = Scene::new();
    scene.add_solid(icosphere([0.0, 0.0, 0.0], 1.0, 2));
    let plc = scene.assemble();
    let b = from_plc(&plc);

    // one analytic sphere surface, no feature edges, no corners (closed smooth)
    assert_eq!(b.faces.len(), 1, "sphere is one face");
    assert!(matches!(b.surface(b.faces[0].surface), Surface::Analytic(SurfaceKind::Sphere { .. })));
    assert_eq!(b.edges.len(), 0, "closed sphere has no feature edges");
    assert_eq!(b.vertices.len(), 0, "closed sphere has no corners");
}

#[test]
fn airfoil_recovers_extruded_face_and_profile_edges() {
    let profile = naca0012_profile(1.0, 40);
    let solid = extrude_spline_profile(
        profile,
        80,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.5],
    );
    let mut scene = Scene::new();
    scene.add_solid(solid);
    let plc = scene.assemble();
    let b = from_plc(&plc);

    // mantle (Extruded) + 2 planar caps
    let n_ext = b
        .faces
        .iter()
        .filter(|f| matches!(b.surface(f.surface), Surface::Analytic(SurfaceKind::Extruded { .. })))
        .count();
    assert_eq!(n_ext, 1, "one extruded mantle face");
    assert!(b.faces.len() >= 3, "mantle + caps, got {}", b.faces.len());

    // the mantle's rim edges are recovered as analytic profile curves
    let n_profile = b.edges.iter().filter(|e| matches!(e.curve, Curve::Profile { .. })).count();
    assert!(n_profile >= 1, "at least one profile edge, got {n_profile}");
    assert!(!b.edges.is_empty(), "airfoil has feature edges");
}

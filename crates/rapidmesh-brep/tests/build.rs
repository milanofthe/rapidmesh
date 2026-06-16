//! Builder tests: TaggedPlc -> Brep on the canonical shapes.

use rapidmesh_brep::{build::from_plc, Curve, Surface};
use rapidmesh_geom::{extrude_spline_profile, icosphere, naca0012_profile, solid_box, Scene};

#[test]
fn hemisphere_recovers_circle_edge() {
    // a sphere cut by z<0: the equator is a sphere/plane intersection -> a Circle
    let mut scene = Scene::new();
    scene.add_solid(icosphere([0.0, 0.0, 0.0], 1.0, 3));
    scene.add_void(solid_box([-2.0, -2.0, -2.0], [2.0, 2.0, 0.0]));
    let b = from_plc(&scene.assemble());
    let r = b
        .edges
        .iter()
        .find_map(|e| match e.curve {
            Curve::Circle { radius, .. } => Some(radius),
            _ => None,
        })
        .expect("equator recovered as a Circle");
    assert!((r - 1.0).abs() < 0.06, "circle radius {r} ~ 1.0");
}

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
        assert!(matches!(b.surface(f.surface), Surface::Plane { .. }));
        assert_eq!(f.loops.len(), 1, "a box face has one loop");
        assert_eq!(f.loops[0].coedges.len(), 4, "a box face loop has 4 co-edges");
        // one side is a region, the other background (0)
        let (a, c) = (f.regions[0].0, f.regions[1].0);
        assert!((a == 0) ^ (c == 0), "box wall separates region from background");
        // the loop's PCurves map to a proper 2D region in the face (u,v): a
        // nonzero-area bounding box, proving the planar chart is well-posed
        let mut lo = [f64::MAX; 2];
        let mut hi = [f64::MIN; 2];
        for &cid in &f.loops[0].coedges {
            let uv = &b.coedge(cid).pcurve.uv;
            assert!(uv.len() >= 2, "co-edge PCurve has >= 2 samples");
            for p in uv {
                for k in 0..2 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
        }
        assert!(hi[0] - lo[0] > 1e-9 && hi[1] - lo[1] > 1e-9, "face (u,v) region has area");
    }
    // every edge is a straight line, shared by exactly two co-edges (two faces)
    for e in &b.edges {
        assert!(matches!(e.curve, Curve::Line { .. }), "box edge is a Line");
        assert_eq!(e.coedges.len(), 2, "box edge is used by two faces");
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
    assert!(matches!(b.surface(b.faces[0].surface), Surface::Sphere { .. }));
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
        .filter(|f| matches!(b.surface(f.surface), Surface::Extruded { .. }))
        .count();
    assert_eq!(n_ext, 1, "one extruded mantle face");
    assert!(b.faces.len() >= 3, "mantle + caps, got {}", b.faces.len());

    // the mantle's rim edges are recovered as analytic profile curves
    let n_profile = b.edges.iter().filter(|e| matches!(e.curve, Curve::Profile { .. })).count();
    assert!(n_profile >= 1, "at least one profile edge, got {n_profile}");
    assert!(!b.edges.is_empty(), "airfoil has feature edges");

    // the extruded mantle face has loops whose co-edges carry PCurves in its
    // (t, h) parameter space (the parametric trim, the native NURBS path)
    let mantle = b
        .faces
        .iter()
        .find(|f| matches!(b.surface(f.surface), Surface::Extruded { .. }))
        .unwrap();
    let mut n_uv = 0;
    for lp in &mantle.loops {
        for &cid in &lp.coedges {
            assert!(b.coedge(cid).pcurve.uv.len() >= 2, "mantle co-edge has a PCurve");
            n_uv += b.coedge(cid).pcurve.uv.len();
        }
    }
    assert!(n_uv > 0, "mantle has parametric trim curves");
}

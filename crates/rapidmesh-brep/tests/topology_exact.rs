//! Acceptance tests of the B-rep topology (issue #77): what the arrangement
//! builds, without welds across gaps, independent of rigid motions.

use rapidmesh_brep::{build::from_plc, Brep, Curve};
use rapidmesh_geom::{
    cylinder, cylinder_iso, frustum, icosphere, sheet_polygon, solid_box, sphere, torus, FaceTag,
    Faceted, Scene,
};

fn brep(s: &Scene) -> Brep {
    from_plc(&s.assemble())
}

/// (vertices, edges, faces)
fn counts(b: &Brep) -> (usize, usize, usize) {
    (b.vertices.len(), b.edges.len(), b.faces.len())
}

/// A ring cut open by a slot keeps both sides of the slot, however narrow:
/// 8 corners (4 per cap, where the slot walls meet the arcs), 12 edges
/// (4 arcs, 4 vertical wall edges, 4 wall edges on the caps), 6 faces.
#[test]
fn split_ring_keeps_both_sides_of_its_gap() {
    for gap in [0.2, 0.02, 0.002] {
        let mut s = Scene::new();
        s.add_solid(cylinder([0.0; 3], [0.0, 0.0, 0.2], 1.0, 64));
        s.add_void(cylinder([0.0, 0.0, -0.1], [0.0, 0.0, 0.4], 0.8, 64));
        s.add_void(solid_box([0.7, -gap / 2.0, -0.1], [1.1, gap / 2.0, 0.3]));
        let b = brep(&s);
        assert_eq!(counts(&b), (8, 12, 6), "gap {gap}");
        let arcs = b
            .edges
            .iter()
            .filter(|e| matches!(e.curve, Curve::Circle { .. }))
            .count();
        assert_eq!(arcs, 4, "gap {gap}: the four rims are circle arcs");
    }
}

fn slotted_box(f: &dyn Fn(Faceted) -> Faceted) -> Scene {
    let mut s = Scene::new();
    s.add_solid(f(solid_box([0.0; 3], [1.0; 3])));
    s.add_void(f(solid_box([0.1, 0.45, 0.8], [0.9, 0.55, 1.2])));
    s.add_void(f(solid_box([0.45, 0.1, 0.85], [0.55, 0.9, 1.2])));
    s
}

/// Two slots crossing on a face of an obliquely rotated box: the rotated
/// faces are not exactly planar in f64, yet the scene assembles and gets
/// the topology of the unrotated one.
#[test]
fn oblique_box_with_crossing_cuts_has_the_upright_topology() {
    let upright = counts(&brep(&slotted_box(&|f| f)));
    for (axis, angle) in [
        ([1.0, 0.7, 0.2], 0.6),
        ([0.0, 0.0, 1.0], 0.3),
        ([1.0, 1.0, 1.0], 2.1),
    ] {
        let rotated = counts(&brep(&slotted_box(&|f| f.rotated([0.5; 3], axis, angle))));
        assert_eq!(rotated, upright, "rotation about {axis:?} by {angle}");
    }
}

/// The origin of every entity in id order: per face its owner, role and
/// tag; per edge the faces around it; per vertex the edges ending at it.
#[allow(clippy::type_complexity)]
fn origins(b: &Brep) -> (Vec<(u32, u32, u32)>, Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let faces = b
        .faces
        .iter()
        .map(|f| (f.owner, f.role, f.face_tag.0))
        .collect();
    let edges = b
        .edges
        .iter()
        .map(|e| {
            let mut fs: Vec<u32> = e
                .coedges
                .iter()
                .map(|c| b.coedges[c.0 as usize].face.0)
                .collect();
            fs.sort_unstable();
            fs.dedup();
            fs
        })
        .collect();
    let mut ends = vec![Vec::new(); b.vertices.len()];
    for (i, e) in b.edges.iter().enumerate() {
        for v in e.ends {
            ends[v.0 as usize].push(i as u32);
        }
    }
    (faces, edges, ends)
}

fn slotted_box_of(height: f64, f: &dyn Fn(Faceted) -> Faceted) -> Scene {
    let mut s = Scene::new();
    s.add_solid(f(solid_box([0.0; 3], [1.0, 1.0, height])));
    s.add_void(f(solid_box(
        [0.1, 0.45, height - 0.2],
        [0.9, 0.55, height + 0.2],
    )));
    s.add_void(f(solid_box(
        [0.45, 0.1, height - 0.15],
        [0.55, 0.9, height + 0.2],
    )));
    s
}

/// Ids follow the origin of the entities: an exact quarter turn, a changed
/// size and a shape added after leave the id of every face, edge and
/// vertex where it was (#128).
#[test]
fn ids_follow_the_origin() {
    let base = origins(&brep(&slotted_box_of(1.0, &|f| f)));
    assert!(base.0.len() > 6, "the slots split and add faces");
    let quarter = std::f64::consts::FRAC_PI_2;
    for axis in [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]] {
        let turned = origins(&brep(&slotted_box_of(1.0, &|f| {
            f.rotated([0.5; 3], axis, quarter)
        })));
        assert_eq!(turned, base, "quarter turn about {axis:?}");
    }
    assert_eq!(
        origins(&brep(&slotted_box_of(1.4, &|f| f))),
        base,
        "a taller box"
    );
    let mut s = slotted_box_of(1.0, &|f| f);
    s.add_solid(solid_box([3.0, 0.0, 0.0], [4.0, 1.0, 1.0]));
    let more = origins(&brep(&s));
    assert_eq!(
        more.0[..base.0.len()],
        base.0[..],
        "faces of a later shape go after"
    );
    assert_eq!(more.1[..base.1.len()], base.1[..], "and so do its edges");
}

/// Rigid rotations, exact quarter turns included, leave the topology and
/// the curve kinds of a cylinder unchanged.
#[test]
fn rotations_keep_the_cylinder_topology() {
    let cyl = cylinder([0.0; 3], [0.0, 0.0, 1.0], 0.5, 32);
    for (axis, angle) in [
        ([1.0, 0.0, 0.0], std::f64::consts::FRAC_PI_2),
        ([0.0, 1.0, 0.0], std::f64::consts::PI),
        ([1.0, 1.0, 0.3], 0.7),
    ] {
        let mut s = Scene::new();
        s.add_solid(cyl.rotated([0.0; 3], axis, angle));
        let b = brep(&s);
        assert_eq!(counts(&b), (2, 2, 3), "rotation about {axis:?} by {angle}");
        assert!(
            b.edges
                .iter()
                .all(|e| matches!(e.curve, Curve::Circle { .. })),
            "rotation about {axis:?} by {angle}: rims are circles"
        );
    }
}

/// A capsule fused from a cylinder and two spheres: one barrel, two caps,
/// two rim circles.
#[test]
fn capsule_is_a_barrel_and_two_caps() {
    let mut s = Scene::new();
    let r = s.add_solid(cylinder([0.0; 3], [0.0, 0.0, 1.0], 0.3, 32));
    let a = s.add_solid(sphere([0.0; 3], 0.3, 32, 16));
    let c = s.add_solid(sphere([0.0, 0.0, 1.0], 0.3, 32, 16));
    s.merge_region(r, a);
    s.merge_region(r, c);
    let b = brep(&s);
    assert_eq!(counts(&b), (2, 2, 3));
    assert!(b
        .edges
        .iter()
        .all(|e| matches!(e.curve, Curve::Circle { .. })));
}

/// The capsule of the corpus: geodesic spheres on a structured barrel of
/// another tessellation. Tangent at the rims, the facets cross each other
/// in a band there; the thin faces it leaves go to their neighbours and the
/// rims are the circles of contact.
#[test]
fn tangent_capsule_of_different_tessellations_is_a_barrel_and_two_caps() {
    let mut s = Scene::new();
    let r = s.add_solid(cylinder_iso([0.0, 0.0, -0.6], [0.0, 0.0, 1.2], 0.6, 40, 13));
    let a = s.add_solid(icosphere([0.0, 0.0, -0.6], 0.6, 3));
    let c = s.add_solid(icosphere([0.0, 0.0, 0.6], 0.6, 3));
    s.merge_region(r, a);
    s.merge_region(r, c);
    let b = brep(&s);
    assert_eq!(counts(&b), (2, 2, 3));
    for e in &b.edges {
        let Curve::Circle { center, radius, .. } = e.curve else {
            panic!("a rim is not a circle: {:?}", e.curve);
        };
        assert!((radius - 0.6).abs() < 1e-12 && (center[2].abs() - 0.6).abs() < 1e-12);
    }
}

/// A block whose top carries a ridge: sharp at the front face, fading out
/// towards the back. The top is one smooth region wrapped around the open
/// end of the ridge crease, which the import hands on as a feature: the
/// B-rep gets it as an edge inside the top face, outside every loop.
#[test]
fn open_import_crease_becomes_an_inner_edge() {
    let n = 12;
    let z = |i: usize, j: usize| -> f64 {
        let (x, y) = (i as f64 / n as f64, j as f64 / n as f64);
        let fade = (1.0 - y / 0.6).max(0.0);
        1.0 + 2.0 * fade * (0.5 - (x - 0.5).abs())
    };
    let top = |i: usize, j: usize| [i as f64 / n as f64, j as f64 / n as f64, z(i, j)];
    let bot = |i: usize, j: usize| [i as f64 / n as f64, j as f64 / n as f64, 0.0];
    let mut tris: Vec<[[f64; 3]; 3]> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            // Top faces up, bottom down; split so the ridge x = 0.5 is a
            // grid line of both.
            tris.push([top(i, j), top(i + 1, j), top(i + 1, j + 1)]);
            tris.push([top(i, j), top(i + 1, j + 1), top(i, j + 1)]);
            tris.push([bot(i, j), bot(i + 1, j + 1), bot(i + 1, j)]);
            tris.push([bot(i, j), bot(i, j + 1), bot(i + 1, j + 1)]);
        }
    }
    for k in 0..n {
        let wall = |a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3], t: &mut Vec<_>| {
            t.push([a, b, c]);
            t.push([a, c, d]);
        };
        wall(
            bot(k, 0),
            bot(k + 1, 0),
            top(k + 1, 0),
            top(k, 0),
            &mut tris,
        );
        wall(
            bot(k + 1, n),
            bot(k, n),
            top(k, n),
            top(k + 1, n),
            &mut tris,
        );
        wall(
            bot(0, k + 1),
            bot(0, k),
            top(0, k),
            top(0, k + 1),
            &mut tris,
        );
        wall(
            bot(n, k),
            bot(n, k + 1),
            top(n, k + 1),
            top(n, k),
            &mut tris,
        );
    }
    let mut stl = String::from("solid ridge\n");
    for t in &tris {
        stl.push_str("facet normal 0 0 0\nouter loop\n");
        for v in t {
            stl.push_str(&format!("vertex {} {} {}\n", v[0], v[1], v[2]));
        }
        stl.push_str("endloop\nendfacet\n");
    }
    stl.push_str("endsolid ridge\n");
    let path = std::env::temp_dir().join("rapidmesh_open_ridge.stl");
    std::fs::write(&path, stl).expect("write stl");
    let shape = rapidmesh_geom::import_stl(&path, rapidmesh_geom::CREASE_DEG).expect("import");
    rapidmesh_geom::validate_closed(&shape).expect("closed");
    assert!(
        !shape.features.is_empty(),
        "the open ridge goes along as features"
    );

    let mut s = Scene::new();
    s.add_solid(shape);
    let plc = s.assemble();
    assert!(!plc.features.is_empty(), "the features survive assembly");
    let b = from_plc(&plc);
    let in_loops: std::collections::HashSet<u32> = b
        .faces
        .iter()
        .flat_map(|f| f.loops.iter().flat_map(|l| l.coedges.iter()))
        .map(|&c| b.coedge(c).edge.0)
        .collect();
    let inner: Vec<usize> = (0..b.edges.len())
        .filter(|&e| !in_loops.contains(&(e as u32)))
        .collect();
    assert!(
        !inner.is_empty(),
        "the ridge is an edge inside the top face"
    );
    for &e in &inner {
        let faces: std::collections::HashSet<u32> = b.edges[e]
            .coedges
            .iter()
            .map(|&c| b.coedge(c).face.0)
            .collect();
        assert_eq!(faces.len(), 1, "an inner edge lies in one face");
    }
}

/// Corners come from the input, not from the turns of a chain (#129): a
/// polygon sheet has an edge per side between its corners, a circle given
/// as a 48-gon one closed rim, and the rims of a bore faceted with six
/// segments stay one edge each though the chain turns by 60 degrees there.
#[test]
fn corners_come_from_the_input() {
    let sheet = |ring: Vec<[f64; 2]>| {
        let mut s = Scene::new();
        s.add_sheet(
            sheet_polygon(&ring, &[], [0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            FaceTag(1),
        );
        brep(&s)
    };
    let notch = vec![
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [0.0, 2.0],
    ];
    assert_eq!(counts(&sheet(notch)), (6, 6, 1), "an L-shaped sheet");
    let circle: Vec<[f64; 2]> = (0..48)
        .map(|k| {
            let a = std::f64::consts::TAU * k as f64 / 48.0;
            [a.cos(), a.sin()]
        })
        .collect();
    assert_eq!(sheet(circle).edges.len(), 1, "a 48-gon sheet has one rim");
    let mut s = Scene::new();
    s.add_solid(solid_box([-2.0, -2.0, 0.0], [2.0, 2.0, 1.0]));
    s.add_void(cylinder([0.0, 0.0, -0.5], [0.0, 0.0, 2.0], 1.0, 6));
    let b = brep(&s);
    let rims = b
        .edges
        .iter()
        .filter(|e| e.chain.iter().all(|p| p[0].hypot(p[1]) <= 1.0 + 1e-9))
        .count();
    assert_eq!(rims, 2, "each rim of a six-segment bore is one edge");
}

/// Circles come from the carriers, not from the chain (#129): a sphere on
/// the axis of a cylinder meets it in two circles at the analytic heights,
/// where the faceted chain of so shallow a cut lies well off them.
#[test]
fn coaxial_sphere_and_cylinder_meet_in_exact_circles() {
    let (rc, rs, zc) = (1.0, 1.081, 0.531);
    let mut s = Scene::new();
    s.add_solid(cylinder([0.0; 3], [0.0, 0.0, 2.4], rc, 28));
    s.add_solid(sphere([0.0, 0.0, zc], rs, 28, 14));
    let b = brep(&s);
    let h = (rs * rs - rc * rc).sqrt();
    let mut heights: Vec<f64> = b
        .edges
        .iter()
        .filter_map(|e| match e.curve {
            Curve::Circle { center, radius, .. } if (radius - rc).abs() < 1e-12 => Some(center[2]),
            _ => None,
        })
        // Not the rims of the cylinder's ends.
        .filter(|&z| z > 1e-9 && z < 2.0)
        .collect();
    heights.sort_by(f64::total_cmp);
    assert_eq!(
        heights.len(),
        2,
        "two circles where the sphere cuts the barrel"
    );
    assert!(
        (heights[0] - (zc - h)).abs() < 1e-12 && (heights[1] - (zc + h)).abs() < 1e-12,
        "{heights:?}"
    );
}

/// The circles of a B-rep as (height, radius), sorted.
fn circles(b: &Brep) -> Vec<(f64, f64)> {
    let mut c: Vec<(f64, f64)> = b
        .edges
        .iter()
        .filter_map(|e| match e.curve {
            Curve::Circle { center, radius, .. } => Some((center[2], radius)),
            _ => None,
        })
        .collect();
    c.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    c
}

fn has(c: &[(f64, f64)], z: f64, r: f64) -> bool {
    c.iter()
        .any(|&(h, q)| (h - z).abs() < 1e-12 && (q - r).abs() < 1e-12)
}

/// Two overlapping frustums on one axis: their cones cross in a circle,
/// r = 1 - 0.3 z = 19/15 - 2/3 z at z = 8/11.
#[test]
fn coaxial_cones_meet_in_an_exact_circle() {
    let mut s = Scene::new();
    s.add_solid(frustum([0.0; 3], [0.0, 0.0, 1.2], 1.0, 0.64, 32));
    s.add_solid(frustum([0.0, 0.0, 0.4], [0.0, 0.0, 1.2], 1.0, 0.2, 32));
    let c = circles(&brep(&s));
    let z = 8.0 / 11.0;
    assert!(has(&c, z, 1.0 - 0.3 * z), "{c:?}");
    // The caps that stick out meet the other cone in circles too.
    assert!(has(&c, 0.4, 1.0 - 0.3 * 0.4), "{c:?}");
    assert!(has(&c, 1.2, 19.0 / 15.0 - 2.0 / 3.0 * 1.2), "{c:?}");
}

/// A torus cut by a plane square to its axis: two circles at R -+ the half
/// chord of the tube.
#[test]
fn a_plane_cuts_a_torus_in_two_exact_circles() {
    let (big, small, h) = (1.0, 0.3, 0.1);
    let mut s = Scene::new();
    s.add_solid(torus([0.0; 3], [0.0, 0.0, 1.0], big, small, 48, 24));
    s.add_void(solid_box([-2.0, -2.0, h], [2.0, 2.0, 1.0]));
    let c = circles(&brep(&s));
    let half = (small * small - h * h).sqrt();
    assert!(has(&c, h, big - half) && has(&c, h, big + half), "{c:?}");
}

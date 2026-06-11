//! CDT WP2 gate: after segment recovery, every PLC segment is a union of DT
//! edges, and every Steiner point lies EXACTLY on its carrier line (verified
//! with the staged-exact predicates, not f64).

use rapidmesh_exact::{orient3d, Point3, Sign};
use rapidmesh_geom::{cylinder, sheet_rect, solid_box, sphere, torus, FaceTag, Scene, TaggedPlc};
use rapidmesh_tet::{acute_vertices, recover_segments, DelaunayBuilder};

/// All unique PLC triangle edges, as sorted vertex index pairs. (A stricter
/// segment set than the CDT needs: facet-interior edges could be left to
/// face recovery, but recovering them all exercises the machinery harder.)
fn unique_edges(plc: &TaggedPlc) -> Vec<(usize, usize)> {
    let mut set = std::collections::HashSet::new();
    for t in &plc.triangles {
        for e in 0..3 {
            let (a, b) = (t[e] as usize, t[(e + 1) % 3] as usize);
            set.insert((a.min(b), a.max(b)));
        }
    }
    let mut v: Vec<(usize, usize)> = set.into_iter().collect();
    v.sort_unstable();
    v
}

/// Two points spanning independent planes through the line (a, b), for the
/// exact on-line check: P is on the line iff it is coplanar with both.
fn line_witnesses(a: [f64; 3], b: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let d: [f64; 3] = std::array::from_fn(|k| b[k] - a[k]);
    let axis = if d[0].abs() <= d[1].abs() && d[0].abs() <= d[2].abs() {
        [1.0, 0.0, 0.0]
    } else if d[1].abs() <= d[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let u = [
        d[1] * axis[2] - d[2] * axis[1],
        d[2] * axis[0] - d[0] * axis[2],
        d[0] * axis[1] - d[1] * axis[0],
    ];
    let v = [
        d[1] * u[2] - d[2] * u[1],
        d[2] * u[0] - d[0] * u[2],
        d[0] * u[1] - d[1] * u[0],
    ];
    (
        std::array::from_fn(|k| a[k] + u[k]),
        std::array::from_fn(|k| a[k] + v[k]),
    )
}

/// Runs recovery on the PLC and checks the gates; returns (number of
/// segments, number of Steiner points inserted).
fn check_recovery(plc: &TaggedPlc) -> (usize, usize) {
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for p in &plc.vertices {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let mut b = DelaunayBuilder::enclosing(lo, hi);
    for &p in &plc.vertices {
        b.insert(p);
    }
    let segments = unique_edges(plc);
    let acute = acute_vertices(&plc.vertices, &segments);
    let recovered = recover_segments(&mut b, &segments, &acute);
    assert_eq!(recovered.len(), segments.len());

    let mut steiner = 0;
    for (k, rs) in recovered.iter().enumerate() {
        let (sa, sb) = segments[k];
        assert_eq!(rs.chain[0], sa, "segment {k} chain must start at v1");
        assert_eq!(*rs.chain.last().expect("chain non-empty"), sb);
        for w in rs.chain.windows(2) {
            assert!(
                b.edge_exists(w[0], w[1]),
                "piece ({}, {}) of segment {k} is not a DT edge",
                w[0],
                w[1],
            );
        }
        let pa = Point3::Explicit(plc.vertices[sa]);
        let pb = Point3::Explicit(plc.vertices[sb]);
        let (w1, w2) = line_witnesses(plc.vertices[sa], plc.vertices[sb]);
        let (w1, w2) = (Point3::Explicit(w1), Point3::Explicit(w2));
        for &v in &rs.chain[1..rs.chain.len() - 1] {
            let p = b.exact_point(v);
            if !matches!(p, Point3::Lnc { .. }) {
                continue; // adopted T-junction vertex, on the line only in f64
            }
            steiner += 1;
            assert_eq!(
                orient3d(&pa, &pb, &w1, &p),
                Some(Sign::Zero),
                "Steiner {v} off the carrier of segment {k}",
            );
            assert_eq!(
                orient3d(&pa, &pb, &w2, &p),
                Some(Sign::Zero),
                "Steiner {v} off the carrier of segment {k}",
            );
        }
    }
    eprintln!(
        "{} segments, {} steiner points inserted",
        segments.len(),
        steiner,
    );
    (segments.len(), steiner)
}

#[test]
fn air_dielectric_sheet_scene_segments_recover() {
    let mut scene = Scene::new();
    scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
    scene.add_solid(solid_box([1.0, 1.0, 1.0], [3.0, 3.0, 2.0]));
    scene.add_sheet(
        sheet_rect([1.5, 1.5, 2.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        FaceTag(7),
    );
    scene.add_sheet(
        sheet_rect([0.5, 0.5, 3.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        FaceTag(7),
    );
    let (n, _) = check_recovery(&scene.assemble());
    assert!(n > 0);
}

#[test]
fn cylinder_via_in_box_segments_recover() {
    let mut scene = Scene::new();
    scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 2.0]));
    scene.add_solid(cylinder([2.0, 2.0, 0.0], [0.0, 0.0, 2.0], 0.7, 16));
    let (n, _) = check_recovery(&scene.assemble());
    assert!(n > 0);
}

#[test]
fn sphere_in_box_segments_recover() {
    let mut scene = Scene::new();
    scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
    scene.add_solid(sphere([2.0, 2.0, 2.0], 1.1, 12, 7));
    let (n, _) = check_recovery(&scene.assemble());
    assert!(n > 0);
}

#[test]
fn offset_stacked_boxes_with_t_junctions_recover() {
    // Offset stacking welds partial face overlaps: vertices of one box land
    // ON segments of the other (T-junctions), the input model violation the
    // adoption path exists for.
    let mut scene = Scene::new();
    scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
    scene.add_solid(solid_box([0.5, 0.5, 0.5], [2.5, 2.5, 1.5]));
    scene.add_solid(solid_box([1.5, 0.5, 1.5], [3.5, 2.5, 2.5]));
    scene.add_sheet(
        sheet_rect([1.0, 1.0, 1.5], [1.5, 0.0, 0.0], [0.0, 1.0, 0.0]),
        FaceTag(3),
    );
    let (n, _) = check_recovery(&scene.assemble());
    assert!(n > 0);
}

#[test]
fn torus_segments_recover() {
    let mut scene = Scene::new();
    scene.add_solid(solid_box([-3.0, -3.0, -1.5], [3.0, 3.0, 1.5]));
    scene.add_solid(torus([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.6, 0.6, 14, 8));
    let (n, _) = check_recovery(&scene.assemble());
    assert!(n > 0);
}

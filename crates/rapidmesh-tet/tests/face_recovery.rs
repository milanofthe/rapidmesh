//! CDT WP3 gate: after full PLC recovery, no DT edge pierces any PLC facet
//! (verified by an independent brute-force sweep over ALL DT edges), every
//! segment chain piece is a DT edge, and the triangulation structure is
//! intact (mutual neighbor wiring via the surgery-validated builder).

use rapidmesh_exact::{orient3d, Point3, Sign};
use rapidmesh_geom::{cylinder, sheet_rect, solid_box, sphere, torus, FaceTag, Scene, TaggedPlc};
use rapidmesh_tet::{acute_vertices, recover_plc, DelaunayBuilder, FacetRef};

/// Unique PLC triangle edges plus the facet list referencing them.
fn plc_constraints(plc: &TaggedPlc) -> (Vec<(usize, usize)>, Vec<FacetRef>) {
    let mut ids: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
    let mut segments = Vec::new();
    let mut facets = Vec::new();
    for t in &plc.triangles {
        let corners = [t[0] as usize, t[1] as usize, t[2] as usize];
        let edges = std::array::from_fn(|e| {
            let (a, b) = (corners[e], corners[(e + 1) % 3]);
            let key = (a.min(b), a.max(b));
            *ids.entry(key).or_insert_with(|| {
                segments.push(key);
                segments.len() - 1
            })
        });
        facets.push(FacetRef { corners, edges });
    }
    (segments, facets)
}

/// Brute force: does the open segment (u, v) strictly cross the open
/// triangle (a, b, c)? Mirrors the recovery predicate but runs over ALL
/// current DT edges, independently of the cavity search.
fn pierces(u: &Point3, v: &Point3, a: &Point3, b: &Point3, c: &Point3) -> bool {
    let su = orient3d(a, b, c, u).unwrap();
    let sv = orient3d(a, b, c, v).unwrap();
    if su == Sign::Zero || sv == Sign::Zero || su == sv {
        return false;
    }
    let s1 = orient3d(u, v, a, b).unwrap();
    let s2 = orient3d(u, v, b, c).unwrap();
    let s3 = orient3d(u, v, c, a).unwrap();
    s1 != Sign::Zero && s1 == s2 && s2 == s3
}

fn check_full_recovery(plc: &TaggedPlc) {
    check_full_recovery_with(plc, &[]);
}

/// Like [`check_full_recovery`] but with extra non-PLC obstacle points
/// inserted before recovery (the refinement stage will create exactly this
/// situation: free vertices whose DT edges pierce constraint facets).
fn check_full_recovery_with(plc: &TaggedPlc, extra: &[[f64; 3]]) {
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for p in &plc.vertices {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    for p in extra {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let mut b = DelaunayBuilder::enclosing(lo, hi);
    for &p in &plc.vertices {
        b.insert(p);
    }
    for &p in extra {
        b.insert(p);
    }
    let (segments, facets) = plc_constraints(plc);
    let acute = acute_vertices(&plc.vertices, &segments);
    let chains = recover_plc(&mut b, &segments, &facets, &acute);

    // Every chain piece is a DT edge.
    for k in 0..chains.segment_count() {
        for w in chains.chain(k).windows(2) {
            assert!(b.edge_exists(w[0], w[1]), "chain piece of segment {k} missing");
        }
    }

    // Independent sweep: collect ALL DT edges, test against ALL facets.
    let mut edges: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    for t in b.tets() {
        for i in 0..4 {
            for j in i + 1..4 {
                edges.insert((t[i].min(t[j]), t[i].max(t[j])));
            }
        }
    }
    let pts: Vec<Point3> = (0..b.len()).map(|i| b.exact_point(i)).collect();
    let mut pierced = 0usize;
    for (fi, f) in facets.iter().enumerate() {
        let [a, c1, c2] = f.corners.map(|v| &pts[v]);
        for &(u, v) in &edges {
            if pierces(&pts[u], &pts[v], a, c1, c2) {
                eprintln!("edge ({u}, {v}) pierces facet {fi}");
                pierced += 1;
            }
        }
    }
    assert_eq!(pierced, 0, "facets remain pierced after recovery");
    eprintln!(
        "{} facets, {} segments, {} vertices, {} cavities / {} wrap tets so far",
        facets.len(),
        segments.len(),
        b.len(),
        rapidmesh_tet::cdt::FACE_CAVITIES.load(std::sync::atomic::Ordering::Relaxed),
        rapidmesh_tet::cdt::WRAP_TETS.load(std::sync::atomic::Ordering::Relaxed),
    );
}

#[test]
fn air_dielectric_sheet_scene_faces_recover() {
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
    check_full_recovery(&scene.assemble());
}

#[test]
fn cylinder_via_in_box_faces_recover() {
    let mut scene = Scene::new();
    scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 2.0]));
    scene.add_solid(cylinder([2.0, 2.0, 0.0], [0.0, 0.0, 2.0], 0.7, 16));
    check_full_recovery(&scene.assemble());
}

#[test]
fn sphere_in_box_faces_recover() {
    let mut scene = Scene::new();
    scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
    scene.add_solid(sphere([2.0, 2.0, 2.0], 1.1, 12, 7));
    check_full_recovery(&scene.assemble());
}

#[test]
fn offset_stacked_boxes_faces_recover() {
    let mut scene = Scene::new();
    scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
    scene.add_solid(solid_box([0.5, 0.5, 0.5], [2.5, 2.5, 1.5]));
    scene.add_solid(solid_box([1.5, 0.5, 1.5], [3.5, 2.5, 2.5]));
    scene.add_sheet(
        sheet_rect([1.0, 1.0, 1.5], [1.5, 0.0, 0.0], [0.0, 1.0, 0.0]),
        FaceTag(3),
    );
    check_full_recovery(&scene.assemble());
}

#[test]
fn sheet_pierced_by_point_cloud_recovers() {
    // A large interior sheet with free points scattered above and below:
    // their DT edges pierce the sheet everywhere, forcing real cavity
    // retetrahedrization along the whole facet (the refinement situation).
    let mut scene = Scene::new();
    scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
    scene.add_sheet(
        sheet_rect([0.5, 0.5, 2.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]),
        FaceTag(9),
    );
    let mut rng = rapidmesh_testutil::Rng::new(0xFACE);
    let mut extra = Vec::new();
    for _ in 0..80 {
        let r = |rng: &mut rapidmesh_testutil::Rng| (rng.next_u64() % 3600) as f64 / 1000.0 + 0.2;
        let z_low = (rng.next_u64() % 1700) as f64 / 1000.0 + 0.1;
        let z_high = (rng.next_u64() % 1700) as f64 / 1000.0 + 2.2;
        let z = if rng.next_u64().is_multiple_of(2) { z_low } else { z_high };
        extra.push([r(&mut rng), r(&mut rng), z]);
    }
    let before = rapidmesh_tet::cdt::FACE_CAVITIES.load(std::sync::atomic::Ordering::Relaxed);
    check_full_recovery_with(&scene.assemble(), &extra);
    let after = rapidmesh_tet::cdt::FACE_CAVITIES.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        after > before,
        "the point cloud must force at least one cavity retetrahedrization",
    );
}

#[test]
fn torus_faces_recover() {
    let mut scene = Scene::new();
    scene.add_solid(solid_box([-3.0, -3.0, -1.5], [3.0, 3.0, 1.5]));
    scene.add_solid(torus([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.6, 0.6, 14, 8));
    check_full_recovery(&scene.assemble());
}

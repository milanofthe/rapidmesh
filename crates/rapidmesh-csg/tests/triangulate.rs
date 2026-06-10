//! triangulate_facet invariants, checked exactly via the rational oracle.

mod common;

use common::check_invariants;
use rapidmesh_csg::{
    tri_tri_intersection, triangulate_facet, Constraint, ConstraintLine, Tri, TriTriIsect,
};
use rapidmesh_exact::Point3;
use rapidmesh_testutil::Rng;

#[test]
fn empty_facet_is_a_single_triangle() {
    let facet = Tri::new([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]);
    let ft = triangulate_facet(&facet, &[], &[]);
    assert_eq!(ft.triangles.len(), 1);
    check_invariants(&facet, &ft, &[]);
}

#[test]
fn single_interior_point_splits_into_three() {
    let facet = Tri::new([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]);
    let p = Point3::explicit(1.0, 1.0, 0.0);
    let ft = triangulate_facet(&facet, &[p], &[]);
    assert_eq!(ft.triangles.len(), 3);
    check_invariants(&facet, &ft, &[]);
}

#[test]
fn point_on_boundary_edge_splits_into_two() {
    let facet = Tri::new([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]);
    let p = Point3::explicit(2.0, 0.0, 0.0);
    let ft = triangulate_facet(&facet, &[p], &[]);
    assert_eq!(ft.triangles.len(), 2);
    check_invariants(&facet, &ft, &[]);
}

#[test]
fn crossing_constraints_get_split_at_their_tpi() {
    // Two plane-cut constraints crossing inside the facet.
    let facet = Tri::new([-4.0, -4.0, 0.0], [4.0, -4.0, 0.0], [0.0, 4.0, 0.0]);
    let cut1 = Tri::new([-3.0, 0.0, -1.0], [3.0, 0.0, -1.0], [0.0, 0.0, 2.0]);
    let cut2 = Tri::new([0.5, -3.0, -1.0], [0.5, 1.0, -1.0], [0.5, -1.0, 2.0]);
    let mut constraints = Vec::new();
    for cut in [&cut1, &cut2] {
        match tri_tri_intersection(&facet, cut) {
            TriTriIsect::Segment(a, b) => constraints.push(Constraint {
                a,
                b,
                line: ConstraintLine::PlaneCut(cut.v),
            }),
            other => panic!("expected segment, got {other:?}"),
        }
    }
    let ft = triangulate_facet(&facet, &[], &constraints);
    check_invariants(&facet, &ft, &constraints);
    // The crossing point (0.5, 0, 0) must be a vertex.
    let crossing = Point3::explicit(0.5, 0.0, 0.0);
    assert!(
        ft.vertices.iter().any(|v| v.coincides(&crossing)),
        "TPI crossing vertex missing"
    );
}

#[test]
fn randomized_plane_cut_constraints_keep_all_invariants() {
    let mut rng = Rng::new(0x7A);
    let facet = Tri::new([-4.0, -4.0, 0.0], [4.0, -4.0, 0.0], [0.0, 4.0, 0.0]);
    let mut nontrivial = 0;
    for _ in 0..60 {
        let mut constraints = Vec::new();
        let mut points = Vec::new();
        for _ in 0..4 {
            // Random triangle with vertices on both sides of z = 0 so a
            // proper cut is likely.
            let cut = Tri::new(
                [rng.grid(12), rng.grid(12), -rng.grid(8).abs() - 0.25],
                [rng.grid(12), rng.grid(12), rng.grid(8).abs() + 0.25],
                [rng.grid(12), rng.grid(12), rng.grid(8)],
            );
            match tri_tri_intersection(&facet, &cut) {
                TriTriIsect::Segment(a, b) => constraints.push(Constraint {
                    a,
                    b,
                    line: ConstraintLine::PlaneCut(cut.v),
                }),
                TriTriIsect::Touching(p) => points.push(p),
                _ => {}
            }
        }
        if !constraints.is_empty() {
            nontrivial += 1;
        }
        let ft = triangulate_facet(&facet, &points, &constraints);
        check_invariants(&facet, &ft, &constraints);
    }
    assert!(nontrivial > 30, "expected many constrained cases, got {nontrivial}");
}

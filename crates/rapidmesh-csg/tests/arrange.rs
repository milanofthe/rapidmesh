//! Arrangement invariants on box soups: per-facet exactness plus the
//! cross-facet glue property (shared intersection vertices coincide exactly
//! across the facets of every intersecting pair).

mod common;

use common::{box_tris, check_invariants};
use rapidmesh_csg::{arrange, tri_tri_intersection, Tri, TriTriIsect};
use rapidmesh_exact::{collinear, within_closed, Point3};

/// Triangulation vertices lying on the closed segment [a, b].
fn vertices_on_segment<'a>(
    vertices: &'a [Point3],
    a: &Point3,
    b: &Point3,
) -> Vec<&'a Point3> {
    vertices
        .iter()
        .filter(|p| {
            collinear(a, b, p).expect("valid") && within_closed(a, b, p).expect("valid")
        })
        .collect()
}

/// Full suite: per-facet invariants plus pairwise glue.
fn check_arrangement(tris: &[Tri]) -> rapidmesh_csg::Arrangement {
    let arr = arrange(tris);
    for (i, t) in tris.iter().enumerate() {
        check_invariants(t, &arr.facets[i], &arr.constraints[i]);
    }
    // Glue: for every properly intersecting pair, the triangulation vertices
    // on the shared segment must match one-to-one across the two facets.
    for i in 0..tris.len() {
        for j in i + 1..tris.len() {
            let TriTriIsect::Segment(a, b) = tri_tri_intersection(&tris[i], &tris[j]) else {
                continue;
            };
            let vi = vertices_on_segment(&arr.facets[i].vertices, &a, &b);
            let vj = vertices_on_segment(&arr.facets[j].vertices, &a, &b);
            assert_eq!(
                vi.len(),
                vj.len(),
                "facets {i} and {j} disagree on shared-segment vertex count"
            );
            for p in &vi {
                assert!(
                    vj.iter().any(|q| p.coincides(q)),
                    "vertex on shared segment of facets {i}/{j} missing on the other side"
                );
            }
        }
    }
    arr
}

#[test]
fn overlapping_boxes() {
    let mut tris = box_tris([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    tris.extend(box_tris([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]));
    let arr = check_arrangement(&tris);
    let cut = arr.facets.iter().filter(|f| f.triangles.len() > 1).count();
    assert!(cut >= 6, "expected many cut facets, got {cut}");
}

#[test]
fn stacked_boxes_with_partial_coplanar_contact() {
    // Top face of the lower box and bottom face of the upper box share the
    // plane z = 1 and overlap in [1,2]x[1,2]: the coplanar clipping path.
    let mut tris = box_tris([0.0, 0.0, 0.0], [2.0, 2.0, 1.0]);
    tris.extend(box_tris([1.0, 1.0, 1.0], [3.0, 3.0, 2.0]));
    let arr = check_arrangement(&tris);
    // Coplanar contact must have produced edge-provenance constraints.
    let edge_constraints: usize = arr
        .constraints
        .iter()
        .flatten()
        .filter(|c| matches!(c.line, rapidmesh_csg::ConstraintLine::Edge(..)))
        .count();
    assert!(
        edge_constraints >= 4,
        "expected coplanar edge constraints, got {edge_constraints}"
    );
}

#[test]
fn touching_boxes_share_a_corner() {
    // Boxes touching in exactly one corner point.
    let mut tris = box_tris([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    tris.extend(box_tris([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]));
    check_arrangement(&tris);
}

#[test]
fn single_box_is_left_intact() {
    // Facets of one box only touch along shared edges/corners; nothing may
    // be subdivided.
    let tris = box_tris([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let arr = check_arrangement(&tris);
    for f in &arr.facets {
        assert_eq!(f.triangles.len(), 1, "box facet must stay a single triangle");
    }
}

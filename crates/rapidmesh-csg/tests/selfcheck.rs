//! The exact self-check of a triangle surface: what meets only where it
//! shares passes, everything else is named.

use rapidmesh_csg::improper_pairs;

/// A unit cube, outward: 8 corners, 12 triangles.
fn cube() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let v: Vec<[f64; 3]> = (0..8)
        .map(|i| [(i & 1) as f64, ((i >> 1) & 1) as f64, ((i >> 2) & 1) as f64])
        .collect();
    let quads = [
        [0, 2, 3, 1],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 4, 6, 2],
        [1, 3, 7, 5],
    ];
    let t = quads
        .iter()
        .flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
        .collect();
    (v, t)
}

#[test]
fn a_closed_cube_is_clean() {
    let (v, t) = cube();
    assert!(improper_pairs(&v, &t).is_empty());
}

#[test]
fn crossings_folds_overlaps_and_duplicates_are_named() {
    let v = vec![
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.5, 0.5, -1.0],
        [0.5, 0.5, 1.0],
        [1.5, 1.5, 0.0],
        [1.0, 0.2, 0.0],
        [0.2, 1.0, 0.0],
        [2.0, 2.0, 0.0],
    ];
    // Sharing nothing, one pierces the other.
    assert_eq!(improper_pairs(&v, &[[0, 1, 2], [3, 4, 5]]), [[0, 1]]);
    // Sharing an edge, folded onto each other in the plane.
    assert_eq!(improper_pairs(&v, &[[0, 1, 2], [0, 1, 7]]), [[0, 1]]);
    // Sharing an edge, side by side: fine.
    assert!(improper_pairs(&v, &[[0, 1, 2], [1, 8, 2]]).is_empty());
    // Sharing a vertex, overlapping in the plane.
    assert_eq!(improper_pairs(&v, &[[0, 1, 2], [0, 6, 7]]), [[0, 1]]);
    // A duplicate.
    assert_eq!(improper_pairs(&v, &[[0, 1, 2], [2, 0, 1]]), [[0, 1]]);
    // A corner on another's edge, sharing nothing (a T-junction), and a
    // triangle of zero area.
    let w = vec![
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
    ];
    assert_eq!(improper_pairs(&w, &[[0, 1, 2], [3, 5, 4]]), [[0, 1]]);
    assert_eq!(improper_pairs(&w, &[[0, 1, 3]]), [[0, 0]]);
}

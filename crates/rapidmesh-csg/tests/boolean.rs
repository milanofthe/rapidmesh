//! Boolean operations on box pairs: exact rational volumes vs analytic
//! ground truth, plus watertightness of every non-empty result.

mod common;

use common::{assert_watertight, box_tris, volume6};
use num_rational::BigRational;
use rapidmesh_csg::{boolean, BoolOp, BooleanResult, Solid};
use rapidmesh_testutil::rat;

fn solid(min: [f64; 3], max: [f64; 3]) -> Solid {
    Solid {
        tris: box_tris(min, max),
    }
}

fn run(a: &Solid, b: &Solid, op: BoolOp, expected_volume6: f64) -> BooleanResult {
    let res = boolean(a, b, op);
    let v6: BigRational = volume6(&res.vertices, &res.triangles);
    assert_eq!(v6, rat(expected_volume6), "volume mismatch for {op:?}");
    if expected_volume6 == 0.0 {
        assert!(res.triangles.is_empty(), "zero-volume result must be empty");
    } else {
        assert_watertight(&res.triangles);
    }
    res
}

#[test]
fn overlapping_boxes() {
    // [0,2]^3 and [1,3]^3 overlap in the unit cube: V = 8, 8, overlap 1.
    let a = solid([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    let b = solid([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);
    run(&a, &b, BoolOp::Union, 6.0 * 15.0);
    run(&a, &b, BoolOp::Intersection, 6.0 * 1.0);
    run(&a, &b, BoolOp::Difference, 6.0 * 7.0);
}

#[test]
fn identical_boxes() {
    let a = solid([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    let b = a.clone();
    run(&a, &b, BoolOp::Union, 6.0 * 8.0);
    run(&a, &b, BoolOp::Intersection, 6.0 * 8.0);
    run(&a, &b, BoolOp::Difference, 0.0);
}

#[test]
fn disjoint_boxes() {
    let a = solid([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = solid([2.0, 2.0, 2.0], [3.0, 3.0, 3.0]);
    let union = run(&a, &b, BoolOp::Union, 6.0 * 2.0);
    assert_eq!(union.triangles.len(), 24, "disjoint union keeps both boxes");
    run(&a, &b, BoolOp::Intersection, 0.0);
    let diff = run(&a, &b, BoolOp::Difference, 6.0 * 1.0);
    assert_eq!(diff.triangles.len(), 12, "difference leaves A intact");
}

#[test]
fn face_touching_boxes() {
    // Sharing the full face x = 1 (opposite normals there).
    let a = solid([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = solid([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
    run(&a, &b, BoolOp::Union, 6.0 * 2.0);
    run(&a, &b, BoolOp::Intersection, 0.0);
    run(&a, &b, BoolOp::Difference, 6.0 * 1.0);
}

#[test]
fn partially_stacked_boxes() {
    // B sits on top of A; contact patch [1,2]x[1,2] in the plane z = 1.
    let a = solid([0.0, 0.0, 0.0], [2.0, 2.0, 1.0]);
    let b = solid([1.0, 1.0, 1.0], [3.0, 3.0, 2.0]);
    run(&a, &b, BoolOp::Union, 6.0 * 8.0);
    run(&a, &b, BoolOp::Intersection, 0.0);
    run(&a, &b, BoolOp::Difference, 6.0 * 4.0);
}

#[test]
fn box_through_box_tunnel() {
    // A thin bar punched through a larger box: the difference is a tunnel
    // (genus 1) — exercises classification with multiple crossings per ray.
    let a = solid([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]);
    let bar = solid([1.0, 1.0, -1.0], [2.0, 2.0, 5.0]);
    // Bar volume inside A: 1*1*4 = 4.
    run(&a, &bar, BoolOp::Union, 6.0 * (64.0 + 6.0 - 4.0));
    run(&a, &bar, BoolOp::Intersection, 6.0 * 4.0);
    run(&a, &bar, BoolOp::Difference, 6.0 * 60.0);
}

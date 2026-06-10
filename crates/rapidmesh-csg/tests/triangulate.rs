//! triangulate_facet invariants, checked exactly via the rational oracle.

use num_rational::BigRational;
use num_traits::{Signed, Zero};
use rapidmesh_csg::{
    tri_tri_intersection, triangulate_facet, Constraint, ConstraintLine, FacetTriangulation,
    Tri, TriTriIsect,
};
use rapidmesh_exact::{collinear, orient2d, within_closed, Axis, Point3};
use rapidmesh_testutil::{affine, rat, Rng, Rv};

/// Signed rational area (times 2) of a sub-triangle in the facet projection.
fn area2(tri: [&Rv; 3], axis: Axis) -> BigRational {
    // Cyclic pairing must match Point3::hom2: drop X -> (y, z), etc.
    let (u, v) = match axis {
        Axis::X => (1, 2),
        Axis::Y => (2, 0),
        Axis::Z => (0, 1),
    };
    let [a, b, c] = tri;
    (&b[u] - &a[u]) * (&c[v] - &a[v]) - (&b[v] - &a[v]) * (&c[u] - &a[u])
}

/// Full invariant suite for one triangulated facet.
fn check_invariants(facet: &Tri, ft: &FacetTriangulation, constraints: &[Constraint]) {
    // 1. Orientation/non-degeneracy: every sub-triangle oriented like the
    //    facet, exactly.
    for t in &ft.triangles {
        let s = orient2d(
            &ft.vertices[t[0]],
            &ft.vertices[t[1]],
            &ft.vertices[t[2]],
            ft.axis,
        );
        assert_eq!(s, Some(ft.orientation), "degenerate or flipped sub-triangle");
    }

    // 2. Exact area conservation.
    let verts_rat: Vec<Rv> = ft.vertices.iter().map(affine).collect();
    let total: BigRational = ft.triangles.iter().fold(BigRational::zero(), |acc, t| {
        acc + area2([&verts_rat[t[0]], &verts_rat[t[1]], &verts_rat[t[2]]], ft.axis)
    });
    let facet_area = area2([&verts_rat[0], &verts_rat[1], &verts_rat[2]], ft.axis);
    assert_eq!(total, facet_area, "sub-triangle areas must sum to facet area");

    // 3. Euler: T = 2 * V_interior + V_boundary - 2.
    let corner = [facet.point(0), facet.point(1), facet.point(2)];
    let on_boundary = |p: &Point3| -> bool {
        (0..3).any(|e| {
            let a = &corner[e];
            let b = &corner[(e + 1) % 3];
            collinear(a, b, p).expect("valid") && within_closed(a, b, p).expect("valid")
        })
    };
    let v_bnd = ft.vertices.iter().filter(|p| on_boundary(p)).count();
    let v_int = ft.vertices.len() - v_bnd;
    assert_eq!(
        ft.triangles.len(),
        2 * v_int + v_bnd - 2,
        "Euler count mismatch: V_int={v_int} V_bnd={v_bnd}"
    );

    // 4. Constraint coverage: triangulation edges on each constraint cover
    //    exactly its rational length (no gaps, no overlaps).
    for c in constraints {
        let (ca, cb) = (affine(&c.a), affine(&c.b));
        let dir: Rv = std::array::from_fn(|i| &cb[i] - &ca[i]);
        let seg_len2 = rapidmesh_testutil::rv_dot(&dir, &dir);
        if seg_len2.is_zero() {
            continue;
        }
        // Parameter of a vertex along the constraint, if it lies on it.
        let param = |k: usize| -> Option<BigRational> {
            let p = &ft.vertices[k];
            if !collinear(&c.a, &c.b, p).expect("valid")
                || !within_closed(&c.a, &c.b, p).expect("valid")
            {
                return None;
            }
            let rel: Rv = std::array::from_fn(|i| &verts_rat[k][i] - &ca[i]);
            Some(rapidmesh_testutil::rv_dot(&rel, &dir) / &seg_len2)
        };
        let mut covered = BigRational::zero();
        let mut seen = std::collections::HashSet::new();
        for t in &ft.triangles {
            for e in 0..3 {
                let (u, v) = (t[e], t[(e + 1) % 3]);
                let key = (u.min(v), u.max(v));
                if !seen.insert(key) {
                    continue;
                }
                if let (Some(tu), Some(tv)) = (param(u), param(v)) {
                    covered += (tu - tv).abs();
                }
            }
        }
        assert_eq!(
            covered,
            rat(1.0),
            "constraint must be exactly covered by triangulation edges"
        );
    }
}

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

//! Shared invariant checks and fixtures for csg integration tests.

use num_rational::BigRational;
use num_traits::{Signed, Zero};
use rapidmesh_csg::{Constraint, FacetTriangulation, Tri};
use rapidmesh_exact::{collinear, orient2d, within_closed, Axis, Point3};
use rapidmesh_testutil::{affine, rat, rv_dot, Rv};

/// Signed rational area (times 2) of a triangle in the facet projection.
/// The cyclic pairing matches `Point3::hom2`: drop X -> (y, z), etc.
#[allow(dead_code)]
pub fn area2(tri: [&Rv; 3], axis: Axis) -> BigRational {
    let (u, v) = match axis {
        Axis::X => (1, 2),
        Axis::Y => (2, 0),
        Axis::Z => (0, 1),
    };
    let [a, b, c] = tri;
    (&b[u] - &a[u]) * (&c[v] - &a[v]) - (&b[v] - &a[v]) * (&c[u] - &a[u])
}

/// Full exact invariant suite for one triangulated facet:
/// orientation/non-degeneracy, area conservation, Euler count, and exact
/// coverage of every constraint by triangulation edges.
#[allow(dead_code)]
pub fn check_invariants(facet: &Tri, ft: &FacetTriangulation, constraints: &[Constraint]) {
    // 1. Every sub-triangle oriented like the facet, exactly.
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
        let seg_len2 = rv_dot(&dir, &dir);
        if seg_len2.is_zero() {
            continue;
        }
        let param = |k: usize| -> Option<BigRational> {
            let p = &ft.vertices[k];
            if !collinear(&c.a, &c.b, p).expect("valid")
                || !within_closed(&c.a, &c.b, p).expect("valid")
            {
                return None;
            }
            let rel: Rv = std::array::from_fn(|i| &verts_rat[k][i] - &ca[i]);
            Some(rv_dot(&rel, &dir) / &seg_len2)
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

/// Exact 6x the signed volume enclosed by an outward-oriented closed
/// triangle surface (divergence theorem over origin tetrahedra).
#[allow(dead_code)]
pub fn volume6(vertices: &[Point3], triangles: &[[usize; 3]]) -> BigRational {
    let verts_rat: Vec<Rv> = vertices.iter().map(affine).collect();
    triangles.iter().fold(BigRational::zero(), |acc, t| {
        let (a, b, c) = (&verts_rat[t[0]], &verts_rat[t[1]], &verts_rat[t[2]]);
        let det = &a[0] * (&b[1] * &c[2] - &b[2] * &c[1])
            - &a[1] * (&b[0] * &c[2] - &b[2] * &c[0])
            + &a[2] * (&b[0] * &c[1] - &b[1] * &c[0]);
        acc + det
    })
}

/// Asserts the surface is a closed orientable manifold: every directed edge
/// appears exactly once and its reverse exists.
#[allow(dead_code)]
pub fn assert_watertight(triangles: &[[usize; 3]]) {
    let mut directed: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for t in triangles {
        assert!(
            t[0] != t[1] && t[1] != t[2] && t[2] != t[0],
            "degenerate output triangle {t:?}"
        );
        for e in 0..3 {
            *directed.entry((t[e], t[(e + 1) % 3])).or_default() += 1;
        }
    }
    for (&(u, v), &n) in &directed {
        assert_eq!(n, 1, "directed edge ({u},{v}) used {n} times");
        assert!(
            directed.contains_key(&(v, u)),
            "directed edge ({u},{v}) has no opposite — surface not closed"
        );
    }
}

/// The 12 triangles of an axis-aligned box (outward orientation).
// Compiled once per test binary; not every binary uses every fixture.
#[allow(dead_code)]
pub fn box_tris(min: [f64; 3], max: [f64; 3]) -> Vec<Tri> {
    // Corner index bits: bit0 = x, bit1 = y, bit2 = z.
    let c: [[f64; 3]; 8] = std::array::from_fn(|i| {
        [
            if i & 1 == 0 { min[0] } else { max[0] },
            if i & 2 == 0 { min[1] } else { max[1] },
            if i & 4 == 0 { min[2] } else { max[2] },
        ]
    });
    let quads: [[usize; 4]; 6] = [
        [0, 2, 3, 1], // -z
        [4, 5, 7, 6], // +z
        [0, 1, 5, 4], // -y
        [2, 6, 7, 3], // +y
        [0, 4, 6, 2], // -x
        [1, 3, 7, 5], // +x
    ];
    let mut tris = Vec::with_capacity(12);
    for q in quads {
        tris.push(Tri::new(c[q[0]], c[q[1]], c[q[2]]));
        tris.push(Tri::new(c[q[0]], c[q[2]], c[q[3]]));
    }
    tris
}

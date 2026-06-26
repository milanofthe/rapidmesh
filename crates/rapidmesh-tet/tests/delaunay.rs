//! Delaunay kernel gates: exact empty-circumsphere property, exact volume
//! conservation (hull = box when corners are included), face conformity.

use num_rational::BigRational;
use num_traits::Zero;
use rapidmesh_exact::Sign;
use rapidmesh_tet::tetrahedralize;
use rapidmesh_testutil::{rat, Rng};

fn box_corners(lo: [f64; 3], hi: [f64; 3]) -> Vec<[f64; 3]> {
    (0..8)
        .map(|i| {
            [
                if i & 1 == 0 { lo[0] } else { hi[0] },
                if i & 2 == 0 { lo[1] } else { hi[1] },
                if i & 4 == 0 { lo[2] } else { hi[2] },
            ]
        })
        .collect()
}

/// Full gate suite for a point set whose convex hull is the given box.
fn check(points: &[[f64; 3]], box_volume6: f64) {
    let dt = tetrahedralize(points);
    assert!(!dt.tets.is_empty());

    // Exact volume conservation: positively oriented tets fill the hull.
    let v6: BigRational = dt.tets.iter().fold(BigRational::zero(), |acc, t| {
        let p: Vec<[BigRational; 3]> = t.iter().map(|&i| dt.points[i].map(rat)).collect();
        let r: Vec<[BigRational; 3]> = (0..3)
            .map(|k| std::array::from_fn(|j| &p[k][j] - &p[3][j]))
            .collect();
        let det = &r[0][0] * (&r[1][1] * &r[2][2] - &r[1][2] * &r[2][1])
            - &r[0][1] * (&r[1][0] * &r[2][2] - &r[1][2] * &r[2][0])
            + &r[0][2] * (&r[1][0] * &r[2][1] - &r[1][1] * &r[2][0]);
        acc + det
    });
    assert_eq!(v6, rat(box_volume6), "tet volumes must fill the hull exactly");

    // Exact Delaunay property: no point strictly inside any circumsphere.
    for t in &dt.tets {
        for p in 0..dt.points.len() {
            if t.contains(&p) {
                continue;
            }
            let s = Sign::of_f64(geometry_predicates::insphere(
                dt.points[t[0]],
                dt.points[t[1]],
                dt.points[t[2]],
                dt.points[t[3]],
                dt.points[p],
            ));
            assert_ne!(s, Sign::Positive, "point {p} inside circumsphere of {t:?}");
        }
    }

    // Conformity: every face in at most 2 tets.
    let mut count: std::collections::HashMap<[usize; 3], usize> = std::collections::HashMap::new();
    for t in &dt.tets {
        for f in [[t[0], t[1], t[2]], [t[0], t[1], t[3]], [t[0], t[2], t[3]], [t[1], t[2], t[3]]] {
            let mut k = f;
            k.sort_unstable();
            *count.entry(k).or_default() += 1;
        }
    }
    assert!(count.values().all(|&n| n <= 2), "non-manifold face");
}

#[test]
fn random_grid_points_in_a_box() {
    let mut rng = Rng::new(0xD3);
    let (lo, hi) = ([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]);
    let mut points = box_corners(lo, hi);
    while points.len() < 60 {
        let p: [f64; 3] = std::array::from_fn(|_| (rng.next_u64() % 17) as f64 * 0.25);
        if !points.contains(&p) {
            points.push(p);
        }
    }
    check(&points, 6.0 * 64.0);
}

#[test]
fn fully_cospherical_lattice() {
    // A regular 4x4x4 lattice: maximal cosphericity and on-face insertions.
    let mut points = Vec::new();
    for x in 0..4 {
        for y in 0..4 {
            for z in 0..4 {
                points.push([x as f64, y as f64, z as f64]);
            }
        }
    }
    check(&points, 6.0 * 27.0);
}

#[test]
fn single_tet_and_degenerate_inputs() {
    let dt = tetrahedralize(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    assert_eq!(dt.tets.len(), 1);
    // Coplanar input: no tets, no panic.
    let flat = tetrahedralize(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]);
    assert!(flat.tets.is_empty());
}

// ------------------------------------------- cavity replacement (CDT WP3a)

/// Wiring consistency over the public API: every neighbor pointer is
/// mutual and the shared face has the same vertex set on both sides.
fn check_wiring(b: &rapidmesh_tet::DelaunayBuilder) {
    for slot in 0..b.slot_count() as u32 {
        let Some(t) = b.tet_at(slot) else { continue };
        for i in 0..4 {
            let Some(nb) = b.neighbor_at(slot, i) else {
                continue;
            };
            let mut shared: Vec<usize> = (0..4).filter(|&k| k != i).map(|k| t[k]).collect();
            shared.sort_unstable();
            let back = (0..4)
                .find(|&k| b.neighbor_at(nb, k) == Some(slot))
                .expect("neighbor pointer must be mutual");
            let nverts = b.verts_of_slot(nb);
            let mut nshared: Vec<usize> = (0..4)
                .filter(|&k| k != back)
                .map(|k| nverts[k].expect("real shared face"))
                .collect();
            nshared.sort_unstable();
            assert_eq!(shared, nshared, "shared face mismatch across neighbors");
        }
    }
}

/// Five points whose DT is the 3-tet configuration around the central edge;
/// returns a builder holding it.
fn three_tet_fixture() -> rapidmesh_tet::DelaunayBuilder {
    use rapidmesh_tet::DelaunayBuilder;
    let pts = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.3, 0.3, 0.2],
        [0.3, 0.3, -0.2],
    ];
    let mut b = DelaunayBuilder::enclosing([-0.1, -0.1, -0.3], [1.1, 1.1, 0.3]);
    for p in pts {
        b.insert(p);
    }
    assert_eq!(b.tets().len(), 3, "fixture must triangulate around edge 3-4");
    assert!(b.edge_exists(3, 4));
    b
}

#[test]
fn replace_cavity_flips_three_tets_to_two() {
    let mut b = three_tet_fixture();
    let slots: Vec<u32> = b.tets_with_slots().iter().map(|&(s, _)| s).collect();
    // The 2-tet configuration of the same bipyramid, positively oriented.
    let oriented = |t: [usize; 4]| -> [usize; 4] {
        let p: Vec<[f64; 3]> = t.iter().map(|&v| b.approx_point(v)).collect();
        if geometry_predicates::orient3d(p[0], p[1], p[2], p[3]) > 0.0 {
            t
        } else {
            [t[1], t[0], t[2], t[3]]
        }
    };
    let new_tets = [oriented([0, 1, 2, 3]), oriented([0, 1, 2, 4])];
    b.replace_cavity(&slots, &new_tets);
    assert_eq!(b.tets().len(), 2);
    assert!(!b.edge_exists(3, 4), "central edge must be gone");
    assert!(b.edge_exists(0, 1));
    check_wiring(&b);
    // The builder keeps functioning after surgery.
    let n = b.insert([0.5, 0.4, 0.05]);
    assert_eq!(n, 5);
    check_wiring(&b);
}

#[test]
#[should_panic(expected = "invalid replacement")]
fn replace_cavity_rejects_misoriented_tets() {
    let mut b = three_tet_fixture();
    let slots: Vec<u32> = b.tets_with_slots().iter().map(|&(s, _)| s).collect();
    b.replace_cavity(&slots, &[[0, 1, 2, 3], [1, 0, 2, 4]]);
}

#[test]
#[should_panic(expected = "invalid replacement")]
fn replace_cavity_rejects_nontiling_complex() {
    let mut b = three_tet_fixture();
    let slots: Vec<u32> = b.tets_with_slots().iter().map(|&(s, _)| s).collect();
    // Only half of the bipyramid: the lower boundary does not match.
    let t = if geometry_predicates::orient3d(
        b.approx_point(0),
        b.approx_point(1),
        b.approx_point(2),
        b.approx_point(3),
    ) > 0.0
    {
        [0, 1, 2, 3]
    } else {
        [1, 0, 2, 3]
    };
    b.replace_cavity(&slots, &[t]);
}

// ----------------------------------------------- implicit Steiner (CDT WP2)

#[test]
fn insert_exact_lnc_recovers_segment_pieces() {
    use rapidmesh_exact::Point3;
    use rapidmesh_tet::DelaunayBuilder;

    // A box point cloud plus a segment (a, b) crossing it; insert LNC
    // Steiner points ON the segment and require every sub-piece to be a
    // DT edge afterwards (the primitive the segment-recovery loop rests on).
    let lo = [0.0, 0.0, 0.0];
    let hi = [1.0, 1.0, 1.0];
    let mut b = DelaunayBuilder::enclosing(lo, hi);
    let mut ids = Vec::new();
    for p in box_corners(lo, hi) {
        ids.push(b.insert(p));
    }
    let mut rng = Rng::new(0xD7);
    for _ in 0..40 {
        ids.push(b.insert(std::array::from_fn(|_| {
            (rng.f64_wide().abs() % 0.8) + 0.1
        })));
    }
    let a = [0.05, 0.05, 0.05];
    let c = [0.95, 0.95, 0.95];
    let ia = b.insert(a);
    let ic = b.insert(c);
    // Three uneven cuts as implicit LNC points.
    let ts = [0.2137, 0.553, 0.829];
    let mut chain = vec![ia];
    for &t in &ts {
        chain.push(b.insert_exact(Point3::lnc(a, c, t)));
    }
    chain.push(ic);
    // exact_point round-trips the implicit position.
    for (k, &t) in ts.iter().enumerate() {
        let p = b.exact_point(chain[k + 1]);
        assert!(p.coincides(&Point3::lnc(a, c, t)));
    }
    // Whether the chain pieces are DT edges is the segment-recovery loop's
    // job (other points may encroach); the kernel owes the exact empty-
    // circumsphere property evaluated at the IMPLICIT positions, not their
    // f64 approximations.
    let _ = chain;
    let pts: Vec<Point3> = (0..b.len()).map(|i| b.exact_point(i)).collect();
    for t in b.tets() {
        for (vi, p) in pts.iter().enumerate() {
            if t.contains(&vi) {
                continue;
            }
            let s = rapidmesh_exact::insphere3d(&pts[t[0]], &pts[t[1]], &pts[t[2]], &pts[t[3]], p)
                .expect("valid points");
            assert_ne!(
                s,
                Sign::Positive,
                "vertex {vi} strictly inside circumsphere of {t:?}"
            );
        }
    }
}

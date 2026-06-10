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

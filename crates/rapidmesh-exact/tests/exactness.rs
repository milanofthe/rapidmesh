//! Correctness gates against a rational oracle.
//!
//! BigRational evaluates the SAME generic geometric expressions (via a Ring
//! impl on a newtype) — that validates the arithmetic. Independent rational
//! derivations (solving the line-plane intersection directly) validate the
//! homogeneous formulas themselves.

use num_rational::BigRational;
use num_traits::Zero;
use rapidmesh_exact::expansion::Expansion;
use rapidmesh_exact::geom::{det3, det4};
use rapidmesh_exact::interval::Interval;
use rapidmesh_exact::orient::{orient2d, orient3d};
use rapidmesh_exact::point::Point3;
use rapidmesh_exact::ring::Ring;
use rapidmesh_exact::{Axis, Sign};
use rapidmesh_testutil::{expansion_to_rat, rat, sign_of_rat, Rat, Rng};

trait RngExt {
    fn point_grid(&mut self) -> [f64; 3];
}

impl RngExt for Rng {
    fn point_grid(&mut self) -> [f64; 3] {
        self.point3(16)
    }
}

// ----------------------------------------------------------- expansion tests

#[test]
fn expansion_add_matches_rational() {
    let mut rng = Rng::new(0xE1);
    for _ in 0..500 {
        let a: Vec<f64> = (0..5).map(|_| rng.f64_wide()).collect();
        let b: Vec<f64> = (0..5).map(|_| rng.f64_wide()).collect();
        let ea = a
            .iter()
            .fold(Expansion::from_f64(0.0), |acc, &v| acc.add(&Expansion::from_f64(v)));
        let eb = b
            .iter()
            .fold(Expansion::from_f64(0.0), |acc, &v| acc.add(&Expansion::from_f64(v)));
        let sum = ea.add(&eb);
        let want: BigRational = a.iter().chain(&b).fold(BigRational::zero(), |acc, &v| acc + rat(v));
        assert_eq!(expansion_to_rat(&sum), want);
    }
}

#[test]
fn expansion_mul_matches_rational() {
    let mut rng = Rng::new(0xE2);
    for _ in 0..500 {
        let a: Vec<f64> = (0..4).map(|_| rng.f64_wide()).collect();
        let b: Vec<f64> = (0..4).map(|_| rng.f64_wide()).collect();
        let ea = a
            .iter()
            .fold(Expansion::from_f64(0.0), |acc, &v| acc.add(&Expansion::from_f64(v)));
        let eb = b
            .iter()
            .fold(Expansion::from_f64(0.0), |acc, &v| acc.add(&Expansion::from_f64(v)));
        let prod = ea.mul(&eb);
        let ra = a.iter().fold(BigRational::zero(), |acc, &v| acc + rat(v));
        let rb = b.iter().fold(BigRational::zero(), |acc, &v| acc + rat(v));
        assert_eq!(expansion_to_rat(&prod), ra * rb);
    }
}

#[test]
fn expansion_catastrophic_cancellation() {
    // 1e16 + 1 - 1e16 == 1 exactly; f64 loses it, the expansion must not.
    let e = Expansion::from_f64(1e16)
        .add(&Expansion::from_f64(1.0))
        .add(&Expansion::from_f64(-1e16));
    assert_eq!(expansion_to_rat(&e), rat(1.0));
    assert_eq!(e.sign(), Sign::Positive);

    // Exact zero.
    let z = Expansion::from_f64(3.5).add(&Expansion::from_f64(-3.5));
    assert!(z.is_zero());
    assert_eq!(z.sign(), Sign::Zero);
}

// ------------------------------------------------------------ interval tests

#[test]
fn interval_det4_contains_rational_value() {
    let mut rng = Rng::new(0x17);
    for _ in 0..300 {
        let m: [[f64; 4]; 4] = std::array::from_fn(|_| std::array::from_fn(|_| rng.f64_wide()));
        let mi: [[Interval; 4]; 4] =
            std::array::from_fn(|i| std::array::from_fn(|j| Interval::point(m[i][j])));
        let mr: [[Rat; 4]; 4] =
            std::array::from_fn(|i| std::array::from_fn(|j| Rat::from_f64(m[i][j])));
        let di = det4(&mi);
        let dr = det4(&mr).0;
        assert!(rat(di.lo()) <= dr && dr <= rat(di.hi()), "det4 interval must contain exact value");
    }
}

#[test]
fn expansion_det4_matches_rational_value() {
    let mut rng = Rng::new(0x18);
    for _ in 0..100 {
        let m: [[f64; 4]; 4] = std::array::from_fn(|_| std::array::from_fn(|_| rng.f64_wide()));
        let me: [[Expansion; 4]; 4] =
            std::array::from_fn(|i| std::array::from_fn(|j| Expansion::from_f64(m[i][j])));
        let mr: [[Rat; 4]; 4] =
            std::array::from_fn(|i| std::array::from_fn(|j| Rat::from_f64(m[i][j])));
        assert_eq!(expansion_to_rat(&det4(&me)), det4(&mr).0);
    }
}

// ------------------------------------------------------------- orient3d tests

/// Rational-oracle orientation through the same homogeneous machinery.
fn orient3d_oracle(pts: [&Point3; 4]) -> Option<Sign> {
    let homs: [[Rat; 4]; 4] = std::array::from_fn(|i| pts[i].hom::<Rat>());
    let mut sign = sign_of_rat(&det4(&homs).0);
    for h in &homs {
        match sign_of_rat(&h[3].0) {
            Sign::Zero => return None,
            s => sign = sign.combine(s),
        }
    }
    Some(sign)
}

#[test]
fn orient3d_explicit_sign_convention() {
    // d above the ccw plane (a, b, c) seen from +z: negative by convention.
    let a = Point3::explicit(0.0, 0.0, 0.0);
    let b = Point3::explicit(1.0, 0.0, 0.0);
    let c = Point3::explicit(0.0, 1.0, 0.0);
    let d = Point3::explicit(0.0, 0.0, 1.0);
    assert_eq!(orient3d(&a, &b, &c, &d), Some(Sign::Negative));
    assert_eq!(orient3d_oracle([&a, &b, &c, &d]), Some(Sign::Negative));
}

#[test]
fn orient3d_explicit_matches_oracle() {
    let mut rng = Rng::new(0x03);
    let mut zeros = 0;
    for i in 0..1000 {
        // Mix fine grid (general position) and coarse grid (degeneracies).
        let pts: [Point3; 4] = std::array::from_fn(|_| {
            Point3::Explicit(if i % 2 == 0 {
                rng.point_grid()
            } else {
                rng.point_coarse()
            })
        });
        let got = orient3d(&pts[0], &pts[1], &pts[2], &pts[3]);
        let want = orient3d_oracle([&pts[0], &pts[1], &pts[2], &pts[3]]);
        assert_eq!(got, want);
        if got == Some(Sign::Zero) {
            zeros += 1;
        }
    }
    // Coarse-grid rounds must have produced exact coplanarity.
    assert!(zeros > 0, "coarse grid points should hit exact coplanar cases");
}

#[test]
fn orient3d_lnc_matches_oracle() {
    // The affine-reduction fast path (orient3d with one Lnc Steiner point and
    // three explicit points) must agree with the rational oracle, both where
    // the segment is clearly on one side and where it straddles the plane
    // (fall-through), and at parameters near 0 and 1.
    let mut rng = Rng::new(0x4C);
    let ts = [1e-12, 1e-3, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0 - 1e-3, 1.0 - 1e-12];
    let (mut zeros, mut nonzeros) = (0u32, 0u32);
    for i in 0..3000 {
        let coarse = i % 3 == 0; // coarse grid hits coplanar/degenerate configs
        let mut mk = |r: &mut Rng| {
            if coarse {
                r.point_coarse()
            } else {
                r.point_grid()
            }
        };
        let e = [
            Point3::Explicit(mk(&mut rng)),
            Point3::Explicit(mk(&mut rng)),
            Point3::Explicit(mk(&mut rng)),
        ];
        let lnc = Point3::lnc(mk(&mut rng), mk(&mut rng), ts[i % ts.len()]);
        // Lnc in the 4th position (the recovery side-test shape) and the 1st
        // (to exercise the position handling).
        for arr in [
            [e[0].clone(), e[1].clone(), e[2].clone(), lnc.clone()],
            [lnc.clone(), e[0].clone(), e[1].clone(), e[2].clone()],
        ] {
            let got = orient3d(&arr[0], &arr[1], &arr[2], &arr[3]);
            let want = orient3d_oracle([&arr[0], &arr[1], &arr[2], &arr[3]]);
            assert_eq!(got, want, "lnc orient3d disagrees with oracle");
            match got {
                Some(Sign::Zero) => zeros += 1,
                Some(_) => nonzeros += 1,
                None => {}
            }
        }
    }
    assert!(nonzeros > 0 && zeros > 0, "test should hit both signed and coplanar cases");
}

#[test]
fn orient3d_exactly_coplanar_affine_combination() {
    let mut rng = Rng::new(0x04);
    for _ in 0..200 {
        let a = rng.point_grid();
        let b = rng.point_grid();
        let c = rng.point_grid();
        // d = b + c - a is an affine combination (coefficients sum to 1) and
        // exact in f64 on the grid: always coplanar with a, b, c.
        let d = [b[0] + c[0] - a[0], b[1] + c[1] - a[1], b[2] + c[2] - a[2]];
        let got = orient3d(
            &Point3::Explicit(a),
            &Point3::Explicit(b),
            &Point3::Explicit(c),
            &Point3::Explicit(d),
        );
        assert_eq!(got, Some(Sign::Zero));
    }
}

// ----------------------------------------------------------------- LPI tests

#[test]
fn lpi_lies_exactly_on_its_plane_and_line() {
    let mut rng = Rng::new(0x15);
    let mut valid = 0;
    for _ in 0..400 {
        let (p, q) = (rng.point_grid(), rng.point_grid());
        let (r, s, t) = (rng.point_grid(), rng.point_grid(), rng.point_grid());
        let lpi = Point3::lpi(p, q, r, s, t);
        if !lpi.is_valid() {
            continue;
        }
        valid += 1;
        let (ep, eq) = (Point3::Explicit(p), Point3::Explicit(q));
        let (er, es, et) = (
            Point3::Explicit(r),
            Point3::Explicit(s),
            Point3::Explicit(t),
        );
        // On the defining plane: orientation with the plane's points is zero.
        assert_eq!(orient3d(&lpi, &er, &es, &et), Some(Sign::Zero));
        // On the defining line: coplanar with p, q and ANY fourth point.
        let w = Point3::Explicit(rng.point_grid());
        assert_eq!(orient3d(&ep, &eq, &lpi, &w), Some(Sign::Zero));
    }
    assert!(valid > 100, "expected many valid LPI cases, got {valid}");
}

#[test]
fn lpi_orientation_matches_independent_rational_solve() {
    let mut rng = Rng::new(0x16);
    let mut valid = 0;
    for _ in 0..300 {
        let (p, q) = (rng.point_grid(), rng.point_grid());
        let (r, s, t) = (rng.point_grid(), rng.point_grid(), rng.point_grid());

        // Independent rational derivation: solve for the intersection point
        // affinely, then take the affine 3x3 orientation determinant.
        let rp: Vec<BigRational> = p.iter().map(|&v| rat(v)).collect();
        let rq: Vec<BigRational> = q.iter().map(|&v| rat(v)).collect();
        let rr: Vec<BigRational> = r.iter().map(|&v| rat(v)).collect();
        let rs: Vec<BigRational> = s.iter().map(|&v| rat(v)).collect();
        let rt: Vec<BigRational> = t.iter().map(|&v| rat(v)).collect();
        let sub = |a: &[BigRational], b: &[BigRational]| -> Vec<BigRational> {
            (0..3).map(|i| &a[i] - &b[i]).collect()
        };
        let cross = |a: &[BigRational], b: &[BigRational]| -> Vec<BigRational> {
            vec![
                &a[1] * &b[2] - &a[2] * &b[1],
                &a[2] * &b[0] - &a[0] * &b[2],
                &a[0] * &b[1] - &a[1] * &b[0],
            ]
        };
        let dot = |a: &[BigRational], b: &[BigRational]| -> BigRational {
            &a[0] * &b[0] + &a[1] * &b[1] + &a[2] * &b[2]
        };
        let n = cross(&sub(&rs, &rr), &sub(&rt, &rr));
        let dir = sub(&rq, &rp);
        let denom = dot(&n, &dir);
        if denom.is_zero() {
            continue;
        }
        valid += 1;
        let tau = dot(&n, &sub(&rr, &rp)) / &denom;
        let inter: Vec<BigRational> = (0..3).map(|i| &rp[i] + &tau * &dir[i]).collect();

        // Affine rational orient3d(inter, x, y, z) via det [[a-d],[b-d],[c-d]].
        let (x, y, z) = (rng.point_grid(), rng.point_grid(), rng.point_grid());
        let rx: Vec<BigRational> = x.iter().map(|&v| rat(v)).collect();
        let ry: Vec<BigRational> = y.iter().map(|&v| rat(v)).collect();
        let rz: Vec<BigRational> = z.iter().map(|&v| rat(v)).collect();
        let r0 = sub(&inter, &rz);
        let r1 = sub(&rx, &rz);
        let r2 = sub(&ry, &rz);
        let det = &r0[0] * (&r1[1] * &r2[2] - &r1[2] * &r2[1])
            - &r0[1] * (&r1[0] * &r2[2] - &r1[2] * &r2[0])
            + &r0[2] * (&r1[0] * &r2[1] - &r1[1] * &r2[0]);
        let want = Some(sign_of_rat(&det));

        let lpi = Point3::lpi(p, q, r, s, t);
        assert!(lpi.is_valid());
        let got = orient3d(
            &lpi,
            &Point3::Explicit(x),
            &Point3::Explicit(y),
            &Point3::Explicit(z),
        );
        assert_eq!(got, want);
    }
    assert!(valid > 100, "expected many valid cases, got {valid}");
}

#[test]
fn lpi_invalid_when_line_parallel_to_plane() {
    // Plane z = 0; line at z = 1 parallel to it.
    let lpi = Point3::lpi(
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    );
    assert!(!lpi.is_valid());
    let e = Point3::explicit(0.0, 0.0, 0.0);
    let f = Point3::explicit(1.0, 0.0, 0.0);
    let g = Point3::explicit(0.0, 1.0, 0.0);
    assert_eq!(orient3d(&lpi, &e, &f, &g), None);
}

// ----------------------------------------------------------------- TPI tests

#[test]
fn tpi_lies_exactly_on_each_defining_plane() {
    let mut rng = Rng::new(0x77);
    let mut valid = 0;
    for _ in 0..300 {
        let planes: [[[f64; 3]; 3]; 3] =
            std::array::from_fn(|_| std::array::from_fn(|_| rng.point_grid()));
        let tpi = Point3::tpi(planes[0], planes[1], planes[2]);
        if !tpi.is_valid() {
            continue;
        }
        valid += 1;
        for plane in &planes {
            let a = Point3::Explicit(plane[0]);
            let b = Point3::Explicit(plane[1]);
            let c = Point3::Explicit(plane[2]);
            assert_eq!(orient3d(&tpi, &a, &b, &c), Some(Sign::Zero));
        }
    }
    assert!(valid > 50, "expected many valid TPI cases, got {valid}");
}

#[test]
fn implicit_orientation_matches_oracle_in_general_position() {
    let mut rng = Rng::new(0x99);
    let mut checked = 0;
    for _ in 0..200 {
        let lpi = Point3::lpi(
            rng.point_grid(),
            rng.point_grid(),
            rng.point_grid(),
            rng.point_grid(),
            rng.point_grid(),
        );
        let planes: [[[f64; 3]; 3]; 3] =
            std::array::from_fn(|_| std::array::from_fn(|_| rng.point_grid()));
        let tpi = Point3::tpi(planes[0], planes[1], planes[2]);
        let a = Point3::Explicit(rng.point_grid());
        let b = Point3::Explicit(rng.point_grid());
        let got = orient3d(&lpi, &tpi, &a, &b);
        let want = orient3d_oracle([&lpi, &tpi, &a, &b]);
        assert_eq!(got, want);
        if got.is_some() {
            checked += 1;
        }
    }
    assert!(checked > 50, "expected many decidable cases, got {checked}");
}

// ------------------------------------------------------------ orient2d tests

/// Rational-oracle 2D orientation through the same homogeneous machinery.
fn orient2d_oracle(pts: [&Point3; 3], drop: Axis) -> Option<Sign> {
    let homs: [[Rat; 3]; 3] = std::array::from_fn(|i| pts[i].hom2::<Rat>(drop));
    let mut sign = sign_of_rat(&det3(&homs).0);
    for h in &homs {
        match sign_of_rat(&h[2].0) {
            Sign::Zero => return None,
            s => sign = sign.combine(s),
        }
    }
    Some(sign)
}

#[test]
fn orient2d_explicit_matches_oracle_all_axes() {
    let mut rng = Rng::new(0x2D);
    let mut zeros = 0;
    for i in 0..900 {
        let pts: [Point3; 3] = std::array::from_fn(|_| {
            Point3::Explicit(if i % 2 == 0 {
                rng.point_grid()
            } else {
                rng.point_coarse()
            })
        });
        for drop in [Axis::X, Axis::Y, Axis::Z] {
            let got = orient2d(&pts[0], &pts[1], &pts[2], drop);
            let want = orient2d_oracle([&pts[0], &pts[1], &pts[2]], drop);
            assert_eq!(got, want);
            if got == Some(Sign::Zero) {
                zeros += 1;
            }
        }
    }
    assert!(zeros > 0, "coarse grid should hit exact 2D collinearity");
}

#[test]
fn orient2d_implicit_matches_oracle() {
    let mut rng = Rng::new(0x2E);
    let mut checked = 0;
    for _ in 0..200 {
        let lpi = Point3::lpi(
            rng.point_grid(),
            rng.point_grid(),
            rng.point_grid(),
            rng.point_grid(),
            rng.point_grid(),
        );
        let a = Point3::Explicit(rng.point_grid());
        let b = Point3::Explicit(rng.point_grid());
        for drop in [Axis::X, Axis::Y, Axis::Z] {
            let got = orient2d(&a, &b, &lpi, drop);
            let want = orient2d_oracle([&a, &b, &lpi], drop);
            assert_eq!(got, want);
            if got.is_some() {
                checked += 1;
            }
        }
    }
    assert!(checked > 100, "expected many decidable cases, got {checked}");
}

#[test]
fn orient2d_lpi_on_line_is_collinear_in_every_projection() {
    let mut rng = Rng::new(0x2F);
    let mut valid = 0;
    for _ in 0..200 {
        let (p, q) = (rng.point_grid(), rng.point_grid());
        let (r, s, t) = (rng.point_grid(), rng.point_grid(), rng.point_grid());
        let lpi = Point3::lpi(p, q, r, s, t);
        if !lpi.is_valid() {
            continue;
        }
        valid += 1;
        // 3D collinearity with p, q survives every axis projection.
        for drop in [Axis::X, Axis::Y, Axis::Z] {
            assert_eq!(
                orient2d(&Point3::Explicit(p), &Point3::Explicit(q), &lpi, drop),
                Some(Sign::Zero)
            );
        }
    }
    assert!(valid > 50, "expected many valid LPI cases, got {valid}");
}

// ----------------------------------------------------------- coincidence

#[test]
fn coincides_lpi_with_known_explicit_point() {
    // Line through (0,0,-1)->(0,0,3) hits plane z=0 at the origin.
    let lpi = Point3::lpi(
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    );
    assert!(lpi.coincides(&Point3::explicit(0.0, 0.0, 0.0)));
    assert!(!lpi.coincides(&Point3::explicit(0.0, 0.0, 1e-300)));
    assert!(!lpi.coincides(&Point3::explicit(0.0, 1.0, 0.0)));
}

#[test]
fn coincides_same_point_from_different_constructions() {
    let mut rng = Rng::new(0xC0);
    let mut valid = 0;
    for _ in 0..200 {
        let (p, q) = (rng.point_grid(), rng.point_grid());
        let (r, s, t) = (rng.point_grid(), rng.point_grid(), rng.point_grid());
        let lpi_a = Point3::lpi(p, q, r, s, t);
        // Same line, same plane: defining points permuted (and the line
        // reversed). Must be recognized as the same point.
        let lpi_b = Point3::lpi(q, p, s, t, r);
        if !lpi_a.is_valid() {
            continue;
        }
        valid += 1;
        assert!(lpi_a.coincides(&lpi_b));
        assert!(lpi_b.coincides(&lpi_a));
    }
    assert!(valid > 50, "expected many valid cases, got {valid}");
}

#[test]
fn coincides_rejects_distinct_implicit_points() {
    let mut rng = Rng::new(0xC1);
    let mut checked = 0;
    for _ in 0..200 {
        let (p, q) = (rng.point_grid(), rng.point_grid());
        let (r, s, t) = (rng.point_grid(), rng.point_grid(), rng.point_grid());
        // Two parallel planes one unit apart along z: the same line meets
        // them in distinct points (unless it is parallel to them).
        let shift = |v: [f64; 3]| [v[0], v[1], v[2] + 1.0];
        let lpi_a = Point3::lpi(p, q, r, s, t);
        let lpi_b = Point3::lpi(p, q, shift(r), shift(s), shift(t));
        if !lpi_a.is_valid() || !lpi_b.is_valid() {
            continue;
        }
        checked += 1;
        assert!(!lpi_a.coincides(&lpi_b));
    }
    assert!(checked > 50, "expected many valid cases, got {checked}");
}

// ----------------------------------------------------------- approx sanity

#[test]
fn approx_is_close_to_rational_intersection() {
    // Line through (0,0,-1)->(0,0,1), plane z = 0.25: intersection (0,0,0.25).
    let lpi = Point3::lpi(
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.25],
        [1.0, 0.0, 0.25],
        [0.0, 1.0, 0.25],
    );
    let a = lpi.approx().expect("valid point");
    assert!((a[0]).abs() < 1e-12 && (a[1]).abs() < 1e-12 && (a[2] - 0.25).abs() < 1e-12);
}

// ------------------------------------------------- lnc + insphere3d (CDT WP1)

use rapidmesh_exact::insphere3d;

/// Independent rational position of an LNC point: a + t (b - a) computed in
/// exact rational arithmetic straight from the f64 inputs.
fn lnc_affine_rat(a: [f64; 3], b: [f64; 3], t: f64) -> [BigRational; 3] {
    std::array::from_fn(|i| rat(a[i]) + rat(t) * (rat(b[i]) - rat(a[i])))
}

/// Independent rational in-sphere oracle over AFFINE rational positions:
/// classic 4x4 with rows (p_i - p_4, |p_i|^2 - |p_4|^2). Sign convention is
/// calibrated against Shewchuk's `insphere` by the explicit randomized test.
fn insphere_affine_oracle(p: [[BigRational; 3]; 5]) -> Sign {
    let row = |i: usize| -> [Rat; 4] {
        let d: [BigRational; 3] =
            std::array::from_fn(|k| p[i][k].clone() - p[4][k].clone());
        let lift = d.iter().map(|v| v * v).fold(BigRational::zero(), |a, v| a + v);
        [Rat(d[0].clone()), Rat(d[1].clone()), Rat(d[2].clone()), Rat(lift)]
    };
    let m: [[Rat; 4]; 4] = std::array::from_fn(row);
    sign_of_rat(&det4(&m).0)
}

fn golden_t(i: usize) -> f64 {
    (((i as f64) + 1.0) * 0.618_033_988_749_894_9).fract().clamp(0.05, 0.95)
}

#[test]
fn lnc_lies_exactly_on_its_carrier_line() {
    let mut rng = Rng::new(0xC0);
    let mut checked = 0;
    for i in 0..400 {
        let a = rng.point3(16);
        let b = rng.point3(16);
        if a == b {
            continue;
        }
        let l = Point3::lnc(a, b, golden_t(i));
        let (pa, pb) = (Point3::Explicit(a), Point3::Explicit(b));
        // Three collinear points make (a, b, x, l) exactly coplanar for
        // EVERY x; two independent witnesses pin collinearity.
        for _ in 0..2 {
            let x = Point3::Explicit(rng.point3(16));
            assert_eq!(orient3d(&pa, &pb, &x, &l), Some(Sign::Zero));
        }
        checked += 1;
    }
    assert!(checked > 300);
}

#[test]
fn pac_lies_exactly_on_its_carrier_plane() {
    let mut rng = Rng::new(0xC7);
    let mut checked = 0;
    for i in 0..400 {
        let a = rng.point3(16);
        let b = rng.point3(16);
        let c = rng.point3(16);
        if a == b || b == c || a == c {
            continue;
        }
        let u = golden_t(i);
        let v = golden_t(i + 13) * (1.0 - u);
        let p = Point3::pac(a, b, c, u, v);
        // Exactly coplanar with the carrier triangle, for crooked (u, v).
        assert_eq!(
            orient3d(
                &Point3::Explicit(a),
                &Point3::Explicit(b),
                &Point3::Explicit(c),
                &p,
            ),
            Some(Sign::Zero),
        );
        // And NOT on the carrier lines (generic parameters): a witness off
        // the plane must see a nonzero volume with two corners and p.
        let x = Point3::Explicit(rng.point3(16));
        if orient3d(
            &Point3::Explicit(a),
            &Point3::Explicit(b),
            &Point3::Explicit(c),
            &x,
        ) != Some(Sign::Zero)
        {
            assert_ne!(
                orient3d(&Point3::Explicit(a), &Point3::Explicit(b), &x, &p),
                Some(Sign::Zero),
                "pac point degenerated onto a carrier edge",
            );
        }
        checked += 1;
    }
    assert!(checked > 300);
}

#[test]
fn lnc_orient3d_matches_independent_rational() {
    let mut rng = Rng::new(0xC1);
    for i in 0..400 {
        let a = rng.point3(16);
        let b = rng.point3(16);
        let c = rng.point3(16);
        let d = rng.point3(16);
        let t = golden_t(i);
        let l = Point3::lnc(a, b, t);
        let got = orient3d(&Point3::Explicit(c), &Point3::Explicit(d), &Point3::Explicit(a), &l);
        // Independent affine-rational determinant.
        let lp = lnc_affine_rat(a, b, t);
        let rows: [[Rat; 3]; 3] = [
            std::array::from_fn(|k| Rat(rat(c[k]) - lp[k].clone())),
            std::array::from_fn(|k| Rat(rat(d[k]) - lp[k].clone())),
            std::array::from_fn(|k| Rat(rat(a[k]) - lp[k].clone())),
        ];
        let want = sign_of_rat(&det3(&rows).0);
        assert_eq!(got, Some(want));
    }
}

#[test]
fn insphere3d_explicit_matches_independent_rational() {
    let mut rng = Rng::new(0xC2);
    for i in 0..600 {
        let p: [[f64; 3]; 5] = std::array::from_fn(|_| {
            if i % 2 == 0 {
                rng.point3(16)
            } else {
                rng.point3(4)
            }
        });
        let e: [Point3; 5] = p.map(Point3::Explicit);
        let got = insphere3d(&e[0], &e[1], &e[2], &e[3], &e[4]);
        let want = insphere_affine_oracle(std::array::from_fn(|j| {
            std::array::from_fn(|k| rat(p[j][k]))
        }));
        assert_eq!(got, Some(want), "points {p:?}");
    }
}

#[test]
fn insphere3d_staged_path_matches_fast_path() {
    let mut rng = Rng::new(0xC3);
    for _ in 0..300 {
        let p: [[f64; 3]; 5] = std::array::from_fn(|_| rng.point3(16));
        let e: [Point3; 5] = p.map(Point3::Explicit);
        let fast = insphere3d(&e[0], &e[1], &e[2], &e[3], &e[4]);
        // An exact-bary wrapper forces the homogeneous lift path.
        let wrapped = Point3::bary(e[4].clone(), e[4].clone(), e[4].clone());
        let staged = insphere3d(&e[0], &e[1], &e[2], &e[3], &wrapped);
        assert_eq!(fast, staged);
    }
}

#[test]
fn insphere3d_cosphere_is_exactly_zero_on_staged_path() {
    // Unit-cube corners: (1,1,0) lies ON the circumsphere of the tet
    // (0,0,0),(1,0,0),(0,1,0),(0,0,1) (centre (.5,.5,.5), r^2 = 3/4).
    let a = Point3::explicit(0.0, 0.0, 0.0);
    let b = Point3::explicit(1.0, 0.0, 0.0);
    let c = Point3::explicit(0.0, 1.0, 0.0);
    let d = Point3::explicit(0.0, 0.0, 1.0);
    let e = Point3::explicit(1.0, 1.0, 0.0);
    assert_eq!(insphere3d(&a, &b, &c, &d, &e), Some(Sign::Zero));
    let wrapped = Point3::bary(e.clone(), e.clone(), e.clone());
    assert_eq!(insphere3d(&a, &b, &c, &d, &wrapped), Some(Sign::Zero));
}

#[test]
fn insphere3d_with_lnc_matches_independent_rational() {
    let mut rng = Rng::new(0xC4);
    for i in 0..400 {
        let p: [[f64; 3]; 4] = std::array::from_fn(|_| rng.point3(16));
        let (sa, sb) = (rng.point3(16), rng.point3(16));
        let t = golden_t(i);
        let l = Point3::lnc(sa, sb, t);
        let e: [Point3; 4] = p.map(Point3::Explicit);
        let got = insphere3d(&e[0], &e[1], &e[2], &e[3], &l);
        let mut affine: [[BigRational; 3]; 5] = std::array::from_fn(|j| {
            std::array::from_fn(|k| rat(p[std::cmp::min(j, 3)][k]))
        });
        affine[4] = lnc_affine_rat(sa, sb, t);
        let want = insphere_affine_oracle(affine);
        assert_eq!(got, Some(want));
    }
}

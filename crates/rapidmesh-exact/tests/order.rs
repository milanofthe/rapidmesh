//! cmp_along / collinear / lli_coplanar vs rational oracle and predicates.

use num_rational::BigRational;
use num_traits::{Signed, Zero};
use rapidmesh_exact::{
    cmp_along, collinear, strictly_between, within_closed, Point3, Sign,
};
use rapidmesh_testutil::{affine, rat, Rng};

trait RngExt {
    fn point_grid(&mut self) -> [f64; 3];
    fn lpi(&mut self) -> Point3;
}

impl RngExt for Rng {
    fn point_grid(&mut self) -> [f64; 3] {
        self.point3(8)
    }
    fn lpi(&mut self) -> Point3 {
        loop {
            let p = Point3::lpi(
                self.point_grid(),
                self.point_grid(),
                self.point_grid(),
                self.point_grid(),
                self.point_grid(),
            );
            if p.is_valid() {
                return p;
            }
        }
    }
}

#[test]
fn cmp_along_matches_rational_oracle() {
    let mut rng = Rng::new(0xA1);
    for i in 0..300 {
        // Mix explicit and implicit points.
        let mut pt = |k: u64| -> Point3 {
            if (i + k).is_multiple_of(3) {
                rng.lpi()
            } else {
                Point3::Explicit(rng.point_grid())
            }
        };
        let (a, b, p, q) = (pt(0), pt(1), pt(2), pt(3));
        let got = cmp_along(&a, &b, &p, &q).expect("valid points");
        let (ra, rb, rp, rq) = (affine(&a), affine(&b), affine(&p), affine(&q));
        let dot: BigRational = (0..3).map(|k| (&rq[k] - &rp[k]) * (&rb[k] - &ra[k])).sum();
        let want = if dot.is_zero() {
            Sign::Zero
        } else if dot.is_positive() {
            Sign::Positive
        } else {
            Sign::Negative
        };
        assert_eq!(got, want);
    }
}

#[test]
fn betweenness_on_an_exact_chain() {
    // Points on the segment (0,0,0)->(4,2,8), constructed implicitly as
    // intersections with parallel planes x = 1, 2, 3.
    let a = Point3::explicit(0.0, 0.0, 0.0);
    let b = Point3::explicit(4.0, 2.0, 8.0);
    let plane_x = |c: f64| {
        Point3::lpi(
            [0.0, 0.0, 0.0],
            [4.0, 2.0, 8.0],
            [c, 0.0, 0.0],
            [c, 1.0, 0.0],
            [c, 0.0, 1.0],
        )
    };
    let p1 = plane_x(1.0);
    let p2 = plane_x(2.0);
    let p3 = plane_x(3.0);
    for p in [&p1, &p2, &p3] {
        assert!(p.is_valid());
        assert_eq!(collinear(&a, &b, p), Some(true));
        assert_eq!(strictly_between(&a, &b, p), Some(true));
        assert_eq!(within_closed(&a, &b, p), Some(true));
    }
    // Ordering along the chain.
    assert_eq!(cmp_along(&a, &b, &p1, &p2), Some(Sign::Positive));
    assert_eq!(cmp_along(&a, &b, &p2, &p3), Some(Sign::Positive));
    assert_eq!(cmp_along(&a, &b, &p3, &p1), Some(Sign::Negative));
    assert_eq!(cmp_along(&a, &b, &p2, &p2), Some(Sign::Zero));
    // Endpoints: closed yes, strict no.
    assert_eq!(within_closed(&a, &b, &a), Some(true));
    assert_eq!(strictly_between(&a, &b, &a), Some(false));
}

#[test]
fn lli_coplanar_intersection_is_on_both_lines() {
    let mut rng = Rng::new(0xA2);
    let mut valid = 0;
    for _ in 0..300 {
        // Exactly coplanar configuration: all points in the plane z = x
        // (exact for grid coordinates).
        let mut pt = || -> [f64; 3] {
            let x = rng.grid(8);
            let y = rng.grid(8);
            [x, y, x]
        };
        let (p, q, a, b) = (pt(), pt(), pt(), pt());
        let Some(i) = Point3::lli_coplanar(p, q, a, b) else {
            continue;
        };
        valid += 1;
        // Membership in both lines pins the intersection point uniquely.
        assert_eq!(
            collinear(&Point3::Explicit(p), &Point3::Explicit(q), &i),
            Some(true)
        );
        assert_eq!(
            collinear(&Point3::Explicit(a), &Point3::Explicit(b), &i),
            Some(true)
        );
    }
    assert!(valid > 100, "expected many valid intersections, got {valid}");
}

#[test]
fn lli_coplanar_rejects_degenerate_inputs() {
    // Parallel lines.
    assert_eq!(
        Point3::lli_coplanar(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ),
        None
    );
    // Identical lines.
    assert_eq!(
        Point3::lli_coplanar(
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ),
        None
    );
    // Not coplanar.
    assert_eq!(
        Point3::lli_coplanar(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
        ),
        None
    );
}

#[test]
fn lli_coplanar_known_crossing() {
    // Diagonals of the unit square in z = 0 cross at (0.5, 0.5, 0).
    let i = Point3::lli_coplanar(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    )
    .expect("proper crossing");
    assert!(i.coincides(&Point3::explicit(0.5, 0.5, 0.0)));
    assert_eq!(affine(&i), [rat(0.5), rat(0.5), rat(0.0)]);
}

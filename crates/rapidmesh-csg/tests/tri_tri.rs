//! tri_tri_intersection vs an independent rational reference.
//!
//! The reference computes T1 ∩ plane(T0) exactly in BigRational and clips the
//! resulting segment by T0's edge half-planes — a completely independent
//! derivation (no shared code path with the implementation under test).

use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use rapidmesh_csg::{tri_tri_intersection, Tri, TriTriIsect};
use rapidmesh_exact::{orient3d, Point3, Ring, Sign};

// ------------------------------------------------------------- rational ring

#[derive(Clone)]
struct Rat(BigRational);

impl Ring for Rat {
    fn from_f64(v: f64) -> Self {
        Rat(BigRational::from_float(v).expect("finite f64"))
    }
    fn add(&self, other: &Self) -> Self {
        Rat(&self.0 + &other.0)
    }
    fn sub(&self, other: &Self) -> Self {
        Rat(&self.0 - &other.0)
    }
    fn mul(&self, other: &Self) -> Self {
        Rat(&self.0 * &other.0)
    }
    fn neg(&self) -> Self {
        Rat(-&self.0)
    }
}

type Rv = [BigRational; 3];

fn rv(p: [f64; 3]) -> Rv {
    std::array::from_fn(|i| BigRational::from_float(p[i]).expect("finite f64"))
}

fn sub(a: &Rv, b: &Rv) -> Rv {
    std::array::from_fn(|i| &a[i] - &b[i])
}

fn cross(a: &Rv, b: &Rv) -> Rv {
    [
        &a[1] * &b[2] - &a[2] * &b[1],
        &a[2] * &b[0] - &a[0] * &b[2],
        &a[0] * &b[1] - &a[1] * &b[0],
    ]
}

fn dot(a: &Rv, b: &Rv) -> BigRational {
    &a[0] * &b[0] + &a[1] * &b[1] + &a[2] * &b[2]
}

fn lerp(a: &Rv, b: &Rv, tau: &BigRational) -> Rv {
    std::array::from_fn(|i| &a[i] + tau * (&b[i] - &a[i]))
}

/// Rational affine coordinates of a (valid) Point3.
fn affine(p: &Point3) -> Rv {
    let h = p.hom::<Rat>();
    let w = h[3].0.clone();
    assert!(!w.is_zero(), "point must be valid");
    std::array::from_fn(|i| &h[i].0 / &w)
}

// -------------------------------------------------------- rational reference

#[derive(Debug, PartialEq)]
enum RefIsect {
    Coplanar,
    Empty,
    Point(Rv),
    Segment(Rv, Rv),
}

/// Exact rational reference intersection of two non-degenerate triangles.
fn reference(t0: &Tri, t1: &Tri) -> RefIsect {
    let a0 = rv(t0.v[0]);
    let b0 = rv(t0.v[1]);
    let c0 = rv(t0.v[2]);
    let n0 = cross(&sub(&b0, &a0), &sub(&c0, &a0));
    let f = |p: &Rv| dot(&n0, &sub(p, &a0));

    let pv: [Rv; 3] = std::array::from_fn(|i| rv(t1.v[i]));
    let fs: [BigRational; 3] = std::array::from_fn(|i| f(&pv[i]));

    if fs.iter().all(|s| s.is_zero()) {
        return RefIsect::Coplanar;
    }
    if fs.iter().all(|s| s.is_positive()) || fs.iter().all(|s| s.is_negative()) {
        return RefIsect::Empty;
    }

    // T1 ∩ plane(T0): at most two distinct points.
    let mut pts: Vec<Rv> = Vec::new();
    let mut push = |p: Rv| {
        if !pts.contains(&p) {
            pts.push(p);
        }
    };
    for i in 0..3 {
        if fs[i].is_zero() {
            push(pv[i].clone());
        }
        let j = (i + 1) % 3;
        if !fs[i].is_zero() && !fs[j].is_zero() && fs[i].is_positive() != fs[j].is_positive() {
            let tau = &fs[i] / (&fs[i] - &fs[j]);
            push(lerp(&pv[i], &pv[j], &tau));
        }
    }

    // Inside-T0 half-plane functions (within plane(T0)).
    let edges = [(&a0, &b0), (&b0, &c0), (&c0, &a0)];
    let g = |p: &Rv, q: &Rv, x: &Rv| dot(&cross(&sub(q, p), &sub(x, p)), &n0);

    match pts.len() {
        1 => {
            let p = pts.pop().expect("len 1");
            if edges.iter().all(|(ea, eb)| !g(ea, eb, &p).is_negative()) {
                RefIsect::Point(p)
            } else {
                RefIsect::Empty
            }
        }
        2 => {
            let b = pts.pop().expect("len 2");
            let a = pts.pop().expect("len 2");
            // Clip [a, b] (parameter tau in [0, 1]) by each edge half-plane.
            let mut t_lo = BigRational::zero();
            let mut t_hi = BigRational::one();
            for (ea, eb) in edges {
                let g0 = g(ea, eb, &a);
                let g1 = g(ea, eb, &b);
                if g0.is_negative() && g1.is_negative() {
                    return RefIsect::Empty;
                }
                if g0.is_negative() || g1.is_negative() {
                    let tau = &g0 / (&g0 - &g1);
                    if g0.is_negative() {
                        if tau > t_lo {
                            t_lo = tau;
                        }
                    } else if tau < t_hi {
                        t_hi = tau;
                    }
                }
            }
            match t_lo.cmp(&t_hi) {
                std::cmp::Ordering::Greater => RefIsect::Empty,
                std::cmp::Ordering::Equal => RefIsect::Point(lerp(&a, &b, &t_lo)),
                std::cmp::Ordering::Less => {
                    RefIsect::Segment(lerp(&a, &b, &t_lo), lerp(&a, &b, &t_hi))
                }
            }
        }
        n => unreachable!("plane section of a triangle has 1 or 2 points, got {n}"),
    }
}

// ------------------------------------------------------------------ test rng

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn grid(&mut self) -> f64 {
        ((self.next_u64() % 17) as f64 - 8.0) / 4.0
    }
    fn coarse(&mut self) -> f64 {
        (self.next_u64() % 3) as f64 - 1.0
    }
    fn tri(&mut self, coarse: bool) -> Tri {
        loop {
            let mut p = || -> [f64; 3] {
                std::array::from_fn(|_| if coarse { self.coarse() } else { self.grid() })
            };
            let t = Tri::new(p(), p(), p());
            // Reject exactly degenerate triangles.
            let n = cross(
                &sub(&rv(t.v[1]), &rv(t.v[0])),
                &sub(&rv(t.v[2]), &rv(t.v[0])),
            );
            if !n.iter().all(|c| c.is_zero()) {
                return t;
            }
        }
    }
}

// --------------------------------------------------------------------- tests

#[test]
fn tri_tri_matches_rational_reference() {
    let mut rng = Rng::new(0x71);
    let (mut disjoint, mut touching, mut segment, mut coplanar) = (0, 0, 0, 0);
    for i in 0..600 {
        let coarse = i % 2 == 1;
        let t0 = rng.tri(coarse);
        let t1 = rng.tri(coarse);
        let got = tri_tri_intersection(&t0, &t1);
        let want = reference(&t0, &t1);
        match (&got, &want) {
            (TriTriIsect::Disjoint, RefIsect::Empty) => disjoint += 1,
            (TriTriIsect::Coplanar, RefIsect::Coplanar) => coplanar += 1,
            (TriTriIsect::Touching(p), RefIsect::Point(q)) => {
                assert_eq!(&affine(p), q, "touch point mismatch: {t0:?} {t1:?}");
                touching += 1;
            }
            (TriTriIsect::Segment(p0, p1), RefIsect::Segment(q0, q1)) => {
                let (r0, r1) = (affine(p0), affine(p1));
                let fwd = &r0 == q0 && &r1 == q1;
                let rev = &r0 == q1 && &r1 == q0;
                assert!(fwd || rev, "segment endpoint mismatch: {t0:?} {t1:?}");
                segment += 1;
            }
            _ => panic!("variant mismatch: got {got:?}, want {want:?} for {t0:?} {t1:?}"),
        }
    }
    // Every regime must actually be exercised.
    assert!(
        disjoint > 20 && touching > 2 && segment > 10 && coplanar > 2,
        "case coverage too thin: disjoint={disjoint} touching={touching} \
         segment={segment} coplanar={coplanar}"
    );
}

#[test]
fn tri_tri_is_symmetric() {
    let mut rng = Rng::new(0x72);
    for i in 0..300 {
        let coarse = i % 2 == 1;
        let t0 = rng.tri(coarse);
        let t1 = rng.tri(coarse);
        let ab = tri_tri_intersection(&t0, &t1);
        let ba = tri_tri_intersection(&t1, &t0);
        match (&ab, &ba) {
            (TriTriIsect::Disjoint, TriTriIsect::Disjoint) => {}
            (TriTriIsect::Coplanar, TriTriIsect::Coplanar) => {}
            (TriTriIsect::Touching(p), TriTriIsect::Touching(q)) => {
                assert!(p.coincides(q));
            }
            (TriTriIsect::Segment(p0, p1), TriTriIsect::Segment(q0, q1)) => {
                let fwd = p0.coincides(q0) && p1.coincides(q1);
                let rev = p0.coincides(q1) && p1.coincides(q0);
                assert!(fwd || rev);
            }
            _ => panic!("asymmetric result: {ab:?} vs {ba:?} for {t0:?} {t1:?}"),
        }
    }
}

#[test]
fn tri_tri_segment_endpoints_lie_on_both_planes() {
    let mut rng = Rng::new(0x73);
    let mut segments = 0;
    for i in 0..300 {
        let coarse = i % 2 == 1;
        let t0 = rng.tri(coarse);
        let t1 = rng.tri(coarse);
        let endpoints = match tri_tri_intersection(&t0, &t1) {
            TriTriIsect::Segment(a, b) => vec![a, b],
            TriTriIsect::Touching(a) => vec![a],
            _ => continue,
        };
        segments += 1;
        for e in &endpoints {
            for t in [&t0, &t1] {
                assert_eq!(
                    orient3d(&t.point(0), &t.point(1), &t.point(2), e),
                    Some(Sign::Zero),
                    "endpoint must lie exactly on the plane of {t:?}"
                );
            }
        }
    }
    assert!(segments > 10, "expected many intersecting pairs, got {segments}");
}

#[test]
fn tri_tri_handcrafted_cases() {
    // Proper crossing: vertical triangle stabbing a horizontal one.
    let horizontal = Tri::new([-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]);
    let vertical = Tri::new([0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]);
    assert!(matches!(
        tri_tri_intersection(&horizontal, &vertical),
        TriTriIsect::Segment(..)
    ));

    // Vertex touching the interior of the other triangle.
    let poke = Tri::new([0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]);
    match tri_tri_intersection(&horizontal, &poke) {
        TriTriIsect::Touching(p) => {
            assert!(p.coincides(&Point3::explicit(0.0, 0.0, 0.0)));
        }
        other => panic!("expected Touching, got {other:?}"),
    }

    // Identical triangles are coplanar.
    assert_eq!(
        tri_tri_intersection(&horizontal, &horizontal),
        TriTriIsect::Coplanar
    );

    // Shared edge between two non-coplanar triangles: the shared edge itself.
    let upper = Tri::new([-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 0.0, 3.0]);
    match tri_tri_intersection(&horizontal, &upper) {
        TriTriIsect::Segment(a, b) => {
            let e0 = Point3::explicit(-2.0, -2.0, 0.0);
            let e1 = Point3::explicit(2.0, -2.0, 0.0);
            assert!(
                (a.coincides(&e0) && b.coincides(&e1))
                    || (a.coincides(&e1) && b.coincides(&e0))
            );
        }
        other => panic!("expected shared-edge Segment, got {other:?}"),
    }

    // Clearly separated.
    let far = Tri::new([10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [10.0, 11.0, 10.0]);
    assert_eq!(tri_tri_intersection(&horizontal, &far), TriTriIsect::Disjoint);
}

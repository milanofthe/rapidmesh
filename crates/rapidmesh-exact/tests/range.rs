//! Exactness beyond the f64 exponent range (#90): products of implicit
//! points of a high degree fall far below the subnormal range (tiny
//! geometry, coordinates like 1.5e-16) or above the largest float. The
//! predicates must give the same signs for every scaling of the input by a
//! power of two, which is exact and changes no sign.

use rapidmesh_exact::expansion::Expansion;
use rapidmesh_exact::orient::orient3d;
use rapidmesh_exact::point::Point3;
use rapidmesh_exact::Sign;
use rapidmesh_testutil::{expansion_to_rat, rat, Rng};

/// Products and sums of values around 2^k agree with the rational oracle.
#[test]
fn expansions_stay_exact_below_and_above_the_float_range() {
    let mut rng = Rng::new(0x90);
    for k in [-520, -300, 0, 300, 480] {
        let s = 2f64.powi(k);
        for _ in 0..200 {
            let v: Vec<f64> = (0..4).map(|_| rng.f64_wide() * s).collect();
            let e: Vec<Expansion> = v.iter().map(|&x| Expansion::from_f64(x)).collect();
            // (v0 v1 + v2) v3 - v0 v1 v3, exactly v2 v3.
            let got = e[0]
                .mul(&e[1])
                .add(&e[2])
                .mul(&e[3])
                .sub(&e[0].mul(&e[1]).mul(&e[3]));
            assert_eq!(expansion_to_rat(&got), rat(v[2]) * rat(v[3]), "scale 2^{k}");
            assert_eq!(got.sign(), e[2].mul(&e[3]).sign());
        }
    }
}

fn scaled(p: [f64; 3], s: f64) -> [f64; 3] {
    p.map(|x| x * s)
}

/// Three-plane points and their barycenters, one plane shared: the
/// barycenter lies exactly on it at every scale, and orientations against
/// other points keep their sign.
#[test]
fn implicit_points_keep_their_signs_under_power_of_two_scaling() {
    let mut rng = Rng::new(0x91);
    let mut checked = 0;
    for _ in 0..60 {
        let shared: [[f64; 3]; 3] = std::array::from_fn(|_| rng.point3(16));
        let others: Vec<[[f64; 3]; 3]> = (0..6)
            .map(|_| std::array::from_fn(|_| rng.point3(16)))
            .collect();
        let probe = rng.point3(16);
        let mut reference: Option<(Sign, Sign)> = None;
        for k in [0, -400, -700, 300] {
            let s = 2f64.powi(k);
            let sh = shared.map(|p| scaled(p, s));
            let tpis: Vec<Point3> = (0..3)
                .map(|i| {
                    Point3::tpi(
                        sh,
                        others[2 * i].map(|p| scaled(p, s)),
                        others[2 * i + 1].map(|p| scaled(p, s)),
                    )
                })
                .collect();
            if tpis.iter().any(|t| !t.is_valid()) {
                assert_eq!(k, 0, "validity changed with the scale");
                break;
            }
            let bary = Point3::bary(tpis[0].clone(), tpis[1].clone(), tpis[2].clone());
            let plane = sh.map(Point3::Explicit);
            let on = orient3d(&bary, &plane[0], &plane[1], &plane[2]);
            assert_eq!(on, Some(Sign::Zero), "barycenter off its plane at 2^{k}");
            let q = Point3::Explicit(scaled(probe, s));
            let side = orient3d(&tpis[0], &tpis[1], &tpis[2], &q).expect("decided");
            let tri = orient3d(&bary, &tpis[1], &plane[0], &q).expect("decided");
            match reference {
                None => reference = Some((side, tri)),
                Some(r) => assert_eq!(r, (side, tri), "sign changed at 2^{k}"),
            }
            if k != 0 {
                checked += 1;
            }
        }
    }
    assert!(checked > 60, "too few configurations checked: {checked}");
}

/// A constructed point far outside the float range of its homogeneous
/// coordinates still rounds to its coordinates (scaled back exactly).
#[test]
fn constructed_points_round_at_every_scale() {
    let mut rng = Rng::new(0x92);
    let mut checked = 0;
    for _ in 0..100 {
        let planes: [[[f64; 3]; 3]; 3] =
            std::array::from_fn(|_| std::array::from_fn(|_| rng.point3(16)));
        let unit = Point3::tpi(planes[0], planes[1], planes[2]);
        let Some(p) = unit.approx() else { continue };
        for k in [-400, 300] {
            let s = 2f64.powi(k);
            let t = Point3::tpi(
                planes[0].map(|q| scaled(q, s)),
                planes[1].map(|q| scaled(q, s)),
                planes[2].map(|q| scaled(q, s)),
            );
            assert_eq!(t.approx(), Some(p.map(|x| x * s)), "at 2^{k}");
        }
        checked += 1;
    }
    assert!(checked > 30);
}

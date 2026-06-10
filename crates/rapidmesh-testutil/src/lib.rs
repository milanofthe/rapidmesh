//! Shared test utilities: rational oracle ring, deterministic RNG, rational
//! vector helpers. Used as a dev-dependency only — never ships in any
//! production crate.

use num_rational::BigRational;
use num_traits::{Signed, Zero};
use rapidmesh_csg::{Constraint, FacetTriangulation, Tri};
use rapidmesh_exact::{collinear, orient2d, within_closed, Axis, Expansion, Point3, Ring, Sign};

/// BigRational newtype implementing [`Ring`]: the exact oracle that evaluates
/// the same generic geometric expressions as the filter and exact stages.
#[derive(Clone)]
pub struct Rat(pub BigRational);

impl Ring for Rat {
    fn from_f64(v: f64) -> Self {
        Rat(rat(v))
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

/// Exact rational value of a finite f64.
pub fn rat(v: f64) -> BigRational {
    BigRational::from_float(v).expect("finite f64")
}

/// Sign of a rational.
pub fn sign_of_rat(r: &BigRational) -> Sign {
    if r.is_zero() {
        Sign::Zero
    } else if r.is_positive() {
        Sign::Positive
    } else {
        Sign::Negative
    }
}

/// Exact rational value represented by an expansion.
pub fn expansion_to_rat(e: &Expansion) -> BigRational {
    e.components()
        .iter()
        .fold(BigRational::zero(), |acc, &c| acc + rat(c))
}

/// Exact rational affine coordinates of a valid (w != 0) point.
pub fn affine(p: &Point3) -> Rv {
    let h = p.hom::<Rat>();
    assert!(!h[3].0.is_zero(), "point must be valid");
    std::array::from_fn(|i| &h[i].0 / &h[3].0)
}

/// A rational 3-vector.
pub type Rv = [BigRational; 3];

/// Lift an f64 point to rationals.
pub fn rv(p: [f64; 3]) -> Rv {
    std::array::from_fn(|i| rat(p[i]))
}

/// Component-wise difference.
pub fn rv_sub(a: &Rv, b: &Rv) -> Rv {
    std::array::from_fn(|i| &a[i] - &b[i])
}

/// Cross product.
pub fn rv_cross(a: &Rv, b: &Rv) -> Rv {
    [
        &a[1] * &b[2] - &a[2] * &b[1],
        &a[2] * &b[0] - &a[0] * &b[2],
        &a[0] * &b[1] - &a[1] * &b[0],
    ]
}

/// Dot product.
pub fn rv_dot(a: &Rv, b: &Rv) -> BigRational {
    &a[0] * &b[0] + &a[1] * &b[1] + &a[2] * &b[2]
}

/// a + tau * (b - a).
pub fn rv_lerp(a: &Rv, b: &Rv, tau: &BigRational) -> Rv {
    std::array::from_fn(|i| &a[i] + tau * (&b[i] - &a[i]))
}

/// Deterministic xorshift RNG for reproducible randomized tests.
pub struct Rng(u64);

impl Rng {
    /// New RNG from a nonzero-coerced seed.
    pub fn new(seed: u64) -> Rng {
        Rng(seed.max(1))
    }

    /// Next raw value.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in [0, 1).
    pub fn f64_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Random f64 spanning many magnitudes (2^-40 .. 2^40), for stressing
    /// expansion arithmetic.
    pub fn f64_wide(&mut self) -> f64 {
        let mantissa = 2.0 * self.f64_unit() - 1.0;
        let exp = (self.next_u64() % 81) as i32 - 40;
        mantissa * (exp as f64).exp2()
    }

    /// Grid coordinate k/4 with k integer in [-half, half]: exact in f64 and
    /// prone to exact degeneracies.
    pub fn grid(&mut self, half: u64) -> f64 {
        ((self.next_u64() % (2 * half + 1)) as f64 - half as f64) / 4.0
    }

    /// Coarse coordinate in {-1, 0, 1}: forces exact degeneracies.
    pub fn coarse(&mut self) -> f64 {
        (self.next_u64() % 3) as f64 - 1.0
    }

    /// Grid point with coordinates from [`Rng::grid`].
    pub fn point3(&mut self, half: u64) -> [f64; 3] {
        std::array::from_fn(|_| self.grid(half))
    }

    /// Coarse point with coordinates from [`Rng::coarse`].
    pub fn point_coarse(&mut self) -> [f64; 3] {
        std::array::from_fn(|_| self.coarse())
    }
}

// ---------------------------------------------------- csg invariant checks

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
            "directed edge ({u},{v}) has no opposite: surface not closed"
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


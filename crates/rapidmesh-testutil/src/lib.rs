//! Shared test utilities: rational oracle ring, deterministic RNG, rational
//! vector helpers. Used as a dev-dependency only — never ships in any
//! production crate.

use num_rational::BigRational;
use num_traits::{Signed, Zero};
use rapidmesh_exact::{Expansion, Point3, Ring, Sign};

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

//! Shewchuk-style floating-point expansion arithmetic.
//!
//! An expansion represents a real number exactly as a sum of f64 components
//! that are nonoverlapping and ordered by increasing magnitude. All operations
//! here are exact (no rounding error in the represented value), which makes
//! sign computation exact.
//!
//! The component storage is `Vec`-backed for full generality (arbitrary-degree
//! polynomial expressions, e.g. homogeneous TPI coordinates of degree 7 inside
//! a 4x4 determinant). Allocation cost is acceptable because expansions only
//! run when the interval filter fails; if profiling ever shows otherwise, the
//! storage can move to an arena without changing the algorithms.

use crate::ring::Ring;
use crate::Sign;

/// 2^27 + 1, Shewchuk's splitter for exact two-product.
const SPLITTER: f64 = 134_217_729.0;

/// Exact sum of two f64: returns (approximate sum, roundoff term).
#[inline]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let b_virt = x - a;
    let a_virt = x - b_virt;
    let b_round = b - b_virt;
    let a_round = a - a_virt;
    (x, a_round + b_round)
}

/// Exact sum of two f64 with |a| >= |b|: returns (approximate sum, roundoff).
#[inline]
pub fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let b_virt = x - a;
    (x, b - b_virt)
}

/// Splits an f64 into high and low halves for exact multiplication.
#[inline]
fn split(a: f64) -> (f64, f64) {
    let c = SPLITTER * a;
    let a_big = c - a;
    let a_hi = c - a_big;
    (a_hi, a - a_hi)
}

/// Exact product of two f64: returns (approximate product, roundoff term).
#[inline]
pub fn two_product(a: f64, b: f64) -> (f64, f64) {
    let (b_hi, b_lo) = split(b);
    two_product_presplit(a, b, b_hi, b_lo)
}

/// Exact product where b is already split.
#[inline]
fn two_product_presplit(a: f64, b: f64, b_hi: f64, b_lo: f64) -> (f64, f64) {
    let x = a * b;
    let (a_hi, a_lo) = split(a);
    let err1 = x - a_hi * b_hi;
    let err2 = err1 - a_lo * b_hi;
    let err3 = err2 - a_hi * b_lo;
    (x, a_lo * b_lo - err3)
}

/// Sums two expansions into `h`, eliminating zero components.
///
/// Inputs must be nonoverlapping expansions sorted by increasing magnitude
/// (the invariant maintained by every routine in this module). The output
/// satisfies the same invariant and is never empty (a zero result is `[0.0]`).
fn fast_expansion_sum_zeroelim(e: &[f64], f: &[f64], h: &mut Vec<f64>) {
    h.clear();
    let (elen, flen) = (e.len(), f.len());
    let mut eindex = 0usize;
    let mut findex = 0usize;
    let mut enow = e[0];
    let mut fnow = f[0];
    // Pick the smaller-magnitude head as the initial accumulator.
    let mut q;
    if (fnow > enow) == (fnow > -enow) {
        q = enow;
        eindex += 1;
        if eindex < elen {
            enow = e[eindex];
        }
    } else {
        q = fnow;
        findex += 1;
        if findex < flen {
            fnow = f[findex];
        }
    }
    if eindex < elen && findex < flen {
        // First merge step may use fast_two_sum (new head dominates q).
        let (qnew, hh);
        if (fnow > enow) == (fnow > -enow) {
            let r = fast_two_sum(enow, q);
            qnew = r.0;
            hh = r.1;
            eindex += 1;
            if eindex < elen {
                enow = e[eindex];
            }
        } else {
            let r = fast_two_sum(fnow, q);
            qnew = r.0;
            hh = r.1;
            findex += 1;
            if findex < flen {
                fnow = f[findex];
            }
        }
        q = qnew;
        if hh != 0.0 {
            h.push(hh);
        }
        while eindex < elen && findex < flen {
            let (qnew, hh);
            if (fnow > enow) == (fnow > -enow) {
                let r = two_sum(q, enow);
                qnew = r.0;
                hh = r.1;
                eindex += 1;
                if eindex < elen {
                    enow = e[eindex];
                }
            } else {
                let r = two_sum(q, fnow);
                qnew = r.0;
                hh = r.1;
                findex += 1;
                if findex < flen {
                    fnow = f[findex];
                }
            }
            q = qnew;
            if hh != 0.0 {
                h.push(hh);
            }
        }
    }
    while eindex < elen {
        let r = two_sum(q, e[eindex]);
        q = r.0;
        if r.1 != 0.0 {
            h.push(r.1);
        }
        eindex += 1;
    }
    while findex < flen {
        let r = two_sum(q, f[findex]);
        q = r.0;
        if r.1 != 0.0 {
            h.push(r.1);
        }
        findex += 1;
    }
    if q != 0.0 || h.is_empty() {
        h.push(q);
    }
}

/// Multiplies expansion `e` by scalar `b` into `h`, eliminating zeros.
fn scale_expansion_zeroelim(e: &[f64], b: f64, h: &mut Vec<f64>) {
    h.clear();
    let (b_hi, b_lo) = split(b);
    let (mut q, hh) = two_product_presplit(e[0], b, b_hi, b_lo);
    if hh != 0.0 {
        h.push(hh);
    }
    for &ei in &e[1..] {
        let (p1, p0) = two_product_presplit(ei, b, b_hi, b_lo);
        let (sum, hh1) = two_sum(q, p0);
        if hh1 != 0.0 {
            h.push(hh1);
        }
        let (qn, hh2) = fast_two_sum(p1, sum);
        q = qn;
        if hh2 != 0.0 {
            h.push(hh2);
        }
    }
    if q != 0.0 || h.is_empty() {
        h.push(q);
    }
}

/// An exact real number as a nonoverlapping sum of f64 components.
///
/// Invariants: components are nonoverlapping, sorted by increasing magnitude,
/// the vector is never empty, and only a zero expansion contains a zero
/// component (exactly `[0.0]`).
#[derive(Debug, Clone, PartialEq)]
pub struct Expansion(Vec<f64>);

// add/sub/mul/neg intentionally mirror the Ring trait instead of std::ops:
// the generic geometric code calls Ring methods, and by-value std operators
// on a Vec-backed type would invite accidental clones.
#[allow(clippy::should_implement_trait)]
impl Expansion {
    /// The exact value `v`.
    pub fn from_f64(v: f64) -> Expansion {
        Expansion(vec![v])
    }

    /// The component slice (increasing magnitude).
    pub fn components(&self) -> &[f64] {
        &self.0
    }

    /// True if the represented value is exactly zero.
    pub fn is_zero(&self) -> bool {
        self.0.len() == 1 && self.0[0] == 0.0
    }

    /// Exact sign of the represented value (sign of the largest component).
    pub fn sign(&self) -> Sign {
        Sign::of_f64(*self.0.last().expect("expansion is never empty"))
    }

    /// Approximate f64 value (sum of components, smallest first).
    pub fn approx(&self) -> f64 {
        self.0.iter().sum()
    }

    /// Exact sum.
    pub fn add(&self, other: &Expansion) -> Expansion {
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }
        let mut h = Vec::with_capacity(self.0.len() + other.0.len());
        fast_expansion_sum_zeroelim(&self.0, &other.0, &mut h);
        Expansion(h)
    }

    /// Exact product with a scalar.
    pub fn scale(&self, b: f64) -> Expansion {
        if self.is_zero() || b == 0.0 {
            return Expansion::from_f64(0.0);
        }
        let mut h = Vec::with_capacity(2 * self.0.len());
        scale_expansion_zeroelim(&self.0, b, &mut h);
        Expansion(h)
    }

    /// Exact product (distributes `other`'s components over `self`).
    pub fn mul(&self, other: &Expansion) -> Expansion {
        if self.is_zero() || other.is_zero() {
            return Expansion::from_f64(0.0);
        }
        let mut acc = self.scale(other.0[0]);
        for &fi in &other.0[1..] {
            acc = acc.add(&self.scale(fi));
        }
        acc
    }

    /// Exact negation.
    pub fn neg(&self) -> Expansion {
        Expansion(self.0.iter().map(|c| -c).collect())
    }
}

/// Correctly rounded quotient of two exact expansions (`w` nonzero): the
/// f64 nearest to the exact value x/w, ties to even. Faithful float division
/// is NOT enough where geometry relies on coordinates landing exactly on
/// values they exactly equal (a constructed point on the plane z = 0.015
/// must approximate to 0.015 bit-exactly, or planarity shatters downstream).
pub fn div_round(x: &Expansion, w: &Expansion) -> f64 {
    debug_assert!(w.sign() != Sign::Zero, "division by zero expansion");
    // Newton refinement with exact residuals: q converges to within an ulp.
    let west = w.approx();
    let mut q = x.approx() / west;
    if !q.is_finite() {
        q = 0.0;
    }
    for _ in 0..3 {
        let r = x.add(&w.scale(q).neg());
        let dq = r.approx() / west;
        if dq == 0.0 {
            break;
        }
        let q2 = q + dq;
        if q2 == q || !q2.is_finite() {
            break;
        }
        q = q2;
    }
    // Exact placement: walk to the neighbor while the exact value lies
    // beyond the midpoint between q and that neighbor.
    let v_minus_q_sign = |q: f64| -> Sign {
        // sign(x - q w) * sign(w) = sign(v - q)
        let r = x.add(&w.scale(q).neg());
        match (r.sign(), w.sign()) {
            (Sign::Zero, _) => Sign::Zero,
            (a, b) if a == b => Sign::Positive,
            _ => Sign::Negative,
        }
    };
    let beyond_mid = |a: f64, b: f64| -> Sign {
        // sign(2v - (a + b)) relative: sign(2x - (a+b)w) * sign(w),
        // computed without forming a+b in f64.
        let m = x.add(x).add(&w.scale(a).neg()).add(&w.scale(b).neg());
        match (m.sign(), w.sign()) {
            (Sign::Zero, _) => Sign::Zero,
            (s, t) if s == t => Sign::Positive,
            _ => Sign::Negative,
        }
    };
    loop {
        match v_minus_q_sign(q) {
            Sign::Zero => return q + 0.0, // normalize -0.0 to +0.0
            Sign::Positive => {
                let n = q.next_up();
                match beyond_mid(q, n) {
                    Sign::Positive => q = n,
                    Sign::Negative => return q + 0.0,
                    Sign::Zero => return if (q.to_bits() & 1) == 0 { q + 0.0 } else { n + 0.0 },
                }
            }
            Sign::Negative => {
                let n = q.next_down();
                match beyond_mid(n, q) {
                    Sign::Negative => q = n,
                    Sign::Positive => return q + 0.0,
                    Sign::Zero => return if (q.to_bits() & 1) == 0 { q + 0.0 } else { n + 0.0 },
                }
            }
        }
    }
}

impl Ring for Expansion {
    fn from_f64(v: f64) -> Self {
        Expansion::from_f64(v)
    }
    fn add(&self, other: &Self) -> Self {
        Expansion::add(self, other)
    }
    fn sub(&self, other: &Self) -> Self {
        Expansion::add(self, &Expansion::neg(other))
    }
    fn mul(&self, other: &Self) -> Self {
        Expansion::mul(self, other)
    }
    fn neg(&self) -> Self {
        Expansion::neg(self)
    }
}

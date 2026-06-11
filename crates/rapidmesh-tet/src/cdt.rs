//! Segment recovery for the constrained Delaunay tetrahedrization.
//!
//! Stage 2 of the CDT pipeline (docs/cdt-recovery-plan.md): after the
//! Delaunay tetrahedrization of the PLC vertices, every PLC segment must be
//! present as a union of DT edges. Missing segments are split by Steiner
//! points placed ON the segment, following the protecting-sphere rules of
//! Si & Gaertner as adopted by Diazzi, Panozzo, Vaxman, Attene (SIGGRAPH
//! Asia 2023): a missing segment is encroached (its diametral sphere is
//! non-empty, since an empty diametral sphere makes it a Gabriel and hence
//! Delaunay edge); the encroaching point spanning the largest circumcircle
//! with the segment endpoints becomes the reference point, and the split
//! position depends on whether the original segment has acute endpoints:
//!
//! * no acute endpoint: protecting spheres around both sub-segment endpoints
//!   through the reference point; split where the smaller sphere cuts the
//!   segment, or at the midpoint when both radii exceed half the length;
//! * one acute endpoint w: the sphere around w through the reference point r
//!   cuts the carrier at p; split at p, or (if r is closer to p than the far
//!   endpoint is) shifted toward w by |r - p|, or at the midpoint of (w, p);
//!   sub-segments stay in this category and remember w;
//! * two acute endpoints: split at the midpoint; the halves each remember
//!   their acute endpoint and continue in the one-acute category.
//!
//! Steiner points are represented implicitly ([`Point3::Lnc`] on the
//! ORIGINAL carrier, with split parameters composed on the carrier), so
//! collinearity with the input segment is exact by construction and every
//! predicate involving them evaluates the implicit position.
//!
//! Input vertices that happen to lie ON a segment interior (f64 welding can
//! produce such T-junctions) violate Si's input model; they are adopted as
//! chain vertices instead of split positions, which would duplicate them.
//!
//! Implemented from the papers only (provenance: see crate docs).

use crate::delaunay::DelaunayBuilder;
use rapidmesh_exact::Point3;
use rustc_hash::FxHashSet;

/// Relative (to the carrier parameter range of the sub-segment) margin a
/// split parameter must keep from the sub-segment ends; rule results outside
/// the margin fall back to the sub-segment midpoint.
const SPLIT_MARGIN_REL: f64 = 1e-6;

/// Maximum distance from the carrier line, relative to the carrier length,
/// for an encroaching vertex to be adopted as an on-segment T-junction.
const ON_SEGMENT_REL: f64 = 1e-9;

/// Splits per original segment beyond which recovery declares divergence.
/// The protecting-sphere theory guarantees termination; this is a loud
/// backstop against implementation bugs, not a silent abandon.
const MAX_SPLITS_PER_SEGMENT: usize = 100_000;

/// One recovered input segment: the ordered vertex chain from the first to
/// the second endpoint. Every consecutive pair is a DT edge; every interior
/// vertex is either an exact on-carrier Steiner point or an adopted input
/// vertex that already sat on the segment.
#[derive(Debug)]
pub struct RecoveredSegment {
    /// Builder vertex indices along the segment, endpoints included.
    pub chain: Vec<usize>,
}

/// Acute-vertex classification: a vertex is acute when two incident segments
/// meet at an angle below 90 degrees (positive dot product of the outgoing
/// unit directions). Split-rule arithmetic is f64 throughout; only the DT
/// predicates need exactness.
pub fn acute_vertices(points: &[[f64; 3]], segments: &[(usize, usize)]) -> Vec<bool> {
    let mut dirs: Vec<Vec<[f64; 3]>> = vec![Vec::new(); points.len()];
    for &(a, b) in segments {
        let d = vsub(points[b], points[a]);
        let n = normalize(d);
        dirs[a].push(n);
        dirs[b].push(neg(n));
    }
    dirs.iter()
        .map(|ds| {
            (0..ds.len()).any(|x| (x + 1..ds.len()).any(|y| dot(ds[x], ds[y]) > 0.0))
        })
        .collect()
}

/// Splitting category of a sub-segment, determined by the ORIGINAL segment's
/// acute endpoints and inherited across splits.
#[derive(Debug, Clone, Copy)]
enum Category {
    /// No acute endpoint: protecting spheres around the sub-segment's own
    /// endpoints.
    Plain,
    /// One acute endpoint, remembered as its parameter on the carrier
    /// (0.0 or 1.0): the protecting sphere is always centered there.
    OneAcute { tw: f64 },
    /// Both endpoints acute: first split at the midpoint, halves become
    /// [`Category::OneAcute`].
    TwoAcute,
}

/// A sub-segment of an input segment, as builder vertices plus carrier
/// parameters (ta < tb on the original segment).
#[derive(Debug, Clone, Copy)]
struct SubSeg {
    va: usize,
    vb: usize,
    ta: f64,
    tb: f64,
    seg: usize,
    cat: Category,
}

/// What recovery does to one missing sub-segment.
enum Action {
    /// Insert an exact Steiner point at carrier parameter t.
    Split(f64),
    /// Adopt an existing vertex sitting on the segment at parameter t.
    Adopt(usize, f64),
}

/// Recovers every input segment as a union of DT edges by inserting implicit
/// Steiner points (and adopting on-segment T-junction vertices). `segments`
/// are pairs of builder vertex indices of EXPLICIT points; `acute` is the
/// per-vertex classification from [`acute_vertices`] over the same segment
/// set. Returns the per-segment chains in input order.
pub fn recover_segments(
    b: &mut DelaunayBuilder,
    segments: &[(usize, usize)],
    acute: &[bool],
) -> Vec<RecoveredSegment> {
    // Carriers: the original endpoint coordinates every Lnc refers to.
    let carriers: Vec<([f64; 3], [f64; 3])> = segments
        .iter()
        .map(|&(a, b_)| (b.approx_point(a), b.approx_point(b_)))
        .collect();

    let mut subs: Vec<SubSeg> = segments
        .iter()
        .enumerate()
        .map(|(k, &(va, vb))| SubSeg {
            va,
            vb,
            ta: 0.0,
            tb: 1.0,
            seg: k,
            cat: match (acute[va], acute[vb]) {
                (false, false) => Category::Plain,
                (true, false) => Category::OneAcute { tw: 0.0 },
                (false, true) => Category::OneAcute { tw: 1.0 },
                (true, true) => Category::TwoAcute,
            },
        })
        .collect();
    let mut splits_per_seg = vec![0usize; segments.len()];

    // Recovery rounds: an insertion for one segment can knock out an edge
    // confirmed for another, so iterate full passes until one is clean.
    loop {
        let mut any_missing = false;
        let mut idx = 0;
        while idx < subs.len() {
            let sub = subs[idx];
            if b.edge_exists(sub.va, sub.vb) {
                idx += 1;
                continue;
            }
            any_missing = true;
            splits_per_seg[sub.seg] += 1;
            assert!(
                splits_per_seg[sub.seg] <= MAX_SPLITS_PER_SEGMENT,
                "segment recovery diverged on segment {} ({:?})",
                sub.seg,
                segments[sub.seg],
            );
            let carrier = carriers[sub.seg];
            let (vm, tm) = match plan_split(b, &sub, carrier) {
                Action::Adopt(v, t) => (v, t),
                Action::Split(t) => {
                    let p = Point3::lnc(carrier.0, carrier.1, t);
                    (b.insert_exact(p), t)
                }
            };
            let (left_cat, right_cat) = child_categories(sub.cat);
            let right = SubSeg {
                va: vm,
                vb: sub.vb,
                ta: tm,
                tb: sub.tb,
                seg: sub.seg,
                cat: right_cat,
            };
            subs[idx] = SubSeg {
                vb: vm,
                tb: tm,
                cat: left_cat,
                ..sub
            };
            subs.push(right);
            // Re-check the shortened left half in place next iteration.
        }
        if !any_missing {
            break;
        }
    }

    // Stitch chains: order each segment's pieces by carrier parameter.
    let mut pieces: Vec<Vec<(f64, usize, usize)>> = vec![Vec::new(); segments.len()];
    for s in &subs {
        pieces[s.seg].push((s.ta, s.va, s.vb));
    }
    pieces
        .into_iter()
        .enumerate()
        .map(|(k, mut ps)| {
            ps.sort_by(|x, y| x.0.total_cmp(&y.0));
            let mut chain = vec![ps[0].1];
            for &(_, va, vb) in &ps {
                assert_eq!(va, *chain.last().expect("chain is non-empty"), "segment {k} pieces do not stitch");
                chain.push(vb);
            }
            RecoveredSegment { chain }
        })
        .collect()
}

fn child_categories(cat: Category) -> (Category, Category) {
    match cat {
        Category::Plain => (Category::Plain, Category::Plain),
        Category::OneAcute { tw } => (Category::OneAcute { tw }, Category::OneAcute { tw }),
        Category::TwoAcute => (
            Category::OneAcute { tw: 0.0 },
            Category::OneAcute { tw: 1.0 },
        ),
    }
}

/// Decides where a missing sub-segment is split (Si & Gaertner rules), or
/// which existing on-segment vertex to adopt.
fn plan_split(b: &DelaunayBuilder, sub: &SubSeg, carrier: ([f64; 3], [f64; 3])) -> Action {
    let (ca, cb) = carrier;
    let cl = dist(ca, cb);
    let pos = |t: f64| -> [f64; 3] { std::array::from_fn(|k| ca[k] + t * (cb[k] - ca[k])) };
    let xa = b.approx_point(sub.va);
    let xb = b.approx_point(sub.vb);
    let mid = 0.5 * (sub.ta + sub.tb);
    let margin = SPLIT_MARGIN_REL * (sub.tb - sub.ta);
    assert!(
        mid > sub.ta && mid < sub.tb,
        "sub-segment collapsed below f64 resolution at t = {}",
        sub.ta,
    );
    let clamp = |t: f64| -> f64 {
        if t > sub.ta + margin && t < sub.tb - margin {
            t
        } else {
            mid
        }
    };

    let Some(enc) = find_encroacher(b, sub, carrier) else {
        // Missing yet no encroacher found by the local sweep (the diametral
        // sphere is provably non-empty): conservative midpoint split.
        return Action::Split(mid);
    };
    if let Some((v, t)) = enc.adopt {
        return Action::Adopt(v, clamp_adopt(t, sub, margin, v));
    }
    let r = enc.pos;

    match sub.cat {
        Category::TwoAcute => Action::Split(mid),
        Category::Plain => {
            let r1 = dist(xa, r);
            let r2 = dist(xb, r);
            let e = dist(xa, xb);
            if r1 > 0.5 * e && r2 > 0.5 * e {
                Action::Split(mid)
            } else if r1 <= r2 {
                Action::Split(clamp(sub.ta + r1 / cl))
            } else {
                Action::Split(clamp(sub.tb - r2 / cl))
            }
        }
        Category::OneAcute { tw } => {
            // Protecting sphere around the original acute vertex w through
            // r cuts the carrier at p (on the side of the sub-segment).
            let xw = pos(tw);
            let dir = if tw <= sub.ta { 1.0 } else { -1.0 };
            let rw = dist(xw, r);
            let tp = tw + dir * rw / cl;
            if tp <= sub.ta + margin || tp >= sub.tb - margin {
                return Action::Split(mid);
            }
            let xp = pos(tp);
            let xo = pos(if dir > 0.0 { sub.tb } else { sub.ta });
            let d_rp = dist(r, xp);
            if d_rp < dist(xp, xo) {
                Action::Split(tp)
            } else if d_rp < 0.5 * rw {
                Action::Split(clamp(tp - dir * d_rp / cl))
            } else {
                Action::Split(clamp(0.5 * (tw + tp)))
            }
        }
    }
}

/// Adoption parameter hygiene: an adopted vertex must sit strictly inside the
/// sub-segment; the geometry guarantees it (it encroaches and lies on the
/// line), the assert keeps the failure loud if f64 says otherwise.
fn clamp_adopt(t: f64, sub: &SubSeg, margin: f64, v: usize) -> f64 {
    assert!(
        t > sub.ta + margin && t < sub.tb - margin,
        "adopted on-segment vertex {v} lands outside its sub-segment",
    );
    t
}

/// The chosen encroacher of a missing sub-segment.
struct Encroacher {
    /// Position of the reference point (max circumcircle).
    pos: [f64; 3],
    /// An on-carrier T-junction vertex to adopt instead, with its carrier
    /// parameter (takes precedence over splitting).
    adopt: Option<(usize, f64)>,
}

/// Sweeps the triangulation around a missing sub-segment for vertices
/// encroaching its diametral sphere: a BFS over tets from both endpoint
/// stars, expanded while a tet could still reach the sphere (conservative
/// f64 bound; the choice of reference point only steers split positions,
/// never correctness).
fn find_encroacher(
    b: &DelaunayBuilder,
    sub: &SubSeg,
    carrier: ([f64; 3], [f64; 3]),
) -> Option<Encroacher> {
    let xa = b.approx_point(sub.va);
    let xb = b.approx_point(sub.vb);
    let center: [f64; 3] = std::array::from_fn(|k| 0.5 * (xa[k] + xb[k]));
    let rad = 0.5 * dist(xa, xb);
    let (ca, cb) = carrier;
    let cdir = vsub(cb, ca);
    let cl2 = dot(cdir, cdir);

    let mut queue: Vec<u32> = Vec::new();
    let mut seen: FxHashSet<u32> = FxHashSet::default();
    for s in b
        .star_slots(sub.va)
        .into_iter()
        .chain(b.star_slots(sub.vb))
    {
        if seen.insert(s) {
            queue.push(s);
        }
    }

    let mut best: Option<([f64; 3], f64)> = None;
    let mut adopt: Option<(usize, f64, f64)> = None; // (vertex, t, line dist)
    let mut head = 0;
    while head < queue.len() {
        let slot = queue[head];
        head += 1;
        let verts = b.verts_of_slot(slot);
        let real: Vec<(usize, [f64; 3])> = verts
            .iter()
            .filter_map(|v| v.map(|i| (i, b.approx_point(i))))
            .collect();
        for &(v, p) in &real {
            if v == sub.va || v == sub.vb || dist(p, center) >= rad {
                continue;
            }
            // On-carrier T-junction: adopt instead of splitting next to it.
            let t = dot(vsub(p, ca), cdir) / cl2;
            let line_d = dist(p, std::array::from_fn(|k| ca[k] + t * cdir[k]));
            if line_d < ON_SEGMENT_REL * cl2.sqrt() && t > sub.ta && t < sub.tb {
                if adopt.is_none_or(|(_, _, d)| line_d < d) {
                    adopt = Some((v, t, line_d));
                }
                continue;
            }
            let r = circumradius(xa, xb, p);
            if best.is_none_or(|(_, br)| r > br) {
                best = Some((p, r));
            }
        }
        // Expand while the tet could still reach the diametral sphere.
        let near = real
            .iter()
            .map(|&(_, p)| dist(p, center))
            .fold(f64::MAX, f64::min);
        let span = real
            .iter()
            .flat_map(|&(_, p)| real.iter().map(move |&(_, q)| dist(p, q)))
            .fold(0.0_f64, f64::max);
        if real.is_empty() || near <= rad + span {
            for k in 0..4 {
                if let Some(nb) = b.neighbor_at(slot, k) {
                    if seen.insert(nb) {
                        queue.push(nb);
                    }
                }
            }
        }
    }
    if let Some((v, t, _)) = adopt {
        return Some(Encroacher {
            pos: b.approx_point(v),
            adopt: Some((v, t)),
        });
    }
    best.map(|(pos, _)| Encroacher { pos, adopt: None })
}

/// Circumradius of the triangle (a, b, c); f64, collinear inputs yield a
/// huge value (which is exactly the right preference for reference points).
fn circumradius(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let ab = dist(a, b);
    let ac = dist(a, c);
    let bc = dist(b, c);
    let cr = cross(vsub(b, a), vsub(c, a));
    let area2 = dot(cr, cr).sqrt();
    if area2 <= f64::MIN_POSITIVE {
        return f64::MAX;
    }
    ab * ac * bc / (2.0 * area2)
}

fn vsub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|k| a[k] - b[k])
}

fn neg(a: [f64; 3]) -> [f64; 3] {
    a.map(|x| -x)
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = vsub(a, b);
    dot(d, d).sqrt()
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let l = dot(a, a).sqrt();
    a.map(|x| x / l)
}

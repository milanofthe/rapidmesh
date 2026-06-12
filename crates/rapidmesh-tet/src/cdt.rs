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
//! Input vertices that lie EXACTLY ON a segment interior (a collinear
//! overlapping segment's chain Steiner point, or a T-junction made explicit by
//! the geometry stage's post-weld repair) violate Si's input model; they are
//! adopted as chain vertices instead of split positions, which would duplicate
//! them. Adoption is gated by an EXACT collinearity predicate (the candidate
//! must be exactly on the carrier line): a merely near-collinear vertex would
//! kink the chain in-plane while the facet region stays the exact straight
//! triangle, breaking face recovery's straddle-impossibility lemma. Approximate
//! (f64-welded) T-junctions are therefore not adopted here; they are resolved
//! upstream by the geometry stage's T-junction repair, which splits the
//! straddled facet so the corner becomes an exact vertex of both incident
//! triangles before the PLC ever reaches recovery.
//!
//! Implemented from the papers only (provenance: see crate docs).

use crate::delaunay::DelaunayBuilder;
use rapidmesh_exact::{Axis, Point3, Sign};
use rustc_hash::FxHashSet;
use std::sync::atomic::{AtomicU64, Ordering};

/// Diagnostic counters: face-recovery cavities retetrahedrized and tets
/// created by gift wrapping (test instrumentation, like the locate
/// counters in `delaunay`).
pub static FACE_CAVITIES: AtomicU64 = AtomicU64::new(0);
pub static WRAP_TETS: AtomicU64 = AtomicU64::new(0);

/// Relative (to the carrier parameter range of the sub-segment) margin a
/// split parameter must keep from the sub-segment ends; rule results outside
/// the margin fall back to the sub-segment midpoint.
const SPLIT_MARGIN_REL: f64 = 1e-6;

/// Splits per original segment beyond which recovery declares divergence.
/// The protecting-sphere theory guarantees termination; this is a loud
/// backstop against implementation bugs, not a silent abandon.
const MAX_SPLITS_PER_SEGMENT: usize = 100_000;

/// Gift-wrap steps per half-cavity beyond which the wrap declares
/// divergence (each step consumes positive cavity volume, so legitimate
/// wraps stay far below this).
const WRAP_MAX_STEPS: usize = 100_000;

/// The recovered state of all input segments: per segment the ordered chain
/// of vertices along the carrier. After a recovery pass every consecutive
/// pair is a DT edge; every interior vertex is either an exact on-carrier
/// Steiner point or an adopted input vertex that already sat on the segment.
/// The state is resumable: face recovery can knock out chain edges, and
/// [`resume_segments`] re-recovers with further splits without ever
/// re-deriving (and thereby duplicating) earlier Steiner points.
#[derive(Debug)]
pub struct SegmentChains {
    /// Per segment: (vertex, carrier parameter) ordered along the carrier;
    /// first/last are the original endpoints at t = 0 / t = 1.
    nodes: Vec<Vec<(usize, f64)>>,
    /// Per segment: splitting category per chain piece (nodes.len() - 1).
    cats: Vec<Vec<Category>>,
    /// Per segment: the original endpoint coordinates (the Lnc carrier).
    carriers: Vec<([f64; 3], [f64; 3])>,
    /// Per segment: total splits so far (divergence backstop).
    splits: Vec<usize>,
}

impl SegmentChains {
    /// Number of input segments.
    pub fn segment_count(&self) -> usize {
        self.nodes.len()
    }

    /// The ordered vertex chain of a segment, endpoints included.
    pub fn chain(&self, seg: usize) -> Vec<usize> {
        self.nodes[seg].iter().map(|&(v, _)| v).collect()
    }

    /// Splits the chain piece between the consecutive vertices `va` and `vb`
    /// of segment `seg` at its carrier-parameter midpoint, inserting an exact
    /// on-carrier Steiner point. The insert is UNGUARDED (like the recovery
    /// midpoint split): a refinement-era split may knock another constraint
    /// out of the DT, which the caller's next recovery pass re-establishes.
    /// Returns the new builder vertex, or `None` if `va`/`vb` are no longer a
    /// consecutive pair (the piece was already split). Mirrors the child
    /// category inheritance of [`resume_segments`].
    pub(crate) fn split_piece_mid(
        &mut self,
        b: &mut DelaunayBuilder,
        seg: usize,
        va: usize,
        vb: usize,
    ) -> Option<usize> {
        let i = (0..self.nodes[seg].len().saturating_sub(1)).find(|&i| {
            let x = self.nodes[seg][i].0;
            let y = self.nodes[seg][i + 1].0;
            (x == va && y == vb) || (x == vb && y == va)
        })?;
        let ta = self.nodes[seg][i].1;
        let tb = self.nodes[seg][i + 1].1;
        let tm = 0.5 * (ta + tb);
        let carrier = self.carriers[seg];
        // The midpoint lands next to va: start the locate walk there.
        b.walk_hint_vertex(va);
        let (vm, tm) = insert_chain_point(b, carrier, tm, ta.min(tb), ta.max(tb));
        let (left_cat, right_cat) = child_categories(self.cats[seg][i]);
        self.nodes[seg].insert(i + 1, (vm, tm));
        self.cats[seg][i] = left_cat;
        self.cats[seg].insert(i + 1, right_cat);
        self.splits[seg] += 1;
        Some(vm)
    }
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
/// set. Returns the per-segment chain state.
pub fn recover_segments(
    b: &mut DelaunayBuilder,
    segments: &[(usize, usize)],
    acute: &[bool],
) -> SegmentChains {
    let mut chains = SegmentChains {
        nodes: segments
            .iter()
            .map(|&(va, vb)| vec![(va, 0.0), (vb, 1.0)])
            .collect(),
        cats: segments
            .iter()
            .map(|&(va, vb)| {
                vec![match (acute[va], acute[vb]) {
                    (false, false) => Category::Plain,
                    (true, false) => Category::OneAcute { tw: 0.0 },
                    (false, true) => Category::OneAcute { tw: 1.0 },
                    (true, true) => Category::TwoAcute,
                }]
            })
            .collect(),
        carriers: segments
            .iter()
            .map(|&(a, b_)| (b.approx_point(a), b.approx_point(b_)))
            .collect(),
        splits: vec![0; segments.len()],
    };
    resume_segments(b, &mut chains);
    chains
}

/// Re-establishes every chain piece as a DT edge, splitting further where
/// needed (the resumable core of segment recovery; used standalone after
/// face-recovery surgery knocked out chain edges).
pub fn resume_segments(b: &mut DelaunayBuilder, chains: &mut SegmentChains) {
    // Recovery rounds: an insertion for one segment can knock out an edge
    // confirmed for another, so iterate full passes until one is clean.
    loop {
        let mut any_missing = false;
        for seg in 0..chains.nodes.len() {
            let mut i = 0;
            while i + 1 < chains.nodes[seg].len() {
                let (va, ta) = chains.nodes[seg][i];
                let (vb, tb) = chains.nodes[seg][i + 1];
                if b.edge_exists(va, vb) {
                    i += 1;
                    continue;
                }
                any_missing = true;
                chains.splits[seg] += 1;
                assert!(
                    chains.splits[seg] <= MAX_SPLITS_PER_SEGMENT,
                    "segment recovery diverged on segment {seg}",
                );
                let sub = SubSeg {
                    va,
                    vb,
                    ta,
                    tb,
                    cat: chains.cats[seg][i],
                };
                let carrier = chains.carriers[seg];
                let (vm, tm) = match plan_split(b, &sub, carrier) {
                    Action::Adopt(v, t) => (v, t),
                    Action::Split(t) => {
                        insert_chain_point(b, carrier, t, ta.min(tb), ta.max(tb))
                    }
                };
                let (left_cat, right_cat) = child_categories(sub.cat);
                chains.nodes[seg].insert(i + 1, (vm, tm));
                chains.cats[seg][i] = left_cat;
                chains.cats[seg].insert(i + 1, right_cat);
                // Re-check the shortened left piece in place.
            }
        }
        if !any_missing {
            break;
        }
    }
}

/// Inserts a chain Steiner point at parameter `t` on the carrier line. A
/// point that lands in the degenerate zone around an existing vertex (a
/// near-duplicate whose sliver star the cavity would swallow) is dodged by
/// shifting `t` in escalating steps RELATIVE TO THE PIECE, strictly inside
/// `(lo, hi)` — the swallow zone scales with the local geometry, not with
/// ulps. Every candidate stays exactly on the carrier, so the chain remains
/// exactly straight; the protection-ball categories tolerate the shift
/// (small steps are tried first, the large ones are a last resort before
/// the loud failure).
fn insert_chain_point(
    b: &mut DelaunayBuilder,
    carrier: ([f64; 3], [f64; 3]),
    t: f64,
    lo: f64,
    hi: f64,
) -> (usize, f64) {
    let span = hi - lo;
    let mut candidates = vec![t];
    for f in [1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.125, 0.2] {
        candidates.push(t + f * span);
        candidates.push(t - f * span);
    }
    let mut blockers: Vec<(f64, usize)> = Vec::new();
    for tc in candidates {
        if !(tc > lo && tc < hi) {
            continue;
        }
        match b.insert_exact_checked(Point3::lnc(carrier.0, carrier.1, tc)) {
            Ok(v) => return (v, tc),
            Err(w) => blockers.push((tc, w)),
        }
    }
    let detail: String = blockers
        .iter()
        .take(6)
        .map(|&(tc, w)| {
            format!(
                "\n  t {tc}: blocked by v{w} at {:?} ({})",
                b.approx_point(w),
                if b.exact_point(w).as_explicit().is_some() { "explicit" } else { "implicit" },
            )
        })
        .collect();
    panic!(
        "chain split at t = {t} in ({lo}, {hi}) on carrier {:?} -> {:?} blocked: \
         every dodged parameter would swallow an existing vertex star{detail}",
        carrier.0, carrier.1,
    );
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
    // Exact carrier line as explicit endpoints plus two off-line witnesses:
    // a point is exactly on the carrier iff it is coplanar with both
    // (ca, cb, witness) planes (same construction as the gate test).
    let (w1, w2) = line_witnesses(ca, cb);
    let (pca, pcb) = (Point3::Explicit(ca), Point3::Explicit(cb));
    let (pw1, pw2) = (Point3::Explicit(w1), Point3::Explicit(w2));

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
    let mut adopt: Option<(usize, f64)> = None; // (vertex, carrier parameter)
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
            // On-carrier T-junction: adopt instead of splitting next to it,
            // but ONLY when the vertex is EXACTLY on the carrier line (a near
            // miss must be split around, not adopted, or the chain kinks while
            // the facet stays straight). The f64 parameter places it on the
            // sub-segment; the exact orient3d pair confirms collinearity.
            let t = dot(vsub(p, ca), cdir) / cl2;
            if t > sub.ta && t < sub.tb {
                let pe = b.exact_point(v);
                let on_line = rapidmesh_exact::orient3d(&pca, &pcb, &pw1, &pe)
                    == Some(Sign::Zero)
                    && rapidmesh_exact::orient3d(&pca, &pcb, &pw2, &pe) == Some(Sign::Zero);
                if on_line {
                    // Deterministic pick: smallest vertex index among the
                    // exactly-collinear encroachers (the rest are handled on
                    // the resulting sub-pieces by the recovery loop).
                    if adopt.is_none_or(|(av, _)| v < av) {
                        adopt = Some((v, t));
                    }
                    continue;
                }
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
    if let Some((v, t)) = adopt {
        return Some(Encroacher {
            pos: b.approx_point(v),
            adopt: Some((v, t)),
        });
    }
    best.map(|(pos, _)| Encroacher { pos, adopt: None })
}

/// Two explicit points spanning independent planes through the carrier line
/// (ca, cb): a point is exactly on the line iff it is coplanar with each
/// (mirrors `line_witnesses` in tests/segment_recovery.rs).
fn line_witnesses(ca: [f64; 3], cb: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let d = vsub(cb, ca);
    let axis = if d[0].abs() <= d[1].abs() && d[0].abs() <= d[2].abs() {
        [1.0, 0.0, 0.0]
    } else if d[1].abs() <= d[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let u = cross(d, axis);
    let w = cross(d, u);
    (
        std::array::from_fn(|k| ca[k] + u[k]),
        std::array::from_fn(|k| ca[k] + w[k]),
    )
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

// --------------------------------------------------- face recovery (WP3)
//
// After segment recovery, every PLC facet must appear as a union of DT
// faces. A facet is missing when a DT edge pierces its interior. The tets
// incident to the piercing edges form cavities; each cavity is removed and
// retetrahedrized in two halves split by the facet plane, so that the new
// tets' faces tile the facet within the cavity. The halves are filled by
// gift wrapping (Shewchuk; the paper's fallback, used here as the primary
// path: it is unconditionally correct and our cavities are small), with
// cospherical ties broken by the paper's symbolic perturbation (Alg. 1,
// memory-order parity). The replacement is verified exactly by
// [`DelaunayBuilder::replace_cavity`]'s 3-chain argument, so any failure
// here is loud, never a silent corruption.
//
// Key structural facts the implementation relies on (provable from the
// simplicial-complex axioms plus recovered segments):
// * a chain/constraint edge never passes through a tet's interior, so a
//   cavity tet's plane cross-section lies wholly inside the facet region;
// * therefore no cavity-boundary face has vertices strictly on both sides
//   of the facet plane (asserted), and the plane split is well defined;
// * cavity tets never have a face with all three vertices on the plane.

/// One PLC facet for [`recover_faces`]: its three corner vertices and the
/// indices of its three boundary segments in the [`SegmentChains`].
#[derive(Debug, Clone, Copy)]
pub struct FacetRef {
    /// Corner vertex indices (builder indices of explicit PLC points).
    pub corners: [usize; 3],
    /// Segment indices of the facet's edges, in any order.
    pub edges: [usize; 3],
}

/// Rounds of the alternating segment/face recovery before declaring
/// divergence (each round is a full pass; tame inputs need 2-3).
const MAX_RECOVERY_ROUNDS: usize = 64;

/// Full constrained recovery: alternates segment and face recovery until
/// every segment chain piece is a DT edge and no facet is pierced.
pub fn recover_plc(
    b: &mut DelaunayBuilder,
    segments: &[(usize, usize)],
    facets: &[FacetRef],
    acute: &[bool],
) -> SegmentChains {
    let mut chains = recover_segments(b, segments, acute);
    let mut facet_clean: Vec<usize> = vec![0; facets.len()];
    for _round in 0..MAX_RECOVERY_ROUNDS {
        let any_face_missing = recover_faces(b, facets, &chains, &mut facet_clean);
        if !any_face_missing {
            return chains;
        }
        resume_segments(b, &mut chains);
    }
    panic!("face recovery did not converge in {MAX_RECOVERY_ROUNDS} rounds");
}

/// One face-recovery pass: detects and retetrahedrizes every pierced facet.
/// Returns true if any facet needed recovery (callers must then re-verify
/// segments and run another pass: cavity surgery for one facet can unmake
/// another facet or a chain edge).
pub fn recover_faces(
    b: &mut DelaunayBuilder,
    facets: &[FacetRef],
    chains: &SegmentChains,
    facet_clean: &mut [usize],
) -> bool {
    // Shared snapshot of the creation-log suffix from the oldest clean
    // position: (log position, tet bbox) per still-alive slot. Each facet's
    // incremental check then sweeps plain bboxes from its own position
    // instead of resolving slots through the builder again. Slots allocated
    // DURING this pass (cavity surgeries of earlier facets) are after the
    // snapshot; recover_one_facet checks that live tail via the builder.
    let from = facet_clean.iter().copied().min().unwrap_or(0);
    let log = b.creation_log();
    let snapshot_end = log.len();
    let mut news: Vec<(usize, [f64; 3], [f64; 3])> = Vec::new();
    for (pos, &s) in log.iter().enumerate().skip(from) {
        let Some(t) = b.tet_at(s) else { continue };
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for v in t {
            let p = b.approx_point(v);
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        news.push((pos, lo, hi));
    }
    let mut any = false;
    for (fi, f) in facets.iter().enumerate() {
        any |= recover_one_facet(b, f, chains, &mut facet_clean[fi], &news, snapshot_end);
    }
    any
}

/// Detects piercing edges of one facet and retetrahedrizes their cavities.
///
/// `clean_pos` is the facet's position in the builder's creation log at its
/// last clean sweep: if no tet created since then overlaps the facet's bbox,
/// the facet cannot have gained a piercing edge (every change to the
/// triangulation creates tets in the changed region) and the whole sweep is
/// skipped. This turns the repeated full-facet passes of the recovery /
/// refinement rounds from "every facet, every round" into "only facets near
/// actual changes".
fn recover_one_facet(
    b: &mut DelaunayBuilder,
    f: &FacetRef,
    chains: &SegmentChains,
    clean_pos: &mut usize,
    news: &[(usize, [f64; 3], [f64; 3])],
    snapshot_end: usize,
) -> bool {
    // Fixed sweep bbox from the facet's boundary vertices (their positions
    // never change during recovery).
    let mut bb_lo = [f64::MAX; 3];
    let mut bb_hi = [f64::MIN; 3];
    let mut seeds: Vec<usize> = f.corners.to_vec();
    for &e in &f.edges {
        seeds.extend(chains.chain(e));
    }
    for &v in &seeds {
        let p = b.approx_point(v);
        for k in 0..3 {
            bb_lo[k] = bb_lo[k].min(p[k]);
            bb_hi[k] = bb_hi[k].max(p[k]);
        }
    }
    let scale: f64 = (0..3).map(|k| bb_hi[k] - bb_lo[k]).fold(0.0, f64::max);
    let pad = 1e-7 * scale.max(1e-30);
    let overlaps = |b_: &DelaunayBuilder, slot: u32| -> bool {
        let Some(t) = b_.tet_at(slot) else { return false };
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for v in t {
            let p = b_.approx_point(v);
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        (0..3).all(|k| lo[k] <= bb_hi[k] + pad && hi[k] >= bb_lo[k] - pad)
    };

    // Incremental skip: nothing new near the facet since its last clean
    // sweep means nothing to recover (cheap f64 bbox tests only). The
    // shared snapshot covers the pass start; the live tail covers slots
    // allocated by earlier facets' surgeries within this pass.
    let bbox_hit = |lo: &[f64; 3], hi: &[f64; 3]| -> bool {
        (0..3).all(|k| lo[k] <= bb_hi[k] + pad && hi[k] >= bb_lo[k] - pad)
    };
    let start = news.partition_point(|&(pos, _, _)| pos < *clean_pos);
    let snap_clean = news[start..].iter().all(|(_, lo, hi)| !bbox_hit(lo, hi));
    let tail_from = (*clean_pos).max(snapshot_end);
    if snap_clean
        && b.creation_log()[tail_from..]
            .iter()
            .all(|&s| !overlaps(b, s))
    {
        *clean_pos = b.creation_log().len();
        return false;
    }

    // Unsplit facet that IS a DT face: a simplicial complex admits no edge
    // through one of its own faces, so nothing can pierce — skip the sweep.
    // This short-circuits the dominant first-pass case (a fine input surface
    // where most facets are already Delaunay).
    if f.edges.iter().all(|&e| chains.chain(e).len() == 2)
        && b.face_exists(f.corners[0], f.corners[1], f.corners[2])
    {
        *clean_pos = b.creation_log().len();
        return false;
    }

    // Precondition of the straddle-impossibility argument: the facet's OWN
    // boundary chains must currently be DT edges (the cross-section proof
    // rests on the boundary not passing through tet interiors). Surgery for
    // an earlier facet may have knocked a piece out; defer this facet to
    // the next pass (the caller alternates with resume_segments). The facet
    // stays dirty (clean_pos untouched).
    let boundary_intact = |b_: &DelaunayBuilder| -> bool {
        f.edges.iter().all(|&e| {
            chains
                .chain(e)
                .windows(2)
                .all(|w| b_.edge_exists(w[0], w[1]))
        })
    };
    if !boundary_intact(b) {
        return true; // work remains: chains first, then this facet
    }

    let [pa, pb, pc] = f.corners.map(|v| b.exact_point(v));

    // One cavity per detection sweep: a surgery invalidates the sweep
    // snapshot (slot reuse, new tets may pierce, components may merge), so
    // re-scan from scratch before every cavity.
    let mut any = false;
    loop {
        // Candidate sweep: BFS over real tets from the stars of the facet's
        // boundary vertices, bounded by a conservative bbox overlap (the
        // tets intersecting the convex facet region are face-connected and
        // all pass the bbox test, so none is missed).
        let mut seen: FxHashSet<u32> = FxHashSet::default();
        let mut queue: Vec<u32> = Vec::new();
        for &v in &seeds {
            for s in b.star_slots(v) {
                if b.tet_at(s).is_some() && seen.insert(s) {
                    queue.push(s);
                }
            }
        }
        let mut head = 0;
        while head < queue.len() {
            let s = queue[head];
            head += 1;
            for k in 0..4 {
                if let Some(nb) = b.neighbor_at(s, k) {
                    if !seen.contains(&nb) && overlaps(b, nb) {
                        seen.insert(nb);
                        queue.push(nb);
                    }
                }
            }
        }

        // Piercing tets: any of the 6 edges strictly crosses the facet.
        // The plane side of each VERTEX is cached (each vertex is shared by
        // many edges, and "both endpoints on one side" rejects almost every
        // edge with two lookups); the remaining straddling edges run the
        // three crossing tests once each (cached per undirected edge).
        let mut side_cache: rustc_hash::FxHashMap<usize, Sign> =
            rustc_hash::FxHashMap::default();
        let mut edge_cache: rustc_hash::FxHashMap<(usize, usize), bool> =
            rustc_hash::FxHashMap::default();
        let mut piercing: FxHashSet<u32> = FxHashSet::default();
        for &s in &queue {
            let Some(t) = b.tet_at(s) else { continue };
            'edges: for i in 0..4 {
                for j in i + 1..4 {
                    let key = (t[i].min(t[j]), t[i].max(t[j]));
                    let mut side = |v: usize| -> Sign {
                        *side_cache
                            .entry(v)
                            .or_insert_with(|| exact_orient(&pa, &pb, &pc, &b.exact_point(v)))
                    };
                    let (su, sv) = (side(key.0), side(key.1));
                    if su == Sign::Zero || sv == Sign::Zero || su == sv {
                        continue;
                    }
                    let hit = *edge_cache
                        .entry(key)
                        .or_insert_with(|| edge_crossing_inside(b, key.0, key.1, &pa, &pb, &pc));
                    if hit {
                        piercing.insert(s);
                        break 'edges;
                    }
                }
            }
        }
        let Some(&start) = piercing.iter().min() else {
            // Clean: remember the log position so the facet is skipped
            // until the triangulation changes near it again.
            *clean_pos = b.creation_log().len();
            return any;
        };

        // The face-connected component of the smallest piercing slot
        // (deterministic pick; the rest re-detects next iteration).
        let mut comp = vec![start];
        let mut in_comp: FxHashSet<u32> = FxHashSet::default();
        in_comp.insert(start);
        let mut h = 0;
        while h < comp.len() {
            let s = comp[h];
            h += 1;
            for k in 0..4 {
                if let Some(nb) = b.neighbor_at(s, k) {
                    if piercing.contains(&nb) && in_comp.insert(nb) {
                        comp.push(nb);
                    }
                }
            }
        }
        retet_cavity(b, &comp, &pa, &pb, &pc);
        any = true;

        // The surgery itself can knock out a piece of this facet's own
        // boundary; restore the precondition before scanning again.
        if !boundary_intact(b) {
            return true;
        }
    }
}

/// Crossing-inside half of the pierce test: given that the endpoints of the
/// open edge (u, v) lie STRICTLY on opposite sides of the facet plane (the
/// caller checks the sides), true iff the crossing is strictly inside the
/// facet triangle (a crossing exactly on the facet boundary would cross a
/// constrained chain edge, which a simplicial complex cannot; a Zero sign
/// therefore means the crossing is outside the facet).
fn edge_crossing_inside(
    b: &DelaunayBuilder,
    u: usize,
    v: usize,
    pa: &Point3,
    pb: &Point3,
    pc: &Point3,
) -> bool {
    let pu = b.exact_point(u);
    let pv = b.exact_point(v);
    let s1 = exact_orient(&pu, &pv, pa, pb);
    let s2 = exact_orient(&pu, &pv, pb, pc);
    let s3 = exact_orient(&pu, &pv, pc, pa);
    s1 != Sign::Zero && s1 == s2 && s2 == s3
}

fn exact_orient(a: &Point3, b: &Point3, c: &Point3, d: &Point3) -> Sign {
    rapidmesh_exact::orient3d(a, b, c, d).expect("DT vertices are valid points")
}

/// Removes one cavity (face-connected piercing tets) and refills it with
/// two gift-wrapped halves split by the facet plane.
fn retet_cavity(b: &mut DelaunayBuilder, comp: &[u32], pa: &Point3, pb: &Point3, pc: &Point3) {
    FACE_CAVITIES.fetch_add(1, Ordering::Relaxed);
    let comp_set: FxHashSet<u32> = comp.iter().copied().collect();

    // Vertices and their exact plane sides.
    let mut side: rustc_hash::FxHashMap<usize, Sign> = rustc_hash::FxHashMap::default();
    for &s in comp {
        let t = b.tet_at(s).expect("cavity tets are real");
        for v in t {
            side.entry(v)
                .or_insert_with(|| exact_orient(pa, pb, pc, &b.exact_point(v)));
        }
    }

    // Oriented boundary faces (wound with the cavity interior on the
    // positive side, the same convention replace_cavity validates against).
    let mut front1: Vec<[usize; 3]> = Vec::new();
    let mut front2: Vec<[usize; 3]> = Vec::new();
    for &s in comp {
        let t = b.tet_at(s).expect("cavity tets are real");
        for i in 0..4 {
            if let Some(nb) = b.neighbor_at(s, i) {
                if comp_set.contains(&nb) {
                    continue;
                }
            }
            let f = oriented_face(t, i);
            let has_pos = f.iter().any(|v| side[v] == Sign::Positive);
            let has_neg = f.iter().any(|v| side[v] == Sign::Negative);
            if has_pos && has_neg {
                // Diagnose before dying: is the outside neighbor secretly a
                // piercing tet the sweep failed to collect?
                let nb = b.neighbor_at(s, i);
                let nb_state = nb.map(|n| {
                    let nt = b.tet_at(n);
                    let pierces = |u: usize, v: usize| {
                        let su = exact_orient(pa, pb, pc, &b.exact_point(u));
                        let sv = exact_orient(pa, pb, pc, &b.exact_point(v));
                        su != Sign::Zero
                            && sv != Sign::Zero
                            && su != sv
                            && edge_crossing_inside(b, u, v, pa, pb, pc)
                    };
                    let nb_pierces = nt
                        .is_some_and(|t| {
                            (0..4).any(|x| (x + 1..4).any(|y| pierces(t[x], t[y])))
                        });
                    (n, nt, nb_pierces)
                });
                panic!(
                    "cavity boundary face straddles the facet plane: face {f:?} \
                     signs {:?} of tet {t:?}, outside neighbor {nb_state:?}",
                    f.map(|v| side[&v]),
                );
            }
            assert!(
                has_pos || has_neg,
                "cavity tet has a face entirely on the facet plane",
            );
            if has_pos {
                front1.push(f);
            } else {
                front2.push(f);
            }
        }
    }
    let v1: Vec<usize> = side
        .iter()
        .filter(|&(_, &s)| s != Sign::Negative)
        .map(|(&v, _)| v)
        .collect();
    let v2: Vec<usize> = side
        .iter()
        .filter(|&(_, &s)| s != Sign::Positive)
        .map(|(&v, _)| v)
        .collect();

    // Wrap the upper half; the on-plane faces it produces tile the facet
    // within the cavity and become part of the lower front, already wound
    // with the lower region on their positive side.
    let mut new_tets = Vec::new();
    let interface = gift_wrap(b, front1, &v1, [pa, pb, pc], true, &mut new_tets);
    let mut front2_full = front2;
    front2_full.extend(interface);
    let leftover = gift_wrap(b, front2_full, &v2, [pa, pb, pc], false, &mut new_tets);
    assert!(leftover.is_empty(), "non-skipping wrap returns no interface");

    b.replace_cavity(comp, &new_tets);
}

/// Face of a positively oriented tet (public indices) opposite vertex `i`,
/// wound with the opposite vertex on its positive side.
fn oriented_face(t: [usize; 4], i: usize) -> [usize; 3] {
    match i {
        0 => [t[1], t[3], t[2]],
        1 => [t[0], t[2], t[3]],
        2 => [t[0], t[3], t[1]],
        _ => [t[0], t[1], t[2]],
    }
}

/// Gift wrapping of one region: `front` are oriented triangles with the
/// unfilled region on their positive side; apexes come from `verts`.
/// With `skip_plane` (the FIRST half of a split cavity), faces landing
/// exactly on the facet plane are not wrapped; they are returned as the
/// interface for the second half, already wound with the other side
/// positive. The second half runs without skipping: its front contains the
/// interface tiles, which are on-plane and must be wrapped downward.
/// Appends the created tets, positively oriented, to `out`.
fn gift_wrap(
    b: &DelaunayBuilder,
    front: Vec<[usize; 3]>,
    verts: &[usize],
    plane: [&Point3; 3],
    skip_plane: bool,
    out: &mut Vec<[usize; 4]>,
) -> Vec<[usize; 3]> {
    // Active front: sorted key -> oriented face.
    let mut active: rustc_hash::FxHashMap<[usize; 3], [usize; 3]> =
        rustc_hash::FxHashMap::default();
    let mut queue: Vec<[usize; 3]> = Vec::new();
    let push = |active: &mut rustc_hash::FxHashMap<[usize; 3], [usize; 3]>,
                    queue: &mut Vec<[usize; 3]>,
                    f: [usize; 3]| {
        let mut key = f;
        key.sort_unstable();
        match active.entry(key) {
            std::collections::hash_map::Entry::Occupied(o) => {
                assert!(
                    is_reversed_pub(*o.get(), f),
                    "wrap fronts collide with equal orientation: {f:?}",
                );
                o.remove();
            }
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(f);
                queue.push(f);
            }
        }
    };
    for f in front {
        push(&mut active, &mut queue, f);
    }

    let on_plane = |b_: &DelaunayBuilder, v: usize| -> bool {
        exact_orient(plane[0], plane[1], plane[2], &b_.exact_point(v)) == Sign::Zero
    };

    let mut interface: Vec<[usize; 3]> = Vec::new();
    let mut steps = 0usize;
    while let Some(f) = queue.pop() {
        let mut key = f;
        key.sort_unstable();
        if active.get(&key) != Some(&f) {
            continue; // cancelled or superseded
        }
        if skip_plane && f.iter().all(|&v| on_plane(b, v)) {
            // Facet-plane tile: created faces of upper tets are already
            // wound with the lower region positive, exactly the winding the
            // lower front needs, so it transfers as-is.
            active.remove(&key);
            interface.push(f);
            continue;
        }
        steps += 1;
        assert!(steps <= WRAP_MAX_STEPS, "gift wrap diverged");

        let fp = f.map(|v| b.exact_point(v));
        // Candidates: strictly on the positive (unfilled) side of f.
        let mut best: Option<usize> = None;
        for &w in verts {
            if f.contains(&w) {
                continue;
            }
            if exact_orient(&fp[0], &fp[1], &fp[2], &b.exact_point(w)) != Sign::Positive {
                continue;
            }
            if crosses_front(b, f, w, &active) {
                continue;
            }
            best = Some(match best {
                None => w,
                Some(cur) => {
                    // Pencil order: w beats cur iff w is perturbed-inside
                    // the circumsphere of (f, cur).
                    if perturbed_insphere(b, [f[0], f[1], f[2], cur], w) == Sign::Positive {
                        w
                    } else {
                        cur
                    }
                }
            });
        }
        let w = best.unwrap_or_else(|| {
            panic!("gift wrap stuck: no valid apex for front face {f:?}")
        });

        active.remove(&key);
        let t = [f[0], f[1], f[2], w];
        WRAP_TETS.fetch_add(1, Ordering::Relaxed);
        out.push(t);
        // The three new faces, wound with the remaining unfilled region on
        // their positive side (the reverse of the tet-inward winding).
        for i in 0..3 {
            let g = oriented_face(t, i);
            push(&mut active, &mut queue, [g[0], g[2], g[1]]);
        }
    }
    interface
}

/// True if the candidate tet (f, w) improperly crosses the current front:
/// a front vertex strictly inside the open tet, a front edge passing
/// through the open tet, a tet edge strictly piercing an open front face,
/// or a coplanar open overlap between a tet face and a front face. Shared
/// vertices self-exclude from the strict tests (they sit on boundaries).
fn crosses_front(
    b: &DelaunayBuilder,
    f: [usize; 3],
    w: usize,
    active: &rustc_hash::FxHashMap<[usize; 3], [usize; 3]>,
) -> bool {
    let t = [f[0], f[1], f[2], w];
    let tp = t.map(|v| b.exact_point(v));
    // The four inward-wound faces of t (positive side = tet interior).
    let inward: [[usize; 3]; 4] = std::array::from_fn(|i| oriented_face(t, i));

    let strictly_inside = |p: &Point3| -> bool {
        inward.iter().all(|g| {
            let gp = g.map(|v| b.exact_point(v));
            exact_orient(&gp[0], &gp[1], &gp[2], p) == Sign::Positive
        })
    };

    for tau in active.values() {
        if tau.iter().all(|v| t.contains(v)) {
            continue; // the base face itself or a face of t already in front
        }
        // Front vertex strictly inside the candidate tet.
        for &v in tau {
            if !t.contains(&v) && strictly_inside(&b.exact_point(v)) {
                return true;
            }
        }
        let taup = tau.map(|v| b.exact_point(v));
        // Tet edge strictly piercing the open front triangle.
        for i in 0..4 {
            for j in i + 1..4 {
                if seg_pierces_tri(&tp[i], &tp[j], &taup[0], &taup[1], &taup[2]) {
                    return true;
                }
            }
        }
        // Front edge strictly piercing an open face of the tet.
        for e in 0..3 {
            let (u, v) = (&taup[e], &taup[(e + 1) % 3]);
            for g in &inward {
                let gp = g.map(|x| b.exact_point(x));
                if seg_pierces_tri(u, v, &gp[0], &gp[1], &gp[2]) {
                    return true;
                }
            }
        }
        // Coplanar open overlap of a tet face with the front triangle.
        for g in &inward {
            if coplanar_open_overlap(b, *g, *tau) {
                return true;
            }
        }
    }
    false
}

/// True if the open segment (u, v) strictly crosses the open triangle:
/// endpoints strictly on opposite plane sides, crossing strictly inside.
/// Shared vertices make one endpoint coplanar and the test false.
fn seg_pierces_tri(u: &Point3, v: &Point3, a: &Point3, b: &Point3, c: &Point3) -> bool {
    let su = exact_orient(a, b, c, u);
    let sv = exact_orient(a, b, c, v);
    if su == Sign::Zero || sv == Sign::Zero || su == sv {
        return false;
    }
    let s1 = exact_orient(u, v, a, b);
    let s2 = exact_orient(u, v, b, c);
    let s3 = exact_orient(u, v, c, a);
    s1 != Sign::Zero && s1 == s2 && s2 == s3
}

/// True if triangles g and tau are coplanar and their open interiors
/// overlap (exact 2D separating-axis test in a non-degenerate projection).
fn coplanar_open_overlap(b: &DelaunayBuilder, g: [usize; 3], tau: [usize; 3]) -> bool {
    let gp = g.map(|v| b.exact_point(v));
    let taup = tau.map(|v| b.exact_point(v));
    if taup
        .iter()
        .any(|p| exact_orient(&gp[0], &gp[1], &gp[2], p) != Sign::Zero)
    {
        return false;
    }
    // Pick a projection axis in which g is non-degenerate.
    let axis = [Axis::X, Axis::Y, Axis::Z]
        .into_iter()
        .find(|&ax| orient2(&gp[0], &gp[1], &gp[2], ax) != Sign::Zero)
        .expect("tet face is non-degenerate");
    let ccw = |tri: &[Point3; 3]| -> [usize; 3] {
        if orient2(&tri[0], &tri[1], &tri[2], axis) == Sign::Positive {
            [0, 1, 2]
        } else {
            [0, 2, 1]
        }
    };
    let go = ccw(&gp);
    let to = ccw(&taup);
    // Separated (or merely touching) iff some directed edge of either
    // triangle has the whole other triangle on its non-positive side.
    let separated = |tri: &[Point3; 3], ord: [usize; 3], other: &[Point3; 3]| -> bool {
        (0..3).any(|e| {
            let a = &tri[ord[e]];
            let bb = &tri[ord[(e + 1) % 3]];
            other
                .iter()
                .all(|p| orient2(a, bb, p, axis) != Sign::Positive)
        })
    };
    !separated(&gp, go, &taup) && !separated(&taup, to, &gp)
}

fn orient2(a: &Point3, b: &Point3, c: &Point3, axis: Axis) -> Sign {
    rapidmesh_exact::orient2d(a, b, c, axis).expect("valid points")
}

/// In-sphere with the paper's symbolic perturbation (Alg. 1, memory-order
/// parity): cospherical ties are broken deterministically by the global
/// vertex order, consistently across cavities. `t` must be positively
/// oriented; returns Positive iff `q` is (perturbed-)inside.
fn perturbed_insphere(b: &DelaunayBuilder, t: [usize; 4], q: usize) -> Sign {
    let p = |i: usize| b.exact_point(i);
    let r = rapidmesh_exact::insphere3d(&p(t[0]), &p(t[1]), &p(t[2]), &p(t[3]), &p(q))
        .expect("valid points");
    if r != Sign::Zero {
        return r;
    }
    let mut idx = [t[0], t[1], t[2], t[3], q];
    let mut swaps = 0usize;
    for i in 0..4 {
        for j in 0..4 - i {
            if idx[j] > idx[j + 1] {
                idx.swap(j, j + 1);
                swaps += 1;
            }
        }
    }
    let r = exact_orient(&p(idx[1]), &p(idx[2]), &p(idx[3]), &p(idx[4]));
    let r = if swaps.is_multiple_of(2) { r } else { r.flip() };
    if r != Sign::Zero {
        return r;
    }
    let r = exact_orient(&p(idx[0]), &p(idx[2]), &p(idx[3]), &p(idx[4]));
    let r = if swaps.is_multiple_of(2) { r.flip() } else { r };
    assert_ne!(r, Sign::Zero, "fully degenerate cospherical configuration");
    r
}

/// True if `g` is the reversal of `f` (public-index variant).
fn is_reversed_pub(f: [usize; 3], g: [usize; 3]) -> bool {
    let r = [f[0], f[2], f[1]];
    g == r || g == [r[1], r[2], r[0]] || g == [r[2], r[0], r[1]]
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

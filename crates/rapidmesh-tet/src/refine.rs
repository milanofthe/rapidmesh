//! Restricted-Delaunay refinement against analytic carrier oracles -- the
//! refinement-core volume path (see `docs/refinement-core.md`).
//!
//! Replaces the frozen-surface construction (pre-mesh the surface, freeze it,
//! hope the restricted Delaunay of an oversampled point set reproduces it) with
//! a guarantee: the boundary/interface facets ARE restricted-Delaunay facets of
//! the final tetrahedralization, refined until they capture each B-rep face's
//! topology and meet the size/shape criteria. Conformity holds by construction
//! -- a surface facet separates its two regions, so no tet straddles an
//! interface and multi-region assembly is the normal case, not a repair.
//!
//! Structure (Boissonnat-Oudot / Cheng-Dey-Ramos, as in CGAL Mesh_3):
//! - 0/1-features (B-rep corners + edges) are protected Ruppert-style: the edge
//!   curves (the exact circles / ellipses / POCS intersection curves) are
//!   sampled into segments whose diametral balls stay empty -- an encroaching
//!   candidate splits the segment ON the true curve instead of being inserted.
//! - 2-features (B-rep faces): a Delaunay facet is a SURFACE facet of face `F`
//!   when its Voronoi dual segment crosses `F`'s carrier inside the trimmed
//!   face; bad surface facets (size / shape / off-face vertices) insert the
//!   crossing point, which lies EXACTLY on the analytic carrier.
//! - 3-cells: tets inside a region refine by circumcenter insertion under a
//!   radius-edge and size bound; a circumcenter that lands in another region or
//!   encroaches a feature refines the boundary instead (facet / segment first).
//!
//! Steiner points are explicit f64 (projected circumcenters / curve points), so
//! the implicit-predicate density that throttled the old CDT path (issue #2)
//! never arises.

use crate::brep_mesh::edge_curve;
use crate::conform::{MeshParams, SurfaceFace, TetMesh};
use crate::curve::distribute_floored;
use crate::delaunay::DelaunayBuilder;
use crate::domain::DomainTree;
use crate::facetbvh::FacetBvh;
use rapidmesh_brep::{Brep, Surface};
use rapidmesh_csg::Tri;
use rapidmesh_geom::vec3::{cross, dist, dot, scale, sub, V3};
use rapidmesh_geom::{FaceTag, RegionTag, TaggedPlc};
use smallvec::SmallVec;
use std::collections::VecDeque;
use std::hash::BuildHasherDefault;

type DMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<rustc_hash::FxHasher>>;

/// Facet-shape bound: minimum surface-facet angle (degrees). Deliberately mild:
/// on curved carriers the first-crossing surface-ball center is not the exact
/// Chew circumcenter, so an aggressive bound churns instead of terminating; the
/// optimizer's surface pass owns the final facet quality.
const FACET_ANGLE_DEG: f64 = 15.0;
/// Surface-ball size factor: a surface facet refines when its Delaunay-ball
/// radius exceeds `FACET_SIZE * h(x)`. Edges can reach ~2x the ball radius, so
/// 0.7 keeps the gmsh `maxh = max edge length` contract (~1.4 h worst case).
const FACET_SIZE: f64 = 0.7;
/// Cell size factor: a tet refines when its circumradius exceeds
/// `CELL_SIZE * h(cc)` (`r ~ 0.61 h` for a regular tet of edge `h`).
const CELL_SIZE: f64 = 0.75;
/// Segment-length floor as a fraction of the local size: a shorter segment is
/// never split, the encroachment is accepted instead (the Ruppert small-angle /
/// tight-curve-turn ping-pong guard: adjacent segments of one sharply turning
/// curve otherwise split each other down to nothing).
const SEG_FLOOR: f64 = 0.25;
/// Duplicate guard: candidates closer than this fraction of the local size to
/// an existing vertex are dropped.
const DUP_FRAC: f64 = 0.05;

/// Local corner indices of the tet facet opposite corner `i`, matching the
/// positive orientation convention of `delaunay::face`.
const FACET_LOCAL: [[usize; 3]; 4] = [[1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]];

/// Where a mesh vertex lives on the geometry (its refinement provenance).
#[derive(Clone, Debug, PartialEq)]
enum Prov {
    /// B-rep corner (on all faces incident to its edges).
    Corner,
    /// Sample on B-rep edge `e` (on its analytic curve).
    Edge(u32),
    /// Sample on B-rep face `f` (on its analytic carrier).
    Face(u32),
    /// Volume Steiner point.
    Interior,
}

/// One protected feature segment of B-rep edge `ei` between the curve samples
/// at vertices `va`/`vb`, spanning arc-length `[sa, sb]` on its analytic curve.
/// `va` corresponds to `sa` and `vb` to `sb` -- the CURVE order. (The map key is
/// index-sorted; without the ordered endpoints here, a split would attach the
/// half-intervals to the wrong ends and place its point outside the segment.)
#[derive(Clone, Copy, Debug)]
struct Seg {
    ei: u32,
    va: usize,
    vb: usize,
    sa: f64,
    sb: f64,
}

fn key2(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

fn mid3(a: V3, b: V3) -> V3 {
    std::array::from_fn(|k| 0.5 * (a[k] + b[k]))
}

/// Circumcenter and circumradius of a tet, `None` if degenerate.
fn tet_circumcenter(p: [V3; 4]) -> Option<(V3, f64)> {
    let row = |i: usize| -> V3 { std::array::from_fn(|k| 2.0 * (p[i][k] - p[0][k])) };
    let sq = |q: V3| -> f64 { q.iter().map(|x| x * x).sum() };
    let (r1, r2, r3) = (row(1), row(2), row(3));
    let b = [sq(p[1]) - sq(p[0]), sq(p[2]) - sq(p[0]), sq(p[3]) - sq(p[0])];
    let det3 = |a: V3, b: V3, c: V3| -> f64 {
        a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
    };
    let d = det3(r1, r2, r3);
    let scale_m: f64 = [r1, r2, r3]
        .iter()
        .map(|r| r.iter().map(|x| x.abs()).fold(0.0, f64::max))
        .fold(0.0, f64::max);
    if d.abs() < 1e-12 * scale_m.powi(3) {
        return None;
    }
    let col = |j: usize| -> f64 {
        let mut m = [r1, r2, r3];
        for (i, row) in m.iter_mut().enumerate() {
            row[j] = b[i];
        }
        det3(m[0], m[1], m[2]) / d
    };
    let c = [col(0), col(1), col(2)];
    let r = dist(c, p[0]);
    Some((c, r))
}

/// Smallest interior angle (degrees) of a triangle.
fn tri_min_angle(a: V3, b: V3, c: V3) -> f64 {
    let ang = |u: V3, v: V3, w: V3| {
        let (e1, e2) = (sub(v, u), sub(w, u));
        let d = dot(e1, e2) / ((dot(e1, e1) * dot(e2, e2)).sqrt() + 1e-300);
        d.clamp(-1.0, 1.0).acos().to_degrees()
    };
    ang(a, b, c).min(ang(b, c, a)).min(ang(c, a, b))
}

/// Signed offset of `p` from a surface: distance along the carrier normal at
/// the footpoint. Generic over every carrier via `project_uv`/`eval_uv`/
/// `normal`; the sign flips exactly at the surface, and near the surface the
/// magnitude approximates the Euclidean distance (the sphere-tracing bound).
fn signed_offset(surf: &Surface, p: V3) -> f64 {
    let (foot, n) = surf.closest(p);
    dot(sub(p, foot), n)
}

/// ALL crossings of segment `a -> b` with a carrier, as parameters `t` in
/// `(0, 1)` ascending -- TRANSVERSAL crossings only (tangential grazes are
/// skipped, matching the sphere-traced fallback's sign-change semantics).
/// Closed-form carriers solve exactly: plane linear, sphere/cylinder
/// quadratic, tube piecewise-quadratic candidates + sign reconstruction.
/// Everything else (cone, torus, extruded, NURBS, discrete) sphere-traces.
/// The analytic paths replaced ~90M oracle projections per coil mesh.
fn carrier_crossings(surf: &Surface, a: V3, b: V3) -> SmallVec<[f64; 4]> {
    if !(dist(a, b) > 0.0) {
        return SmallVec::new();
    }
    let d = sub(b, a);
    match surf {
        // Plane stays on the TRACER for now: the analytic root matches the
        // traced crossing set exactly (verified via a diff harness on the
        // exact-volume scenes) but shifts each t by ~1e-9 -- and the
        // classification walks turned out to be knife-edge sensitive to
        // that at coplanar sheet rims (air_dielectric volume bias 2e-4).
        // Tracing a plane is cheap (linear field, a handful of O(1)
        // projections); the expensive carriers below are analytic.
        // RAPIDMESH_ANALYTIC_PLANE=1 switches to the analytic root -- the
        // sensitivity's diagnosis harness (and the switch to flip once the
        // walks are robust to the ~1e-9 shift).
        Surface::Plane { .. } => {
            static ANALYTIC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if !*ANALYTIC
                .get_or_init(|| std::env::var_os("RAPIDMESH_ANALYTIC_PLANE").is_some())
            {
                return trace_crossings(surf, a, b);
            }
            // Asymmetric endpoint semantics, matching the march's
            // arithmetic: start sampled once (exact-0 drop), a crossing
            // within float noise of the far end is dropped (the march's
            // Zeno samples round to 0.0 there).
            let (s0, s1) = (signed_offset(surf, a), signed_offset(surf, b));
            let mut out = SmallVec::new();
            if s0 != 0.0
                && s1.abs() > endpoint_noise(a, b)
                && s0.signum() != s1.signum()
            {
                let t = s0 / (s0 - s1);
                if t > 0.0 && t < 1.0 {
                    out.push(t);
                }
            }
            out
        }
        Surface::Sphere { center, radius, .. } => {
            let w = sub(a, *center);
            let mut out = SmallVec::new();
            push_quad_roots_01(dot(d, d), dot(w, d), dot(w, w) - radius * radius, &mut out);
            drop_endpoint_touch(surf, a, b, &mut out);
            out
        }
        Surface::Cylinder { center, axis, radius, .. } => {
            // Components orthogonal to the (unit) axis; the barrel is the
            // zero set of |perp|^2 - r^2, quadratic along the segment.
            let reject = |v: V3| -> V3 {
                let s = dot(v, *axis);
                std::array::from_fn(|k| v[k] - s * axis[k])
            };
            let (u0, u1) = (reject(sub(a, *center)), reject(d));
            let mut out = SmallVec::new();
            push_quad_roots_01(dot(u1, u1), dot(u0, u1), dot(u0, u0) - radius * radius, &mut out);
            drop_endpoint_touch(surf, a, b, &mut out);
            out
        }
        Surface::Tube { path, radius } => tube_crossings(path, *radius, a, b),
        _ => trace_crossings(surf, a, b),
    }
}

/// Absolute offset scale below which a segment endpoint counts as ON the
/// carrier (see the plane arm's comment): a few ulps of the coordinate
/// magnitude, the noise floor of the projected-offset arithmetic.
fn endpoint_noise(a: V3, b: V3) -> f64 {
    let coord = a
        .iter()
        .chain(b.iter())
        .fold(0.0f64, |m, &v| m.max(v.abs()));
    64.0 * f64::EPSILON * coord
}

/// Drops analytic roots that are an ENDPOINT TOUCH: the endpoint's projected
/// offset is at float-noise scale (on the carrier by the tracer's
/// arithmetic), so its adjacent root at t = O(eps) or 1 - O(eps) is the
/// endpoint itself, which the traced semantics never reported as a crossing.
/// Lazy: offsets are only evaluated when a root sits within 1e-9 of that
/// endpoint.
fn drop_endpoint_touch(surf: &Surface, a: V3, b: V3, out: &mut SmallVec<[f64; 4]>) {
    // Same asymmetry as the plane arm: exact-zero start, noise-level end.
    if out.first().is_some_and(|&t| t < 1e-9) && signed_offset(surf, a) == 0.0 {
        out.remove(0);
    }
    if out.last().is_some_and(|&t| t > 1.0 - 1e-9)
        && signed_offset(surf, b).abs() <= endpoint_noise(a, b)
    {
        out.pop();
    }
}

/// Roots of `A t^2 + 2 B t + C = 0` inside the OPEN interval `(0, 1)`,
/// pushed ascending. A tangent root (zero discriminant) is skipped: it is a
/// graze, not a transversal crossing.
fn push_quad_roots_01(aa: f64, bb: f64, cc: f64, out: &mut SmallVec<[f64; 4]>) {
    if aa == 0.0 {
        if bb != 0.0 {
            let t = -cc / (2.0 * bb);
            if t > 0.0 && t < 1.0 {
                out.push(t);
            }
        }
        return;
    }
    let disc = bb * bb - aa * cc;
    if disc <= 0.0 {
        return;
    }
    let s = disc.sqrt();
    // Citardauq pairing avoids cancellation when |bb| ~ s.
    let q = -(bb + bb.signum() * s);
    let (t1, t2) = (q / aa, cc / q);
    let (lo, hi) = (t1.min(t2), t1.max(t2));
    if lo > 0.0 && lo < 1.0 {
        out.push(lo);
    }
    if hi > 0.0 && hi < 1.0 && hi != lo {
        out.push(hi);
    }
}

/// Analytic tube crossings: every crossing's closest path point lies either
/// in a segment interior (perpendicular-distance quadratic) or at a node
/// (sphere quadratic), so the roots of those pieces over the nearby segments
/// are a candidate superset. The true offset's SIGN pattern over the
/// candidate partition then keeps exactly the transversal crossings --
/// spurious candidates (roots of a piece that another piece undercuts) fall
/// out because the sign does not flip there.
fn tube_crossings(
    path: &rapidmesh_geom::TubePath,
    radius: f64,
    a: V3,
    b: V3,
) -> SmallVec<[f64; 4]> {
    let d = sub(b, a);
    let mut cand: SmallVec<[f64; 4]> = SmallVec::new();
    path.for_segments_near(a, b, radius, &mut |s| {
        let (q0, q1) = (path.pts[s], path.pts[s + 1]);
        let e = sub(q1, q0);
        let ee = dot(e, e);
        if ee > 0.0 {
            let reject = |v: V3| -> V3 {
                let t = dot(v, e) / ee;
                std::array::from_fn(|k| v[k] - t * e[k])
            };
            let (u0, u1) = (reject(sub(a, q0)), reject(d));
            push_quad_roots_01(dot(u1, u1), dot(u0, u1), dot(u0, u0) - radius * radius, &mut cand);
        }
        for q in [q0, q1] {
            let w = sub(a, q);
            push_quad_roots_01(dot(d, d), dot(w, d), dot(w, w) - radius * radius, &mut cand);
        }
    });
    if cand.is_empty() {
        return SmallVec::new();
    }
    cand.sort_unstable_by(|x, y| x.total_cmp(y));
    cand.dedup_by(|x, y| (*x - *y).abs() < 1e-12);
    let at = |t: f64| -> V3 { std::array::from_fn(|k| a[k] + t * (b[k] - a[k])) };
    let f = |t: f64| -> f64 {
        let p = at(t);
        dist(p, path.closest(p)) - radius
    };
    let mut out = SmallVec::new();
    let mut s_left = f(0.0);
    for (i, &t) in cand.iter().enumerate() {
        let t_right = if i + 1 < cand.len() { 0.5 * (t + cand[i + 1]) } else { 0.5 * (t + 1.0) };
        let s_right = f(t_right);
        if s_left != 0.0 && s_right != 0.0 && s_left.signum() != s_right.signum() {
            out.push(t);
        }
        s_left = s_right;
    }
    out
}

/// Sphere-traced crossings (the generic fallback): the step never exceeds
/// the current distance to the carrier, so a thin feature (a torus tube, a
/// plate) crossed twice between two same-sign endpoints is not skipped --
/// the failure mode of plain endpoint-sign bisection. Each sign change is
/// then bisected tight.
fn trace_crossings(surf: &Surface, a: V3, b: V3) -> SmallVec<[f64; 4]> {
    let len = dist(a, b);
    if !(len > 0.0) {
        return SmallVec::new();
    }
    let at = |t: f64| -> V3 { std::array::from_fn(|k| a[k] + t * (b[k] - a[k])) };
    let min_step = 1e-4;
    let mut out = SmallVec::new();
    let mut t = 0.0f64;
    let mut s_prev = signed_offset(surf, a);
    let mut guard = 0;
    let mut evals = 1u64;
    while t < 1.0 {
        guard += 1;
        if guard > 4096 {
            CC_GUARD_TRIPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            break;
        }
        // |offset| lower-bounds the true distance (the offset of a 1-Lipschitz
        // signed distance-like field), so stepping 0.9 of it cannot skip a
        // crossing.
        let step = ((s_prev.abs() * 0.9) / len).max(min_step);
        let t2 = (t + step).min(1.0);
        let s2 = signed_offset(surf, at(t2));
        evals += 1;
        if s_prev != 0.0 && s2 != 0.0 && s_prev.signum() != s2.signum() {
            // 14 halvings resolve the bracket to ~6e-5 of the segment; both
            // consumers then pull the point exactly onto the carrier via
            // `closest` before using it, so tighter bisection buys nothing
            // (it was 40 rounds -- a third of ALL oracle evaluations).
            let (mut lo, mut hi, lo_neg) = (t, t2, s_prev < 0.0);
            for _ in 0..14 {
                let m = 0.5 * (lo + hi);
                if (signed_offset(surf, at(m)) < 0.0) == lo_neg {
                    lo = m;
                } else {
                    hi = m;
                }
            }
            evals += 14;
            out.push(0.5 * (lo + hi));
        }
        t = t2;
        s_prev = s2;
    }
    CC_EVALS.fetch_add(evals, std::sync::atomic::Ordering::Relaxed);
    out
}

/// Sphere-tracing telemetry (RAPIDMESH_REFINE_TRACE): total offset
/// evaluations and 4096-step-guard trips of [`carrier_crossings`]. A guard
/// trip means a segment grazed a carrier for its whole length (offset ~ 0
/// everywhere, so the safe step degenerates to `min_step`). Atomics, not
/// thread-locals: the wall stage evaluates across rayon workers.
static CC_EVALS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CC_GUARD_TRIPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Parameter range of segment `a -> b` inside the (slightly inflated) ball
/// `(c, r)`, or `None` when the segment misses it entirely.
fn clip_segment_to_ball(a: V3, b: V3, c: V3, r: f64) -> Option<(f64, f64)> {
    let d = sub(b, a);
    let m = sub(a, c);
    let aa = dot(d, d);
    if !(aa > 0.0) {
        return Some((0.0, 1.0));
    }
    let r = r * 1.0001;
    let bb = dot(m, d);
    let cc = dot(m, m) - r * r;
    let disc = bb * bb - aa * cc;
    if disc <= 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let (t0, t1) = (((-bb - s) / aa).max(0.0), ((-bb + s) / aa).min(1.0));
    if t0 >= t1 {
        return None;
    }
    Some((t0, t1))
}

/// The refinement state: Delaunay + provenance + feature segments + oracles.
struct Refiner<'a> {
    brep: &'a Brep,
    domain: &'a DomainTree,
    plc: &'a TaggedPlc,
    /// Chord-sagitta band per PLC facet (0 for planar): how far the analytic
    /// carrier can deviate from the facet. Region queries inside the global
    /// band walk the ANALYTIC surfaces; outside it the faceted classifier is
    /// exact and cheap. One consistent authority everywhere.
    facet_band: Vec<f64>,
    band: f64,
    db: DelaunayBuilder,
    /// Provenance per public DT vertex.
    prov: Vec<Prov>,
    /// Protecting-ball radius per vertex (0 = unprotected). Feature points
    /// (corners, curve samples) carry a ball; no facet/cell candidate is
    /// inserted inside it -- the Cheng-Dey-Ramos protection that keeps surface
    /// and volume refinement from crowding the 0/1-features (the crowding is
    /// what breeds the unremovable slivers along intersection curves). Balls
    /// only ever SHRINK (when a nearer feature point appears), so protection
    /// never blocks legitimate refinement of the features themselves.
    ball: Vec<f64>,
    /// Spatial hash of protected vertices (ball AABB cells -> vertex).
    ball_grid: DMap<[i64; 3], Vec<usize>>,
    /// Faces incident to each vertex (restricted-facet candidates).
    vfaces: Vec<Vec<u32>>,
    /// Live feature segments, keyed by their (sorted) endpoint vertex pair.
    segs: DMap<(usize, usize), Seg>,
    /// Spatial hash of segments (diametral-ball AABB cells -> segment keys;
    /// entries validated against `segs` on query, lazily cleaned).
    seg_grid: DMap<[i64; 3], Vec<(usize, usize)>>,
    seg_cell: f64,
    /// Global facet BVH for membership + tagging; `facet_face[i]` is the B-rep
    /// face of PLC facet `i`.
    bvh: FacetBvh,
    facet_face: Vec<u32>,
    /// Curve evaluators per B-rep edge (arc-length parametrised).
    curves: Vec<Option<Box<dyn crate::curve::Curve>>>,
    /// Pierce repairs blocked by the duplicate guard, resolved by SNAPPING the
    /// blocking interior point onto the carrier instead: `vertex -> (position
    /// on carrier, face)`. Applied at the next [`Self::rebuild_points`] (the
    /// builder cannot move points in place).
    snaps: DMap<usize, (V3, u32)>,
    /// Work queue of tet slots to (re-)examine, with their vertices for
    /// validation (slots are reused by the builder's free list).
    queue: VecDeque<(u32, [usize; 4])>,
    diag: f64,
    max_points: usize,
    radius_edge: f64,
    /// Hard sizing floor: curvature-driven refinement (a curvature-radius spike
    /// on an intersection curve, a micro-rim) may not push `h` below this --
    /// the auto local-feature-size clamp. `min_h_surf` overrides when set.
    h_floor: f64,
    /// Diagnostics (RAPIDMESH_REFINE_TRACE): why insertions happen / fail.
    n_facet_ins: u64,
    n_cell_ins: u64,
    n_seg_split: u64,
    n_ins_rejected: u64,
    n_ball_none: u64,
    n_bad_size: u64,
    n_bad_shape: u64,
    n_bad_offface: u64,
    // Atomics, not Cells: the wall/depth stages share &Refiner across rayon
    // workers (everything else in the struct is plain read-only data there).
    n_cross_found: std::sync::atomic::AtomicU64,
    n_member_reject: std::sync::atomic::AtomicU64,
    n_cc_none: std::sync::atomic::AtomicU64,
    n_side_pierce: std::sync::atomic::AtomicU64,
}

impl<'a> Refiner<'a> {
    fn h_at(&self, p: V3) -> f64 {
        self.domain.h_at(p).max(self.h_floor)
    }

    fn pos(&self, v: usize) -> V3 {
        self.db.approx_point(v)
    }

    // ---------------- region classification (analytic-consistent) ----------

    /// Region at `p`, consistent with the ANALYTIC carriers. Away from curved
    /// surfaces (outside the sagitta band) the faceted classifier is exact and
    /// used directly. Inside the band -- where the chord surface (the faceted
    /// classifier's flip locus) and the analytic surface (where refinement
    /// places vertices) disagree -- a short ray is walked from an out-of-band
    /// anchor through the analytic crossings. Without this, cell refinement
    /// near a curved boundary fights the facet refinement in the sliver gap
    /// between chord and arc and never terminates.
    fn region_at_c(&self, p: V3) -> u32 {
        if self.band <= 0.0 || self.bvh.nearest_dist(p) > self.band {
            return self.domain.region_at(p);
        }
        const DIRS: [V3; 6] = [
            [0.9631, 0.1908, 0.1902],
            [-0.1791, 0.9645, 0.1937],
            [0.2113, -0.1979, 0.9571],
            [-0.5774, -0.5774, 0.5774],
            [0.7071, 0.0, -0.7071],
            [0.0, -0.7071, 0.7071],
        ];
        for d in DIRS {
            if let Some(r) = self.region_walk(p, d) {
                return r;
            }
        }
        // All directions ambiguous (grazing every time is vanishingly rare):
        // fall back to the faceted classifier.
        self.domain.region_at(p)
    }

    /// One attempt of the short analytic ray walk: anchor at `q = p + L*d`
    /// (outside the band), collect the analytic crossings of `p -> q`, and walk
    /// them from the anchor back to `p`. `None` when the picture is ambiguous
    /// (grazing crossing, membership mismatch) -- the caller retries rotated.
    fn region_walk(&self, p: V3, d: V3) -> Option<u32> {
        static WTRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let wtrace =
            *WTRACE.get_or_init(|| std::env::var_os("RAPIDMESH_WALK_TRACE").is_some());
        let ll = 6.0 * self.band;
        let mut q: V3 = std::array::from_fn(|k| p[k] + ll * d[k]);
        // extend until the anchor is out of the band (bounded tries)
        let mut ext = 0;
        while self.bvh.nearest_dist(q) <= self.band {
            q = std::array::from_fn(|k| q[k] + ll * d[k]);
            ext += 1;
            if ext > 8 {
                return None;
            }
        }
        let mut region = self.domain.region_at(q);
        // Crossings of segment q -> p with every nearby analytic face
        // (sphere-traced: thin features crossed twice are not skipped).
        let mut crossings: SmallVec<[(f64, usize); 8]> = SmallVec::new(); // (t along q->p, plc facet)
        for f in self.faces_near_segment(q, p) {
            let surf = self.brep.surface(self.brep.faces[f as usize].surface);
            for tm in carrier_crossings(surf, q, p) {
                let x0: V3 = std::array::from_fn(|k| q[k] + tm * (p[k] - q[k]));
                // Pull exactly onto the carrier (the bisection stops at
                // ~6e-5 of the segment, coarser than the planar band).
                let x = surf.closest(x0).0;
                // membership: nearest facet must belong to THIS face and be
                // within its own band (else the crossing is off the trim).
                if let Some((fi, dst)) = self.bvh.nearest_index(x) {
                    let ok = self.facet_face[fi] == f
                        && dst <= self.facet_band[fi].max(1e-7 * self.diag);
                    if wtrace {
                        eprintln!(
                            "WCROSS p ({:.9},{:.9},{:.9}) d ({:.3},{:.3},{:.3}) f {f} fi {fi} own {} dst {dst:.3e} band {:.3e} tags {:?} {}",
                            p[0], p[1], p[2], d[0], d[1], d[2],
                            self.facet_face[fi],
                            self.facet_band[fi].max(1e-7 * self.diag),
                            self.plc.region_tags[fi],
                            if ok { "OK" } else { "REJ" }
                        );
                    }
                    if ok {
                        crossings.push((tm, fi));
                    }
                }
            }
        }
        crossings.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        // Walk q -> p: at each crossing the region flips to the facet's tag on
        // the arrival side; the departure side must MATCH the current region,
        // else a crossing was missed (grazing) and this direction is unusable.
        let dir_qp: V3 = std::array::from_fn(|k| p[k] - q[k]);
        for &(_, fi) in &crossings {
            let t = self.plc.triangles[fi];
            let (a, b, c) = (
                self.plc.vertices[t[0] as usize],
                self.plc.vertices[t[1] as usize],
                self.plc.vertices[t[2] as usize],
            );
            let n = cross(sub(b, a), sub(c, a));
            let s = dot(dir_qp, n);
            if s == 0.0 {
                return None; // grazing
            }
            let [fr, bk] = self.plc.region_tags[fi];
            // Walking along dir_qp: s > 0 means we arrive on the FRONT side.
            let (depart, arrive) = if s > 0.0 { (bk.0, fr.0) } else { (fr.0, bk.0) };
            if depart != region {
                if wtrace {
                    eprintln!(
                        "WMISMATCH p ({:.9},{:.9},{:.9}) d ({:.3},{:.3},{:.3}) fi {fi} depart {depart} != region {region}",
                        p[0], p[1], p[2], d[0], d[1], d[2]
                    );
                }
                return None; // inconsistent: a crossing was missed
            }
            region = arrive;
        }
        if wtrace {
            eprintln!(
                "WALK p ({:.9},{:.9},{:.9}) d ({:.3},{:.3},{:.3}) -> {region} ({} crossings)",
                p[0], p[1], p[2], d[0], d[1], d[2], crossings.len()
            );
        }
        Some(region)
    }

    /// Trimmed-face membership: a point `x` ON face `f`'s carrier lies on the
    /// face's sampled region iff its nearest PLC facet belongs to `f` and the
    /// distance stays within that facet's own sagitta band. The band is the
    /// MEASURED chord-to-carrier gap, so flat faces get a near-zero tolerance
    /// -- an h-fraction there admitted phantom face points up to 0.1 h beyond
    /// a trim edge, which then pierced the adjacent face's plane at box
    /// corners -- while curved faces keep their legitimate sagitta slack.
    fn on_trimmed_face(&self, f: u32, x: V3) -> bool {
        self.bvh.nearest_index(x).is_some_and(|(fi, d)| {
            self.facet_face[fi] == f && d <= self.facet_band[fi].max(1e-7 * self.diag)
        })
    }

    // ---------------- protecting balls -------------------------------------

    /// Grants vertex `v` a protecting ball of radius `r` (or shrinks it).
    fn set_ball(&mut self, v: usize, r: f64) {
        if v >= self.ball.len() || !(r > 0.0) {
            return;
        }
        if self.ball[v] > 0.0 && self.ball[v] <= r {
            return; // balls only shrink
        }
        self.ball[v] = r;
        let p = self.pos(v);
        let lo: [i64; 3] = std::array::from_fn(|k| ((p[k] - r) / self.seg_cell).floor() as i64);
        let hi: [i64; 3] = std::array::from_fn(|k| ((p[k] + r) / self.seg_cell).floor() as i64);
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    let cell = self.ball_grid.entry([x, y, z]).or_default();
                    if !cell.contains(&v) {
                        cell.push(v);
                    }
                }
            }
        }
    }

    /// True if `p` lies inside some vertex's protecting ball.
    fn in_ball(&self, p: V3) -> bool {
        self.ball_owner(p).is_some()
    }

    /// The protected vertex whose ball contains `p`, if any.
    fn ball_owner(&self, p: V3) -> Option<usize> {
        let cell: [i64; 3] = std::array::from_fn(|k| (p[k] / self.seg_cell).floor() as i64);
        if let Some(vs) = self.ball_grid.get(&cell) {
            for &v in vs {
                let r = self.ball[v];
                if r > 0.0 && dist(p, self.pos(v)) < r {
                    return Some(v);
                }
            }
        }
        None
    }

    /// Splits a live segment incident to protected vertex `v` (Cheng-Dey-Ramos:
    /// a crowded ball refines its feature, shrinking the balls until the
    /// blocked repair fits). Returns true if a split happened.
    fn split_ball_feature(&mut self, v: usize) -> bool {
        self.split_ball_feature_with(v, false)
    }

    fn split_ball_feature_with(&mut self, v: usize, force: bool) -> bool {
        let key = self
            .segs
            .iter()
            .filter(|(_, s)| s.va == v || s.vb == v)
            .map(|(&k, s)| {
                let l = dist(self.pos(s.va), self.pos(s.vb));
                (k, l)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k);
        match key {
            Some(k) => self.split_seg_with(k, force).is_some(),
            None => false,
        }
    }

    // ---------------- feature segments ------------------------------------

    fn seg_cells(&self, a: V3, b: V3) -> Vec<[i64; 3]> {
        // Cells covering the diametral-ball AABB.
        let m = mid3(a, b);
        let r = 0.5 * dist(a, b);
        let lo: [i64; 3] = std::array::from_fn(|k| ((m[k] - r) / self.seg_cell).floor() as i64);
        let hi: [i64; 3] = std::array::from_fn(|k| ((m[k] + r) / self.seg_cell).floor() as i64);
        let mut out = Vec::new();
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    out.push([x, y, z]);
                }
            }
        }
        out
    }

    fn add_seg(&mut self, seg: Seg) {
        let key = key2(seg.va, seg.vb);
        let (a, b) = (self.pos(seg.va), self.pos(seg.vb));
        for c in self.seg_cells(a, b) {
            self.seg_grid.entry(c).or_default().push(key);
        }
        self.segs.insert(key, seg);
    }

    /// The live segment whose OPEN diametral ball contains `p`, if any.
    fn encroached_seg(&self, p: V3) -> Option<(usize, usize)> {
        let cell: [i64; 3] = std::array::from_fn(|k| (p[k] / self.seg_cell).floor() as i64);
        let mut best: Option<((usize, usize), f64)> = None;
        for dx in -1..=1i64 {
            for dy in -1..=1i64 {
                for dz in -1..=1i64 {
                    let Some(keys) = self.seg_grid.get(&[cell[0] + dx, cell[1] + dy, cell[2] + dz])
                    else {
                        continue;
                    };
                    for &key in keys {
                        if !self.segs.contains_key(&key) {
                            continue; // stale (split away)
                        }
                        let (a, b) = (self.pos(key.0), self.pos(key.1));
                        // strictly inside the open diametral ball
                        if dot(sub(p, a), sub(p, b)) < 0.0 {
                            let d = dist(p, mid3(a, b));
                            if best.map_or(true, |(_, bd)| d < bd) {
                                best = Some((key, d));
                            }
                        }
                    }
                }
            }
        }
        best.map(|(k, _)| k)
    }

    /// True if any existing DT vertex encroaches segment `key`. The Delaunay
    /// stars of the endpoints suffice: a non-empty diametral ball contains the
    /// endpoint-nearest offender, which is a Delaunay neighbour of an endpoint.
    fn seg_is_encroached(&self, key: (usize, usize)) -> bool {
        let (a, b) = (self.pos(key.0), self.pos(key.1));
        for &v in &[key.0, key.1] {
            for slot in self.db.star_slots(v) {
                for ov in self.db.verts_of_slot(slot).into_iter().flatten() {
                    if ov == key.0 || ov == key.1 {
                        continue;
                    }
                    let p = self.pos(ov);
                    if dot(sub(p, a), sub(p, b)) < 0.0 {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Splits segment `key` at the arc-length midpoint ON its analytic curve.
    /// Returns the new vertex, or `None` when the segment is at the floor (the
    /// encroachment is then accepted -- the small-angle guard).
    fn split_seg(&mut self, key: (usize, usize)) -> Option<usize> {
        self.split_seg_with(key, false)
    }

    /// [`Self::split_seg`] with `force`: a manifold repair may split below the
    /// small-angle floor (correctness beats the churn guard).
    fn split_seg_with(&mut self, key: (usize, usize), force: bool) -> Option<usize> {
        let seg = *self.segs.get(&key)?;
        let (a, b) = (self.pos(seg.va), self.pos(seg.vb));
        let len = dist(a, b);
        let h = self.h_at(mid3(a, b));
        if !force && len < SEG_FLOOR * h {
            return None;
        }
        if len < 1e-6 * h {
            return None; // absolute backstop even when forced
        }
        let sm = 0.5 * (seg.sa + seg.sb);
        let p = match &self.curves[seg.ei as usize] {
            Some(c) => c.point_at(sm),
            None => mid3(a, b),
        };
        let v = self.db.try_insert(p)?;
        self.segs.remove(&key);
        self.n_seg_split += 1;
        self.prov.push(Prov::Edge(seg.ei));
        self.vfaces.push(self.edge_faces(seg.ei));
        self.ball.push(0.0);
        self.add_seg(Seg { ei: seg.ei, va: seg.va, vb: v, sa: seg.sa, sb: sm });
        self.add_seg(Seg { ei: seg.ei, va: v, vb: seg.vb, sa: sm, sb: seg.sb });
        let rb = 0.6 * dist(self.pos(seg.va), p).min(dist(self.pos(seg.vb), p));
        self.set_ball(v, rb);
        self.set_ball(seg.va, rb.max(dist(self.pos(seg.va), p) * 0.6));
        self.set_ball(seg.vb, rb.max(dist(self.pos(seg.vb), p) * 0.6));
        self.queue_created();
        Some(v)
    }

    fn edge_faces(&self, ei: u32) -> Vec<u32> {
        let mut fs: Vec<u32> = self.brep.edges[ei as usize]
            .coedges
            .iter()
            .map(|&c| self.brep.coedge(c).face.0)
            .collect();
        fs.sort_unstable();
        fs.dedup();
        fs
    }

    // ---------------- surface oracle ---------------------------------------

    /// The B-rep faces a dual segment could cross: the faces owning the PLC
    /// facets nearest to a few samples along the segment.
    fn faces_near_segment(&self, a: V3, b: V3) -> SmallVec<[u32; 8]> {
        let mut out: SmallVec<[u32; 8]> = SmallVec::new();
        let len = dist(a, b);
        let pad = len.max(1e-3 * self.diag);
        for i in 0..=8 {
            let t = i as f64 / 8.0;
            let p: V3 = std::array::from_fn(|k| a[k] + t * (b[k] - a[k]));
            if let Some((fi, d)) = self.bvh.nearest_index(p) {
                if d <= pad {
                    let f = self.facet_face[fi];
                    if !out.contains(&f) {
                        out.push(f);
                    }
                }
            }
        }
        out
    }

    /// First membership-valid crossing of the segment `a -> b` with face `f`'s
    /// carrier (sphere-traced, so double crossings through thin features are
    /// found), restricted to the trimmed face via nearest-PLC-facet ownership.
    /// `near` localises the search: a crossing farther than `near.1` from
    /// `near.0` belongs to another facet's dual (an unlocalised search lets a
    /// facet on one side of a thin feature claim crossings on the FAR side,
    /// whose huge surface balls then drive a refinement cascade).
    fn face_crossing(&self, f: u32, a: V3, b: V3, near: Option<(V3, f64)>) -> Option<V3> {
        let surf = self.brep.surface(self.brep.faces[f as usize].surface);
        // Clip the segment to the locality ball BEFORE tracing: any crossing
        // outside it is discarded below anyway, and dual segments (sliver
        // circumcenters, hull rays) can be orders of magnitude longer than
        // the ball -- tracing their far part is pure waste.
        let (a, b) = match near {
            Some((c, rmax)) => match clip_segment_to_ball(a, b, c, rmax) {
                Some((t0, t1)) => (
                    std::array::from_fn(|k| a[k] + t0 * (b[k] - a[k])),
                    std::array::from_fn(|k| a[k] + t1 * (b[k] - a[k])),
                ),
                None => return None,
            },
            None => (a, b),
        };
        for t in carrier_crossings(surf, a, b) {
            self.n_cross_found.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let x0: V3 = std::array::from_fn(|k| a[k] + t * (b[k] - a[k]));
            if let Some((c, rmax)) = near {
                if dist(x0, c) > rmax {
                    continue;
                }
            }
            // Pull exactly onto the carrier.
            let x = surf.closest(x0).0;
            // Membership: the crossing must lie on THIS trimmed face -- or,
            // near a feature edge, on the NEIGHBOUR face's carrier too (the
            // nearest-facet test attributes an on-edge crossing to whichever
            // wall wins the tie; rejecting it there opened flood-fill leaks
            // through the edges of an interior void).
            let Some((fi, d)) = self.bvh.nearest_index(x) else { continue };
            if d <= self.h_at(x) {
                let owner = self.facet_face[fi];
                if owner == f {
                    return Some(x);
                }
                if owner != u32::MAX {
                    let osurf = self.brep.surface(self.brep.faces[owner as usize].surface);
                    if signed_offset(osurf, x).abs() <= 1e-7 * self.diag {
                        return Some(x);
                    }
                }
            }
            self.n_member_reject.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        None
    }

    /// The restricted-Delaunay surface ball of the facet opposite corner `i` of
    /// the tet in `slot`: `(face, ball center, ball radius)` when the facet's
    /// Voronoi dual crosses a face carrier inside the trimmed face.
    fn facet_surface_ball(&self, slot: u32, i: usize, tet: [usize; 4]) -> Option<(u32, V3, f64)> {
        let fv: [usize; 3] = std::array::from_fn(|k| tet[FACET_LOCAL[i][k]]);
        let p: [V3; 4] = std::array::from_fn(|k| self.pos(tet[k]));
        let Some((c1, r1)) = tet_circumcenter(p) else {
            self.n_cc_none.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        };
        // Dual endpoint on the other side: neighbour circumcenter, or an
        // outward ray for hull facets (the neighbour holds a super corner).
        let (c2, r2): (V3, f64) = {
            let nb_tet = self
                .db
                .neighbor_at(slot, i)
                .and_then(|nb| self.db.tet_at(nb));
            match nb_tet {
                Some(nt) => {
                    let np: [V3; 4] = std::array::from_fn(|k| self.pos(nt[k]));
                    tet_circumcenter(np)?
                }
                None => (self.hull_dual_end(c1, fv)?, f64::INFINITY),
            }
        };
        // Sound prune: a crossing point x lies ON the dual segment, so
        // |x - v0| <= max(|c1 - v0|, |c2 - v0|) = max(r1, r2). If the facet's
        // vertex is farther from the whole surface than that, no crossing
        // exists -- skips the expensive oracle for the deep-interior bulk.
        if r2.is_finite() && self.bvh.nearest_dist(self.pos(fv[0])) > r1.max(r2) {
            return None;
        }
        // Candidate faces from provenance first (cheap), else BVH proximity.
        let mut cands: SmallVec<[u32; 8]> = SmallVec::new();
        for &f in &self.vfaces[fv[0]] {
            if self.vfaces[fv[1]].contains(&f) && self.vfaces[fv[2]].contains(&f) {
                cands.push(f);
            }
        }
        if cands.is_empty() {
            cands = self.faces_near_segment(c1, c2);
        }
        // Locality bound: crossings farther from the facet than its own scale
        // plus the local target size belong to other facets' duals.
        let (a, b, c) = (self.pos(fv[0]), self.pos(fv[1]), self.pos(fv[2]));
        let fc: V3 = std::array::from_fn(|k| (a[k] + b[k] + c[k]) / 3.0);
        let rf = crate::geomutil::circumradius(a, b, c);
        let near = Some((fc, 2.0 * rf.min(self.diag) + 2.0 * self.h_at(fc)));
        for f in cands {
            if let Some(x) = self.face_crossing(f, c1, c2, near) {
                let r = dist(x, self.pos(fv[0]));
                return Some((f, x, r));
            }
        }
        None
    }

    /// Outward dual-ray endpoint for a hull facet: from the circumcenter along
    /// the facet normal, oriented away from the tet.
    fn hull_dual_end(&self, c1: V3, fv: [usize; 3]) -> Option<V3> {
        let (a, b, c) = (self.pos(fv[0]), self.pos(fv[1]), self.pos(fv[2]));
        let mut n = cross(sub(b, a), sub(c, a));
        let l = dot(n, n).sqrt();
        if l == 0.0 {
            return None;
        }
        n = scale(n, 1.0 / l);
        let fc: V3 = std::array::from_fn(|k| (a[k] + b[k] + c[k]) / 3.0);
        if dot(sub(fc, c1), n) < 0.0 {
            n = scale(n, -1.0);
        }
        Some(std::array::from_fn(|k| fc[k] + n[k] * self.diag))
    }

    /// Extraction-side wall test: does the Voronoi dual of the facet `fv`
    /// between `t1` and (optional) `t2` cross a surface? Mirrors
    /// [`Self::facet_surface_ball`] but works on vertex indices (the extraction
    /// has no live slots). `None` neighbour = hull side (dual ray).
    fn facet_is_wall(&self, t1: &[usize; 4], t2: Option<&[usize; 4]>, fv: [usize; 3]) -> bool {
        self.facet_wall_face(t1, t2, fv).is_some()
    }

    /// [`Self::facet_is_wall`], returning WHICH face the facet walls off.
    fn facet_wall_face(
        &self,
        t1: &[usize; 4],
        t2: Option<&[usize; 4]>,
        fv: [usize; 3],
    ) -> Option<u32> {
        // Centroid side test, NOT the circumcenter dual: the dual of a flat
        // wall-piercing tet (two wall points + an interior point) can sit on
        // one side entirely and miss the crossing; the two tets' centroids are
        // local and reliable. The segment between them crosses the surface iff
        // the facet separates material from cavity here.
        let p1: [V3; 4] = std::array::from_fn(|k| self.pos(t1[k]));
        let c1: V3 = std::array::from_fn(|k| 0.25 * (p1[0][k] + p1[1][k] + p1[2][k] + p1[3][k]));
        let c2: V3 = match t2 {
            Some(t) => {
                let p2: [V3; 4] = std::array::from_fn(|k| self.pos(t[k]));
                std::array::from_fn(|k| 0.25 * (p2[0][k] + p2[1][k] + p2[2][k] + p2[3][k]))
            }
            None => match self.hull_dual_end(c1, fv) {
                Some(e) => e,
                None => return None,
            },
        };
        // Prune: both centroids farther from the surface than their distance
        // cannot straddle it.
        let len = dist(c1, c2);
        if self.bvh.nearest_dist(c1) > len && self.bvh.nearest_dist(c2) > len {
            return None;
        }
        let mut cands: SmallVec<[u32; 8]> = SmallVec::new();
        for &f in &self.vfaces[fv[0]] {
            if self.vfaces[fv[1]].contains(&f) && self.vfaces[fv[2]].contains(&f) {
                cands.push(f);
            }
        }
        for f in self.faces_near_segment(c1, c2) {
            if !cands.contains(&f) {
                cands.push(f);
            }
        }
        // SEPARATION predicate: the facet walls off face `f` iff the two tets
        // lie on OPPOSITE sides of `f`'s carrier (one-sided negative vs
        // one-sided positive, with an on-face tolerance). Two tets hugging the
        // SAME side -- whose centroid segment can still cross the carrier
        // elsewhere -- are the false positives that shifted the flood boundary
        // one tet layer off the interface (a systematic ~1e-3 volume bias).
        // A mixed (piercing) tet falls back to the centroid side: pierces are
        // repaired by the manifold sweep, and the residue must not open the
        // wall entirely.
        let tol_on = 1e-6 * self.diag;
        let side_of = |surf: &Surface, t: &[usize; 4], cen: V3| -> i8 {
            let (mut pos, mut neg) = (false, false);
            for &v in t {
                let s = signed_offset(surf, self.pos(v));
                if s > tol_on {
                    pos = true;
                }
                if s < -tol_on {
                    neg = true;
                }
            }
            match (pos, neg) {
                (true, false) => 1,
                (false, true) => -1,
                // ALL vertices on the carrier: this is NOT necessarily a
                // degenerate sliver -- a chord tet spanning an unseeded void
                // has all four vertices ON the surface and its whole volume
                // on one side. The centroid tells that side (for a truly
                // flat sliver it is sign noise, but CONSISTENT per tet,
                // which is all manifoldness needs). An undecided 0 here made
                // every facet of such tets a non-wall and let the flood fill
                // swallow entire voids; walling them on ALL sides (the old
                // provenance shortcut) fragmented the sagitta band into
                // thousands of enclaves instead.
                (false, false) => {
                    if signed_offset(surf, cen) >= 0.0 { 1 } else { -1 }
                }
                (true, true) => {
                    // piercing: dominant side by centroid
                    self.n_side_pierce.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if signed_offset(surf, cen) >= 0.0 { 1 } else { -1 }
                }
            }
        };
        let wtrace = std::env::var_os("RAPIDMESH_WALL_TRACE").is_some();
        if wtrace && cands.is_empty() {
            eprintln!("WALL no cands, fv {fv:?}");
        }
        for f in cands {
            let surf = self.brep.surface(self.brep.faces[f as usize].surface);
            // both tets must actually TOUCH this carrier's neighbourhood: the
            // shared facet's vertices are the contact set
            let near_facet = fv
                .iter()
                .any(|&v| signed_offset(surf, self.pos(v)).abs() <= 0.5 * self.h_at(self.pos(v)));
            if !near_facet {
                if wtrace {
                    eprintln!("WALL f {f} not near, fv {fv:?}");
                }
                continue;
            }
            let s1 = side_of(surf, t1, c1);
            let s2 = match t2 {
                Some(t) => side_of(surf, t, c2),
                None => -s1, // hull side: outward is the other side by definition
            };
            if wtrace {
                eprintln!("WALL f {f} sides {s1} {s2}, fv {fv:?}");
            }
            if s1 * s2 == -1 {
                // opposite sides: confirm the trimmed-face membership at the
                // facet centroid's footpoint (not at a distant crossing)
                let fc: V3 = {
                    let (a, b, c) = (self.pos(fv[0]), self.pos(fv[1]), self.pos(fv[2]));
                    std::array::from_fn(|k| (a[k] + b[k] + c[k]) / 3.0)
                };
                let x = surf.closest(fc).0;
                if let Some((fi, d)) = self.bvh.nearest_index(x) {
                    let owner = self.facet_face[fi];
                    if wtrace {
                        eprintln!("WALL f {f} membership owner {owner} d {d:.4} h {:.4}", self.h_at(x));
                    }
                    if d <= self.h_at(x) && owner == f {
                        return Some(f);
                    }
                    // near a feature edge the neighbour face may own the
                    // footpoint; accept when it shares the carrier locally
                    if d <= self.h_at(x) && owner != u32::MAX {
                        let osurf =
                            self.brep.surface(self.brep.faces[owner as usize].surface);
                        if signed_offset(osurf, x).abs() <= tol_on {
                            return Some(f);
                        }
                    }
                }
            }
        }
        None
    }

    /// Manifold guard: a tet whose vertices carry signed offsets of BOTH signs
    /// to a nearby face carrier PIERCES that surface -- a hole in the
    /// restricted wall no facet criterion sees (the dual of a flat piercing
    /// tet does not cross). Returns the pierced face and the centroid's
    /// on-carrier projection as the repair point.
    fn pierce_repair(&self, tet: [usize; 4]) -> Option<(u32, V3)> {
        let p: [V3; 4] = std::array::from_fn(|k| self.pos(tet[k]));
        let cen: V3 =
            std::array::from_fn(|k| 0.25 * (p[0][k] + p[1][k] + p[2][k] + p[3][k]));
        // cheap prune: a tet farther from the surface than its own extent
        // cannot pierce.
        let mut ext = 0.0f64;
        for a in 0..4 {
            for b in (a + 1)..4 {
                ext = ext.max(dist(p[a], p[b]));
            }
        }
        if self.bvh.nearest_dist(cen) > ext {
            return None;
        }
        // candidate faces: provenance of the vertices + the nearest facet
        let mut cands: Vec<u32> = Vec::new();
        for &v in &tet {
            for &f in &self.vfaces[v] {
                if !cands.contains(&f) {
                    cands.push(f);
                }
            }
        }
        if let Some((fi, _)) = self.bvh.nearest_index(cen) {
            let f = self.facet_face[fi];
            if f != u32::MAX && !cands.contains(&f) {
                cands.push(f);
            }
        }
        for f in cands {
            let surf = self.brep.surface(self.brep.faces[f as usize].surface);
            let tol = 1e-7 * self.diag;
            let (mut pos, mut neg) = (false, false);
            for q in p {
                let s = signed_offset(surf, q);
                if s > tol {
                    pos = true;
                }
                if s < -tol {
                    neg = true;
                }
            }
            if !(pos && neg) {
                continue;
            }
            // pierced: repair at the piercing EDGE's crossing with the carrier
            // (the centroid footpoint often lands on an existing wall point and
            // dies to the duplicate guard; an edge crossing lies strictly
            // between two off-surface endpoints).
            for a in 0..4 {
                for b in (a + 1)..4 {
                    let (sa, sb) = (signed_offset(surf, p[a]), signed_offset(surf, p[b]));
                    if !(sa > tol && sb < -tol || sa < -tol && sb > tol) {
                        continue;
                    }
                    let (mut qa, mut qb, a_neg) = (p[a], p[b], sa < 0.0);
                    for _ in 0..40 {
                        let m = mid3(qa, qb);
                        if (signed_offset(surf, m) < 0.0) == a_neg {
                            qa = m;
                        } else {
                            qb = m;
                        }
                    }
                    let x = surf.closest(mid3(qa, qb)).0;
                    // Tight trimmed-face membership: a loose h-bound accepted
                    // plane crossings BEYOND a convex feature edge -- tets
                    // spanning two adjacent faces pierce the extended carrier
                    // PLANE of one of them, which is no hole at all, and
                    // their phantom repairs churned against the edge's
                    // protected zone.
                    if self.on_trimmed_face(f, x) {
                        return Some((f, x));
                    }
                }
            }
        }
        None
    }

    /// True if `p` keeps enough clearance from the surface: at least `frac`
    /// of the SURFACE target size at the nearest facet (not the volume size at
    /// `p` -- with a fine interface under a coarse bulk, the volume size lets
    /// interior points sit closer to the wall than the wall's own point
    /// spacing, and their Delaunay edges then pierce it).
    fn surface_clear(&self, p: V3, frac: f64) -> bool {
        let Some((fi, d)) = self.bvh.nearest_index(p) else {
            return true;
        };
        let t = self.plc.triangles[fi];
        let cen: V3 = std::array::from_fn(|k| {
            (self.plc.vertices[t[0] as usize][k]
                + self.plc.vertices[t[1] as usize][k]
                + self.plc.vertices[t[2] as usize][k])
                / 3.0
        });
        d >= frac * self.h_at(cen).min(self.h_at(p))
    }

    // ---------------- insertion with feature protection --------------------

    /// Inserts a surface/volume candidate unless it encroaches a feature
    /// segment (then the segment splits instead) or duplicates a vertex.
    /// Returns true if the triangulation changed.
    fn insert_protected(&mut self, p: V3, prov: Prov, faces: Vec<u32>) -> bool {
        self.insert_protected_with(p, prov, faces, false)
    }

    /// [`Self::insert_protected`] with `manifold_repair` semantics: a candidate
    /// blocked by a protecting ball refines the ball's feature instead of
    /// being dropped (used by the pierce sweep, whose insertions are
    /// correctness repairs, not quality refinements).
    fn insert_protected_with(
        &mut self,
        p: V3,
        prov: Prov,
        faces: Vec<u32>,
        manifold_repair: bool,
    ) -> bool {
        if self.db.len() >= self.max_points {
            return false;
        }
        // Inside a protecting ball: quality-driven candidates are dropped (the
        // feature is authoritative there); MANIFOLD repairs may instead refine
        // the crowded ball's feature -- its balls shrink and the repair fits on
        // a later sweep (a pierce left unrepaired is a hole in the wall).
        if let Some(owner) = self.ball_owner(p) {
            if manifold_repair {
                return self.split_ball_feature_with(owner, true);
            }
            return false;
        }
        if let Some(key) = self.encroached_seg(p) {
            if self.split_seg(key).is_some() {
                return true;
            }
            // Segment at the sizing floor, no further split possible. A
            // QUALITY candidate is dropped (Ruppert: never insert into a
            // diametral zone), but a MANIFOLD repair outranks the rule: the
            // point lies ON a face carrier and closes a hole in the wall --
            // leaving the pierce open costs region-volume correctness. The
            // floor caps any cascade (the segment never splits again).
            if !manifold_repair {
                return false;
            }
        }
        let h = self.h_at(p);
        let guard = (DUP_FRAC * h) * (DUP_FRAC * h);
        match self.db.insert_guarded(p, guard, |_| true) {
            Some(_) => {
                self.prov.push(prov);
                self.ball.push(0.0);
                self.vfaces.push(faces);
                self.queue_created();
                match self.prov.last() {
                    Some(Prov::Interior) => self.n_cell_ins += 1,
                    _ => self.n_facet_ins += 1,
                }
                true
            }
            None => {
                self.n_ins_rejected += 1;
                false
            }
        }
    }

    fn queue_created(&mut self) {
        for slot in self.db.last_created().to_vec() {
            if let Some(t) = self.db.tet_at(slot) {
                self.queue.push_back((slot, t));
            }
        }
    }

    /// Rebuilds the DT from scratch at `new_pos` (every existing vertex) plus
    /// `extra` new interior points. FIXED (non-interior) vertices insert
    /// first: a relaxed interior point that drifted near the surface must
    /// lose the near-duplicate tie against a surface/feature point, never
    /// displace it -- one displaced wall point is a hole in the restricted
    /// surface and a flood-fill leak. Pending pierce SNAPS apply first (the
    /// snapped vertex becomes a face point at its on-carrier position), and
    /// the protecting-ball state is REMAPPED onto the new indices, so ball
    /// guards keep working for insertions after the rebuild.
    fn rebuild_points(&mut self, lo: V3, hi: V3, new_pos: &mut [V3], extra: &[V3]) {
        let n_all = self.db.len();
        debug_assert_eq!(new_pos.len(), n_all);
        if !self.snaps.is_empty() {
            let snaps = std::mem::take(&mut self.snaps);
            for (v, (x, f)) in snaps {
                self.prov[v] = Prov::Face(f);
                self.vfaces[v] = vec![f];
                new_pos[v] = x;
            }
        }
        let mut db = DelaunayBuilder::enclosing(lo, hi);
        let mut prov2: Vec<Prov> = Vec::with_capacity(n_all + extra.len());
        let mut vfaces2: Vec<Vec<u32>> = Vec::with_capacity(n_all + extra.len());
        let mut ball2: Vec<f64> = Vec::with_capacity(n_all + extra.len());
        // old -> new index (u32::MAX = swallowed duplicate): fixed-first
        // insertion renumbers EVERY vertex, and the feature-segment store
        // below is keyed by vertex index.
        let mut old2new: Vec<u32> = vec![u32::MAX; n_all];
        for pass in 0..2 {
            for v in 0..n_all {
                let is_interior = matches!(self.prov[v], Prov::Interior);
                if (pass == 0) == is_interior {
                    continue;
                }
                if db.try_insert(new_pos[v]).is_some() {
                    old2new[v] = prov2.len() as u32;
                    prov2.push(self.prov[v].clone());
                    vfaces2.push(self.vfaces[v].clone());
                    ball2.push(if is_interior {
                        0.0
                    } else {
                        self.ball.get(v).copied().unwrap_or(0.0)
                    });
                }
            }
        }
        for &p in extra {
            if db.try_insert(p).is_some() {
                prov2.push(Prov::Interior);
                vfaces2.push(Vec::new());
                ball2.push(0.0);
            }
        }
        self.db = db;
        self.prov = prov2;
        self.vfaces = vfaces2;
        self.ball = vec![0.0; ball2.len()];
        self.ball_grid.clear();
        for (v, r0) in ball2.into_iter().enumerate() {
            if r0 > 0.0 {
                self.set_ball(v, r0);
            }
        }
        // Remap the live feature segments onto the new numbering. This was
        // MISSING: after any rebuild every seg key pointed at the old
        // indices, and the next encroachment/ball query walked pos() with a
        // stale (possibly out-of-range) vertex -- latent since the first
        // rebuild_points, manifest once the piercer escalation made
        // shrinking rebuilds common (pressure_vessel index panic). A
        // swallowed endpoint (interior duplicates only; features insert
        // first) drops its segment.
        let old_segs = std::mem::take(&mut self.segs);
        self.seg_grid.clear();
        for (_, s) in old_segs {
            let (na, nb) = (old2new[s.va], old2new[s.vb]);
            if na == u32::MAX || nb == u32::MAX {
                continue;
            }
            self.add_seg(Seg { va: na as usize, vb: nb as usize, ..s });
        }
        // Work-queue slots/tets reference the old builder: all invalid now.
        self.queue.clear();
    }

    // ---------------- criteria ---------------------------------------------

    /// Checks the tet's four facets against the surface criteria, inserting at
    /// most one refinement point. Returns true if an insertion happened.
    fn process_facets(&mut self, slot: u32, tet: [usize; 4]) -> bool {
        // Piercing tets first: a tet with carrier offsets of both signs crosses
        // a face outright -- no facet criterion sees it (its dual does not
        // cross), and every one that survives to extraction puts volume on the
        // wrong side of an interface. In-loop repair converges through the
        // normal requeue instead of a bounded end sweep.
        if let Some((f, x)) = self.pierce_repair(tet) {
            if self.insert_protected_with(x, Prov::Face(f), vec![f], true) {
                return true;
            }
        }
        for i in 0..4 {
            let Some((f, x, r)) = self.facet_surface_ball(slot, i, tet) else {
                self.n_ball_none += 1;
                continue;
            };
            let fv: [usize; 3] = std::array::from_fn(|k| tet[FACET_LOCAL[i][k]]);
            let h = self.h_at(x);
            let on_face = fv.iter().all(|&v| {
                self.vfaces[v].contains(&f) || matches!(self.prov[v], Prov::Corner)
            });
            let (a, b, c) = (self.pos(fv[0]), self.pos(fv[1]), self.pos(fv[2]));
            // The off-face rule (a vertex not on `f` sits in a surface facet)
            // only refines ABOVE a size floor: in thin regions every interior
            // point is near the surface, and unbounded off-face insertion
            // ping-pongs with cell refinement (torus blowup). Sub-floor
            // boundary slivers are the optimizer's sliver stage's job.
            let bad_size = r > FACET_SIZE * h;
            // Shape and off-face rules only refine ABOVE a size floor: on a
            // curved carrier the surface-ball center of a skinny facet can be
            // far away, so inserting it does not fix the local shape -- the
            // churn bounds only through the floor. Sub-floor skinny/off-face
            // facets are the optimizer's sliver stage's job.
            let bad_shape = tri_min_angle(a, b, c) < FACET_ANGLE_DEG && r > 0.8 * h;
            let bad_off = !on_face && r > 0.5 * h;
            if (bad_size || bad_shape || bad_off) && self.insert_protected(x, Prov::Face(f), vec![f]) {
                if bad_size {
                    self.n_bad_size += 1;
                } else if bad_shape {
                    self.n_bad_shape += 1;
                } else {
                    self.n_bad_offface += 1;
                }
                return true;
            }
        }
        false
    }

    /// Checks the tet against the cell criteria (size, radius-edge), inserting
    /// its circumcenter when bad. Returns true if an insertion happened.
    fn process_cell(&mut self, slot: u32, tet: [usize; 4]) -> bool {
        // Only tets inside a meshed region.
        let p: [V3; 4] = std::array::from_fn(|k| self.pos(tet[k]));
        let centroid: V3 =
            std::array::from_fn(|k| 0.25 * (p[0][k] + p[1][k] + p[2][k] + p[3][k]));
        let region = self.region_at_c(centroid);
        if region == 0 {
            return false;
        }
        let Some((cc, r)) = tet_circumcenter(p) else {
            return false;
        };
        let mut lmin = f64::INFINITY;
        for a in 0..4 {
            for b in (a + 1)..4 {
                lmin = lmin.min(dist(p[a], p[b]));
            }
        }
        let h = self.h_at(cc);
        if r <= CELL_SIZE * h && r / lmin.max(1e-300) <= self.radius_edge {
            return false;
        }
        // A circumcenter in another region means the boundary lies between:
        // refine the crossing facet instead (never insert across an interface).
        if self.region_at_c(cc) != region {
            // Refine the crossing facet instead of the cell -- but ONLY when
            // that facet is bad by its OWN size criterion. If the boundary is
            // already at target density, the cell is simply skipped (a boundary
            // sliver for the sliver stage): an unconditional fallback bombards
            // a thin region's boundary at the duplicate-guard density.
            for i in 0..4 {
                if let Some((f, x, rb)) = self.facet_surface_ball(slot, i, tet) {
                    if rb > FACET_SIZE * self.h_at(x)
                        && self.insert_protected(x, Prov::Face(f), vec![f])
                    {
                        return true;
                    }
                }
            }
            return false;
        }
        // Interior points hugging the surface pierce the (finer) wall
        // sampling: a tet of two wall points and one hugger connects the two
        // sides straight through the wall, a hole no facet criterion sees.
        // Clearance is measured against the SURFACE target size.
        if !self.surface_clear(cc, 0.5) {
            return false;
        }
        self.insert_protected(cc, Prov::Interior, Vec::new())
    }
}

/// Meshes a tagged PLC by restricted-Delaunay refinement (the refinement-core
/// path). Same output schema as [`crate::cvt::mesh_cdt`], so downstream
/// (optimize / topo / python) is unchanged.
pub fn mesh_refine(plc: &TaggedPlc, params: &MeshParams) -> TetMesh {
    use rapidmesh_exact::log as rmlog;
    let t_start = std::time::Instant::now();

    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in &plc.vertices {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diag = (0..3).map(|k| hi[k] - lo[k]).fold(0.0_f64, f64::max).max(1e-12);

    let trace0 = std::env::var_os("RAPIDMESH_REFINE_TRACE").is_some();
    let brep = rapidmesh_brep::build::from_plc(plc);
    if trace0 {
        eprintln!("CHECKPOINT brep: {} faces {} edges", brep.faces.len(), brep.edges.len());
    }
    let domain = crate::cvt::build_sizing_domain(plc, params, &brep);
    if trace0 {
        eprintln!("CHECKPOINT domain built, finest {}", domain.finest());
    }
    if std::env::var_os("RAPIDMESH_REFINE_TRACE").is_some() {
        for t in plc.triangles.iter().take(3) {
            let c: V3 = std::array::from_fn(|k| {
                (plc.vertices[t[0] as usize][k] + plc.vertices[t[1] as usize][k] + plc.vertices[t[2] as usize][k]) / 3.0
            });
            eprintln!("TRACE h_at({c:?}) = {}", domain.h_at(c));
        }
    }

    // Global facet BVH: membership tests + boundary tagging.
    let mut facet_face = vec![u32::MAX; plc.triangles.len()];
    for (fid, f) in brep.faces.iter().enumerate() {
        for &ti in &f.facets {
            facet_face[ti as usize] = fid as u32;
        }
    }
    let facets: Vec<(Tri, f64)> = plc
        .triangles
        .iter()
        .map(|t| {
            (
                Tri::new(
                    plc.vertices[t[0] as usize],
                    plc.vertices[t[1] as usize],
                    plc.vertices[t[2] as usize],
                ),
                0.0,
            )
        })
        .collect();
    let bvh = FacetBvh::build(&facets);

    // Chord-sagitta band per facet: how far the analytic carrier deviates from
    // the chord facet, MEASURED as the offset of the facet's centroid and edge
    // midpoints from the carrier (an `L^2/8R` formula over-estimates wildly on
    // elongated facets -- a barrel wall's diagonal is not a curvature chord).
    let facet_band: Vec<f64> = plc
        .triangles
        .iter()
        .enumerate()
        .map(|(fi, t)| {
            let bf = facet_face[fi];
            if bf == u32::MAX {
                return 0.0;
            }
            let surf = brep.surface(brep.faces[bf as usize].surface);
            let (a, b, c) = (
                plc.vertices[t[0] as usize],
                plc.vertices[t[1] as usize],
                plc.vertices[t[2] as usize],
            );
            let cen: V3 = std::array::from_fn(|k| (a[k] + b[k] + c[k]) / 3.0);
            [cen, mid3(a, b), mid3(b, c), mid3(c, a)]
                .into_iter()
                .map(|q| signed_offset(surf, q).abs())
                .fold(0.0f64, f64::max)
        })
        .collect();
    let band = facet_band.iter().copied().fold(0.0f64, f64::max) * 1.5;

    // Curve evaluator per B-rep edge.
    let curves: Vec<Option<Box<dyn crate::curve::Curve>>> =
        brep.edges.iter().map(|e| edge_curve(&brep, e)).collect();
    if trace0 {
        eprintln!("CHECKPOINT curves built");
    }

    let mut r = Refiner {
        brep: &brep,
        domain: &domain,
        plc,
        facet_band,
        band,
        db: DelaunayBuilder::enclosing(lo, hi),
        prov: Vec::new(),
        ball: Vec::new(),
        ball_grid: DMap::default(),
        vfaces: Vec::new(),
        segs: DMap::default(),
        seg_grid: DMap::default(),
        seg_cell: (0.05 * diag).max(1e-9),
        bvh,
        facet_face,
        curves,
        snaps: DMap::default(),
        queue: VecDeque::new(),
        diag,
        max_points: params.max_points,
        radius_edge: params.radius_edge_bound.max(1.0),
        h_floor: {
            let h_ref = if params.maxh.is_finite() {
                params.maxh
            } else {
                diag / crate::constants::DEFAULT_SUBDIV
            };
            // The auto floor clamps CURVATURE-driven runaway; explicit user
            // targets (per-edge / per-face / per-region / point sources) may
            // be finer and win.
            let user_min = params
                .edge_maxh
                .iter()
                .chain(&params.surf_maxh)
                .chain(&params.face_maxh)
                .chain(&params.surface_maxh)
                .chain(&params.region_maxh)
                .map(|&(_, h)| h)
                .chain(params.size_points.iter().map(|&(_, h)| h))
                .fold(f64::INFINITY, f64::min);
            params.min_h_surf.max((h_ref / 8.0).min(user_min)).max(1e-12)
        },
        n_facet_ins: 0,
        n_cell_ins: 0,
        n_seg_split: 0,
        n_ins_rejected: 0,
        n_ball_none: 0,
        n_bad_size: 0,
        n_bad_shape: 0,
        n_bad_offface: 0,
        n_cross_found: std::sync::atomic::AtomicU64::new(0),
        n_member_reject: std::sync::atomic::AtomicU64::new(0),
        n_cc_none: std::sync::atomic::AtomicU64::new(0),
        n_side_pierce: std::sync::atomic::AtomicU64::new(0),
    };

    // ---- protect 0-features: corners --------------------------------------
    let mut corner_vid: Vec<usize> = Vec::with_capacity(brep.vertices.len());
    for v in &brep.vertices {
        let vid = r.db.insert(v.pos);
        r.prov.push(Prov::Corner);
        r.ball.push(0.0);
        r.vfaces.push(Vec::new()); // filled from incident edges below
        corner_vid.push(vid);
    }
    // Corner face incidence = union of radial faces of incident edges.
    for (ei, e) in brep.edges.iter().enumerate() {
        let fs = r.edge_faces(ei as u32);
        for &end in &e.ends {
            let vid = corner_vid[end.0 as usize];
            for &f in &fs {
                if !r.vfaces[vid].contains(&f) {
                    r.vfaces[vid].push(f);
                }
            }
        }
    }

    // ---- protect 1-features: sample every edge curve ----------------------
    let t_feat = std::time::Instant::now();
    for (ei, edge) in brep.edges.iter().enumerate() {
        let Some(curve) = &r.curves[ei] else { continue };
        // The auto LFS floor must not overrule an EXPLICIT user target: a
        // per-edge maxh finer than the floor wins (the floor only clamps
        // curvature-driven runaway).
        let user_cap = params.edge_maxh_for(ei);
        let floor = r.h_floor.min(user_cap);
        let cap = edge
            .chain
            .iter()
            .map(|&p| r.h_at(p))
            .fold(f64::INFINITY, f64::min)
            .min(user_cap)
            .max(floor);
        let grad = if params.grading > 0.0 { params.grading } else { 0.5 };
        let ss = distribute_floored(&**curve, params.edge_tol_for(ei), cap, grad, floor);
        let fs = r.edge_faces(ei as u32);
        // Vertex ids along the edge WITH their arc params, in curve order (a
        // rejected near-duplicate sample drops its param too, so segments stay
        // aligned with the surviving points).
        let mut vids: Vec<(usize, f64)> = Vec::with_capacity(ss.len());
        vids.push((corner_vid[edge.ends[0].0 as usize], ss[0]));
        for &s in ss.iter().take(ss.len().saturating_sub(1)).skip(1) {
            let p = curve.point_at(s);
            if let Some(v) = r.db.try_insert(p) {
                r.prov.push(Prov::Edge(ei as u32));
                r.ball.push(0.0);
                r.vfaces.push(fs.clone());
                vids.push((v, s));
            }
        }
        vids.push((corner_vid[edge.ends[1].0 as usize], ss[ss.len() - 1]));
        for w in vids.windows(2) {
            if w[0].0 != w[1].0 {
                r.add_seg(Seg { ei: ei as u32, va: w[0].0, vb: w[1].0, sa: w[0].1, sb: w[1].1 });
                let d = 0.6 * dist(r.pos(w[0].0), r.pos(w[1].0));
                r.set_ball(w[0].0, d);
                r.set_ball(w[1].0, d);
            }
        }
    }
    // Resolve initial encroachments (a fixpoint: splits can re-encroach).
    loop {
        let keys: Vec<(usize, usize)> = r.segs.keys().copied().collect();
        let mut split_any = false;
        for key in keys {
            if r.segs.contains_key(&key) && r.seg_is_encroached(key) {
                split_any |= r.split_seg(key).is_some();
            }
        }
        if !split_any {
            break;
        }
    }
    rmlog::stat("refine.feature_points", r.db.len() as f64);
    rmlog::stage("refine.features", t_feat.elapsed().as_secs_f64());
    if std::env::var_os("RAPIDMESH_REFINE_TRACE").is_some() {
        eprintln!("CHECKPOINT features done: {} points", r.db.len());
    }

    // ---- seed edge-less faces (closed spheres etc.) ------------------------
    for (fid, face) in brep.faces.iter().enumerate() {
        let has_edge = face.loops.iter().any(|l| !l.coedges.is_empty());
        if has_edge || face.facets.is_empty() {
            continue;
        }
        // A closed face: seed WELL-SPREAD on-carrier points so the restricted
        // Delaunay has something to grow from. Farthest-point sampling over the
        // facet centroids -- a fixed stride over a structured tessellation can
        // pick one ring of COCIRCULAR points (every tet circumcenter degenerate,
        // no dual, no refinement: the torus failure).
        let surf = brep.surface(face.surface);
        let cents: Vec<V3> = face
            .facets
            .iter()
            .map(|&ti| {
                let t = plc.triangles[ti as usize];
                std::array::from_fn(|k| {
                    (plc.vertices[t[0] as usize][k]
                        + plc.vertices[t[1] as usize][k]
                        + plc.vertices[t[2] as usize][k])
                        / 3.0
                })
            })
            .collect();
        let n_seed = 32.min(cents.len());
        let mut chosen: Vec<usize> = vec![0];
        let mut dist2: Vec<f64> = cents.iter().map(|&c| {
            let d = sub(c, cents[0]);
            dot(d, d)
        }).collect();
        while chosen.len() < n_seed {
            let (best, _) = dist2
                .iter()
                .enumerate()
                .fold((0, -1.0), |acc, (i, &d)| if d > acc.1 { (i, d) } else { acc });
            chosen.push(best);
            for (i, &c) in cents.iter().enumerate() {
                let d = sub(c, cents[best]);
                dist2[i] = dist2[i].min(dot(d, d));
            }
        }
        for &ci in &chosen {
            let p = surf.closest(cents[ci]).0;
            r.insert_protected(p, Prov::Face(fid as u32), vec![fid as u32]);
        }
    }

    // ---- seed the interior (graded grid, clear of the surface) -------------
    // Cell refinement alone cannot populate the volume next to a CONVEX
    // surface: the circumcenters of surface-hugging tets lie OUTSIDE the
    // region, are rejected, and the flat tets between on-surface points stay
    // Delaunay (their empty circumspheres bulge outward). Seeding the interior
    // up front -- clear of the surface by the local size -- puts a point inside
    // every such circumsphere, so the flat-surface-sliver class never forms.
    let t_seed = std::time::Instant::now();
    {
        let step = {
            let mut fin = f64::INFINITY;
            for t in plc.triangles.iter().take(64) {
                let c: V3 = std::array::from_fn(|k| {
                    (plc.vertices[t[0] as usize][k]
                        + plc.vertices[t[1] as usize][k]
                        + plc.vertices[t[2] as usize][k])
                        / 3.0
                });
                fin = fin.min(r.h_at(c));
            }
            (0.7 * fin.min(domain.finest().max(r.h_floor))).max(1e-9)
        };
        let span = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        let (nx, ny, nz) = (
            ((span[0] / step).ceil() as i64).max(1),
            ((span[1] / step).ceil() as i64).max(1),
            ((span[2] / step).ceil() as i64).max(1),
        );
        if std::env::var_os("RAPIDMESH_REFINE_TRACE").is_some() {
            eprintln!("CHECKPOINT seeding grid step {step:.6} -> {nx} x {ny} x {nz} cells");
        }
        let ncell = step.max(1e-9);
        let ckey = |p: V3| -> [i64; 3] {
            std::array::from_fn(|k| (p[k] / ncell).floor() as i64)
        };
        let mut grid: DMap<[i64; 3], Vec<V3>> = DMap::default();
        for i in 1..nx {
            for j in 1..ny {
                for k in 1..nz {
                    let p: V3 = [
                        lo[0] + i as f64 * step,
                        lo[1] + j as f64 * step,
                        lo[2] + k as f64 * step,
                    ];
                    let h = r.h_at(p);
                    // Clearance from the surface: a seed within one local size
                    // of the boundary makes tets the facet criteria then fight;
                    // measured against the surface target so a fine interface
                    // under a coarse bulk keeps its exclusion band.
                    if r.bvh.nearest_dist(p) < h || !r.surface_clear(p, 0.8) {
                        continue;
                    }
                    if r.domain.region_at(p) == 0 {
                        continue;
                    }
                    // Crowding: keep clear of already accepted seeds.
                    let r2 = (0.7 * h) * (0.7 * h);
                    let cc = ckey(p);
                    let mut clear = true;
                    'scan: for dx in -1..=1i64 {
                        for dy in -1..=1i64 {
                            for dz in -1..=1i64 {
                                if let Some(v) = grid.get(&[cc[0] + dx, cc[1] + dy, cc[2] + dz]) {
                                    for &q in v {
                                        let d2: f64 =
                                            (0..3).map(|k| (p[k] - q[k]).powi(2)).sum();
                                        if d2 < r2 {
                                            clear = false;
                                            break 'scan;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if !clear {
                        continue;
                    }
                    if r.db.try_insert(p).is_some() {
                        r.prov.push(Prov::Interior);
                        r.ball.push(0.0);
                        r.vfaces.push(Vec::new());
                        grid.entry(cc).or_default().push(p);
                    }
                }
            }
        }
    }
    rmlog::stage("refine.seed", t_seed.elapsed().as_secs_f64());

    // ---- main refinement loop ----------------------------------------------
    // Two phases with a GLOBAL priority: every bad surface facet is refined
    // before any cell circumcenter is inserted. Interleaving them puts interior
    // points next to a still-coarse surface, and each such point later violates
    // the off-face rule -- a churn that over-refines thin regions severalfold.
    let t_refine = std::time::Instant::now();
    for (slot, t) in r.db.tets_with_slots() {
        r.queue.push_back((slot, t));
    }
    let mut cells: VecDeque<(u32, [usize; 4])> = VecDeque::new();
    let mut pierce_rounds = 0usize;
    let trace_hb = std::env::var_os("RAPIDMESH_REFINE_TRACE").is_some();
    let mut hb_iter = 0u64;
    let mut hb_last = std::time::Instant::now();
    loop {
        hb_iter += 1;
        if trace_hb && hb_last.elapsed().as_secs() >= 5 {
            eprintln!(
                "HEARTBEAT iter {hb_iter} points {} fq {} cq {} pierce_rounds {pierce_rounds} facet_ins {} cell_ins {} seg_split {}",
                r.db.len(), r.queue.len(), cells.len(), r.n_facet_ins, r.n_cell_ins, r.n_seg_split
            );
            hb_last = std::time::Instant::now();
        }
        // Phase 1: drain all facet work (insertions re-feed r.queue).
        while let Some((slot, tet)) = r.queue.pop_front() {
            if r.db.tet_at(slot) != Some(tet) {
                continue; // slot died or was reused
            }
            if r.process_facets(slot, tet) {
                if r.db.tet_at(slot) == Some(tet) {
                    r.queue.push_back((slot, tet));
                }
            } else {
                cells.push_back((slot, tet));
            }
        }
        // Phase 2: one cell insertion, then re-drain facets.
        let mut advanced = false;
        while let Some((slot, tet)) = cells.pop_front() {
            if r.db.tet_at(slot) != Some(tet) {
                continue;
            }
            if r.process_cell(slot, tet) {
                if r.db.tet_at(slot) == Some(tet) {
                    cells.push_back((slot, tet));
                }
                advanced = true;
                break;
            }
        }
        if advanced || !r.queue.is_empty() {
            continue;
        }
        // Phase 3: manifold sweep -- runs only when facet and cell work is
        // exhausted; any repair feeds phase 1 again. Hard-capped: a repair can
        // spawn new work next to the coarse duplicate guard, and an uncapped
        // sweep then cascades for minutes on large scenes. Two detectors:
        //  (a) EDGE PARITY of the wall set: a wall edge with exactly one
        //      incident wall facet is a hole rim (the restricted surface is
        //      open there) -- repaired at the edge midpoint on the carrier.
        //      Feature edges are exempt (3+ walls meet there legally).
        //  (b) TET PIERCE: a tet with carrier offsets of both signs crosses a
        //      face outright -- repaired at the piercing edge's crossing.
        pierce_rounds += 1;
        if pierce_rounds > 8 {
            break;
        }
        let mut pierced = false;
        {
            // (a) wall set + edge incidence
            let all: Vec<[usize; 4]> = r.db.tets();
            let mut owners: DMap<[usize; 3], (u32, u32)> = DMap::default();
            for (ti, t) in all.iter().enumerate() {
                for fl in &FACET_LOCAL {
                    let mut f = [t[fl[0]], t[fl[1]], t[fl[2]]];
                    f.sort_unstable();
                    let e = owners.entry(f).or_insert((u32::MAX, u32::MAX));
                    if e.0 == u32::MAX {
                        e.0 = ti as u32;
                    } else {
                        e.1 = ti as u32;
                    }
                }
            }
            let mut edge_walls: DMap<(usize, usize), (u32, u32)> = DMap::default(); // count, face
            for (fkey, &(a, b)) in &owners {
                let t2 = (b != u32::MAX).then(|| &all[b as usize]);
                let Some(wf) = r.facet_wall_face(&all[a as usize], t2, *fkey) else {
                    continue;
                };
                for k in 0..3 {
                    let (x, y) = (fkey[k], fkey[(k + 1) % 3]);
                    let e = edge_walls.entry((x.min(y), x.max(y))).or_insert((0, wf));
                    e.0 += 1;
                }
            }
            let is_feature = |v: usize| matches!(r.prov[v], Prov::Corner | Prov::Edge(_));
            let rim_edges: Vec<((usize, usize), u32)> = edge_walls
                .iter()
                .filter(|(&(a, b), &(cnt, _))| cnt == 1 && !(is_feature(a) && is_feature(b)))
                .map(|(&e, &(_, wf))| (e, wf))
                .collect();
            for ((a, b), wf) in rim_edges {
                let surf = r.brep.surface(r.brep.faces[wf as usize].surface);
                let x = surf.closest(mid3(r.pos(a), r.pos(b))).0;
                if r.on_trimmed_face(wf, x)
                    && r.insert_protected_with(x, Prov::Face(wf), vec![wf], true)
                {
                    pierced = true;
                }
            }
        }
        let mut blocked = 0u64;
        // nearest-vertex accelerator for the snap fallback (positions are
        // frozen within one sweep round)
        let pts_now: Vec<V3> = (0..r.db.len()).map(|v| r.db.approx_point(v)).collect();
        let tree = crate::spatial::Octree::build(&pts_now);
        for (slot, tet) in r.db.tets_with_slots() {
            if r.db.tet_at(slot) != Some(tet) {
                continue;
            }
            if let Some((f, x)) = r.pierce_repair(tet) {
                if r.insert_protected_with(x, Prov::Face(f), vec![f], true) {
                    pierced = true;
                    continue;
                }
                // Blocked (usually the duplicate guard): the blocker is a point
                // ALMOST on the wall. If it is interior, snap it onto the
                // carrier -- the final rebuild takes the moved position, the
                // wall closes, and the volume balance with it. Features are
                // never snapped (they are the authority; the ball path handles
                // them).
                if let Some(v) = tree.nearest(x) {
                    let v = v as usize;
                    if matches!(r.prov[v], Prov::Interior)
                        && dist(pts_now[v], x) <= DUP_FRAC * r.h_at(x)
                        && !r.snaps.contains_key(&v)
                    {
                        r.snaps.insert(v, (x, f));
                        pierced = true;
                        continue;
                    }
                }
                blocked += 1;
            }
        }
        if !pierced {
            if blocked > 0 && std::env::var_os("RAPIDMESH_REFINE_TRACE").is_some() {
                eprintln!("PIERCE {} unrepairable (insertion blocked)", blocked);
            }
            break;
        }
        if r.snaps.len() > 4096 {
            break; // runaway backstop
        }
    }
    rmlog::stat("refine.points", r.db.len() as f64);
    rmlog::stage("refine.loop", t_refine.elapsed().as_secs_f64());

    // ---- ODT (Lloyd) relaxation of the interior ---------------------------
    // Delaunay refinement bounds radius-edge, not dihedral: the raw point set
    // is sliver-heavy. Relaxing the INTERIOR points toward their ODT positions
    // (surface + feature points frozen on their carriers) is what makes the
    // volume near-isotropic -- the proven cvt.rs recipe, now running on the
    // refinement's guaranteed-conforming point set. The DT is rebuilt per pass
    // from a frozen base (surface points inserted once, interiors re-inserted).
    let t_lloyd = std::time::Instant::now();
    {
        // Pierce snaps from the refinement loop: apply them via a rebuild so
        // the snapped points act as FIXED face points during the relaxation.
        if !r.snaps.is_empty() {
            let mut cur: Vec<V3> = (0..r.db.len()).map(|v| r.pos(v)).collect();
            r.rebuild_points(lo, hi, &mut cur, &[]);
        }
        let n_all = r.db.len();
        let fixed: Vec<V3> = (0..n_all)
            .filter(|&v| !matches!(r.prov[v], Prov::Interior))
            .map(|v| r.pos(v))
            .collect();
        let mut interior: Vec<V3> = (0..n_all)
            .filter(|&v| matches!(r.prov[v], Prov::Interior))
            .map(|v| r.pos(v))
            .collect();
        if !interior.is_empty() {
            let mut base = DelaunayBuilder::enclosing(lo, hi);
            let mut base_ok = 0usize;
            for &p in &fixed {
                if base.try_insert(p).is_some() {
                    base_ok += 1;
                }
            }
            let _ = base_ok;
            for _pass in 0..6 {
                let mut db = base.clone();
                let mut idx: Vec<usize> = Vec::with_capacity(interior.len());
                for (i, &p) in interior.iter().enumerate() {
                    if db.try_insert(p).is_some() {
                        idx.push(i);
                    }
                }
                // ODT accumulation over all real tets.
                let all: Vec<V3> = {
                    let mut v = fixed.clone();
                    v.extend_from_slice(&interior);
                    v
                };
                // builder index -> position index: fixed first (insertion
                // order), then the accepted interiors in `idx` order.
                let mut b2p: Vec<usize> = Vec::with_capacity(db.len());
                for i in 0..fixed.len() {
                    b2p.push(i);
                }
                for &i in &idx {
                    b2p.push(fixed.len() + i);
                }
                let mut num = vec![[0.0f64; 3]; all.len()];
                let mut den = vec![0.0f64; all.len()];
                for t in db.tets() {
                    let pi: [usize; 4] = std::array::from_fn(|k| b2p[t[k]]);
                    let p: [V3; 4] = std::array::from_fn(|k| all[pi[k]]);
                    let e1 = sub(p[1], p[0]);
                    let e2 = sub(p[2], p[0]);
                    let e3 = sub(p[3], p[0]);
                    let w = dot(e1, cross(e2, e3)).abs();
                    let sum4: V3 =
                        std::array::from_fn(|k| p[0][k] + p[1][k] + p[2][k] + p[3][k]);
                    for &i in &pi {
                        for k in 0..3 {
                            num[i][k] += w * (sum4[k] - all[i][k]);
                        }
                        den[i] += w;
                    }
                }
                let mut max_move = 0.0f64;
                for (i, p) in interior.iter_mut().enumerate() {
                    let gi = fixed.len() + i;
                    if den[gi] <= 0.0 {
                        continue;
                    }
                    let tgt: V3 = std::array::from_fn(|k| num[gi][k] / (3.0 * den[gi]));
                    // stay inside, out of protecting balls, clear of the
                    // surface (a hugger pierces the finer wall sampling)
                    if r.in_ball(tgt)
                        || r.domain.region_at(tgt) == 0
                        || !r.surface_clear(tgt, 0.5)
                    {
                        continue;
                    }
                    max_move = max_move.max(dist(*p, tgt));
                    *p = tgt;
                }
                // Adaptive insertion at the COARSEST spots (the cvt.rs recipe):
                // a tet whose longest edge exceeds the local target marks a
                // hole in the interior distribution -- THIN regions (a torus
                // tube ~2-3 h across) get almost no grid seeds (the surface
                // clearance eats them), the flat surface slivers there survive,
                // and only points injected between the walls break them. The
                // next pass relaxes the newcomers in.
                let mut inserted = 0usize;
                for t in db.tets() {
                    let pi: [usize; 4] = std::array::from_fn(|k| b2p[t[k]]);
                    let p: [V3; 4] = std::array::from_fn(|k| all[pi[k]]);
                    let (mut best_len, mut mid) = (0.0f64, [0.0f64; 3]);
                    for a in 0..4 {
                        for b in (a + 1)..4 {
                            let l = dist(p[a], p[b]);
                            if l > best_len {
                                best_len = l;
                                mid = mid3(p[a], p[b]);
                            }
                        }
                    }
                    let h = r.h_at(mid);
                    if best_len <= 1.4 * h {
                        continue;
                    }
                    if r.in_ball(mid)
                        || r.domain.region_at(mid) == 0
                        || !r.surface_clear(mid, 0.5)
                    {
                        continue;
                    }
                    // clearance from the existing points of THIS tet suffices
                    // as a cheap crowding guard (the midpoint of the longest
                    // edge is far from all four by construction).
                    interior.push(mid);
                    inserted += 1;
                }
                if inserted == 0 && max_move < 0.02 * r.h_floor {
                    break;
                }
            }
            // Write the relaxed positions back into the refiner's DT: fixed
            // vertices keep their position, interiors take their relaxed
            // one, adaptive insertions append as new interior points.
            let mut new_pos: Vec<V3> = (0..n_all).map(|v| r.pos(v)).collect();
            let mut it = interior.iter();
            for (v, np) in new_pos.iter_mut().enumerate() {
                if matches!(r.prov[v], Prov::Interior) {
                    *np = *it.next().expect("one relaxed position per interior");
                }
            }
            let extra: Vec<V3> = it.copied().collect();
            let _ = &fixed;
            r.rebuild_points(lo, hi, &mut new_pos, &extra);
        }
        // Post-Lloyd conformity: the relaxation moves interior points AFTER
        // the last manifold sweep, and a moved point can pull an edge back
        // across a wall -- piercing tets the refinement loop never saw (the
        // extraction census found dozens; their misassigned half-volumes are
        // a ~1e-2 relative region bias). Each cycle: (1) insertion rounds
        // put the carrier crossing of every piercer into the DT until that
        // runs dry, (2) piercers whose repair point is BLOCKED (duplicate
        // guard, segment encroachment at the sizing floor) instead SNAP
        // their closest interior vertex onto the carrier, applied by a
        // rebuild. The snap changes local geometry, so the next cycle
        // re-checks; quality repair of the snapped neighbourhood is the
        // optimizer's job.
        let trace = std::env::var_os("RAPIDMESH_REFINE_TRACE").is_some();
        let t_post = std::time::Instant::now();
        // Snap reach as a fraction of local h. Conservative first (0.4 h:
        // gentle pulls only); when a cycle can neither insert nor snap but
        // piercers REMAIN, escalate once to 0.75 h -- a strongly pulled
        // vertex costs local quality (the optimizer's job to smooth), while
        // a surviving piercer costs CORRECTNESS: its centroid-side fallback
        // misassigns real volume across the wall (the exact-volume bias this
        // block exists to prevent).
        let mut snap_cap = 0.4;
        'cycles: for cycle in 0..6 {
            // (1) piercer insertion rounds until dry
            for _round in 0..8 {
                let mut fixed_any = false;
                for (slot, tet) in r.db.tets_with_slots() {
                    if r.db.tet_at(slot) != Some(tet) {
                        continue;
                    }
                    if let Some((f, x)) = r.pierce_repair(tet) {
                        if r.insert_protected_with(x, Prov::Face(f), vec![f], true) {
                            fixed_any = true;
                        }
                    }
                }
                if !fixed_any {
                    break;
                }
            }
            // Blocked piercers: pull the interior vertex nearest to the
            // carrier onto it. Only near-carrier vertices move (a far pull
            // would fold the neighbourhood); the rest stay for the
            // classification's centroid fallback.
            let mut snapped = 0usize;
            let mut left = 0usize;
            for (slot, tet) in r.db.tets_with_slots() {
                if r.db.tet_at(slot) != Some(tet) {
                    continue;
                }
                let Some((f, _)) = r.pierce_repair(tet) else {
                    continue;
                };
                let surf = r.brep.surface(r.brep.faces[f as usize].surface);
                let mut best: Option<(f64, usize)> = None;
                for &v in &tet {
                    if !matches!(r.prov[v], Prov::Interior) || r.snaps.contains_key(&v) {
                        continue;
                    }
                    let d = signed_offset(surf, r.pos(v)).abs();
                    if best.is_none_or(|(bd, _)| d < bd) {
                        best = Some((d, v));
                    }
                }
                // Snap only onto the TRIMMED face: `closest` projects onto
                // the unbounded carrier, and near a corner the footpoint can
                // land beyond the trim edge -- a phantom face point in the
                // neighbouring region that itself pierces the adjacent
                // face's plane (self-inflicted piercers at box corners).
                match best {
                    Some((d, v))
                        if d <= snap_cap * r.h_at(r.pos(v))
                            && r.on_trimmed_face(f, surf.closest(r.pos(v)).0) =>
                    {
                        let x = surf.closest(r.pos(v)).0;
                        r.snaps.insert(v, (x, f));
                        snapped += 1;
                    }
                    _ => {
                        left += 1;
                        if trace {
                            let provs: Vec<String> =
                                tet.iter().map(|&v| format!("{:?}", r.prov[v])).collect();
                            let offs: Vec<f64> = tet
                                .iter()
                                .map(|&v| signed_offset(surf, r.pos(v)))
                                .collect();
                            let ps: Vec<String> = tet
                                .iter()
                                .map(|&v| {
                                    let p = r.pos(v);
                                    format!("({:.3},{:.3},{:.3})", p[0], p[1], p[2])
                                })
                                .collect();
                            eprintln!(
                                "POSTLLOYD left: f {f} best {best:?} provs {provs:?} offs {offs:?} pos {ps:?}"
                            );
                        }
                    }
                }
            }
            if trace {
                eprintln!(
                    "POSTLLOYD cycle {cycle}: snapped {snapped} left {left} (cap {snap_cap})"
                );
            }
            if snapped == 0 {
                if left > 0 && snap_cap < 0.75 {
                    snap_cap = 0.75;
                    continue;
                }
                break 'cycles;
            }
            let mut cur: Vec<V3> = (0..r.db.len()).map(|v| r.pos(v)).collect();
            r.rebuild_points(lo, hi, &mut cur, &[]);
        }
        rmlog::stage("refine.postlloyd", t_post.elapsed().as_secs_f64());
    }
    rmlog::stage("refine.lloyd", t_lloyd.elapsed().as_secs_f64());
    if std::env::var_os("RAPIDMESH_REFINE_TRACE").is_some() {
        eprintln!(
            "REFINE facet_ins {} (size {} shape {} offface {}) cell_ins {} seg_split {} ins_rejected {} ball_none {} cross_found {} member_reject {} cc_none {} trace_evals {} guard_trips {}",
            r.n_facet_ins, r.n_bad_size, r.n_bad_shape, r.n_bad_offface,
            r.n_cell_ins, r.n_seg_split, r.n_ins_rejected, r.n_ball_none,
            r.n_cross_found.load(std::sync::atomic::Ordering::Relaxed), r.n_member_reject.load(std::sync::atomic::Ordering::Relaxed), r.n_cc_none.load(std::sync::atomic::Ordering::Relaxed),
            CC_EVALS.load(std::sync::atomic::Ordering::Relaxed), CC_GUARD_TRIPS.load(std::sync::atomic::Ordering::Relaxed)
        );
    }

    // ---- extraction ---------------------------------------------------------
    let points: Vec<V3> = (0..r.db.len()).map(|v| r.db.approx_point(v)).collect();
    let all_tets = r.db.tets();
    if std::env::var_os("RAPIDMESH_REFINE_TRACE").is_some() {
        // final-state pierce census: piercers left over AFTER the last sweep
        // (Lloyd moves can create new ones the loop never saw)
        let mut n_pierce = 0usize;
        let mut vol = 0.0f64;
        for t in &all_tets {
            if r.pierce_repair(*t).is_some() {
                n_pierce += 1;
                let p: [V3; 4] = std::array::from_fn(|k| points[t[k]]);
                vol += dot(sub(p[1], p[0]), cross(sub(p[2], p[0]), sub(p[3], p[0]))).abs() / 6.0;
            }
        }
        eprintln!("EXTRACT piercing tets {n_pierce} (volume {vol:.6})");
    }
    let mut tets: Vec<[usize; 4]> = Vec::new();
    let mut tet_regions: Vec<RegionTag> = Vec::new();
    // Region per tet by FLOOD FILL over non-wall facets: two tets sharing a
    // facet whose Voronoi dual does NOT cross the surface belong to the same
    // region by the restricted-Delaunay definition, so walls exist exactly at
    // the surface. A per-tet point query instead wavers inside the sagitta
    // band and wedges flat surface slivers between two boundary faces --
    // unreachable for every optimizer operation. Each connected component is
    // labelled ONCE, at its deepest tet (farthest from the surface), where the
    // point query is unambiguous.
    let t_class = std::time::Instant::now();
    let mut owners: DMap<[usize; 3], (u32, u32)> = DMap::default();
    for (ti, t) in all_tets.iter().enumerate() {
        for fl in &FACET_LOCAL {
            let mut f = [t[fl[0]], t[fl[1]], t[fl[2]]];
            f.sort_unstable();
            let e = owners.entry(f).or_insert((u32::MAX, u32::MAX));
            if e.0 == u32::MAX {
                e.0 = ti as u32;
            } else {
                e.1 = ti as u32;
            }
        }
    }
    let mut uf: Vec<u32> = (0..all_tets.len() as u32).collect();
    fn find(uf: &mut [u32], mut x: u32) -> u32 {
        while uf[x as usize] != x {
            uf[x as usize] = uf[uf[x as usize] as usize];
            x = uf[x as usize];
        }
        x
    }
    let mut n_walls = 0u64;
    // RAPIDMESH_DEBUG_LEAK="x0,y0,z0,x1,y1,z1": report non-wall facets whose
    // two tets straddle this box's boundary (the classification-leak tracer).
    let dbg_box: Option<[f64; 6]> = std::env::var("RAPIDMESH_DEBUG_LEAK")
        .ok()
        .and_then(|s| {
            let v: Vec<f64> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            (v.len() == 6).then(|| std::array::from_fn(|i| v[i]))
        });
    let dbg_leak = dbg_box.is_some();
    let in_void = |c: &V3| -> bool {
        let b = dbg_box.unwrap_or([0.0; 6]);
        c[0] > b[0] && c[0] < b[3] && c[1] > b[1] && c[1] < b[4] && c[2] > b[2] && c[2] < b[5]
    };
    let ctr = |t: &[usize; 4]| -> V3 {
        std::array::from_fn(|k| {
            0.25 * (points[t[0]][k] + points[t[1]][k] + points[t[2]][k] + points[t[3]][k])
        })
    };
    // Non-wall internal facets, kept for the bridge-guard watershed below.
    // The wall predicate (carrier oracles per facet) is read-only on the
    // refiner, so it is evaluated in PARALLEL over the fixed facet list; the
    // union-find replay below is sequential and order-identical.
    let t_walls = std::time::Instant::now();
    let mut nonwall: Vec<(u32, u32)> = Vec::new();
    let internal: Vec<([usize; 3], u32, u32)> = owners
        .iter()
        .filter(|&(_, &(_, b))| b != u32::MAX)
        .map(|(fkey, &(a, b))| (*fkey, a, b))
        .collect();
    let is_wall: Vec<bool> = {
        use rayon::prelude::*;
        internal
            .par_iter()
            .map(|&(fkey, a, b)| {
                r.facet_is_wall(&all_tets[a as usize], Some(&all_tets[b as usize]), fkey)
            })
            .collect()
    };
    for (&(fkey, a, b), &wall) in internal.iter().zip(&is_wall) {
        if wall {
            n_walls += 1;
            continue;
        }
        if dbg_leak {
            let (ca, cb) = (ctr(&all_tets[a as usize]), ctr(&all_tets[b as usize]));
            if in_void(&ca) != in_void(&cb) {
                let fp: [V3; 3] = std::array::from_fn(|k| points[fkey[k]]);
                eprintln!(
                    "LEAK facet ({:.3?} {:.3?} {:.3?}) ca ({:.3},{:.3},{:.3}) cb ({:.3},{:.3},{:.3})",
                    fp[0], fp[1], fp[2], ca[0], ca[1], ca[2], cb[0], cb[1], cb[2]
                );
            }
        }
        nonwall.push((a, b));
        let (ra, rb) = (find(&mut uf, a), find(&mut uf, b));
        if ra != rb {
            uf[ra.max(rb) as usize] = ra.min(rb);
        }
    }
    rmlog::stage("refine.walls", t_walls.elapsed().as_secs_f64());
    // Deepest tet per component -> one unambiguous label.
    let centroid_of = |t: &[usize; 4]| -> V3 {
        std::array::from_fn(|k| {
            0.25 * (points[t[0]][k] + points[t[1]][k] + points[t[2]][k] + points[t[3]][k])
        })
    };
    // Depth (surface distance) per tet in parallel -- pure BVH queries over
    // the fixed tet list; the component fold below stays sequential.
    let depths: Vec<f64> = {
        use rayon::prelude::*;
        all_tets
            .par_iter()
            .map(|t| r.bvh.nearest_dist(centroid_of(t)))
            .collect()
    };
    let mut comp_best: DMap<u32, (f64, u32)> = DMap::default();
    for (ti, d) in depths.iter().enumerate() {
        let root = find(&mut uf, ti as u32);
        let e = comp_best.entry(root).or_insert((-1.0, 0));
        if *d > e.0 {
            *e = (*d, ti as u32);
        }
    }
    let mut comp_region: DMap<u32, u32> = DMap::default();
    for (&root, &(_, ti)) in &comp_best {
        comp_region.insert(root, r.region_at_c(centroid_of(&all_tets[ti as usize])));
    }
    // BRIDGE GUARD: one leaked facet chain merges inside and outside into a
    // single component, and the single-anchor label then wipes out a whole
    // region (torus tangential belt: chains of on-carrier sliver tets whose
    // centroid side is noise classified the body as ONE component, region
    // volume 0). Detect: sample the DEEP tets of each component; if their
    // domain queries disagree, the component contains a bridge. Repair:
    // watershed re-fill from the deep tets outward over non-wall adjacency,
    // deepest first -- each true region floods up to the ambiguous band and
    // the bridge splits between them. Consistent components are untouched,
    // so the exact-volume path stays exact.
    let mut region_override: DMap<u32, u32> = DMap::default();
    {
        let mut comp_tets: DMap<u32, Vec<u32>> = DMap::default();
        for ti in 0..all_tets.len() as u32 {
            comp_tets.entry(find(&mut uf, ti)).or_default().push(ti);
        }
        let mut bridged: Vec<(u32, Vec<(u32, u32)>)> = Vec::new(); // root -> seeds
        for (&root, tids) in &comp_tets {
            let mut deep: Vec<u32> = tids
                .iter()
                .copied()
                .filter(|&ti| {
                    let c = centroid_of(&all_tets[ti as usize]);
                    depths[ti as usize] >= 0.3 * r.h_at(c)
                })
                .collect();
            if deep.len() < 2 {
                continue;
            }
            deep.sort_by(|&x, &y| depths[y as usize].total_cmp(&depths[x as usize]));
            let step = (deep.len() / 16).max(1);
            let seeds: Vec<(u32, u32)> = deep
                .iter()
                .step_by(step)
                .take(16)
                .map(|&ti| (ti, r.region_at_c(centroid_of(&all_tets[ti as usize]))))
                .collect();
            if seeds.iter().any(|&(_, rg)| rg != seeds[0].1) {
                bridged.push((root, seeds));
            }
        }
        if !bridged.is_empty() {
            // adjacency restricted to bridged components
            let in_bridged: DMap<u32, ()> = bridged
                .iter()
                .flat_map(|(root, _)| comp_tets[root].iter().map(|&ti| (ti, ())))
                .collect();
            let mut adj: DMap<u32, Vec<u32>> = DMap::default();
            for &(a, b) in &nonwall {
                if in_bridged.contains_key(&a) {
                    adj.entry(a).or_default().push(b);
                    adj.entry(b).or_default().push(a);
                }
            }
            for (root, seeds) in &bridged {
                if std::env::var_os("RAPIDMESH_REFINE_TRACE").is_some() {
                    eprintln!(
                        "CLASSIFY bridge in component of {} tets, watershed from {} seeds",
                        comp_tets[root].len(),
                        seeds.len()
                    );
                }
                let mut heap: std::collections::BinaryHeap<(u64, u32, u32)> =
                    std::collections::BinaryHeap::new();
                for &(ti, rg) in seeds {
                    heap.push((depths[ti as usize].to_bits(), ti, rg));
                }
                while let Some((_, ti, rg)) = heap.pop() {
                    if region_override.contains_key(&ti) {
                        continue;
                    }
                    region_override.insert(ti, rg);
                    if let Some(nbs) = adj.get(&ti) {
                        for &nb in nbs.clone().iter() {
                            if !region_override.contains_key(&nb) {
                                heap.push((depths[nb as usize].to_bits(), nb, rg));
                            }
                        }
                    }
                }
            }
        }
    }
    if std::env::var_os("RAPIDMESH_REFINE_TRACE").is_some() {
        let mut sizes: DMap<u32, usize> = DMap::default();
        for ti in 0..all_tets.len() as u32 {
            *sizes.entry(find(&mut uf, ti)).or_insert(0) += 1;
        }
        let mut info: Vec<(usize, u32, V3)> = comp_best
            .iter()
            .map(|(&root, &(_, ti))| {
                (sizes[&root], comp_region[&root], centroid_of(&all_tets[ti as usize]))
            })
            .collect();
        info.sort_by(|a, b| b.0.cmp(&a.0));
        eprintln!("CLASSIFY walls {} components {}:", n_walls, comp_best.len());
        for (n, reg, c) in info.iter().take(8) {
            eprintln!("  comp size {n} region {reg} anchor ({:.3}, {:.3}, {:.3})", c[0], c[1], c[2]);
        }
    }
    let mut region_of: Vec<u32> = Vec::with_capacity(all_tets.len());
    for ti in 0..all_tets.len() as u32 {
        if let Some(&rg) = region_override.get(&ti) {
            region_of.push(rg);
            continue;
        }
        let root = find(&mut uf, ti);
        region_of.push(*comp_region.get(&root).unwrap_or(&0));
    }
    // SPECKLE SMOOTHING: an on-carrier band tet takes its side from the
    // centroid's offset SIGN, which is noise at sagitta scale -- adjacent
    // band tets alternate regions, the interface gets speckled, and each
    // speckle rim is a 4-face edge of one region (the per-region
    // non-manifold pinches clustered on curved material interfaces like a
    // via barrel). The side of a ~zero-volume band tet is arbitrary but must
    // be COHERENT: flip any band tet whose facet neighbours hold a 3+
    // majority of one other region, to a fixpoint. Real region tets (deep
    // anchors) never flip, so exact volumes stay exact.
    {
        let mut nbrs: Vec<[u32; 4]> = vec![[u32::MAX; 4]; all_tets.len()];
        let mut push_nb = |ti: u32, nb: u32| {
            for s in nbrs[ti as usize].iter_mut() {
                if *s == u32::MAX {
                    *s = nb;
                    return;
                }
            }
        };
        for (_, &(a, b)) in &owners {
            if b == u32::MAX {
                continue;
            }
            push_nb(a, b);
            push_nb(b, a);
        }
        let mut flipped = 0usize;
        for _round in 0..8 {
            let mut changed = false;
            for ti in 0..all_tets.len() {
                let c = centroid_of(&all_tets[ti]);
                if depths[ti] > 0.05 * r.h_at(c) {
                    continue; // not a band tet: its label is trustworthy
                }
                let mut votes: [(u32, u8); 4] = [(u32::MAX, 0); 4];
                for &nb in &nbrs[ti] {
                    if nb == u32::MAX {
                        continue;
                    }
                    let rg = region_of[nb as usize];
                    for v in votes.iter_mut() {
                        if v.0 == rg {
                            v.1 += 1;
                            break;
                        }
                        if v.0 == u32::MAX {
                            *v = (rg, 1);
                            break;
                        }
                    }
                }
                if let Some(&(rg, cnt)) = votes.iter().max_by_key(|v| v.1) {
                    if cnt >= 3 && rg != u32::MAX && rg != region_of[ti] {
                        region_of[ti] = rg;
                        changed = true;
                        flipped += 1;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        if flipped > 0 && std::env::var_os("RAPIDMESH_REFINE_TRACE").is_some() {
            eprintln!("CLASSIFY speckle smoothing flipped {flipped} band tets");
        }
    }
    if std::env::var_os("RAPIDMESH_REFINE_TRACE").is_some() {
        // per-region f64 volume: the balance sheet of the classification
        let mut rv: DMap<u32, f64> = DMap::default();
        for (ti, t) in all_tets.iter().enumerate() {
            let p: [V3; 4] = std::array::from_fn(|k| points[t[k]]);
            let v = dot(sub(p[1], p[0]), cross(sub(p[2], p[0]), sub(p[3], p[0]))).abs() / 6.0;
            *rv.entry(region_of[ti]).or_insert(0.0) += v;
        }
        let mut rvv: Vec<(u32, f64)> = rv.into_iter().collect();
        rvv.sort_unstable_by_key(|e| e.0);
        eprintln!(
            "CLASSIFY region volumes: {rvv:?} (piercing side-fallbacks {})",
            r.n_side_pierce.load(std::sync::atomic::Ordering::Relaxed)
        );
    }
    rmlog::stage("refine.classify", t_class.elapsed().as_secs_f64());
    for (t, &reg) in all_tets.iter().zip(&region_of) {
        if reg != 0 {
            tets.push(*t);
            tet_regions.push(RegionTag(reg));
        }
    }

    // Boundary faces: facets between differing regions, tagged by the nearest
    // PLC facet (its face tag / surface / patch = B-rep face id).
    let mut face_owners: DMap<[usize; 3], (u32, u32)> = DMap::default(); // -> (count, region of last owner)
    let mut face_pair: DMap<[usize; 3], [u32; 2]> = DMap::default();
    for (t, &reg) in all_tets.iter().zip(&region_of) {
        for fl in &FACET_LOCAL {
            let mut f = [t[fl[0]], t[fl[1]], t[fl[2]]];
            f.sort_unstable();
            let e = face_owners.entry(f).or_insert((0, 0));
            e.0 += 1;
            let pair = face_pair.entry(f).or_insert([u32::MAX, u32::MAX]);
            if pair[0] == u32::MAX {
                pair[0] = reg;
            } else {
                pair[1] = reg;
            }
            e.1 = reg;
        }
    }
    let mut faces: Vec<SurfaceFace> = Vec::new();
    for (key, pair) in &face_pair {
        let (ra, rb) = (pair[0], if pair[1] == u32::MAX { 0 } else { pair[1] });
        if ra == rb {
            continue;
        }
        let c: V3 = std::array::from_fn(|k| {
            (points[key[0]][k] + points[key[1]][k] + points[key[2]][k]) / 3.0
        });
        let (surface, face_tag, patch) = match r.bvh.nearest_index(c) {
            Some((fi, _)) => {
                let bf = r.facet_face[fi];
                if bf != u32::MAX {
                    let face = &brep.faces[bf as usize];
                    (face.plc_surface, face.face_tag, bf)
                } else {
                    (plc.surface_refs[fi].0, plc.face_tags[fi], u32::MAX)
                }
            }
            None => (0, FaceTag(0), u32::MAX),
        };
        faces.push(SurfaceFace {
            tri: *key,
            face_tag,
            regions: [RegionTag(ra.min(rb)), RegionTag(ra.max(rb))],
            patch,
            surface,
        });
    }

    let point_size: Vec<f64> = points.iter().map(|&p| domain.h_at(p)).collect();
    let mesh = TetMesh {
        points,
        tets,
        tet_regions,
        faces,
        surfaces: plc.surfaces.clone(),
        surface_owners: plc.surface_owners.clone(),
        plc_points: brep.vertices.len(),
        point_size,
    };
    rmlog::stat("refine.tets", mesh.tets.len() as f64);
    rmlog::stage("refine.total", t_start.elapsed().as_secs_f64());
    mesh
}

#[cfg(test)]
mod oracle_tests {
    use super::*;
    use rapidmesh_geom::SurfaceKind;

    #[test]
    fn torus_segment_crossings() {
        let kind = SurfaceKind::Torus {
            center: [0.0, 0.0, 0.0],
            axis: [0.0, 0.0, 1.0],
            major_radius: 1.2,
            minor_radius: 0.4,
        };
        let surf = Surface::from_kind(&kind, &[]);
        // vertical line through the tube at x = 1.2: crosses at z = -0.4, +0.4
        let c = carrier_crossings(&surf, [1.2, 0.0, -2.0], [1.2, 0.0, 2.0]);
        assert_eq!(c.len(), 2, "tube crossed twice, got {c:?}");
        // horizontal line through the hole: crosses tube walls 4 times
        let c = carrier_crossings(&surf, [-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]);
        assert_eq!(c.len(), 4, "diameter crosses 4 walls, got {c:?}");
        // line far away: no crossing
        let c = carrier_crossings(&surf, [-3.0, 0.0, 3.0], [3.0, 0.0, 3.0]);
        assert!(c.is_empty(), "far line must not cross, got {c:?}");
    }

    /// Dense sign-sampled reference crossings (10k samples + bisection):
    /// slow but assumption-free, the ground truth for the analytic solvers.
    fn reference_crossings(surf: &Surface, a: V3, b: V3) -> Vec<f64> {
        let at = |t: f64| -> V3 { std::array::from_fn(|k| a[k] + t * (b[k] - a[k])) };
        let n = 10_000;
        let mut out = Vec::new();
        let mut s_prev = signed_offset(surf, a);
        for i in 1..=n {
            let t2 = i as f64 / n as f64;
            let s2 = signed_offset(surf, at(t2));
            if s_prev != 0.0 && s2 != 0.0 && s_prev.signum() != s2.signum() {
                let (mut lo, mut hi, lo_neg) = ((i - 1) as f64 / n as f64, t2, s_prev < 0.0);
                for _ in 0..50 {
                    let m = 0.5 * (lo + hi);
                    if (signed_offset(surf, at(m)) < 0.0) == lo_neg {
                        lo = m;
                    } else {
                        hi = m;
                    }
                }
                out.push(0.5 * (lo + hi));
            }
            s_prev = s2;
        }
        out
    }

    fn xorshift(s: &mut u64) -> f64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s >> 11) as f64 / (1u64 << 53) as f64
    }

    /// The analytic fast paths must reproduce the sign-sampled reference on
    /// random segments: every reference crossing matched within 1e-6, no
    /// unmatched analytic extras (mismatches closer than the sampler's
    /// resolution would be double crossings inside one sample step, which
    /// the reference cannot see -- excluded by the 1e-4 pair filter).
    #[test]
    fn analytic_crossings_match_reference() {
        let mut surfs: Vec<Surface> = vec![
            Surface::from_kind(
                &SurfaceKind::Sphere { center: [0.1, -0.2, 0.3], radius: 0.9 },
                &[],
            ),
            Surface::from_kind(
                &SurfaceKind::Cylinder {
                    center: [0.0, 0.1, 0.0],
                    axis: [0.3, 0.2, 0.93],
                    radius: 0.7,
                },
                &[],
            ),
            Surface::from_kind(
                &SurfaceKind::Plane,
                &[[0.0, 0.0, 0.2], [1.0, 0.0, 0.3], [1.0, 1.0, 0.5], [0.0, 1.0, 0.4]],
            ),
        ];
        // helix tube (the coil carrier class)
        let pts: Vec<V3> = (0..=140)
            .map(|i| {
                let t = i as f64 / 24.0 * std::f64::consts::TAU;
                [0.8 * t.cos(), 0.8 * t.sin(), 0.35 * i as f64 / 24.0]
            })
            .collect();
        surfs.push(Surface::Tube {
            path: std::sync::Arc::new(rapidmesh_geom::TubePath::new(pts)),
            radius: 0.15,
        });
        let mut seed = 987654321u64;
        for (si, surf) in surfs.iter().enumerate() {
            for case in 0..60 {
                let mut r = || 5.0 * xorshift(&mut seed) - 2.5;
                let (a, b): (V3, V3) = ([r(), r(), r()], [r(), r(), r()]);
                let want = reference_crossings(surf, a, b);
                let got = carrier_crossings(surf, a, b);
                // Double crossings within one sampler step are invisible to
                // the reference; skip cases with near-tangent pairs.
                let tangent_pair = got.windows(2).any(|w| w[1] - w[0] < 1e-4)
                    || want.windows(2).any(|w| w[1] - w[0] < 1e-4);
                if tangent_pair {
                    continue;
                }
                assert_eq!(
                    got.len(),
                    want.len(),
                    "surf {si} case {case}: got {got:?} want {want:?} (a {a:?} b {b:?})"
                );
                for (g, w) in got.iter().zip(&want) {
                    assert!(
                        (g - w).abs() < 1e-6,
                        "surf {si} case {case}: got {got:?} want {want:?}"
                    );
                }
            }
        }
    }
}

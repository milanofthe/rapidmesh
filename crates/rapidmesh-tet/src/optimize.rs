//! Post-pass mesh quality optimization: smart Laplacian smoothing of free
//! interior vertices, surface re-tiling and in-surface smoothing, 2-3 flips
//! and generalized edge removal, all constrained-aware.
//!
//! Constrained faces are never flipped away and their edges never removed,
//! so conformity is an invariant of this pass. Surface vertices move only
//! WITHIN their geometry: in their plane patch, on their analytic curved
//! surface, or along their (straight or curved) feature curve. Plane-only
//! moves preserve the exact region volumes; fidelity snaps onto curved
//! analytic geometry deliberately move the mesh from the faceted PLC
//! approximation toward the true surface (accepted on validity, not
//! quality: lying on the geometry is a constraint). All other operations
//! are gated on improving the local minimum dihedral angle and on exact
//! positive orientation of the replacement tets, and the pass terminates
//! when nothing improves.

use crate::conform::TetMesh;
use rapidmesh_exact::{collinear, orient2d, Axis, Point3, Sign};
use rapidmesh_geom::SurfaceKind;
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;

type DState = BuildHasherDefault<rustc_hash::FxHasher>;
type DMap<K, V> = HashMap<K, V, DState>;
type DSet<T> = HashSet<T, DState>;

/// Strict-improvement epsilon on the comparison quality scale (minus max
/// dihedral cosine, in [-1, 1]): guards float noise and accept/reject
/// cycling.
const QUALITY_EPS: f64 = 1e-12;
/// Smoothing moves below this fraction of the local edge length are
/// discarded: their quality effect is immeasurable, but accepting them keeps
/// neighborhoods dirty for dozens of micro-converging passes.
const MIN_REL_MOVE: f64 = 1e-3;
/// Largest tet ring handled by edge removal. Sliver fans hub around a vertex
/// with a dozen incident tets; the Klincsek DP is O(k^3), so a generous cap
/// stays cheap.
const MAX_RING: usize = 12;
/// Radius-edge ratio (circumradius over shortest edge) of a tet on explicit
/// coordinates; `MAX` for degenerate tets. Guards vertex insertion against
/// trading sliver dihedrals for huge-circumradius cones.
fn radius_edge(p: [[f64; 3]; 4]) -> f64 {
    // Solve 2 (p_i - p_0) . c = |p_i|^2 - |p_0|^2 for the circumcenter.
    let mut m = [[0.0f64; 3]; 3];
    let mut b = [0.0f64; 3];
    for i in 0..3 {
        for k in 0..3 {
            m[i][k] = 2.0 * (p[i + 1][k] - p[0][k]);
        }
        b[i] = (0..3).map(|k| p[i + 1][k] * p[i + 1][k] - p[0][k] * p[0][k]).sum();
    }
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < f64::MIN_POSITIVE {
        return f64::MAX;
    }
    let inv = 1.0 / det;
    let mut c = [0.0f64; 3];
    for k in 0..3 {
        // Cramer: replace column k with b.
        let mut mm = m;
        for i in 0..3 {
            mm[i][k] = b[i];
        }
        c[k] = inv
            * (mm[0][0] * (mm[1][1] * mm[2][2] - mm[1][2] * mm[2][1])
                - mm[0][1] * (mm[1][0] * mm[2][2] - mm[1][2] * mm[2][0])
                + mm[0][2] * (mm[1][0] * mm[2][1] - mm[1][1] * mm[2][0]));
    }
    let r2: f64 = (0..3).map(|k| (c[k] - p[0][k]).powi(2)).sum();
    let mut lmin2 = f64::MAX;
    for i in 0..4 {
        for j in i + 1..4 {
            lmin2 = lmin2.min((0..3).map(|k| (p[i][k] - p[j][k]).powi(2)).sum());
        }
    }
    if lmin2 <= 0.0 {
        return f64::MAX;
    }
    (r2 / lmin2).sqrt()
}

/// Tets whose dihedral angles leave this band (below it, or above its
/// 180-degree complement) become vertex-insertion
/// candidates: boundary slivers with every vertex pinned to the surface are
/// unreachable for smoothing, and their near-coplanar rings defeat edge
/// removal; splitting their star from an interior Steiner point hands the
/// region a FREE vertex that later smoothing passes position optimally.
const INSERT_BELOW_DEG: f64 = 10.0;
/// Cone tets of a vertex insertion may have radius-edge ratios up to this
/// value unconditionally; beyond it only when not exceeding the cavity's
/// own worst ratio (no minting of huge-circumsphere tets).
const INSERT_RE_ALLOW: f64 = 16.0;

/// Parameters for [`optimize`].
#[derive(Debug, Clone)]
pub struct OptimizeParams {
    /// Maximum number of smoothing+flip rounds. The loop exits at the fixed
    /// point (a pass with zero accepted operations); passes after the first
    /// only revisit the neighborhoods of previous changes, so a high cap
    /// costs little.
    pub passes: usize,
    /// The meshing size target (see [`crate::MeshParams::maxh`]): the
    /// optimizer keeps every new edge within [`EDGE_CONTRACT`] times the
    /// local target, so the mesher's sizing survives optimization.
    pub maxh: f64,
    /// Per-region size targets (see [`crate::MeshParams::region_maxh`]).
    pub region_maxh: Vec<(u32, f64)>,
}

impl Default for OptimizeParams {
    fn default() -> Self {
        OptimizeParams {
            passes: 50,
            maxh: f64::INFINITY,
            region_maxh: Vec::new(),
        }
    }
}

/// Edges up to this multiple of the local size target are legal (the same
/// documented slack the mesher's own max-edge contract uses).
const EDGE_CONTRACT: f64 = 1.5;

/// Local complexes whose worst tet is already at or above this
/// comparison-scale quality (min dihedral ~35 deg, max ~145 deg by the
/// symmetric -max|cos| metric) are left alone: optimization effort
/// concentrates on the sliver tail that actually hurts H(curl)
/// conditioning (the HXT recipe). Fidelity snaps are constraints and remain
/// exempt.
const TARGET_Q: f64 = -0.8191520442889918; // -cos(35 deg)

fn orient_positive(points: &[[f64; 3]], t: [usize; 4]) -> bool {
    Sign::of_f64(geometry_predicates::orient3d(
        points[t[0]],
        points[t[1]],
        points[t[2]],
        points[t[3]],
    )) == Sign::Positive
}

/// Comparison-scale tet quality: MINUS the maximum ABSOLUTE cosine over the
/// six dihedral angles, in [-1, 0]. Penalizes both extremes: slivers (near
/// 0 deg) and caps/wedges (near 180 deg) score equally badly, while the
/// minimum-dihedral metric is blind to obtuse angles (cos(179 deg) = -1
/// looks great to a max-cos gate). Every improvement gate compares this
/// scale directly; degrees stay a reporting concern (`quality_stats`).
fn quality(points: &[[f64; 3]], t: [usize; 4]) -> f64 {
    quality_above(points, t, f64::MIN).expect("MIN threshold never rejects")
}

/// [`quality`] with a cutoff: `None` as soon as one dihedral already bounds
/// the result to `<= threshold` (the candidate cannot beat the incumbent).
/// Most failing candidates exit after one or two edges instead of all six.
fn quality_above(points: &[[f64; 3]], t: [usize; 4], threshold: f64) -> Option<f64> {
    let p: [[f64; 3]; 4] = std::array::from_fn(|k| points[t[k]]);
    quality_above_coords(p, threshold)
}

/// [`quality_above`] on explicit coordinates (speculative positions that are
/// not mesh vertices yet, e.g. vertex-insertion position search).
fn quality_above_coords(p: [[f64; 3]; 4], threshold: f64) -> Option<f64> {
    let mut q = f64::MAX;
    const OPP: [((usize, usize), (usize, usize)); 6] = [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
        ((1, 2), (0, 3)),
        ((1, 3), (0, 2)),
        ((2, 3), (0, 1)),
    ];
    for ((i, j), (k, l)) in OPP {
        let (a, b) = (p[i], p[j]);
        let t2: f64 = (0..3).map(|m| (b[m] - a[m]).powi(2)).sum();
        if t2 == 0.0 {
            continue;
        }
        let perp = |c: [f64; 3]| -> [f64; 3] {
            let w: [f64; 3] = std::array::from_fn(|m| c[m] - a[m]);
            let s: f64 = (0..3).map(|m| w[m] * (b[m] - a[m])).sum::<f64>() / t2;
            std::array::from_fn(|m| w[m] - s * (b[m] - a[m]))
        };
        let (u, v) = (perp(p[k]), perp(p[l]));
        let nu: f64 = (0..3).map(|m| u[m] * u[m]).sum::<f64>().sqrt();
        let nv: f64 = (0..3).map(|m| v[m] * v[m]).sum::<f64>().sqrt();
        if nu * nv == 0.0 {
            continue;
        }
        let cos = ((0..3).map(|m| u[m] * v[m]).sum::<f64>() / (nu * nv)).clamp(-1.0, 1.0);
        q = q.min(-cos.abs());
        if q <= threshold {
            return None;
        }
    }
    Some(q)
}

/// Longest squared edge over a set of tets.
fn lmax2_of(points: &[[f64; 3]], tets: &[[usize; 4]], ids: &[usize]) -> f64 {
    let mut m = 0.0f64;
    for &ti in ids {
        let t = tets[ti];
        for i in 0..4 {
            for j in i + 1..4 {
                m = m.max(
                    (0..3).map(|k| (points[t[i]][k] - points[t[j]][k]).powi(2)).sum(),
                );
            }
        }
    }
    m
}

fn dist2_pts(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum()
}

fn sorted3(f: [usize; 3]) -> [usize; 3] {
    let mut s = f;
    s.sort_unstable();
    s
}

/// In-place quality optimization. Returns the number of accepted operations.
///
/// Pass 0 sweeps everything; every later pass only revisits candidates whose
/// one-ring contains a vertex changed by an accepted operation of the
/// previous pass. The filtering is exact, not heuristic: an operation's
/// outcome depends only on the positions and tets of its local complex, so a
/// candidate with an unchanged neighborhood would be re-rejected verbatim.
/// The fixed point is therefore identical to full sweeps, at a per-pass cost
/// that scales with the remaining work.
pub fn optimize(mesh: &mut TetMesh, params: &OptimizeParams) -> usize {
    let trace = std::env::var("RAPIDMESH_OPT_TRACE").is_ok();
    // Squared edge-length budget per region: quality operations may create
    // edges up to the sizing contract, or up to the local status quo where
    // the mesh is already coarser (never blocking improvements there).
    let edge_budget2 = |region: rapidmesh_geom::RegionTag| -> f64 {
        let h = params
            .region_maxh
            .iter()
            .find(|(r, _)| *r == region.0)
            .map(|&(_, h)| h)
            .unwrap_or(params.maxh);
        if h.is_finite() {
            (EDGE_CONTRACT * h) * (EDGE_CONTRACT * h)
        } else {
            f64::INFINITY
        }
    };
    let mut total_ops = 0usize;
    // Vertices changed by the previous pass; `None` = everything (pass 0).
    let mut dirty: Option<DSet<usize>> = None;
    // Persistent working state (the refinement queue's medicine applied to
    // the optimizer): tet slots are append-only with an alive mask and ONE
    // final compaction, so the vertex->tet incidence, the per-tet quality
    // cache, and the constraint bookkeeping survive across passes -- every
    // per-pass cost then scales with the change set, not with the mesh.
    let mut alive: Vec<bool> = vec![true; mesh.tets.len()];
    let mut g_incident: Vec<Vec<u32>> = vec![Vec::new(); mesh.points.len()];
    for (ti, t) in mesh.tets.iter().enumerate() {
        for &v in t {
            g_incident[v].push(ti as u32);
        }
    }
    // Per-tet quality cache for the incumbent side of every gate;
    // invalidated whenever a vertex of the tet moves or the slot is
    // rewritten in place.
    let mut tet_q: Vec<f64> = vec![f64::NAN; mesh.tets.len()];
    // Surface face records by vertex triple, kept in sync by splits and
    // 2-2 flips.
    let mut face_idx: DMap<[usize; 3], u32> = DMap::default();
    for (fi, sf) in mesh.faces.iter().enumerate() {
        face_idx.insert(sorted3(sf.tri), fi as u32);
    }
    // Constrained surface complex, maintained incrementally by the surface
    // operations.
    let (mut constrained_faces, mut constrained_edges, mut constrained_verts) = {
        let mut cf: DSet<[usize; 3]> = DSet::default();
        let mut ce: DSet<(usize, usize)> = DSet::default();
        let mut cv: DSet<usize> = DSet::default();
        for sf in &mesh.faces {
            cf.insert(sorted3(sf.tri));
            for e in 0..3 {
                let (a, b) = (sf.tri[e], sf.tri[(e + 1) % 3]);
                ce.insert((a.min(b), a.max(b)));
            }
            for &v in &sf.tri {
                cv.insert(v);
            }
        }
        (cf, ce, cv)
    };
    for _pass in 0..params.passes {
        let mut ops = 0usize;
        let t0 = std::time::Instant::now();

        // Dilation via the persistent incidence: tets touching a dirty
        // vertex, their vertices ("active"), and those vertices' tets.
        // Owner lookups built from these sets are COMPLETE for active
        // candidates (every owner of a face/edge contains the candidate's
        // active vertex).
        let t_d: Option<Vec<u32>> = dirty.as_ref().map(|d| {
            let mut td: Vec<u32> = d
                .iter()
                .flat_map(|&v| g_incident[v].iter().copied())
                .filter(|&ti| alive[ti as usize])
                .collect();
            td.sort_unstable();
            td.dedup();
            td
        });
        let active_verts: Option<DSet<usize>> = t_d.as_ref().map(|td| {
            let mut av: DSet<usize> = DSet::default();
            for &ti in td {
                for &v in &mesh.tets[ti as usize] {
                    av.insert(v);
                }
            }
            av
        });
        let active_tets: Vec<u32> = match &active_verts {
            None => (0..mesh.tets.len() as u32)
                .filter(|&ti| alive[ti as usize])
                .collect(),
            Some(av) => {
                let mut at: Vec<u32> = av
                    .iter()
                    .flat_map(|&v| g_incident[v].iter().copied())
                    .filter(|&ti| alive[ti as usize])
                    .collect();
                at.sort_unstable();
                at.dedup();
                at
            }
        };
        // Tets feeding the flip owner maps: a much tighter set than
        // `active_tets` (vertex dilation saturates small meshes). Every
        // owner that a CHANGED candidate needs shares an edge with a tet
        // containing a dirty vertex: face owners share three edges, ring
        // tets share the ring edge itself, 2-2 tet pairs share the tile
        // edge.
        let map_tets: Vec<u32> = match &t_d {
            None => active_tets.clone(),
            Some(td) => {
                let mut e1: DSet<(usize, usize)> = DSet::default();
                for &ti in td {
                    let t = &mesh.tets[ti as usize];
                    for i in 0..4 {
                        for j in i + 1..4 {
                            e1.insert((t[i].min(t[j]), t[i].max(t[j])));
                        }
                    }
                }
                active_tets
                    .iter()
                    .copied()
                    .filter(|&ti| {
                        let t = &mesh.tets[ti as usize];
                        (0..4).any(|i| {
                            (i + 1..4)
                                .any(|j| e1.contains(&(t[i].min(t[j]), t[i].max(t[j]))))
                        })
                    })
                    .collect()
            }
        };
        let is_active =
            |vs: &[usize]| active_verts.as_ref().is_none_or(|av| vs.iter().any(|v| av.contains(v)));
        // Exact candidate filter: a candidate is re-evaluated only if its
        // complex (the vertices its outcome depends on) contains a vertex
        // CHANGED by the previous pass. `is_active` (one ring wider) only
        // keeps the owner maps complete; filtering with it would saturate
        // on small meshes.
        let complex_changed =
            |vs: &[usize]| dirty.as_ref().is_none_or(|d| vs.iter().any(|v| d.contains(v)));
        let mut next_dirty: DSet<usize> = DSet::default();

        // Surface improvement first: boundary slivers cannot be fixed by
        // interior-only operations.
        ops += surface_pass(
            mesh,
            &mut g_incident,
            &alive,
            &mut tet_q,
            &mut face_idx,
            &mut constrained_faces,
            &mut constrained_edges,
            &map_tets,
            &is_active,
            &complex_changed,
            &edge_budget2,
            &mut next_dirty,
        );
        let t_surf = t0.elapsed();
        let t1 = std::time::Instant::now();

        // ------------------------------------------------- smoothing
        // Topology is fixed during smoothing; the persistent incidence
        // serves directly (dead slots filtered on read).
        fn cached_q(
            points: &[[f64; 3]],
            tets: &[[usize; 4]],
            tet_q: &mut [f64],
            ti: usize,
        ) -> f64 {
            if tet_q[ti].is_nan() {
                tet_q[ti] = quality(points, tets[ti]);
            }
            tet_q[ti]
        }
        let smooth_verts: Vec<usize> = match &active_verts {
            None => (0..mesh.points.len()).collect(),
            Some(av) => {
                let mut sv: Vec<usize> = av.iter().copied().collect();
                sv.sort_unstable();
                sv
            }
        };
        let mut nbrs: Vec<usize> = Vec::new();
        let mut inc: Vec<u32> = Vec::new();
        for &v in &smooth_verts {
            if constrained_verts.contains(&v) {
                continue;
            }
            inc.clear();
            inc.extend(
                g_incident[v]
                    .iter()
                    .copied()
                    .filter(|&ti| alive[ti as usize]),
            );
            if inc.is_empty() {
                continue;
            }
            nbrs.clear();
            for &ti in &inc {
                for &w in &mesh.tets[ti as usize] {
                    if w != v {
                        nbrs.push(w);
                    }
                }
            }
            nbrs.sort_unstable();
            nbrs.dedup();
            if !(complex_changed(&[v]) || complex_changed(&nbrs)) {
                continue;
            }
            let old_pos = mesh.points[v];
            let mut avg = [0.0f64; 3];
            let mut lref2 = 0.0f64;
            for &w in &nbrs {
                let mut d2 = 0.0;
                for (k, a) in avg.iter_mut().enumerate() {
                    *a += mesh.points[w][k];
                    d2 += (mesh.points[w][k] - old_pos[k]).powi(2);
                }
                lref2 = lref2.max(d2);
            }
            for a in &mut avg {
                *a /= nbrs.len() as f64;
            }
            let move2: f64 = (0..3).map(|k| (avg[k] - old_pos[k]).powi(2)).sum();
            if move2 < MIN_REL_MOVE * MIN_REL_MOVE * lref2 {
                continue;
            }
            // Sizing contract: a move may not grow any incident edge past
            // the local budget (unless it already was longer): smoothing
            // was the one operation without the edge gate, and chains of
            // moves walked edges past the documented 1.5 h bound.
            {
                let budget2 = edge_budget2(mesh.tet_regions[inc[0] as usize]);
                let mut old_lmax2 = 0.0f64;
                let mut new_lmax2 = 0.0f64;
                for &w in &nbrs {
                    let dw_old: f64 =
                        (0..3).map(|k| (mesh.points[w][k] - old_pos[k]).powi(2)).sum();
                    let dw_new: f64 =
                        (0..3).map(|k| (mesh.points[w][k] - avg[k]).powi(2)).sum();
                    old_lmax2 = old_lmax2.max(dw_old);
                    new_lmax2 = new_lmax2.max(dw_new);
                }
                if new_lmax2 > old_lmax2.max(budget2) {
                    continue;
                }
            }
            let mut old_q = f64::MAX;
            for &ti in &inc {
                old_q = old_q.min(cached_q(&mesh.points, &mesh.tets, &mut tet_q, ti as usize));
            }
            if old_q >= TARGET_Q {
                continue;
            }
            mesh.points[v] = avg;
            let mut new_q = f64::MAX;
            let mut ok = true;
            for &ti in &inc {
                if !orient_positive(&mesh.points, mesh.tets[ti as usize]) {
                    ok = false;
                    break;
                }
                match quality_above(&mesh.points, mesh.tets[ti as usize], old_q) {
                    Some(q) => new_q = new_q.min(q),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && new_q > old_q + QUALITY_EPS {
                ops += 1;
                next_dirty.insert(v);
                for &ti in &inc {
                    tet_q[ti as usize] = f64::NAN;
                }
            } else {
                mesh.points[v] = old_pos;
            }
        }

        let t_smooth = t1.elapsed();
        let t2 = std::time::Instant::now();

        // ------------------------------------------------------ flips
        let mut added: Vec<([usize; 4], rapidmesh_geom::RegionTag)> = Vec::new();

        // Face map for 2-3 flips and edge map for 3-2 flips, over active
        // tets (complete owner lists for active candidates). Sorted entry
        // vectors instead of hash maps: grouping by sort is allocation-free
        // and gives the deterministic key order directly.
        let mut face_entries: Vec<([usize; 3], u32)> = Vec::with_capacity(map_tets.len() * 4);
        let mut edge_entries: Vec<((usize, usize), u32)> = Vec::with_capacity(map_tets.len() * 6);
        for &ti in &map_tets {
            let t = &mesh.tets[ti as usize];
            for i in 0..4 {
                let f: [usize; 3] = std::array::from_fn(|k| t[(i + 1 + k) % 4]);
                face_entries.push((sorted3(f), ti));
            }
            for i in 0..4 {
                for j in i + 1..4 {
                    let (a, b) = (t[i].min(t[j]), t[i].max(t[j]));
                    edge_entries.push(((a, b), ti));
                }
            }
        }
        face_entries.sort_unstable();
        edge_entries.sort_unstable();
        let t_entries = t2.elapsed();

        let mut gi = 0;
        while gi < face_entries.len() {
            let f = face_entries[gi].0;
            let gj = gi + face_entries[gi..].iter().take_while(|e| e.0 == f).count();
            let owners = &face_entries[gi..gj];
            gi = gj;
            if owners.len() != 2 || constrained_faces.contains(&f) {
                continue;
            }
            let (t1, t2) = (owners[0].1 as usize, owners[1].1 as usize);
            if !alive[t1] || !alive[t2] {
                continue;
            }
            if !(complex_changed(&mesh.tets[t1]) || complex_changed(&mesh.tets[t2])) {
                continue;
            }
            // Faces must still match (alive guard covers staleness).
            if mesh.tet_regions[t1] != mesh.tet_regions[t2] {
                continue;
            }
            let d = *mesh.tets[t1].iter().find(|v| !f.contains(v)).expect("apex");
            let e = *mesh.tets[t2].iter().find(|v| !f.contains(v)).expect("apex");
            if d == e {
                continue;
            }
            let old_q = cached_q(&mesh.points, &mesh.tets, &mut tet_q, t1)
                .min(cached_q(&mesh.points, &mesh.tets, &mut tet_q, t2));
            if old_q >= TARGET_Q {
                continue;
            }
            // Sizing contract: the new edge stays within the local size
            // budget (or the local status quo where already coarser).
            if dist2_pts(mesh.points[d], mesh.points[e])
                > lmax2_of(&mesh.points, &mesh.tets, &[t1, t2])
                    .max(edge_budget2(mesh.tet_regions[t1]))
            {
                continue;
            }
            // Geometric validity: the new edge d-e must cross the interior
            // of f, otherwise the three new tets do not tile the union of
            // the two old ones (they overlap, breaking volume and
            // conformity). Equivalent test: consistent orientation of d-e
            // against all three face edges.
            let side = |a: usize, b: usize| {
                Sign::of_f64(geometry_predicates::orient3d(
                    mesh.points[a],
                    mesh.points[b],
                    mesh.points[d],
                    mesh.points[e],
                ))
            };
            let (s0, s1, s2) = (side(f[0], f[1]), side(f[1], f[2]), side(f[2], f[0]));
            if s0 == Sign::Zero || s0 != s1 || s1 != s2 {
                continue;
            }
            // 2-3 flip: three tets around the new edge d-e.
            let mk = |a: usize, b: usize| -> Option<[usize; 4]> {
                let cand = [a, b, d, e];
                if orient_positive(&mesh.points, cand) {
                    Some(cand)
                } else {
                    let swapped = [b, a, d, e];
                    orient_positive(&mesh.points, swapped).then_some(swapped)
                }
            };
            let (Some(n1), Some(n2), Some(n3)) =
                (mk(f[0], f[1]), mk(f[1], f[2]), mk(f[2], f[0]))
            else {
                continue;
            };
            let (Some(q1), Some(q2), Some(q3)) = (
                quality_above(&mesh.points, n1, old_q),
                quality_above(&mesh.points, n2, old_q),
                quality_above(&mesh.points, n3, old_q),
            ) else {
                continue;
            };
            if q1.min(q2).min(q3) <= old_q + QUALITY_EPS {
                continue;
            }
            alive[t1] = false;
            alive[t2] = false;
            let region = mesh.tet_regions[t1];
            added.push((n1, region));
            added.push((n2, region));
            added.push((n3, region));
            for v in f.iter().chain([d, e].iter()) {
                next_dirty.insert(*v);
            }
            ops += 1;
        }

        let t_23 = t2.elapsed();
        // EDGE REMOVAL (the 3-2 flip generalized to rings of k tets): the k
        // tets around an unconstrained interior edge d-e are replaced by
        // 2(k-2) tets over the max-min-quality triangulation of the ring
        // polygon (Klincsek interval DP). Validity is positivity of every
        // new tet in the FIXED ring orientation (no flipping): a folded
        // triangulation triangle then shows up as a negative tet, so the
        // accepted complex tiles exactly the old star.
        let mut gi = 0;
        while gi < edge_entries.len() {
            let key = edge_entries[gi].0;
            let gj = gi + edge_entries[gi..].iter().take_while(|e| e.0 == key).count();
            let owners = &edge_entries[gi..gj];
            gi = gj;
            let k = owners.len();
            if !(3..=MAX_RING).contains(&k) || constrained_edges.contains(&key) {
                continue;
            }
            let mut ts = [0usize; MAX_RING];
            for (i, e) in owners.iter().enumerate() {
                ts[i] = e.1 as usize;
            }
            let ts = &ts[..k];
            if ts.iter().any(|&t| !alive[t]) {
                continue;
            }
            if !ts.iter().any(|&t| complex_changed(&mesh.tets[t])) {
                continue;
            }
            if ts
                .iter()
                .any(|&t| mesh.tet_regions[t] != mesh.tet_regions[ts[0]])
            {
                continue;
            }
            let old_q = {
                let mut q = f64::MAX;
                for &t in ts {
                    q = q.min(cached_q(&mesh.points, &mesh.tets, &mut tet_q, t));
                }
                q
            };
            if old_q >= TARGET_Q {
                continue;
            }
            let (d, e) = key;
            // Each tet contributes the pair of its two non-(d,e) vertices;
            // the pairs must chain into a single closed ring (interior
            // edge), walked here into cyclic order. All on the stack: a
            // distinct-vertex table (k <= 7) with degree-2 adjacency.
            let mut vs = [usize::MAX; MAX_RING];
            let mut adj = [[usize::MAX; 2]; MAX_RING];
            let mut deg = [0u8; MAX_RING];
            let mut nv = 0usize;
            let mut ok = true;
            'tets: for &t in ts {
                let mut pr = mesh.tets[t].iter().copied().filter(|&x| x != d && x != e);
                let (a, b) = (pr.next().expect("pair"), pr.next().expect("pair"));
                for x in [a, b] {
                    let y = if x == a { b } else { a };
                    let slot = match vs[..nv].iter().position(|&w| w == x) {
                        Some(i) => i,
                        None => {
                            if nv == k {
                                ok = false;
                                break 'tets; // more than k distinct: not a ring
                            }
                            vs[nv] = x;
                            nv += 1;
                            nv - 1
                        }
                    };
                    if deg[slot] == 2 {
                        ok = false;
                        break 'tets;
                    }
                    adj[slot][deg[slot] as usize] = y;
                    deg[slot] += 1;
                }
            }
            if !ok || nv != k || deg[..k].iter().any(|&x| x != 2) {
                continue;
            }
            let start = *vs[..k].iter().min().expect("nonempty");
            let mut ring = [usize::MAX; MAX_RING];
            ring[0] = start;
            let mut prev = usize::MAX;
            for i in 1..k {
                let cur = ring[i - 1];
                let slot = vs[..k].iter().position(|&w| w == cur).expect("in table");
                let p = adj[slot];
                let next = if p[0] != prev { p[0] } else { p[1] };
                prev = cur;
                ring[i] = next;
            }
            // Closed ring?
            {
                let slot = vs[..k].iter().position(|&w| w == ring[k - 1]).expect("in table");
                if !adj[slot].contains(&start) {
                    continue;
                }
            }
            // Orient the cycle so that consecutive pairs rotate positively
            // around d-e: orient3d(a_i, a_{i+1}, d, e) > 0 for all i (an
            // embedded star is consistent; anything else is degenerate).
            let side = |a: usize, b: usize| {
                Sign::of_f64(geometry_predicates::orient3d(
                    mesh.points[a],
                    mesh.points[b],
                    mesh.points[d],
                    mesh.points[e],
                ))
            };
            if side(ring[0], ring[1]) == Sign::Negative {
                ring[..k].reverse();
            }
            if (0..k).any(|i| side(ring[i], ring[(i + 1) % k]) != Sign::Positive) {
                continue;
            }
            // Klincsek DP over the ring polygon: best[i][j] = max-min
            // quality of triangulating the sub-polygon ring[i..=j]. Each
            // triangle (p, q, r) in ring order spawns the tet pair
            // (p, q, r, e) and (p, r, q, d), both required positive.
            // Values not above the incumbent old_q are clamped to MIN (the
            // final gate rejects them anyway), which lets the quality
            // evaluation exit early.
            let star_lmax2 = lmax2_of(&mesh.points, &mesh.tets, ts)
                .max(edge_budget2(mesh.tet_regions[ts[0]]));
            let pair_q = |i: usize, m: usize, j: usize| -> f64 {
                let (p, q, r) = (ring[i], ring[m], ring[j]);
                // Sizing invariant: new chords stay within the old star's
                // longest edge.
                if dist2_pts(mesh.points[p], mesh.points[q]) > star_lmax2
                    || dist2_pts(mesh.points[q], mesh.points[r]) > star_lmax2
                    || dist2_pts(mesh.points[p], mesh.points[r]) > star_lmax2
                {
                    return f64::MIN;
                }
                let t1 = [p, q, r, e];
                let t2 = [p, r, q, d];
                if !orient_positive(&mesh.points, t1) || !orient_positive(&mesh.points, t2) {
                    return f64::MIN;
                }
                match (
                    quality_above(&mesh.points, t1, old_q),
                    quality_above(&mesh.points, t2, old_q),
                ) {
                    (Some(q1), Some(q2)) => q1.min(q2),
                    _ => f64::MIN,
                }
            };
            let mut best = [[f64::MAX; MAX_RING]; MAX_RING];
            let mut cut = [[usize::MAX; MAX_RING]; MAX_RING];
            for len in 2..k {
                for i in 0..k - len {
                    let j = i + len;
                    let (mut bq, mut bm) = (f64::MIN, usize::MAX);
                    #[allow(clippy::needless_range_loop)]
                    for m in i + 1..j {
                        let q = pair_q(i, m, j).min(best[i][m]).min(best[m][j]);
                        if q > bq {
                            bq = q;
                            bm = m;
                        }
                    }
                    best[i][j] = bq;
                    cut[i][j] = bm;
                }
            }
            if best[0][k - 1] <= old_q + QUALITY_EPS {
                continue;
            }
            let region = mesh.tet_regions[ts[0]];
            let mut stack = [(0usize, 0usize); 2 * MAX_RING];
            stack[0] = (0, k - 1);
            let mut sp = 1usize;
            while sp > 0 {
                sp -= 1;
                let (i, j) = stack[sp];
                if j - i < 2 {
                    continue;
                }
                let m = cut[i][j];
                added.push(([ring[i], ring[m], ring[j], e], region));
                added.push(([ring[i], ring[j], ring[m], d], region));
                stack[sp] = (i, m);
                stack[sp + 1] = (m, j);
                sp += 2;
            }
            for &t in ts {
                alive[t] = false;
            }
            for &v in ring[..k].iter().chain([d, e].iter()) {
                next_dirty.insert(v);
            }
            ops += 1;
        }

        let t_eremove = t2.elapsed();
        // --------------------------------------------- vertex insertion
        // (see INSERT_BELOW_DEG). The cavity is the bad tet plus its alive
        // same-region face neighbors (1-ring: the owner map is complete for
        // it), with no constrained face in its interior; the replacement is
        // the cone from the cavity vertex centroid, gated on exact
        // positivity against every boundary face (star-shapedness) and a
        // strict min-quality improvement. Faces whose neighbor died in an
        // earlier operation of this pass stay cavity boundary; their
        // replacements tile the same space behind the shared interface.
        {
            // Face of positively oriented `t` opposite vertex slot `i`,
            // wound so the opposite vertex lies on its positive side.
            let opp_face = |t: [usize; 4], i: usize| -> [usize; 3] {
                match i {
                    0 => [t[1], t[3], t[2]],
                    1 => [t[0], t[2], t[3]],
                    2 => [t[0], t[3], t[1]],
                    _ => [t[0], t[1], t[2]],
                }
            };
            let face_owners = |key: [usize; 3]| -> &[([usize; 3], u32)] {
                let lo = face_entries.partition_point(|e| e.0 < key);
                let hi = lo + face_entries[lo..].iter().take_while(|e| e.0 == key).count();
                &face_entries[lo..hi]
            };
            let insert_below = -(INSERT_BELOW_DEG.to_radians().cos());
            let (mut split_cands, mut split_faces, mut split_gate, mut split_ok) =
                (0usize, 0usize, 0usize, 0usize);
            let mut bad: Vec<(f64, u32)> = Vec::new();
            for &ti in &map_tets {
                if !alive[ti as usize] || !complex_changed(&mesh.tets[ti as usize]) {
                    continue;
                }
                let q = cached_q(&mesh.points, &mesh.tets, &mut tet_q, ti as usize);
                if q < insert_below {
                    bad.push((q, ti));
                }
            }
            bad.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let n_bad = bad.len();
            for (_, ti) in bad {
                let ti = ti as usize;
                if !alive[ti] {
                    continue;
                }
                let region = mesh.tet_regions[ti];
                let t = mesh.tets[ti];
                let interior_done = 'interior: {
                let mut cavity: Vec<usize> = vec![ti];
                for i in 0..4 {
                    let key = sorted3(opp_face(t, i));
                    if constrained_faces.contains(&key) {
                        continue; // stays cavity boundary
                    }
                    for e in face_owners(key) {
                        let nb = e.1 as usize;
                        if nb != ti && alive[nb] && mesh.tet_regions[nb] == region {
                            cavity.push(nb);
                        }
                    }
                }
                cavity.sort_unstable();
                cavity.dedup();
                // Cavity faces: interior iff shared by two cavity tets.
                // Constrained interior faces veto the candidate.
                let mut faces: Vec<([usize; 3], [usize; 3])> = Vec::new(); // (sorted, oriented)
                for &c in &cavity {
                    for i in 0..4 {
                        let of = opp_face(mesh.tets[c], i);
                        faces.push((sorted3(of), of));
                    }
                }
                faces.sort_unstable();
                let mut bfaces: Vec<[usize; 3]> = Vec::new();
                let mut fi = 0;
                while fi < faces.len() {
                    let same = faces[fi..].iter().take_while(|f| f.0 == faces[fi].0).count();
                    if same == 1 {
                        bfaces.push(faces[fi].1);
                    } else if constrained_faces.contains(&faces[fi].0) {
                        break 'interior false;
                    }
                    fi += same;
                }
                let mut verts: Vec<usize> = cavity.iter().flat_map(|&c| mesh.tets[c]).collect();
                verts.sort_unstable();
                verts.dedup();
                let mut x = [0.0f64; 3];
                let mut diag2 = 0.0f64;
                for &v in &verts {
                    for (k, a) in x.iter_mut().enumerate() {
                        *a += mesh.points[v][k];
                    }
                }
                for a in &mut x {
                    *a /= verts.len() as f64;
                }
                for &v in &verts {
                    let d2: f64 =
                        (0..3).map(|k| (mesh.points[v][k] - x[k]).powi(2)).sum();
                    diag2 = diag2.max(d2);
                }
                let mut old_q = f64::MAX;
                for &c in &cavity {
                    old_q = old_q.min(cached_q(&mesh.points, &mesh.tets, &mut tet_q, c));
                }
                let mut old_re = 0.0f64;
                for &c in &cavity {
                    old_re = old_re.max(radius_edge(std::array::from_fn(|k| {
                        mesh.points[mesh.tets[c][k]]
                    })));
                }
                let re_cap = old_re.max(INSERT_RE_ALLOW);
                // Optimization-based positioning (the Stellar recipe): the
                // centroid cone of a squashed sliver cavity is itself thin,
                // so the insertion point pattern-searches the position that
                // maximizes the worst cone DIHEDRAL quality. The radius-edge
                // cap is a hard constraint INSIDE the objective: needles
                // have fine dihedrals, so without it the search happily
                // walks into huge-circumsphere positions.
                // Objective: worst cone dihedral quality, with the
                // radius-edge cap as a PENALTY rather than a hard wall
                // (infeasible start positions would otherwise sit on a MIN
                // plateau the pattern search cannot leave). Acceptance at
                // the end is strict on both.
                let cav_lmax2 =
                    lmax2_of(&mesh.points, &mesh.tets, &cavity).max(edge_budget2(region));
                let cone_eval = |x: [f64; 3]| -> (f64, f64) {
                    let mut q = f64::MAX;
                    let mut re = 0.0f64;
                    for f in &bfaces {
                        let p: [[f64; 3]; 4] = [
                            mesh.points[f[0]],
                            mesh.points[f[1]],
                            mesh.points[f[2]],
                            x,
                        ];
                        if (0..3).any(|k| dist2_pts(p[k], x) > cav_lmax2) {
                            return (f64::MIN, f64::MAX);
                        }
                        if Sign::of_f64(geometry_predicates::orient3d(p[0], p[1], p[2], p[3]))
                            != Sign::Positive
                        {
                            return (f64::MIN, f64::MAX);
                        }
                        re = re.max(radius_edge(p));
                        match quality_above_coords(p, f64::MIN) {
                            Some(cq) => q = q.min(cq),
                            None => return (f64::MIN, f64::MAX),
                        }
                    }
                    (q, re)
                };
                let score = |(q, re): (f64, f64)| -> f64 {
                    if q == f64::MIN {
                        f64::MIN
                    } else {
                        q - 0.5 * ((re / re_cap) - 1.0).max(0.0)
                    }
                };
                let mut best = cone_eval(x);
                let mut best_s = score(best);
                let mut step = 0.25 * diag2.sqrt();
                for _ in 0..3 {
                    loop {
                        let mut improved = false;
                        for k in 0..3 {
                            for sgn in [-1.0, 1.0] {
                                let mut cand = x;
                                cand[k] += sgn * step;
                                let e = cone_eval(cand);
                                let sc = score(e);
                                if sc > best_s {
                                    best_s = sc;
                                    best = e;
                                    x = cand;
                                    improved = true;
                                }
                            }
                        }
                        if !improved {
                            break;
                        }
                    }
                    step *= 0.5;
                }
                // Acceptance is on the dihedral objective alone: the
                // radius-edge penalty steers the search away from needle
                // positions, but a sliver removal is not sacrificed to the
                // occasional long cone (slivers hurt Nedelec conditioning
                // directly; needles with sound dihedrals do not).
                if best.0 <= old_q + QUALITY_EPS {
                    break 'interior false;
                }
                let xi = mesh.points.len();
                mesh.points.push(x);
                for &c in &cavity {
                    alive[c] = false;
                }
                for f in bfaces {
                    added.push(([f[0], f[1], f[2], xi], region));
                }
                for &v in &verts {
                    next_dirty.insert(v);
                }
                next_dirty.insert(xi);
                ops += 1;
                true
                };
                if interior_done {
                    continue;
                }

                split_cands += 1;
                // ------------------------------- surface split fallback
                // Boundary caps and wedges with every vertex pinned to the
                // surface are unreachable for interior insertion: their
                // only roomy face IS a surface face. Split that face 1-3 at
                // a Steiner point ON the geometry (in-plane for plane
                // patches, projected for curved ones); every owner tet
                // (both sides of an interface) splits 1-3 with it. The new
                // vertex is a regular surface vertex that in-surface
                // smoothing may slide afterwards. The in-plane search basis
                // has exact zeros in the constant coordinate of
                // axis-aligned planes, so exact region volumes survive.
                for slot in 0..4 {
                    let of = opp_face(t, slot);
                    let key = sorted3(of);
                    let Some(&fidx) = face_idx.get(&key) else {
                        continue;
                    };
                    split_faces += 1;
                    let owners = face_owners(key);
                    if owners.is_empty() || owners.iter().any(|e| !alive[e.1 as usize]) {
                        continue;
                    }
                    let sf = mesh.faces[fidx as usize].clone();
                    let (fa, fb, fc) = (sf.tri[0], sf.tri[1], sf.tri[2]);
                    let kind = mesh.surfaces[sf.surface as usize].clone();
                    let (pa, pb, pc) = (mesh.points[fa], mesh.points[fb], mesh.points[fc]);
                    let n = face_normal(&mesh.points, sf.tri);
                    let mut e1: [f64; 3] = std::array::from_fn(|k| pb[k] - pa[k]);
                    let l1 = (e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]).sqrt();
                    if l1 <= 0.0 {
                        continue;
                    }
                    for v in &mut e1 {
                        *v /= l1;
                    }
                    let e2 = [
                        n[1] * e1[2] - n[2] * e1[1],
                        n[2] * e1[0] - n[0] * e1[2],
                        n[0] * e1[1] - n[1] * e1[0],
                    ];
                    let axis = dominant_axis(n);
                    let pe = |q: [f64; 3]| Point3::Explicit(q);
                    let Some(face_ori) = orient2d(&pe(pa), &pe(pb), &pe(pc), axis) else {
                        continue;
                    };
                    if face_ori == Sign::Zero {
                        continue;
                    }
                    let strictly_inside = |x: [f64; 3]| -> bool {
                        [(pa, pb), (pb, pc), (pc, pa)].iter().all(|&(u, v)| {
                            orient2d(&pe(u), &pe(v), &pe(x), axis) == Some(face_ori)
                        })
                    };
                    let mut old_q = f64::MAX;
                    let mut old_re = 0.0f64;
                    for e in owners {
                        let o = e.1 as usize;
                        old_q =
                            old_q.min(cached_q(&mesh.points, &mesh.tets, &mut tet_q, o));
                        old_re = old_re.max(radius_edge(std::array::from_fn(|k| {
                            mesh.points[mesh.tets[o][k]]
                        })));
                    }
                    let re_cap = old_re.max(INSERT_RE_ALLOW);
                    // Sub-tets: each owner with one face vertex replaced by x.
                    let owner_ids: Vec<usize> =
                        owners.iter().map(|e| e.1 as usize).collect();
                    let own_lmax2 = lmax2_of(&mesh.points, &mesh.tets, &owner_ids)
                        .max(edge_budget2(mesh.tet_regions[owner_ids[0]]));
                    let split_eval = |x: [f64; 3]| -> (f64, f64) {
                        if !strictly_inside(x) {
                            return (f64::MIN, f64::MAX);
                        }
                        let mut q = f64::MAX;
                        let mut re = 0.0f64;
                        for e in owners {
                            let ot = mesh.tets[e.1 as usize];
                            for fv in [fa, fb, fc] {
                                let p: [[f64; 3]; 4] = std::array::from_fn(|k| {
                                    if ot[k] == fv {
                                        x
                                    } else {
                                        mesh.points[ot[k]]
                                    }
                                });
                                if (0..4).any(|k| {
                                    p[k] != x && dist2_pts(p[k], x) > own_lmax2
                                }) {
                                    return (f64::MIN, f64::MAX);
                                }
                                if Sign::of_f64(geometry_predicates::orient3d(
                                    p[0], p[1], p[2], p[3],
                                )) != Sign::Positive
                                {
                                    return (f64::MIN, f64::MAX);
                                }
                                re = re.max(radius_edge(p));
                                match quality_above_coords(p, f64::MIN) {
                                    Some(cq) => q = q.min(cq),
                                    None => return (f64::MIN, f64::MAX),
                                }
                            }
                        }
                        (q, re)
                    };
                    let score = |(q, re): (f64, f64)| -> f64 {
                        if q == f64::MIN {
                            f64::MIN
                        } else {
                            q - 0.5 * ((re / re_cap) - 1.0).max(0.0)
                        }
                    };
                    let project = |x: [f64; 3]| -> [f64; 3] {
                        match kind {
                            SurfaceKind::Plane => x,
                            ref curved => project_to_surface(curved, x),
                        }
                    };
                    let mut x = project(std::array::from_fn(|k| {
                        (pa[k] + pb[k] + pc[k]) / 3.0
                    }));
                    let mut best = split_eval(x);
                    let mut best_s = score(best);
                    let diag = (0..3)
                        .map(|k| (pb[k] - pa[k]).powi(2) + (pc[k] - pa[k]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let mut step = 0.25 * diag;
                    for _ in 0..3 {
                        loop {
                            let mut improved = false;
                            for dir in [e1, e2] {
                                for sgn in [-1.0, 1.0] {
                                    let cand = project(std::array::from_fn(|k| {
                                        x[k] + sgn * step * dir[k]
                                    }));
                                    let ev = split_eval(cand);
                                    let sc = score(ev);
                                    if sc > best_s {
                                        best_s = sc;
                                        best = ev;
                                        x = cand;
                                        improved = true;
                                    }
                                }
                            }
                            if !improved {
                                break;
                            }
                        }
                        step *= 0.5;
                    }
                    if best.0 <= old_q + QUALITY_EPS {
                        split_gate += 1;
                        continue;
                    }
                    // Commit: new surface vertex, three sub-faces, three
                    // sub-tets per owner.
                    let xi = mesh.points.len();
                    mesh.points.push(x);
                    for e in owners {
                        let o = e.1 as usize;
                        let ot = mesh.tets[o];
                        let oregion = mesh.tet_regions[o];
                        alive[o] = false;
                        for fv in [fa, fb, fc] {
                            let nt: [usize; 4] =
                                std::array::from_fn(|k| if ot[k] == fv { xi } else { ot[k] });
                            added.push((nt, oregion));
                        }
                        for &v in &ot {
                            next_dirty.insert(v);
                        }
                    }
                    let mut sub1 = sf.clone();
                    let mut sub2 = sf.clone();
                    mesh.faces[fidx as usize].tri = [fa, fb, xi];
                    sub1.tri = [fb, fc, xi];
                    sub2.tri = [fc, fa, xi];
                    mesh.faces.push(sub1);
                    mesh.faces.push(sub2);
                    face_idx.remove(&key);
                    face_idx.insert(sorted3([fa, fb, xi]), fidx);
                    face_idx.insert(sorted3([fb, fc, xi]), (mesh.faces.len() - 2) as u32);
                    face_idx.insert(sorted3([fc, fa, xi]), (mesh.faces.len() - 1) as u32);
                    constrained_faces.remove(&key);
                    constrained_faces.insert(sorted3([fa, fb, xi]));
                    constrained_faces.insert(sorted3([fb, fc, xi]));
                    constrained_faces.insert(sorted3([fc, fa, xi]));
                    for v in [fa, fb, fc] {
                        constrained_edges.insert((v.min(xi), v.max(xi)));
                    }
                    constrained_verts.insert(xi);
                    next_dirty.insert(xi);
                    split_ok += 1;
                    ops += 1;
                    break;
                }
            }
            if trace && n_bad > 0 {
                eprintln!("  insert: {n_bad} bad cands");
            }
            if trace && split_cands > 0 {
                eprintln!(
                    "  split: {split_cands} cands, {split_faces} faces tried, {split_gate} gate-fail, {split_ok} ok"
                );
            }
        }

        // Apply: append the pass's new tets; retired slots stay dead until
        // the single final compaction (stable ids keep the caches valid).
        g_incident.resize(mesh.points.len(), Vec::new());
        for (t, r) in added {
            let ti = mesh.tets.len() as u32;
            mesh.tets.push(t);
            mesh.tet_regions.push(r);
            alive.push(true);
            tet_q.push(f64::NAN);
            for &v in &t {
                g_incident[v].push(ti);
            }
        }

        if trace {
            eprintln!(
                "opt pass {_pass}: surf {:?}, smooth {:?}, flips {:?} (entries {:?}, 2-3 {:?}, eremove {:?}, insert {:?}), ops {ops}, dirty {}",
                t_surf,
                t_smooth,
                t2.elapsed(),
                t_entries,
                t_23 - t_entries,
                t_eremove - t_23,
                t2.elapsed() - t_eremove,
                next_dirty.len()
            );
        }
        total_ops += ops;
        if ops == 0 {
            break;
        }
        dirty = Some(next_dirty);
    }
    // The single final compaction.
    if alive.iter().any(|&a| !a) {
        let mut tets = Vec::with_capacity(mesh.tets.len());
        let mut regions = Vec::with_capacity(mesh.tets.len());
        for (ti, t) in mesh.tets.iter().enumerate() {
            if alive[ti] {
                tets.push(*t);
                regions.push(mesh.tet_regions[ti]);
            }
        }
        mesh.tets = tets;
        mesh.tet_regions = regions;
    }
    total_ops
}

/// Maintains the vertex->tet incidence across an in-place rewrite of slot
/// `ti` from `old` to `new`.
fn update_incidence(g_incident: &mut [Vec<u32>], ti: u32, old: [usize; 4], new: [usize; 4]) {
    for &v in &old {
        if !new.contains(&v) {
            if let Some(pos) = g_incident[v].iter().position(|&x| x == ti) {
                g_incident[v].swap_remove(pos);
            }
        }
    }
    for &v in &new {
        if !old.contains(&v) {
            g_incident[v].push(ti);
        }
    }
}

/// Normalized face normal (f64; used for projections and fold-over guards,
/// never for exact decisions).
fn face_normal(points: &[[f64; 3]], t: [usize; 3]) -> [f64; 3] {
    let u: [f64; 3] = std::array::from_fn(|k| points[t[1]][k] - points[t[0]][k]);
    let v: [f64; 3] = std::array::from_fn(|k| points[t[2]][k] - points[t[0]][k]);
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(f64::MIN_POSITIVE);
    [n[0] / l, n[1] / l, n[2] / l]
}

fn dominant_axis(n: [f64; 3]) -> Axis {
    if n[0].abs() >= n[1].abs() && n[0].abs() >= n[2].abs() {
        Axis::X
    } else if n[1].abs() >= n[2].abs() {
        Axis::Y
    } else {
        Axis::Z
    }
}

/// Surface quality pass: 2-2 diagonal flips of patch tiles (with the
/// matching tet pairs re-split on both sides) and in-plane Laplacian
/// smoothing of patch-interior surface vertices. The patch REGION is the
/// constraint, not its triangulation, so re-tiling it is legal; crease and
/// rim vertices stay fixed.
/// Projects a point onto an analytic surface (identity for planes).
fn project_to_surface(kind: &SurfaceKind, p: [f64; 3]) -> [f64; 3] {
    match kind {
        SurfaceKind::Plane => p,
        SurfaceKind::Sphere { center, radius } => {
            let w: [f64; 3] = std::array::from_fn(|k| p[k] - center[k]);
            let l = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
            if l < f64::MIN_POSITIVE {
                return p;
            }
            std::array::from_fn(|k| center[k] + radius * w[k] / l)
        }
        SurfaceKind::Cylinder { center, axis, radius } => {
            let al = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
            let a: [f64; 3] = std::array::from_fn(|k| axis[k] / al);
            let w: [f64; 3] = std::array::from_fn(|k| p[k] - center[k]);
            let t: f64 = (0..3).map(|k| w[k] * a[k]).sum();
            let radial: [f64; 3] = std::array::from_fn(|k| w[k] - t * a[k]);
            let rl = (radial[0] * radial[0] + radial[1] * radial[1] + radial[2] * radial[2]).sqrt();
            if rl < f64::MIN_POSITIVE {
                return p;
            }
            std::array::from_fn(|k| center[k] + t * a[k] + radius * radial[k] / rl)
        }
        SurfaceKind::Cone { apex, axis, tan_half_angle } => {
            let al = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
            let a: [f64; 3] = std::array::from_fn(|k| axis[k] / al);
            let w: [f64; 3] = std::array::from_fn(|k| p[k] - apex[k]);
            let t: f64 = (0..3).map(|k| w[k] * a[k]).sum();
            let radial: [f64; 3] = std::array::from_fn(|k| w[k] - t * a[k]);
            let rl = (radial[0] * radial[0] + radial[1] * radial[1] + radial[2] * radial[2]).sqrt();
            if rl < f64::MIN_POSITIVE || t <= 0.0 {
                return p;
            }
            let r_target = t * tan_half_angle;
            std::array::from_fn(|k| apex[k] + t * a[k] + r_target * radial[k] / rl)
        }
    }
}

/// Faces of one PLANE patch may be re-tiled freely; faces of one CURVED
/// analytic surface (sphere/cylinder/cone barrel) form a single smooth group
/// in which vertices slide on the true surface and diagonals flip across the
/// (near-coplanar) facet pairs.
fn face_group(mesh: &TetMesh, fi: usize) -> (u8, u32) {
    let sf = &mesh.faces[fi];
    match mesh.surfaces[sf.surface as usize] {
        SurfaceKind::Plane => (0, sf.patch),
        _ => (1, sf.surface),
    }
}

/// One surface a vertex is constrained to: the (fixed) plane of a plane
/// patch, or an analytic curved kind. A feature vertex on the seam of
/// several surfaces is constrained to all of them at once.
enum SurfConstraint {
    Plane { n: [f64; 3], p0: [f64; 3] },
    Curved(SurfaceKind),
}

impl SurfConstraint {
    fn project(&self, q: [f64; 3]) -> [f64; 3] {
        match self {
            SurfConstraint::Plane { n, p0 } => {
                let off: f64 = (0..3).map(|k| n[k] * (q[k] - p0[k])).sum();
                std::array::from_fn(|k| q[k] - off * n[k])
            }
            SurfConstraint::Curved(kind) => project_to_surface(kind, q),
        }
    }

    fn residual2(&self, q: [f64; 3]) -> f64 {
        let p = self.project(q);
        (0..3).map(|k| (p[k] - q[k]).powi(2)).sum()
    }
}

/// Projects `q` onto the common intersection of all constraints by
/// alternating projections; `None` if it does not converge within `tol`
/// (tangential or degenerate seam).
fn project_onto_all(cons: &[SurfConstraint], mut q: [f64; 3], tol: f64) -> Option<[f64; 3]> {
    for _ in 0..32 {
        for c in cons {
            q = c.project(q);
        }
        if cons.iter().all(|c| c.residual2(q) <= tol * tol) {
            return Some(q);
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn surface_pass(
    mesh: &mut TetMesh,
    g_incident: &mut [Vec<u32>],
    alive: &[bool],
    tet_q: &mut [f64],
    face_idx: &mut DMap<[usize; 3], u32>,
    constrained_faces: &mut DSet<[usize; 3]>,
    constrained_edges: &mut DSet<(usize, usize)>,
    map_tets: &[u32],
    is_active: &impl Fn(&[usize]) -> bool,
    complex_changed: &impl Fn(&[usize]) -> bool,
    edge_budget2: &impl Fn(rapidmesh_geom::RegionTag) -> f64,
    next_dirty: &mut DSet<usize>,
) -> usize {
    let mut ops = 0usize;

    // Maps over active tets/faces only: complete for active candidates (an
    // owner of an active face/edge always contains the active vertex).
    // Sorted entry vectors: grouping and binary search instead of hash maps.
    // ---------------------------------------------------- 2-2 flips
    {
        let mut tof: Vec<([usize; 3], u32)> = Vec::with_capacity(map_tets.len() * 4);
        for &ti in map_tets {
            let t = &mesh.tets[ti as usize];
            for i in 0..4 {
                let f: [usize; 3] = std::array::from_fn(|k| t[(i + 1 + k) % 4]);
                tof.push((sorted3(f), ti));
            }
        }
        tof.sort_unstable();
        let tof_get = |key: [usize; 3]| -> &[([usize; 3], u32)] {
            let lo = tof.partition_point(|e| e.0 < key);
            let hi = lo + tof[lo..].iter().take_while(|e| e.0 == key).count();
            &tof[lo..hi]
        };
        let mut edge_faces: Vec<((usize, usize), u32)> = Vec::with_capacity(mesh.faces.len() * 3);
        for (fi, sf) in mesh.faces.iter().enumerate() {
            if !is_active(&sf.tri) {
                continue;
            }
            for e in 0..3 {
                let (a, b) = (sf.tri[e], sf.tri[(e + 1) % 3]);
                edge_faces.push(((a.min(b), a.max(b)), fi as u32));
            }
        }
        edge_faces.sort_unstable();
        let mut touched_faces: DSet<usize> = DSet::default();
        let mut touched_tets: DSet<usize> = DSet::default();

        let mut gi = 0;
        while gi < edge_faces.len() {
            let key = edge_faces[gi].0;
            let gj = gi + edge_faces[gi..].iter().take_while(|e| e.0 == key).count();
            let fids = &edge_faces[gi..gj];
            gi = gj;
            if fids.len() != 2 {
                continue;
            }
            let (fi1, fi2) = (fids[0].1 as usize, fids[1].1 as usize);
            if touched_faces.contains(&fi1) || touched_faces.contains(&fi2) {
                continue;
            }
            let (sf1, sf2) = (mesh.faces[fi1].clone(), mesh.faces[fi2].clone());
            if face_group(mesh, fi1) != face_group(mesh, fi2)
                || sf1.regions != sf2.regions
                || sf1.face_tag != sf2.face_tag
            {
                continue;
            }
            let (a, b) = key;
            let c = *sf1.tri.iter().find(|&&v| v != a && v != b).expect("apex");
            let d = *sf2.tri.iter().find(|&&v| v != a && v != b).expect("apex");
            if c == d {
                continue;
            }
            // Sizing contract: the new diagonal must stay within the local
            // budget (or the old quad's own longest edge); the 2-2 flip was
            // the one topological operation without the edge gate.
            {
                let quad = [a, b, c, d];
                let mut lmax2 = 0.0f64;
                for i in 0..4 {
                    for j in i + 1..4 {
                        if (quad[i], quad[j]) == (c, d) || (quad[i], quad[j]) == (d, c) {
                            continue;
                        }
                        lmax2 = lmax2.max(
                            (0..3)
                                .map(|k| {
                                    (mesh.points[quad[i]][k] - mesh.points[quad[j]][k]).powi(2)
                                })
                                .sum(),
                        );
                    }
                }
                let budget2 = edge_budget2(sf1.regions[0]).min(edge_budget2(sf1.regions[1]));
                let new2: f64 = (0..3)
                    .map(|k| (mesh.points[c][k] - mesh.points[d][k]).powi(2))
                    .sum();
                if new2 > lmax2.max(budget2) {
                    continue;
                }
            }
            // Exact candidate filter on the full complex (both faces plus
            // their adjacent tet pairs), before any exact predicate work.
            let unchanged = !complex_changed(&sf1.tri)
                && !complex_changed(&sf2.tri)
                && tof_get(sorted3(sf1.tri))
                    .iter()
                    .chain(tof_get(sorted3(sf2.tri)).iter())
                    .all(|e| !complex_changed(&mesh.tets[e.1 as usize]));
            if unchanged {
                continue;
            }
            // Quad convexity, exactly, in the dominant projection.
            let axis = dominant_axis(face_normal(&mesh.points, sf1.tri));
            let pe = |v: usize| Point3::Explicit(mesh.points[v]);
            let opp = |s1: Option<Sign>, s2: Option<Sign>| -> bool {
                matches!(
                    (s1, s2),
                    (Some(Sign::Positive), Some(Sign::Negative))
                        | (Some(Sign::Negative), Some(Sign::Positive))
                )
            };
            if !opp(
                orient2d(&pe(a), &pe(b), &pe(c), axis),
                orient2d(&pe(a), &pe(b), &pe(d), axis),
            ) || !opp(
                orient2d(&pe(c), &pe(d), &pe(a), axis),
                orient2d(&pe(c), &pe(d), &pe(b), axis),
            ) {
                continue;
            }
            // Tet pairing by shared apex on every adjacent side.
            let t1s = tof_get(sorted3(sf1.tri));
            let t2s = tof_get(sorted3(sf2.tri));
            if t1s.len() != t2s.len() || t1s.is_empty() {
                continue;
            }
            if t1s
                .iter()
                .chain(t2s.iter())
                .any(|e| touched_tets.contains(&(e.1 as usize)))
            {
                continue;
            }
            let apex = |ti: usize, tri: [usize; 3]| -> usize {
                *mesh.tets[ti].iter().find(|v| !tri.contains(v)).expect("apex")
            };
            let mut pairs: Vec<(usize, usize, usize)> = Vec::new(); // (t1, t2, apex)
            let mut ok = true;
            for e1 in t1s {
                let t1 = e1.1 as usize;
                let p = apex(t1, sf1.tri);
                match t2s.iter().map(|e| e.1 as usize).find(|&t2| apex(t2, sf2.tri) == p) {
                    Some(t2) if mesh.tet_regions[t1] == mesh.tet_regions[t2] => {
                        pairs.push((t1, t2, p))
                    }
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            // Replacement tets, orientation-fixed; gate on quality gain.
            let mk = |x: usize, p: usize| -> Option<[usize; 4]> {
                let cand = [c, d, x, p];
                if orient_positive(&mesh.points, cand) {
                    Some(cand)
                } else {
                    let s = [d, c, x, p];
                    orient_positive(&mesh.points, s).then_some(s)
                }
            };
            let mut old_q = f64::MAX;
            for &(t1, t2, _) in &pairs {
                if tet_q[t1].is_nan() {
                    tet_q[t1] = quality(&mesh.points, mesh.tets[t1]);
                }
                if tet_q[t2].is_nan() {
                    tet_q[t2] = quality(&mesh.points, mesh.tets[t2]);
                }
                old_q = old_q.min(tet_q[t1]).min(tet_q[t2]);
            }
            if old_q >= TARGET_Q {
                continue;
            }
            let mut new_q = f64::MAX;
            let mut repl: Vec<(usize, [usize; 4], usize, [usize; 4])> = Vec::new();
            for &(t1, t2, p) in &pairs {
                let (Some(n1), Some(n2)) = (mk(a, p), mk(b, p)) else {
                    ok = false;
                    break;
                };
                new_q = new_q.min(quality(&mesh.points, n1)).min(quality(&mesh.points, n2));
                repl.push((t1, n1, t2, n2));
            }
            if !ok || new_q <= old_q + QUALITY_EPS {
                continue;
            }
            for (t1, n1, t2, n2) in repl {
                let (o1, o2) = (mesh.tets[t1], mesh.tets[t2]);
                mesh.tets[t1] = n1;
                mesh.tets[t2] = n2;
                tet_q[t1] = f64::NAN;
                tet_q[t2] = f64::NAN;
                update_incidence(g_incident, t1 as u32, o1, n1);
                update_incidence(g_incident, t2 as u32, o2, n2);
                touched_tets.insert(t1);
                touched_tets.insert(t2);
                for v in n1.iter().chain(n2.iter()) {
                    next_dirty.insert(*v);
                }
            }
            // Persistent constraint bookkeeping: the volume pass gates on
            // these sets (they are no longer rebuilt per pass).
            let (old1, old2) = (sorted3(sf1.tri), sorted3(sf2.tri));
            constrained_faces.remove(&old1);
            constrained_faces.remove(&old2);
            constrained_faces.insert(sorted3([c, d, a]));
            constrained_faces.insert(sorted3([c, d, b]));
            constrained_edges.remove(&(a.min(b), a.max(b)));
            constrained_edges.insert((c.min(d), c.max(d)));
            face_idx.remove(&old1);
            face_idx.remove(&old2);
            face_idx.insert(sorted3([c, d, a]), fi1 as u32);
            face_idx.insert(sorted3([c, d, b]), fi2 as u32);
            mesh.faces[fi1].tri = [c, d, a];
            mesh.faces[fi2].tri = [c, d, b];
            touched_faces.insert(fi1);
            touched_faces.insert(fi2);
            ops += 1;
        }
    }

    // ------------------------------------------- in-plane smoothing
    {
        let mut edge_faces: Vec<((usize, usize), u32)> = Vec::with_capacity(mesh.faces.len() * 3);
        let mut vertex_faces: Vec<(usize, u32)> = Vec::with_capacity(mesh.faces.len() * 3);
        for (fi, sf) in mesh.faces.iter().enumerate() {
            if !is_active(&sf.tri) {
                continue;
            }
            for e in 0..3 {
                let (a, b) = (sf.tri[e], sf.tri[(e + 1) % 3]);
                edge_faces.push(((a.min(b), a.max(b)), fi as u32));
            }
            for &v in &sf.tri {
                vertex_faces.push((v, fi as u32));
            }
        }
        edge_faces.sort_unstable();
        vertex_faces.sort_unstable();
        let ef_get = |key: (usize, usize)| -> &[((usize, usize), u32)] {
            let lo = edge_faces.partition_point(|e| e.0 < key);
            let hi = lo + edge_faces[lo..].iter().take_while(|e| e.0 == key).count();
            &edge_faces[lo..hi]
        };
        let mut gi = 0;
        while gi < vertex_faces.len() {
            let v = vertex_faces[gi].0;
            let gj = gi + vertex_faces[gi..].iter().take_while(|e| e.0 == v).count();
            let vfs: Vec<usize> = vertex_faces[gi..gj].iter().map(|e| e.1 as usize).collect();
            gi = gj;
            if !is_active(&[v]) {
                continue;
            }
            let group = face_group(mesh, vfs[0]);
            let single_group = vfs.iter().all(|&fi| face_group(mesh, fi) == group);
            // Surface neighbors of v, and its FEATURE edges: surface edges
            // not interior to one smooth group (sheet rims, creases between
            // patches). Feature edges are the mesh's 1D constraints.
            let mut nbrs: Vec<usize> = Vec::new();
            let mut feature_nbrs: Vec<usize> = Vec::new();
            for &fi in &vfs {
                let tri = mesh.faces[fi].tri;
                for e in 0..3 {
                    let (x, y) = (tri[e], tri[(e + 1) % 3]);
                    if x != v && y != v {
                        continue;
                    }
                    let w = if x == v { y } else { x };
                    nbrs.push(w);
                    let efs = ef_get((x.min(y), x.max(y)));
                    if efs.len() != 2
                        || face_group(mesh, efs[0].1 as usize) != face_group(mesh, efs[1].1 as usize)
                    {
                        feature_nbrs.push(w);
                    }
                }
            }
            nbrs.sort_unstable();
            nbrs.dedup();
            feature_nbrs.sort_unstable();
            feature_nbrs.dedup();
            if nbrs.is_empty() {
                continue;
            }
            let inc_v: Vec<usize> = g_incident[v]
                .iter()
                .filter(|&&ti| alive[ti as usize])
                .map(|&ti| ti as usize)
                .collect();
            // Exact candidate filter: validity and quality depend on the
            // incident tets, fold-over and sliding on the face neighbors.
            if !(complex_changed(&[v])
                || complex_changed(&nbrs)
                || inc_v.iter().any(|&ti| complex_changed(&mesh.tets[ti])))
            {
                continue;
            }
            let cur = mesh.points[v];
            // Local length scale and tolerance for "is the vertex on its
            // analytic geometry".
            let lref2 = nbrs
                .iter()
                .map(|&w| (0..3).map(|k| (mesh.points[w][k] - cur[k]).powi(2)).sum::<f64>())
                .fold(0.0_f64, f64::max);
            let tol = 1e-7 * lref2.sqrt();
            // A vertex OFF its analytic geometry is a constraint violation
            // (chord-plane Steiner points): snapping it on is accepted on
            // VALIDITY alone. A vertex already on the geometry slides
            // quality-gated.
            let mut snap = false;
            let target: [f64; 3] = if feature_nbrs.is_empty() && single_group {
                // Free in-surface vertex: Laplacian over the surface
                // neighbors. Plane groups project it back into the plane;
                // curved groups project it onto the analytic surface, which
                // both keeps the vertex on the geometry and IMPROVES
                // boundary fidelity.
                let kind = mesh.surfaces[mesh.faces[vfs[0]].surface as usize].clone();
                match kind {
                    SurfaceKind::Plane => {
                        let n = face_normal(&mesh.points, mesh.faces[vfs[0]].tri);
                        let mut avg = [0.0f64; 3];
                        for &w in &nbrs {
                            for (k, acc) in avg.iter_mut().enumerate() {
                                *acc += mesh.points[w][k];
                            }
                        }
                        for acc in &mut avg {
                            *acc /= nbrs.len() as f64;
                        }
                        let off: f64 = (0..3).map(|k| n[k] * (avg[k] - cur[k])).sum();
                        std::array::from_fn(|k| avg[k] - off * n[k])
                    }
                    ref curved => {
                        let proj_cur = project_to_surface(curved, cur);
                        let d2: f64 = (0..3).map(|k| (proj_cur[k] - cur[k]).powi(2)).sum();
                        if d2 > tol * tol {
                            snap = true;
                            proj_cur
                        } else {
                            let mut avg = [0.0f64; 3];
                            for &w in &nbrs {
                                for (k, acc) in avg.iter_mut().enumerate() {
                                    *acc += mesh.points[w][k];
                                }
                            }
                            for acc in &mut avg {
                                *acc /= nbrs.len() as f64;
                            }
                            project_to_surface(curved, avg)
                        }
                    }
                }
            } else if feature_nbrs.len() == 2 {
                let (u, w) = (feature_nbrs[0], feature_nbrs[1]);
                // The distinct surfaces meeting at this feature vertex.
                let mut group_reps: Vec<((u8, u32), usize)> =
                    vfs.iter().map(|&fi| (face_group(mesh, fi), fi)).collect();
                group_reps.sort_unstable_by_key(|&(g, _)| g);
                group_reps.dedup_by_key(|&mut (g, _)| g);
                if group_reps.iter().all(|&((k, _), _)| k == 0) {
                    // All-plane feature with exactly collinear feature
                    // edges: the feature is straight here, so the vertex
                    // may SLIDE along it (1D Laplacian: the midpoint of its
                    // feature neighbors). Bent plane-plane features must
                    // keep their corner; sliding would change the patch
                    // geometry and the region volumes. Coordinates shared
                    // by both neighbors are preserved bit-exactly, so
                    // axis-aligned creases stay exactly on their line and
                    // in their patches' planes.
                    let pe = |i: usize| Point3::Explicit(mesh.points[i]);
                    if collinear(&pe(u), &pe(v), &pe(w)) != Some(true) {
                        continue;
                    }
                    std::array::from_fn(|k| 0.5 * (mesh.points[u][k] + mesh.points[w][k]))
                } else {
                    // Curved feature curve (e.g. the rim circle where a
                    // cylinder meets a plane): the true curve is the
                    // intersection of the adjacent surfaces. Plane anchors
                    // come from a face vertex OTHER than v (those lie on
                    // the patch plane and do not move).
                    let cons: Vec<SurfConstraint> = group_reps
                        .iter()
                        .map(|&((kind, _), fi)| {
                            let tri = mesh.faces[fi].tri;
                            if kind == 0 {
                                let p0 = *tri.iter().find(|&&x| x != v).expect("anchor");
                                SurfConstraint::Plane {
                                    n: face_normal(&mesh.points, tri),
                                    p0: mesh.points[p0],
                                }
                            } else {
                                SurfConstraint::Curved(
                                    mesh.surfaces[mesh.faces[fi].surface as usize].clone(),
                                )
                            }
                        })
                        .collect();
                    let Some(snapped) = project_onto_all(&cons, cur, tol) else {
                        continue; // tangential / degenerate seam: pinned
                    };
                    let d2: f64 = (0..3).map(|k| (snapped[k] - cur[k]).powi(2)).sum();
                    if d2 > tol * tol {
                        snap = true;
                        snapped
                    } else {
                        // On the curve: slide along it (1D Laplacian of the
                        // feature neighbors, projected back onto the curve).
                        let mid: [f64; 3] = std::array::from_fn(|k| {
                            0.5 * (mesh.points[u][k] + mesh.points[w][k])
                        });
                        let Some(t) = project_onto_all(&cons, mid, tol) else {
                            continue;
                        };
                        t
                    }
                }
            } else {
                // Corner and junction vertices are pinned.
                continue;
            };
            if !snap {
                let move2: f64 = (0..3).map(|k| (target[k] - cur[k]).powi(2)).sum();
                if move2 < MIN_REL_MOVE * MIN_REL_MOVE * lref2 {
                    continue;
                }
                // Sizing contract (see the volume smoothing gate). Snaps
                // are constraints and exempt.
                let mut budget2 = f64::INFINITY;
                for &ti in &inc_v {
                    budget2 = budget2.min(edge_budget2(mesh.tet_regions[ti]));
                }
                let mut old_lmax2 = 0.0f64;
                let mut new_lmax2 = 0.0f64;
                for &w in &nbrs {
                    let dw_old: f64 =
                        (0..3).map(|k| (mesh.points[w][k] - cur[k]).powi(2)).sum();
                    let dw_new: f64 =
                        (0..3).map(|k| (mesh.points[w][k] - target[k]).powi(2)).sum();
                    old_lmax2 = old_lmax2.max(dw_old);
                    new_lmax2 = new_lmax2.max(dw_new);
                }
                if new_lmax2 > old_lmax2.max(budget2) {
                    continue;
                }
            }
            // Fold-over guard data: normals before the move, per face.
            let old_normals: Vec<[f64; 3]> = vfs
                .iter()
                .map(|&fi| face_normal(&mesh.points, mesh.faces[fi].tri))
                .collect();

            let mut old_q = f64::MAX;
            for &ti in &inc_v {
                if tet_q[ti].is_nan() {
                    tet_q[ti] = quality(&mesh.points, mesh.tets[ti]);
                }
                old_q = old_q.min(tet_q[ti]);
            }
            if !snap && old_q >= TARGET_Q {
                continue;
            }
            mesh.points[v] = target;
            let tets_ok = inc_v
                .iter()
                .all(|&ti| orient_positive(&mesh.points, mesh.tets[ti]));
            // Fold-over guard: every incident surface face keeps its own
            // previous normal direction (per-face, so curved groups with
            // varying normals are handled correctly).
            let faces_ok = vfs.iter().zip(&old_normals).all(|(&fi, no)| {
                let nf = face_normal(&mesh.points, mesh.faces[fi].tri);
                nf[0] * no[0] + nf[1] * no[1] + nf[2] * no[2] > 0.1
            });
            let new_q = if tets_ok && faces_ok {
                inc_v
                    .iter()
                    .map(|&ti| quality(&mesh.points, mesh.tets[ti]))
                    .fold(f64::MAX, f64::min)
            } else {
                f64::MIN
            };
            // Fidelity snaps are a constraint, not an optimization: they
            // are accepted whenever the result is valid; later passes
            // recover quality SUBJECT to the geometry.
            let accept = if snap {
                tets_ok && faces_ok
            } else {
                new_q > old_q + QUALITY_EPS
            };
            if accept {
                ops += 1;
                next_dirty.insert(v);
                // The vertex moved: invalidate the cached qualities of its
                // (persistent) incidence.
                for &ti in &g_incident[v] {
                    tet_q[ti as usize] = f64::NAN;
                }
            } else {
                mesh.points[v] = cur;
            }
        }
    }

    ops
}

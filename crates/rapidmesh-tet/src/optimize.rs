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

use crate::conform::{tet_min_dihedral_deg, TetMesh};
use rapidmesh_exact::{collinear, orient2d, Axis, Point3, Sign};
use rapidmesh_geom::SurfaceKind;
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;

type DState = BuildHasherDefault<rustc_hash::FxHasher>;
type DMap<K, V> = HashMap<K, V, DState>;
type DSet<T> = HashSet<T, DState>;

/// Parameters for [`optimize`].
#[derive(Debug, Clone)]
pub struct OptimizeParams {
    /// Maximum number of smoothing+flip rounds.
    pub passes: usize,
}

impl Default for OptimizeParams {
    fn default() -> Self {
        OptimizeParams { passes: 8 }
    }
}

fn orient_positive(points: &[[f64; 3]], t: [usize; 4]) -> bool {
    Sign::of_f64(geometry_predicates::orient3d(
        points[t[0]],
        points[t[1]],
        points[t[2]],
        points[t[3]],
    )) == Sign::Positive
}

fn quality(points: &[[f64; 3]], t: [usize; 4]) -> f64 {
    tet_min_dihedral_deg(std::array::from_fn(|k| points[t[k]]))
}

fn sorted3(f: [usize; 3]) -> [usize; 3] {
    let mut s = f;
    s.sort_unstable();
    s
}

/// In-place quality optimization. Returns the number of accepted operations.
pub fn optimize(mesh: &mut TetMesh, params: &OptimizeParams) -> usize {
    let mut total_ops = 0usize;
    for _pass in 0..params.passes {
        let mut ops = 0usize;

        // Surface improvement first: boundary slivers cannot be fixed by
        // interior-only operations.
        ops += surface_pass(mesh);

        // Constrained surface (recomputed per pass: the surface pass re-tiles
        // patches): faces, their edges, their vertices.
        let mut constrained_faces: DSet<[usize; 3]> = DSet::default();
        let mut constrained_edges: DSet<(usize, usize)> = DSet::default();
        let mut constrained_verts: DSet<usize> = DSet::default();
        for sf in &mesh.faces {
            constrained_faces.insert(sorted3(sf.tri));
            for e in 0..3 {
                let (a, b) = (sf.tri[e], sf.tri[(e + 1) % 3]);
                constrained_edges.insert((a.min(b), a.max(b)));
            }
            for &v in &sf.tri {
                constrained_verts.insert(v);
            }
        }

        // ------------------------------------------------- smoothing
        // Topology is fixed during smoothing, so incidence is built once.
        let mut incident: Vec<Vec<u32>> = vec![Vec::new(); mesh.points.len()];
        for (ti, t) in mesh.tets.iter().enumerate() {
            for &v in t {
                incident[v].push(ti as u32);
            }
        }
        let mut nbrs: Vec<usize> = Vec::new();
        for (v, inc) in incident.iter().enumerate() {
            if constrained_verts.contains(&v) || inc.is_empty() {
                continue;
            }
            nbrs.clear();
            for &ti in inc {
                for &w in &mesh.tets[ti as usize] {
                    if w != v {
                        nbrs.push(w);
                    }
                }
            }
            nbrs.sort_unstable();
            nbrs.dedup();
            let mut avg = [0.0f64; 3];
            for &w in &nbrs {
                for (k, a) in avg.iter_mut().enumerate() {
                    *a += mesh.points[w][k];
                }
            }
            for a in &mut avg {
                *a /= nbrs.len() as f64;
            }
            let old_pos = mesh.points[v];
            let old_q = inc
                .iter()
                .map(|&ti| quality(&mesh.points, mesh.tets[ti as usize]))
                .fold(f64::MAX, f64::min);
            mesh.points[v] = avg;
            let valid = inc
                .iter()
                .all(|&ti| orient_positive(&mesh.points, mesh.tets[ti as usize]));
            let new_q = if valid {
                inc
                    .iter()
                    .map(|&ti| quality(&mesh.points, mesh.tets[ti as usize]))
                    .fold(f64::MAX, f64::min)
            } else {
                f64::MIN
            };
            if new_q > old_q + 1e-9 {
                ops += 1;
            } else {
                mesh.points[v] = old_pos;
            }
        }

        // ------------------------------------------------------ flips
        let mut alive: Vec<bool> = vec![true; mesh.tets.len()];
        let mut added: Vec<([usize; 4], rapidmesh_geom::RegionTag)> = Vec::new();

        // Face map for 2-3 flips and edge map for 3-2 flips.
        let mut face_map: DMap<[usize; 3], Vec<u32>> = DMap::default();
        let mut edge_map: DMap<(usize, usize), Vec<u32>> = DMap::default();
        for (ti, t) in mesh.tets.iter().enumerate() {
            for i in 0..4 {
                let f: Vec<usize> = (0..4).filter(|&k| k != i).map(|k| t[k]).collect();
                face_map
                    .entry(sorted3([f[0], f[1], f[2]]))
                    .or_default()
                    .push(ti as u32);
            }
            for i in 0..4 {
                for j in i + 1..4 {
                    let (a, b) = (t[i].min(t[j]), t[i].max(t[j]));
                    edge_map.entry((a, b)).or_default().push(ti as u32);
                }
            }
        }

        // Deterministic iteration: sort keys.
        let mut faces: Vec<[usize; 3]> = face_map.keys().copied().collect();
        faces.sort_unstable();
        for f in faces {
            let owners = &face_map[&f];
            if owners.len() != 2 || constrained_faces.contains(&f) {
                continue;
            }
            let (t1, t2) = (owners[0] as usize, owners[1] as usize);
            if !alive[t1] || !alive[t2] {
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
            let old_q = quality(&mesh.points, mesh.tets[t1])
                .min(quality(&mesh.points, mesh.tets[t2]));
            let new_q = quality(&mesh.points, n1)
                .min(quality(&mesh.points, n2))
                .min(quality(&mesh.points, n3));
            if new_q <= old_q + 1e-9 {
                continue;
            }
            alive[t1] = false;
            alive[t2] = false;
            let region = mesh.tet_regions[t1];
            added.push((n1, region));
            added.push((n2, region));
            added.push((n3, region));
            ops += 1;
        }

        // EDGE REMOVAL (the 3-2 flip generalized to rings of k tets): the k
        // tets around an unconstrained interior edge d-e are replaced by
        // 2(k-2) tets over the max-min-quality triangulation of the ring
        // polygon (Klincsek interval DP). Validity is positivity of every
        // new tet in the FIXED ring orientation (no flipping): a folded
        // triangulation triangle then shows up as a negative tet, so the
        // accepted complex tiles exactly the old star.
        let mut edges: Vec<(usize, usize)> = edge_map.keys().copied().collect();
        edges.sort_unstable();
        for key in edges {
            if constrained_edges.contains(&key) {
                continue;
            }
            let owners = &edge_map[&key];
            let k = owners.len();
            if !(3..=7).contains(&k) {
                continue;
            }
            let ts: Vec<usize> = owners.iter().map(|&t| t as usize).collect();
            if ts.iter().any(|&t| !alive[t]) {
                continue;
            }
            if ts
                .iter()
                .any(|&t| mesh.tet_regions[t] != mesh.tet_regions[ts[0]])
            {
                continue;
            }
            let (d, e) = key;
            // Each tet contributes the pair of its two non-(d,e) vertices;
            // the pairs must chain into a single closed ring (interior
            // edge), walked here into cyclic order.
            let mut partners: DMap<usize, Vec<usize>> = DMap::default();
            for &t in &ts {
                let mut pr = mesh.tets[t].iter().copied().filter(|&x| x != d && x != e);
                let (a, b) = (pr.next().expect("pair"), pr.next().expect("pair"));
                partners.entry(a).or_default().push(b);
                partners.entry(b).or_default().push(a);
            }
            if partners.len() != k || partners.values().any(|p| p.len() != 2) {
                continue;
            }
            let start = *partners.keys().min().expect("nonempty");
            let mut ring: Vec<usize> = vec![start];
            let mut prev = usize::MAX;
            while ring.len() < k {
                let cur = *ring.last().expect("nonempty");
                let p = &partners[&cur];
                let next = if p[0] != prev { p[0] } else { p[1] };
                prev = cur;
                ring.push(next);
            }
            // Closed ring?
            if !partners[&ring[k - 1]].contains(&start) {
                continue;
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
                ring.reverse();
            }
            if (0..k).any(|i| side(ring[i], ring[(i + 1) % k]) != Sign::Positive) {
                continue;
            }
            // Klincsek DP over the ring polygon: best[i][j] = max-min
            // quality of triangulating the sub-polygon ring[i..=j]. Each
            // triangle (p, q, r) in ring order spawns the tet pair
            // (p, q, r, e) and (p, r, q, d), both required positive.
            let pair_q = |i: usize, m: usize, j: usize| -> f64 {
                let (p, q, r) = (ring[i], ring[m], ring[j]);
                let t1 = [p, q, r, e];
                let t2 = [p, r, q, d];
                if !orient_positive(&mesh.points, t1) || !orient_positive(&mesh.points, t2) {
                    return f64::MIN;
                }
                quality(&mesh.points, t1).min(quality(&mesh.points, t2))
            };
            let mut best = vec![vec![f64::MAX; k]; k];
            let mut cut = vec![vec![usize::MAX; k]; k];
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
            let old_q = ts
                .iter()
                .map(|&t| quality(&mesh.points, mesh.tets[t]))
                .fold(f64::MAX, f64::min);
            if best[0][k - 1] <= old_q + 1e-9 {
                continue;
            }
            let region = mesh.tet_regions[ts[0]];
            let mut stack: Vec<(usize, usize)> = vec![(0, k - 1)];
            while let Some((i, j)) = stack.pop() {
                if j - i < 2 {
                    continue;
                }
                let m = cut[i][j];
                added.push(([ring[i], ring[m], ring[j], e], region));
                added.push(([ring[i], ring[j], ring[m], d], region));
                stack.push((i, m));
                stack.push((m, j));
            }
            for &t in &ts {
                alive[t] = false;
            }
            ops += 1;
        }

        // Compact.
        if !added.is_empty() || alive.iter().any(|&a| !a) {
            let mut tets = Vec::with_capacity(mesh.tets.len());
            let mut regions = Vec::with_capacity(mesh.tets.len());
            for (ti, t) in mesh.tets.iter().enumerate() {
                if alive[ti] {
                    tets.push(*t);
                    regions.push(mesh.tet_regions[ti]);
                }
            }
            for (t, r) in added {
                tets.push(t);
                regions.push(r);
            }
            mesh.tets = tets;
            mesh.tet_regions = regions;
        }

        total_ops += ops;
        if ops == 0 {
            break;
        }
    }
    total_ops
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

fn surface_pass(mesh: &mut TetMesh) -> usize {
    let mut ops = 0usize;

    // ---------------------------------------------------- 2-2 flips
    {
        let mut tets_of_face: DMap<[usize; 3], Vec<usize>> = DMap::default();
        for (ti, t) in mesh.tets.iter().enumerate() {
            for i in 0..4 {
                let f: Vec<usize> = (0..4).filter(|&k| k != i).map(|k| t[k]).collect();
                tets_of_face
                    .entry(sorted3([f[0], f[1], f[2]]))
                    .or_default()
                    .push(ti);
            }
        }
        let mut edge_faces: DMap<(usize, usize), Vec<usize>> = DMap::default();
        for (fi, sf) in mesh.faces.iter().enumerate() {
            for e in 0..3 {
                let (a, b) = (sf.tri[e], sf.tri[(e + 1) % 3]);
                edge_faces.entry((a.min(b), a.max(b))).or_default().push(fi);
            }
        }
        let mut keys: Vec<(usize, usize)> = edge_faces.keys().copied().collect();
        keys.sort_unstable();
        let mut touched_faces: DSet<usize> = DSet::default();
        let mut touched_tets: DSet<usize> = DSet::default();

        for key in keys {
            let fids = &edge_faces[&key];
            if fids.len() != 2 {
                continue;
            }
            let (fi1, fi2) = (fids[0], fids[1]);
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
            let t1s = tets_of_face.get(&sorted3(sf1.tri)).cloned().unwrap_or_default();
            let t2s = tets_of_face.get(&sorted3(sf2.tri)).cloned().unwrap_or_default();
            if t1s.len() != t2s.len() || t1s.is_empty() {
                continue;
            }
            if t1s.iter().chain(t2s.iter()).any(|t| touched_tets.contains(t)) {
                continue;
            }
            let apex = |ti: usize, tri: [usize; 3]| -> usize {
                *mesh.tets[ti].iter().find(|v| !tri.contains(v)).expect("apex")
            };
            let mut pairs: Vec<(usize, usize, usize)> = Vec::new(); // (t1, t2, apex)
            let mut ok = true;
            for &t1 in &t1s {
                let p = apex(t1, sf1.tri);
                match t2s.iter().find(|&&t2| apex(t2, sf2.tri) == p) {
                    Some(&t2) if mesh.tet_regions[t1] == mesh.tet_regions[t2] => {
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
            let mut new_q = f64::MAX;
            let mut repl: Vec<(usize, [usize; 4], usize, [usize; 4])> = Vec::new();
            for &(t1, t2, p) in &pairs {
                old_q = old_q
                    .min(quality(&mesh.points, mesh.tets[t1]))
                    .min(quality(&mesh.points, mesh.tets[t2]));
                let (Some(n1), Some(n2)) = (mk(a, p), mk(b, p)) else {
                    ok = false;
                    break;
                };
                new_q = new_q.min(quality(&mesh.points, n1)).min(quality(&mesh.points, n2));
                repl.push((t1, n1, t2, n2));
            }
            if !ok || new_q <= old_q + 1e-9 {
                continue;
            }
            for (t1, n1, t2, n2) in repl {
                mesh.tets[t1] = n1;
                mesh.tets[t2] = n2;
                touched_tets.insert(t1);
                touched_tets.insert(t2);
            }
            mesh.faces[fi1].tri = [c, d, a];
            mesh.faces[fi2].tri = [c, d, b];
            touched_faces.insert(fi1);
            touched_faces.insert(fi2);
            ops += 1;
        }
    }

    // ------------------------------------------- in-plane smoothing
    {
        let mut edge_faces: DMap<(usize, usize), Vec<usize>> = DMap::default();
        let mut vertex_faces: DMap<usize, Vec<usize>> = DMap::default();
        for (fi, sf) in mesh.faces.iter().enumerate() {
            for e in 0..3 {
                let (a, b) = (sf.tri[e], sf.tri[(e + 1) % 3]);
                edge_faces.entry((a.min(b), a.max(b))).or_default().push(fi);
            }
            for &v in &sf.tri {
                vertex_faces.entry(v).or_default().push(fi);
            }
        }
        let mut incident: Vec<Vec<usize>> = vec![Vec::new(); mesh.points.len()];
        for (ti, t) in mesh.tets.iter().enumerate() {
            for &v in t {
                incident[v].push(ti);
            }
        }
        let mut verts: Vec<usize> = vertex_faces.keys().copied().collect();
        verts.sort_unstable();
        for v in verts {
            let vfs = &vertex_faces[&v];
            let group = face_group(mesh, vfs[0]);
            let single_group = vfs.iter().all(|&fi| face_group(mesh, fi) == group);
            // Surface neighbors of v, and its FEATURE edges: surface edges
            // not interior to one smooth group (sheet rims, creases between
            // patches). Feature edges are the mesh's 1D constraints.
            let mut nbrs: Vec<usize> = Vec::new();
            let mut feature_nbrs: Vec<usize> = Vec::new();
            for &fi in vfs {
                let tri = mesh.faces[fi].tri;
                for e in 0..3 {
                    let (x, y) = (tri[e], tri[(e + 1) % 3]);
                    if x != v && y != v {
                        continue;
                    }
                    let w = if x == v { y } else { x };
                    nbrs.push(w);
                    let efs = &edge_faces[&(x.min(y), x.max(y))];
                    if efs.len() != 2 || face_group(mesh, efs[0]) != face_group(mesh, efs[1]) {
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
            // Fold-over guard data: normals before the move, per face.
            let old_normals: Vec<[f64; 3]> = vfs
                .iter()
                .map(|&fi| face_normal(&mesh.points, mesh.faces[fi].tri))
                .collect();

            let old_q = incident[v]
                .iter()
                .map(|&ti| quality(&mesh.points, mesh.tets[ti]))
                .fold(f64::MAX, f64::min);
            mesh.points[v] = target;
            let tets_ok = incident[v]
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
                incident[v]
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
                new_q > old_q + 1e-9
            };
            if accept {
                ops += 1;
            } else {
                mesh.points[v] = cur;
            }
        }
    }

    ops
}

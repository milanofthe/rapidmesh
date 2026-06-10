//! Post-pass mesh quality optimization: smart Laplacian smoothing of free
//! interior vertices and 2-3 / 3-2 bistellar flips, all constrained-aware.
//!
//! The conforming surface (patch tiles) is untouchable: constrained faces
//! are never flipped away, their edges never removed, and their vertices
//! never moved, so conformity and the exact region volumes are invariants
//! of this pass (interior modifications telescope over the fixed boundary).
//! Every operation is gated on improving the local minimum dihedral angle
//! and on exact positive orientation of the replacement tets, so the pass
//! can only improve the mesh and terminates when nothing improves.

use crate::conform::{tet_min_dihedral_deg, TetMesh};
use rapidmesh_exact::{orient2d, Axis, Point3, Sign};
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;

type DState = BuildHasherDefault<std::collections::hash_map::DefaultHasher>;
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

        let mut edges: Vec<(usize, usize)> = edge_map.keys().copied().collect();
        edges.sort_unstable();
        for key in edges {
            if constrained_edges.contains(&key) {
                continue;
            }
            let owners = &edge_map[&key];
            if owners.len() != 3 {
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
            // Ring vertices: each tet contributes one vertex besides d, e.
            let mut ring: Vec<usize> = Vec::new();
            for &t in &ts {
                for &v in &mesh.tets[t] {
                    if v != d && v != e && !ring.contains(&v) {
                        ring.push(v);
                    }
                }
            }
            if ring.len() != 3 {
                continue;
            }
            let (a, b, c) = (ring[0], ring[1], ring[2]);
            // Geometric validity: the removed edge d-e must cross the
            // interior of triangle (a, b, c), or the two new tets do not
            // tile the ring's union.
            let side = |u: usize, v: usize| {
                Sign::of_f64(geometry_predicates::orient3d(
                    mesh.points[u],
                    mesh.points[v],
                    mesh.points[d],
                    mesh.points[e],
                ))
            };
            let (s0, s1, s2) = (side(a, b), side(b, c), side(c, a));
            if s0 == Sign::Zero || s0 != s1 || s1 != s2 {
                continue;
            }
            let mk = |t: [usize; 4]| -> Option<[usize; 4]> {
                if orient_positive(&mesh.points, t) {
                    Some(t)
                } else {
                    let s = [t[0], t[2], t[1], t[3]];
                    orient_positive(&mesh.points, s).then_some(s)
                }
            };
            let (Some(n1), Some(n2)) = (mk([a, b, c, d]), mk([a, b, c, e])) else {
                continue;
            };
            let old_q = ts
                .iter()
                .map(|&t| quality(&mesh.points, mesh.tets[t]))
                .fold(f64::MAX, f64::min);
            let new_q = quality(&mesh.points, n1).min(quality(&mesh.points, n2));
            if new_q <= old_q + 1e-9 {
                continue;
            }
            for &t in &ts {
                alive[t] = false;
            }
            let region = mesh.tet_regions[ts[0]];
            added.push((n1, region));
            added.push((n2, region));
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
            if sf1.patch != sf2.patch {
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
            // Free in-plane: all incident faces on ONE patch, and every
            // incident surface edge interior to the complex (2 faces) — rim
            // and crease vertices stay fixed.
            let patch = mesh.faces[vfs[0]].patch;
            if vfs.iter().any(|&fi| mesh.faces[fi].patch != patch) {
                continue;
            }
            let mut nbrs: Vec<usize> = Vec::new();
            let mut rim = false;
            for &fi in vfs {
                let tri = mesh.faces[fi].tri;
                for e in 0..3 {
                    let (x, y) = (tri[e], tri[(e + 1) % 3]);
                    if x != v && y != v {
                        continue;
                    }
                    let w = if x == v { y } else { x };
                    nbrs.push(w);
                    if edge_faces[&(x.min(y), x.max(y))].len() != 2 {
                        rim = true;
                    }
                }
            }
            if rim || nbrs.is_empty() {
                continue;
            }
            nbrs.sort_unstable();
            nbrs.dedup();
            // Laplacian over the surface neighbors, projected back onto the
            // patch plane through the current position.
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
            let cur = mesh.points[v];
            let off: f64 = (0..3).map(|k| n[k] * (avg[k] - cur[k])).sum();
            let target: [f64; 3] = std::array::from_fn(|k| avg[k] - off * n[k]);

            let old_q = incident[v]
                .iter()
                .map(|&ti| quality(&mesh.points, mesh.tets[ti]))
                .fold(f64::MAX, f64::min);
            mesh.points[v] = target;
            let tets_ok = incident[v]
                .iter()
                .all(|&ti| orient_positive(&mesh.points, mesh.tets[ti]));
            // Fold-over guard: every incident surface face keeps its normal
            // direction.
            let faces_ok = vfs.iter().all(|&fi| {
                let nf = face_normal(&mesh.points, mesh.faces[fi].tri);
                nf[0] * n[0] + nf[1] * n[1] + nf[2] * n[2] > 0.1
            });
            let new_q = if tets_ok && faces_ok {
                incident[v]
                    .iter()
                    .map(|&ti| quality(&mesh.points, mesh.tets[ti]))
                    .fold(f64::MAX, f64::min)
            } else {
                f64::MIN
            };
            if new_q > old_q + 1e-9 {
                ops += 1;
            } else {
                mesh.points[v] = cur;
            }
        }
    }

    ops
}

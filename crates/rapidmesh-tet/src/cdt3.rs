//! Stage 3: boundary-constrained tetrahedralization (conforming CDT).
//!
//! Given the frozen Stage-2 surface mesh `S` (vertices on their exact carriers,
//! plus a triangulation with a per-facet carrier) and a set of relaxed interior
//! points, this builds a Delaunay tetrahedralization that contains every triangle
//! of `S` as a union of tet faces (\cref{prop:watertight}). Boundary recovery is
//! conforming, by Steiner insertion ON the constraint (Diazzi et al. 2023): a
//! missing edge or facet is split, and the new vertex is constructed via the
//! carrier ([`Site::exact`]) so it lands EXACTLY on the carrier (a
//! [`Point3::Lnc`] on a straight edge line, a [`Point3::Pac`] on a plane). The
//! surface geometry is therefore preserved, planar region volumes stay bit-exact,
//! and the next recovery round sees the carrier intact. There is then a
//! tetrahedron on each side of every surface triangle, so region labelling is a
//! flood fill that never leaks.
//!
//! This replaces the unconstrained-Delaunay + centroid-classification path, which
//! recovered the boundary only statistically (\cref{sec:conformity}).

use crate::delaunay::DelaunayBuilder;
use crate::site::{Carrier, Site};
use rapidmesh_exact::{orient3d, Point3, Sign};

type V3 = [f64; 3];

// ---- 3D bistellar flips (the engine of flip-based boundary recovery) --------
//
// A surface facet from Stage 2 that the Delaunay does not contain as a face is
// pierced by a tet edge; a flip removes that edge and makes the facet appear
// (the 3D analogue of the 2D constrained-edge flip). Both flips reuse the
// builder's [`DelaunayBuilder::replace_cavity`], which validates orientation and
// the oriented-boundary balance, so an invalid flip is rejected; we pre-check
// validity (convexity) so we never trigger that panic.

/// Exact orientation sign of the tet `(a,b,c,d)` (public builder indices).
fn orient(db: &DelaunayBuilder, a: usize, b: usize, c: usize, d: usize) -> Sign {
    orient3d(&db.exact_point(a), &db.exact_point(b), &db.exact_point(c), &db.exact_point(d))
        .unwrap_or(Sign::Zero)
}

/// The tet vertices reordered to positive orientation, or `None` if degenerate.
fn positive(db: &DelaunayBuilder, t: [usize; 4]) -> Option<[usize; 4]> {
    match orient(db, t[0], t[1], t[2], t[3]) {
        Sign::Positive => Some(t),
        Sign::Negative => Some([t[0], t[1], t[3], t[2]]),
        Sign::Zero => None,
    }
}

/// 2-3 flip across the interior face shared by slots `s1`, `s2`: replaces the two
/// tets by three sharing the new edge `(d,e)` (the two apexes). Returns the
/// created edge on success, or `None` if the union is not convex (the edge `d-e`
/// does not pierce the shared triangle, so the flip is invalid).
fn flip23(db: &mut DelaunayBuilder, s1: u32, s2: u32) -> Option<(usize, usize)> {
    let t1 = db.tet_at(s1)?;
    let t2 = db.tet_at(s2)?;
    let shared: Vec<usize> = t1.iter().copied().filter(|v| t2.contains(v)).collect();
    if shared.len() != 3 {
        return None;
    }
    let d = *t1.iter().find(|v| !shared.contains(v))?;
    let e = *t2.iter().find(|v| !shared.contains(v))?;
    let (a, b, c) = (shared[0], shared[1], shared[2]);
    // Valid iff edge (d,e) passes through triangle (a,b,c): the three tets
    // d-e-(edge of abc) are consistently oriented (the orient3d signs agree).
    let s_ab = orient(db, d, e, a, b);
    let s_bc = orient(db, d, e, b, c);
    let s_ca = orient(db, d, e, c, a);
    if s_ab == Sign::Zero || s_ab != s_bc || s_bc != s_ca {
        return None;
    }
    let n1 = positive(db, [d, e, a, b])?;
    let n2 = positive(db, [d, e, b, c])?;
    let n3 = positive(db, [d, e, c, a])?;
    db.replace_cavity(&[s1, s2], &[n1, n2, n3]);
    Some((d, e))
}

/// Does the open tet edge `(p,q)` pierce the interior of triangle `(a,b,c)`?
/// True iff `p,q` are on opposite sides of the triangle's plane and the segment
/// crosses the triangle interior (the three tets `p-q-(edge of abc)` agree in
/// orientation, the same convexity test `flip23` uses). All-exact.
fn edge_pierces_facet(db: &DelaunayBuilder, p: usize, q: usize, a: usize, b: usize, c: usize) -> bool {
    let sp = orient(db, a, b, c, p);
    let sq = orient(db, a, b, c, q);
    if sp == Sign::Zero || sq == Sign::Zero || sp == sq {
        return false;
    }
    let s_ab = orient(db, p, q, a, b);
    let s_bc = orient(db, p, q, b, c);
    let s_ca = orient(db, p, q, c, a);
    s_ab != Sign::Zero && s_ab == s_bc && s_bc == s_ca
}

/// A tet edge that pierces triangle `(a,b,c)`'s interior, if any (the obstruction
/// that keeps the facet from being a mesh face).
fn piercing_edge(db: &DelaunayBuilder, a: usize, b: usize, c: usize) -> Option<(usize, usize)> {
    for (_, t) in db.tets_with_slots() {
        for &(i, j) in &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
            let (p, q) = (t[i], t[j]);
            if p == a || p == b || p == c || q == a || q == b || q == c {
                continue; // an edge sharing a facet vertex cannot pierce the interior
            }
            if edge_pierces_facet(db, p, q, a, b, c) {
                return Some((p, q));
            }
        }
    }
    None
}

/// Outcome of a facet-recovery attempt.
#[derive(Debug, PartialEq, Eq)]
pub enum Recover {
    /// The facet is now a union of mesh faces.
    Done,
    /// No flip resolved it (a Schönhardt-type lock): the caller splits the facet
    /// at a Steiner point on its carrier.
    NeedSteiner,
}

/// Flip-based facet recovery: makes triangle `(a,b,c)` a mesh face by flipping
/// away the tet edges piercing it (\cref{alg:recover}). Each piercing edge of
/// tet-degree three is removed by a 3-2 flip; a piercing edge no flip resolves
/// returns [`Recover::NeedSteiner`] (the rare fallback). Planar facets never need
/// this (their region is already covered), so the caller only invokes it on
/// curved/feature facets.
fn recover_facet(db: &mut DelaunayBuilder, a: usize, b: usize, c: usize) -> Recover {
    let cap = db.slot_count() + 64;
    for _ in 0..cap {
        if db.face_exists(a, b, c) {
            return Recover::Done;
        }
        match piercing_edge(db, a, b, c) {
            Some((p, q)) => {
                if flip32(db, p, q).is_none() {
                    return Recover::NeedSteiner; // degree != 3 / non-convex
                }
            }
            None => {
                return if db.face_exists(a, b, c) { Recover::Done } else { Recover::NeedSteiner };
            }
        }
    }
    Recover::NeedSteiner
}

/// 3-2 flip removing edge `(d,e)` when it is shared by exactly three tets:
/// replaces them by two tets sharing the new face `(a,b,c)` (the edge's ring).
/// Returns the created face, or `None` if `(d,e)` is not shared by exactly three
/// tets or the flip is not convex.
fn flip32(db: &mut DelaunayBuilder, d: usize, e: usize) -> Option<[usize; 3]> {
    let slots: Vec<u32> = db
        .star_slots(d)
        .into_iter()
        .filter(|&s| db.tet_at(s).map_or(false, |t| t.contains(&e)))
        .collect();
    if slots.len() != 3 {
        return None;
    }
    let mut ring: Vec<usize> = Vec::new();
    for &s in &slots {
        for v in db.tet_at(s)? {
            if v != d && v != e && !ring.contains(&v) {
                ring.push(v);
            }
        }
    }
    if ring.len() != 3 {
        return None;
    }
    let (a, b, c) = (ring[0], ring[1], ring[2]);
    // Valid iff d and e are on opposite sides of the new face's plane.
    let sd = orient(db, a, b, c, d);
    let se = orient(db, a, b, c, e);
    if sd == Sign::Zero || se == Sign::Zero || sd == se {
        return None;
    }
    let n1 = positive(db, [a, b, c, d])?;
    let n2 = positive(db, [a, b, c, e])?;
    db.replace_cavity(&slots, &[n1, n2]);
    Some([a, b, c])
}

/// Output of [`tetrahedralize_constrained`].
pub struct Constrained {
    /// Tet vertex indices into `points`.
    pub tets: Vec<[usize; 4]>,
    /// f64 vertex positions (surface verts, interior, recovered Steiner).
    pub points: Vec<V3>,
    /// The refined constraint triangulation (the surface after recovery splits),
    /// indices into `points`. Every triangle here is a face of two tets.
    pub surf_tris: Vec<[usize; 3]>,
    /// Per refined triangle, the index of the original constraint triangle it was
    /// split from (so the caller carries region/tag/surface through recovery).
    pub surf_parent: Vec<usize>,
    /// `points[..n_surf_verts]` are the original surface vertices.
    pub n_surf_verts: usize,
}

/// Recovery cap: a safety bound on Steiner insertions, far above any real need
/// (conforming recovery terminates; this only guards a degenerate input).
const RECOVER_CAP: usize = 1 << 20;

/// A mesh vertex during recovery: its current f64 position, exact carrier, and
/// builder index.
struct Vert {
    pos: V3,
    carrier: Carrier,
    bidx: usize,
}

/// Boundary-constrained Delaunay tetrahedralization. `verts` are the frozen
/// surface vertices (each on its exact carrier); `tris` index into `verts` and
/// form the watertight surface; `tri_carrier[i]` is the carrier of triangle `i`
/// (its plane / analytic surface), used to construct Steiner points exactly on
/// the facet; `interior` are the relaxed interior seeds; `lo`/`hi` bound the
/// domain.
pub fn tetrahedralize_constrained(
    verts: &[Site],
    tris: &[[usize; 3]],
    tri_carrier: &[Carrier],
    interior: &[V3],
    lo: V3,
    hi: V3,
) -> Constrained {
    assert_eq!(tris.len(), tri_carrier.len());
    let mut db = DelaunayBuilder::enclosing(lo, hi);

    // All mesh vertices; surface vertices first (exact, on their carriers).
    let mut vs: Vec<Vert> = Vec::with_capacity(verts.len() + interior.len());
    for s in verts {
        let bidx = db.insert_exact(s.exact());
        vs.push(Vert { pos: s.pos(), carrier: s.carrier.clone(), bidx });
    }
    let n_surf_verts = vs.len();
    for &p in interior {
        if let Some(bidx) = db.try_insert(p) {
            vs.push(Vert { pos: p, carrier: Carrier::Volume, bidx });
        }
    }

    // Constraint triangles, each tagged with the index of its parent (original)
    // triangle so region/tag survive recovery splits, and with its facet carrier.
    let mut tris: Vec<[usize; 3]> = tris.to_vec();
    let mut parent: Vec<usize> = (0..tris.len()).collect();
    let mut carrier: Vec<Carrier> = tri_carrier.to_vec();

    recover_curved_facets(&mut db, &mut vs, &mut tris, &mut parent, &mut carrier);

    let b2a = invert(&vs, db.len());
    let tets: Vec<[usize; 4]> = db
        .tets()
        .into_iter()
        .map(|t| std::array::from_fn(|j| b2a[t[j]]))
        .collect();
    let points: Vec<V3> = vs.iter().map(|v| v.pos).collect();
    Constrained { tets, points, surf_tris: tris, surf_parent: parent, n_surf_verts }
}

/// Inverse map builder index -> `vs` index (`usize::MAX` for builder slots that
/// no `vs` vertex owns, e.g. the super-tet corners or a deduplicated insert).
fn invert(vs: &[Vert], builder_len: usize) -> Vec<usize> {
    let mut b2a = vec![usize::MAX; builder_len];
    for (a, v) in vs.iter().enumerate() {
        if v.bidx < builder_len {
            b2a[v.bidx] = a;
        }
    }
    b2a
}

/// Jitter fractions tried when a Steiner insertion would swallow a cospherical
/// vertex's star: the point slides along its carrier (so it stays exactly on the
/// edge / facet) to dodge the degeneracy. `0.5` (the midpoint / centroid) first.
const JITTER: [f64; 7] = [0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65];

/// Inserts a Steiner vertex on `carrier` near `pos`, retrying along the carrier
/// (via `candidate(frac)`) if the exact insert would swallow a cospherical
/// vertex. Returns its `vs` index, or `None` if every candidate is degenerate.
fn insert_steiner(
    db: &mut DelaunayBuilder,
    vs: &mut Vec<Vert>,
    carrier: &Carrier,
    candidate: impl Fn(f64) -> V3,
) -> Option<usize> {
    for &frac in &JITTER {
        let pos = candidate(frac);
        let exact = carrier.exact(pos);
        match db.insert_exact_checked(exact.clone()) {
            Ok(bidx) => {
                // The builder rounds the exact point; use ITS coords so positions
                // and predicates agree.
                let pos = match exact {
                    Point3::Explicit(p) => p,
                    other => other.approx().unwrap_or(pos),
                };
                vs.push(Vert { pos, carrier: carrier.clone(), bidx });
                return Some(vs.len() - 1);
            }
            Err(_) => continue, // would swallow a near-cospherical vertex: slide
        }
    }
    None
}

/// Makes every CURVED constraint facet a union of mesh faces by flip-based
/// recovery (\cref{alg:recover}). Planar facets are skipped: their vertices are
/// coplanar, so the Delaunay restricted to the plane already tiles them
/// (\cref{prop:watertight}). A curved facet that no flip resolves is split at a
/// Steiner point on its carrier and retried (the rare fallback).
fn recover_curved_facets(
    db: &mut DelaunayBuilder,
    vs: &mut Vec<Vert>,
    tris: &mut Vec<[usize; 3]>,
    parent: &mut Vec<usize>,
    carrier: &mut Vec<Carrier>,
) {
    let mut i = 0usize;
    let mut guard = 0usize;
    while i < tris.len() {
        guard += 1;
        assert!(guard < RECOVER_CAP, "facet recovery did not terminate");
        if matches!(carrier[i], Carrier::Plane { .. }) {
            i += 1;
            continue; // planar facet: covered by coplanarity, no recovery
        }
        let t = tris[i];
        let (ba, bb, bc) = (vs[t[0]].bidx, vs[t[1]].bidx, vs[t[2]].bidx);
        match recover_facet(db, ba, bb, bc) {
            Recover::Done => i += 1,
            Recover::NeedSteiner => {
                // Split the curved facet at a carrier point and retry the pieces.
                let car = carrier[i].clone();
                let par = parent[i];
                let (pa, pb, pc) = (vs[t[0]].pos, vs[t[1]].pos, vs[t[2]].pos);
                let blend = |f: f64| {
                    let d = f - 0.5;
                    let (wa, wb, wc) = (1.0 / 3.0 + d, 1.0 / 3.0 - 0.5 * d, 1.0 / 3.0 - 0.5 * d);
                    [
                        wa * pa[0] + wb * pb[0] + wc * pc[0],
                        wa * pa[1] + wb * pb[1] + wc * pc[1],
                        wa * pa[2] + wb * pb[2] + wc * pc[2],
                    ]
                };
                let g = match insert_steiner(db, vs, &car, blend) {
                    Some(g) => g,
                    None => {
                        i += 1; // give up on this facet (degenerate); leave best-effort
                        continue;
                    }
                };
                tris[i] = [t[0], t[1], g];
                for &(x, y) in &[(t[1], t[2]), (t[2], t[0])] {
                    tris.push([x, y, g]);
                    parent.push(par);
                    carrier.push(car.clone());
                }
                // re-process slot i (now a sub-triangle)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recover_facet_flips_in_a_missing_triangle() {
        // A flat bipyramid: triangle a,b,c in z=0 with near apexes d,e. The
        // apexes are nearly coplanar with abc, so abc+d has a huge circumsphere
        // that contains e: the Delaunay connects the apexes by the edge (d,e) and
        // face (a,b,c) is absent. Flip-based recovery must 3-2 flip (d,e).
        let a = [1.0, 0.0, 0.0];
        let b = [-0.5, 0.866, 0.0];
        let c = [-0.5, -0.866, 0.0];
        let d = [0.0, 0.0, 0.25];
        let e = [0.0, 0.0, -0.25];
        let mut db = DelaunayBuilder::enclosing([-5.0; 3], [5.0; 3]);
        for p in [a, b, c, d, e] {
            db.insert(p);
        }
        // Vertices a,b,c are public indices 0,1,2.
        assert!(!db.face_exists(0, 1, 2), "the tall bipyramid must omit face abc");
        assert!(db.edge_exists(3, 4), "the apexes are joined by edge (d,e)");
        assert_eq!(recover_facet(&mut db, 0, 1, 2), Recover::Done);
        assert!(db.face_exists(0, 1, 2), "facet recovery made abc a mesh face");
        assert!(!db.edge_exists(3, 4), "recovery removed the piercing edge (d,e)");
    }

    #[test]
    fn flip_23_then_32_round_trips() {
        // Five points: a triangle a,b,c with apexes d (above) and e (below).
        let pts = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.3, 1.0],
            [0.3, 0.3, -1.0],
        ];
        let mut db = DelaunayBuilder::enclosing([-2.0; 3], [2.0; 3]);
        for p in pts {
            db.insert(p);
        }
        let n0 = db.tets_with_slots().len();
        // Find an interior face shared by two all-real tets and 2-3 flip it.
        let mut flipped = None;
        'find: for (s1, _) in db.tets_with_slots() {
            for i in 0..4 {
                if let Some(s2) = db.neighbor_at(s1, i) {
                    if db.tet_at(s2).is_some() {
                        if let Some(edge) = flip23(&mut db, s1, s2) {
                            flipped = Some(edge);
                            break 'find;
                        }
                    }
                }
            }
        }
        let (d, e) = flipped.expect("a 2-3 flip should apply to this bipyramid");
        assert_eq!(db.tets_with_slots().len(), n0 + 1, "2-3 flip adds one tet");
        assert!(db.edge_exists(d, e), "the flip created edge (d,e)");
        // 3-2 flip the created edge back.
        flip32(&mut db, d, e).expect("the created edge is shared by exactly 3 tets");
        assert_eq!(db.tets_with_slots().len(), n0, "3-2 flip restores the count");
        assert!(!db.edge_exists(d, e), "the 3-2 flip removed edge (d,e)");
    }

    /// A unit-cube surface subdivided `n`x`n` per face: vertices on each face
    /// grid (shared edges/corners deduplicated), triangulated with outward
    /// winding, each triangle carrying its face plane. This is the kind of
    /// oversampled surface Stage 2 produces (8 cospherical corners alone are a
    /// degenerate stress case, not what the mesher ever feeds Stage 3).
    fn subdivided_cube(n: usize) -> (Vec<Site>, Vec<[usize; 3]>, Vec<Carrier>) {
        use std::collections::HashMap;
        let mut idx: HashMap<(i64, i64, i64), usize> = HashMap::new();
        let mut pts: Vec<V3> = Vec::new();
        let key = |p: V3| ((p[0] * 1e6) as i64, (p[1] * 1e6) as i64, (p[2] * 1e6) as i64);
        let mut vid = |p: V3, pts: &mut Vec<V3>, idx: &mut HashMap<(i64, i64, i64), usize>| {
            *idx.entry(key(p)).or_insert_with(|| {
                pts.push(p);
                pts.len() - 1
            })
        };
        // Each face: origin + two in-plane unit axes + outward normal.
        let faces: [(V3, V3, V3, V3); 6] = [
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]), // z=0
            ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),  // z=1
            ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]), // y=0
            ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),  // y=1
            ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]), // x=0
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),  // x=1
        ];
        let mut tris = Vec::new();
        let mut carr = Vec::new();
        for (o, du, dv, nrm) in faces {
            let at = |i: usize, j: usize| -> V3 {
                let (s, t) = (i as f64 / n as f64, j as f64 / n as f64);
                [o[0] + s * du[0] + t * dv[0], o[1] + s * du[1] + t * dv[1], o[2] + s * du[2] + t * dv[2]]
            };
            for i in 0..n {
                for j in 0..n {
                    let a = vid(at(i, j), &mut pts, &mut idx);
                    let b = vid(at(i + 1, j), &mut pts, &mut idx);
                    let c = vid(at(i + 1, j + 1), &mut pts, &mut idx);
                    let d = vid(at(i, j + 1), &mut pts, &mut idx);
                    // Outward winding (du x dv aligns with nrm by construction).
                    tris.push([a, b, c]);
                    tris.push([a, c, d]);
                    for _ in 0..2 {
                        carr.push(Carrier::Plane { p0: o, n: nrm });
                    }
                }
            }
        }
        let verts: Vec<Site> = pts.iter().map(|&p| Site::vertex(p)).collect();
        (verts, tris, carr)
    }

    fn total_volume(c: &Constrained) -> f64 {
        let mut vol = 0.0;
        for t in &c.tets {
            let p: [V3; 4] = std::array::from_fn(|j| c.points[t[j]]);
            let d = |i: usize, k: usize| p[i][k] - p[0][k];
            vol += (d(1, 0) * (d(2, 1) * d(3, 2) - d(2, 2) * d(3, 1))
                - d(1, 1) * (d(2, 0) * d(3, 2) - d(2, 2) * d(3, 0))
                + d(1, 2) * (d(2, 0) * d(3, 1) - d(2, 1) * d(3, 0)))
            .abs()
                / 6.0;
        }
        vol
    }

    #[test]
    fn cube_planar_facets_need_no_recovery_and_volume_is_bit_exact() {
        // All six faces are planar (axis-aligned), so no facet recovery runs; the
        // Delaunay covers each face by coplanarity. The result must be a watertight
        // cube of bit-exact volume 1: the boundary faces (each used by one tet) lie
        // exactly on the six planes and total area 6.
        let (verts, tris, carr) = subdivided_cube(3);
        let interior = vec![
            [0.5, 0.5, 0.5], [0.25, 0.5, 0.7], [0.7, 0.3, 0.4],
            [0.3, 0.7, 0.3], [0.6, 0.6, 0.6], [0.4, 0.4, 0.8],
        ];
        let c = tetrahedralize_constrained(&verts, &tris, &carr, &interior, [0.0; 3], [1.0; 3]);

        // The geometry is exact (every boundary vertex lands on a cube plane,
        // checked below); the tiny residual here is only f64 summation rounding
        // over the many tets. Bit-exact rational volume is gated end to end in
        // tests/conform.rs (mesh_region_volume6 == rat).
        let vol = total_volume(&c);
        assert!((vol - 1.0).abs() < 1e-12, "cube volume must be 1, got {vol}");

        // Boundary faces = tet faces used exactly once. They must tile the cube
        // surface: every vertex on one of the six planes, total area 6.
        let mut count: std::collections::HashMap<[usize; 3], usize> = std::collections::HashMap::new();
        for t in &c.tets {
            for f in &[[t[0], t[1], t[2]], [t[0], t[1], t[3]], [t[0], t[2], t[3]], [t[1], t[2], t[3]]] {
                let mut s = *f;
                s.sort_unstable();
                *count.entry(s).or_insert(0) += 1;
            }
        }
        let mut area = 0.0;
        for (f, &n) in &count {
            if n != 1 {
                continue;
            }
            let (a, b, cc) = (c.points[f[0]], c.points[f[1]], c.points[f[2]]);
            for &p in &[a, b, cc] {
                let on = p[0] == 0.0 || p[0] == 1.0 || p[1] == 0.0 || p[1] == 1.0 || p[2] == 0.0 || p[2] == 1.0;
                assert!(on, "boundary vertex {p:?} is not on a cube face");
            }
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [cc[0] - a[0], cc[1] - a[1], cc[2] - a[2]];
            let cr = [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]];
            area += 0.5 * (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
        }
        assert!((area - 6.0).abs() < 1e-9, "cube surface area must be 6, got {area}");
    }
}

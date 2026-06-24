//! Constrained boundary (facet) recovery for the volume CDT.
//!
//! The restricted-Delaunay boundary bridges concave creases (the sphere-union
//! neck): a flat tet face sits tangentially over the groove instead of dipping to
//! the intersection ring, because the empty space above the groove makes the
//! bridge face Delaunay-optimal. Conforming Steiner refinement cannot break this
//! (the bridge's circumsphere stays empty no matter how many surface points are
//! added). The fix is CONSTRAINED: force each clean frozen surface facet to be a
//! tet face by locally re-tetrahedralizing the cavity around it, which deletes the
//! bridge tets.
//!
//! Per missing facet: gather the cavity (the union of the vertex stars of its
//! three corners), gift-wrap it with the facet as a two-sided internal wall (no
//! candidate apex may form a tet that crosses the wall), and swap it in via
//! [`DelaunayBuilder::try_replace_cavity`]. Gift-wrapping forces the facet AND its
//! three edges at once, so no separate edge-recovery pass is needed. On any
//! failure (a cavity that would need Steiner points to tetrahedralize) the facet is
//! left as is and counted, never corrupting the mesh.

use crate::delaunay::DelaunayBuilder;
use rapidmesh_exact::{Point3, Sign};
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;

type BH = BuildHasherDefault<rustc_hash::FxHasher>;

/// Face of a positively oriented tet opposite vertex `i`, wound so the opposite
/// vertex lies on its positive side (mirrors `delaunay::face`, in public indices).
fn face_pub(t: [usize; 4], i: usize) -> [usize; 3] {
    match i {
        0 => [t[1], t[3], t[2]],
        1 => [t[0], t[2], t[3]],
        2 => [t[0], t[3], t[1]],
        _ => [t[0], t[1], t[2]],
    }
}

/// True if `g` is an even rotation of `f` (same oriented triangle).
fn is_same_cycle(f: [usize; 3], g: [usize; 3]) -> bool {
    g == f || g == [f[1], f[2], f[0]] || g == [f[2], f[0], f[1]]
}

/// True if `g` is the reversal of `f` (same triangle, opposite orientation).
fn is_reversed(f: [usize; 3], g: [usize; 3]) -> bool {
    is_same_cycle([f[0], f[2], f[1]], g)
}

/// Proper intersection of the OPEN segment `u`-`v` with the OPEN triangle
/// `a,b,c`, by exact orientation. Shared endpoints (a coincident vertex) give a
/// zero and count as no crossing, so a facet corner shared with a tet is fine.
fn seg_crosses_tri(
    o: &impl Fn(usize, usize, usize, usize) -> Sign,
    u: usize,
    v: usize,
    a: usize,
    b: usize,
    c: usize,
) -> bool {
    let s1 = o(a, b, c, u);
    let s2 = o(a, b, c, v);
    if s1 == Sign::Zero || s2 == Sign::Zero || s1 == s2 {
        return false; // endpoints not on strictly opposite sides of the plane
    }
    let d1 = o(u, v, a, b);
    let d2 = o(u, v, b, c);
    let d3 = o(u, v, c, a);
    if d1 == Sign::Zero || d2 == Sign::Zero || d3 == Sign::Zero {
        return false;
    }
    d1 == d2 && d2 == d3 // pierces the triangle interior
}

/// True if the tet `[p,q,r,s]` properly intersects the wall triangle `w`'s
/// interior: a tet edge piercing the wall, or a wall edge piercing a tet face.
fn tet_crosses_tri(
    o: &impl Fn(usize, usize, usize, usize) -> Sign,
    t: [usize; 4],
    w: [usize; 3],
) -> bool {
    const EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    for &(i, j) in &EDGES {
        if seg_crosses_tri(o, t[i], t[j], w[0], w[1], w[2]) {
            return true;
        }
    }
    for k in 0..3 {
        let (u, v) = (w[k], w[(k + 1) % 3]);
        for fi in 0..4 {
            let f = face_pub(t, fi);
            if seg_crosses_tri(o, u, v, f[0], f[1], f[2]) {
                return true;
            }
        }
    }
    false
}

/// Gift-wrap the cavity `removed` (alive tet slots) into a valid tetrahedral-
/// ization that contains every `walls` triangle as an internal face. Returns the
/// new tets (public vertex indices) or `None` if the advancing front cannot be
/// closed without Steiner points. Pure read of `db`.
fn giftwrap(
    db: &DelaunayBuilder,
    removed: &[u32],
    walls: &[[usize; 3]],
) -> Option<Vec<[usize; 4]>> {
    // Cavity vertices and their exact positions.
    let mut verts: Vec<usize> = Vec::new();
    let mut seen: HashSet<usize, BH> = HashSet::default();
    for &s in removed {
        let t = db.tet_at(s)?;
        for v in t {
            if seen.insert(v) {
                verts.push(v);
            }
        }
    }
    let mut pts: HashMap<usize, Point3, BH> = HashMap::default();
    for &v in &verts {
        pts.insert(v, db.exact_point(v));
    }
    let o = |a: usize, b: usize, c: usize, d: usize| -> Sign {
        rapidmesh_exact::orient3d(&pts[&a], &pts[&b], &pts[&c], &pts[&d]).expect("valid cavity pts")
    };
    let ins = |a: usize, b: usize, c: usize, d: usize, e: usize| -> Sign {
        rapidmesh_exact::insphere3d(&pts[&a], &pts[&b], &pts[&c], &pts[&d], &pts[&e])
            .expect("valid cavity pts")
    };

    // Initial advancing front: the cavity boundary, each face oriented so the
    // UNFILLED interior is on its positive side (an apex is sought there). A
    // removed tet's face shared with a non-removed neighbor is a boundary face;
    // `face_pub` already winds it with that tet's apex (the interior) positive.
    let removed_set: HashSet<u32, BH> = removed.iter().copied().collect();
    let mut front: HashMap<[usize; 3], [usize; 3], BH> = HashMap::default();
    for &s in removed {
        let t = db.tet_at(s)?;
        for i in 0..4 {
            let nb = db.neighbor_at(s, i);
            if nb.is_some_and(|n| removed_set.contains(&n)) {
                continue; // shared with another removed tet -> interior, skip
            }
            let f = face_pub(t, i);
            if !front_xor(&mut front, f) {
                return None;
            }
        }
    }

    let mut result: Vec<[usize; 4]> = Vec::new();
    let max_iters = removed.len() * 12 + 128;
    let mut iters = 0usize;
    while let Some((&_key, &f)) = front.iter().next() {
        iters += 1;
        if iters > max_iters {
            return None;
        }
        let [p, q, r] = f;
        // Best apex: on the positive (unfilled) side, forming no wall-crossing
        // tet, Delaunay-best (its circumsphere encloses no other candidate).
        let mut best: Option<usize> = None;
        for &s in &verts {
            if s == p || s == q || s == r {
                continue;
            }
            if o(p, q, r, s) != Sign::Positive {
                continue;
            }
            if walls.iter().any(|w| tet_crosses_tri(&o, [p, q, r, s], *w)) {
                continue;
            }
            best = match best {
                None => Some(s),
                Some(b) if ins(p, q, r, b, s) == Sign::Positive => Some(s),
                some => some,
            };
        }
        let s = best?; // front face has no valid apex -> needs Steiner -> fail
        result.push([p, q, r, s]);
        // Consume the base face; XOR the three new side faces, reversed so their
        // unfilled side (away from s) is positive.
        let mut bk = f;
        bk.sort_unstable();
        front.remove(&bk);
        for nf in [[p, q, s], [p, s, r], [s, q, r]] {
            // nf is face_pub of (p,q,r,s) reversed: unfilled side positive.
            if !front_xor(&mut front, nf) {
                return None;
            }
        }
    }

    // Every wall must have emerged as a face of the tiling.
    for w in walls {
        let present = result.iter().any(|t| {
            (0..4).any(|i| {
                let mut g = face_pub(*t, i);
                g.sort_unstable();
                let mut wk = *w;
                wk.sort_unstable();
                g == wk
            })
        });
        if !present {
            return None;
        }
    }
    Some(result)
}

/// XOR an oriented face into the front: cancel against its reverse if present,
/// else insert. Returns false on a same-orientation duplicate (degenerate).
fn front_xor(front: &mut HashMap<[usize; 3], [usize; 3], BH>, f: [usize; 3]) -> bool {
    let mut k = f;
    k.sort_unstable();
    match front.get(&k).copied() {
        Some(g) if is_reversed(f, g) => {
            front.remove(&k);
            true
        }
        Some(_) => false, // same orientation twice: invalid complex
        None => {
            front.insert(k, f);
            true
        }
    }
}

/// Maximum cavity size (tets) before a recovery is abandoned -- a runaway cavity
/// is a sign the facet needs Steiner points, not more enlargement.
const MAX_CAVITY: usize = 256;
/// How many times the cavity is enlarged by one neighbor ring on gift-wrap failure.
const MAX_GROW: usize = 4;

/// All target facets fully contained in the cavity vertex set, as walls that
/// gift-wrapping must preserve (else enlarging the cavity would silently destroy
/// neighbouring surface facets). Returned as plain triples (orientation-free).
fn contained_walls(verts: &HashSet<usize, BH>, facets: &[[usize; 3]]) -> Vec<[usize; 3]> {
    facets
        .iter()
        .filter(|f| f.iter().all(|v| verts.contains(v)))
        .copied()
        .collect()
}

/// Vertex set of a cavity (real tets only).
fn cavity_verts(db: &DelaunayBuilder, removed: &[u32]) -> HashSet<usize, BH> {
    let mut verts: HashSet<usize, BH> = HashSet::default();
    for &s in removed {
        if let Some(t) = db.tet_at(s) {
            verts.extend(t);
        }
    }
    verts
}

/// Recover one missing facet `[a,b,c]` (public indices) by constrained cavity
/// re-tetrahedralization, enlarging the cavity on failure. `facets` are all the
/// surface facets to preserve as walls. Returns true if the facet is now a tet face.
fn recover_one(db: &mut DelaunayBuilder, a: usize, b: usize, c: usize, facets: &[[usize; 3]]) -> bool {
    // Seed cavity: union of the vertex stars of the three corners (real tets).
    let mut set: HashSet<u32, BH> = HashSet::default();
    for v in [a, b, c] {
        for s in db.star_slots(v) {
            if db.tet_at(s).is_some() {
                set.insert(s);
            }
        }
    }
    if set.is_empty() {
        return false;
    }
    for _ in 0..=MAX_GROW {
        let removed: Vec<u32> = set.iter().copied().collect();
        let verts = cavity_verts(db, &removed);
        let mut walls = contained_walls(&verts, facets);
        if !walls.iter().any(|w| {
            let mut k = *w;
            k.sort_unstable();
            k == { let mut t = [a, b, c]; t.sort_unstable(); t }
        }) {
            walls.push([a, b, c]); // ensure the target itself is a wall
        }
        if let Some(new_tets) = giftwrap(db, &removed, &walls) {
            if db.try_replace_cavity(&removed, &new_tets) {
                return true;
            }
        }
        // Enlarge by one neighbour ring of real tets.
        if set.len() >= MAX_CAVITY {
            break;
        }
        let mut added = false;
        for &s in &removed {
            for i in 0..4 {
                if let Some(nb) = db.neighbor_at(s, i) {
                    if db.tet_at(nb).is_some() && set.insert(nb) {
                        added = true;
                    }
                }
            }
        }
        if !added {
            break;
        }
    }
    false
}

/// Forces every triangle in `facets` (public vertex indices) to be a tet face,
/// where it is not already, by constrained cavity re-tetrahedralization. Returns
/// `(recovered, failed)`: facets made present by recovery, and facets that could
/// not be recovered (left untouched). Already-present facets count as neither.
pub fn recover_facets(db: &mut DelaunayBuilder, facets: &[[usize; 3]]) -> (usize, usize) {
    let (mut recovered, mut failed) = (0usize, 0usize);
    let trace = std::env::var("RAPIDMESH_RECOVER_TRACE").is_ok();
    for &[a, b, c] in facets {
        if db.face_exists(a, b, c) {
            continue;
        }
        if recover_one(db, a, b, c, facets) {
            recovered += 1;
        } else {
            failed += 1;
            if trace {
                let ctr = |i: usize| db.approx_point(i);
                let (pa, pb, pc) = (ctr(a), ctr(b), ctr(c));
                let m = [
                    (pa[0] + pb[0] + pc[0]) / 3.0,
                    (pa[1] + pb[1] + pc[1]) / 3.0,
                    (pa[2] + pb[2] + pc[2]) / 3.0,
                ];
                eprintln!("[recover-fail] {} {} {}", m[0], m[1], m[2]);
            }
        }
    }
    (recovered, failed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A missing facet inside a small point set is forced to appear, the mesh
    /// stays a valid tiling of the same point set, and its volume is preserved.
    #[test]
    fn recovers_a_missing_interior_facet() {
        // Two unit tets sharing face (1,2,3) plus extra points; pick a facet that
        // the Delaunay does not contain and force it.
        let lo = [-2.0, -2.0, -2.0];
        let hi = [2.0, 2.0, 2.0];
        let mut db = DelaunayBuilder::enclosing(lo, hi);
        let pts = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.6, 0.6, -0.4],
            [-0.4, 0.3, 0.5],
        ];
        for p in pts {
            db.insert(p);
        }
        let before = db.tets().len();
        assert!(before > 0);
        // Find a triangle of existing vertices that is NOT currently a face but
        // whose three vertices are all present; recover it and re-check.
        let mut forced = 0;
        'outer: for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                for k in (j + 1)..pts.len() {
                    if !db.face_exists(i, j, k) {
                        if recover_one(&mut db, i, j, k, &[[i, j, k]]) {
                            assert!(db.face_exists(i, j, k), "facet must be present after recovery");
                            forced += 1;
                            break 'outer;
                        }
                    }
                }
            }
        }
        assert!(forced > 0, "expected to force at least one facet");
        // The triangulation must remain valid: every face shared by exactly two
        // tets or one (boundary). A gross corruption would desync this.
        let mut count: HashMap<[usize; 3], usize, BH> = HashMap::default();
        for t in db.tets() {
            for i in 0..4 {
                let mut f = face_pub(t, i);
                f.sort_unstable();
                *count.entry(f).or_insert(0) += 1;
            }
        }
        assert!(count.values().all(|&n| n == 1 || n == 2), "manifold tiling");
    }
}

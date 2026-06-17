//! Stage 3: boundary-constrained tetrahedralization (conforming CDT).
//!
//! Given the frozen Stage-2 surface mesh `S` (exact vertices on their carriers,
//! plus a triangulation) and a set of relaxed interior points, this builds a
//! Delaunay tetrahedralization that contains every triangle of `S` as a union of
//! tet faces (\cref{prop:watertight} of the report). Boundary recovery is
//! conforming, by Steiner insertion ON the constraint (Diazzi et al. 2023): a
//! missing constraint edge is split at a [`Point3::Lnc`] midpoint (exactly on the
//! edge line) and a missing facet at a [`Point3::Pac`] interior point (exactly on
//! the facet plane), so the surface geometry is preserved and the next recovery
//! round sees the carrier intact. The result has a tetrahedron on each side of
//! every surface triangle, so region labelling is a flood fill that never leaks.
//!
//! This replaces the unconstrained-Delaunay + centroid-classification path, which
//! recovered the boundary only statistically (\cref{sec:conformity}).

use crate::delaunay::DelaunayBuilder;
use rapidmesh_exact::Point3;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

type V3 = [f64; 3];
type DMap<K, V> = HashMap<K, V, BuildHasherDefault<rustc_hash::FxHasher>>;

fn sorted2(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

/// Output of [`tetrahedralize_constrained`].
pub struct Constrained {
    /// Tet vertex indices into `points` (and `exact`).
    pub tets: Vec<[usize; 4]>,
    /// f64 vertex positions (surface verts, interior, recovered Steiner).
    pub points: Vec<V3>,
    /// The refined constraint triangulation (the surface after recovery splits),
    /// indices into `points`. Every triangle here is a face of two tets.
    pub surf_tris: Vec<[usize; 3]>,
    /// `points[i]` came from surface vertex / interior / Steiner: this is the
    /// number of original surface vertices (`points[..n_surf_verts]`).
    pub n_surf_verts: usize,
}

/// Recovery cap: a safety bound on Steiner insertions, far above any real need
/// (conforming recovery terminates; this only guards a degenerate input).
const RECOVER_CAP: usize = 1 << 20;

/// Boundary-constrained Delaunay tetrahedralization. `verts` are the exact
/// surface vertices (each already on its analytic carrier); `tris` index into
/// `verts` and form the frozen, watertight surface; `interior` are the relaxed
/// interior seeds; `lo`/`hi` bound the domain.
pub fn tetrahedralize_constrained(
    verts: &[Point3],
    tris: &[[usize; 3]],
    interior: &[V3],
    lo: V3,
    hi: V3,
) -> Constrained {
    let mut db = DelaunayBuilder::enclosing(lo, hi);

    // All mesh vertices, exact; `bidx[k]` is the builder index of `all[k]`.
    // Surface vertices first (so `n_surf_verts` splits them from interior).
    let mut all: Vec<Point3> = verts.to_vec();
    let mut bidx: Vec<usize> = Vec::with_capacity(all.len() + interior.len());
    for p in &all {
        bidx.push(db.insert_exact(p.clone()));
    }
    let n_surf_verts = all.len();
    for &p in interior {
        if let Some(i) = db.try_insert(p) {
            bidx.push(i);
            all.push(Point3::Explicit(p));
        }
    }

    let mut tris: Vec<[usize; 3]> = tris.to_vec();

    // ---- edge recovery: every constraint edge must be a mesh edge ----------
    recover_edges(&mut db, &mut all, &mut bidx, &mut tris);

    // ---- facet recovery: every constraint triangle must be a mesh face -----
    recover_facets(&mut db, &mut all, &mut bidx, &mut tris);

    // Final tets in builder indices -> map back to `all` indices.
    let b2a = invert(&bidx, db.len());
    let tets: Vec<[usize; 4]> = db
        .tets()
        .into_iter()
        .map(|t| std::array::from_fn(|j| b2a[t[j]]))
        .collect();
    let points: Vec<V3> = all.iter().map(|p| p.approx().expect("valid vertex")).collect();
    Constrained { tets, points, surf_tris: tris, n_surf_verts }
}

/// Inverse of `bidx`: builder index -> `all` index (`usize::MAX` if unused, e.g.
/// a deduplicated near-duplicate).
fn invert(bidx: &[usize], builder_len: usize) -> Vec<usize> {
    let mut b2a = vec![usize::MAX; builder_len];
    for (a, &b) in bidx.iter().enumerate() {
        if b < builder_len {
            b2a[b] = a;
        }
    }
    b2a
}

/// Inserts the [`Point3::Lnc`] midpoint of edge `(a,b)` (indices into `all`),
/// exactly on the edge line, and returns its `all` index.
fn split_edge_midpoint(db: &mut DelaunayBuilder, all: &mut Vec<Point3>, bidx: &mut Vec<usize>, a: usize, b: usize) -> usize {
    let pa = all[a].approx().expect("valid");
    let pb = all[b].approx().expect("valid");
    let m = Point3::Lnc { a: pa, b: pb, t: 0.5 };
    let bi = db.insert_exact(m.clone());
    all.push(m);
    bidx.push(bi);
    all.len() - 1
}

/// Inserts the [`Point3::Pac`] centroid of triangle `(a,b,c)`, exactly on the
/// facet plane, and returns its `all` index.
fn split_facet_centroid(db: &mut DelaunayBuilder, all: &mut Vec<Point3>, bidx: &mut Vec<usize>, a: usize, b: usize, c: usize) -> usize {
    let pa = all[a].approx().expect("valid");
    let pb = all[b].approx().expect("valid");
    let pc = all[c].approx().expect("valid");
    let g = Point3::Pac { a: pa, b: pb, c: pc, u: 1.0 / 3.0, v: 1.0 / 3.0 };
    let bi = db.insert_exact(g.clone());
    all.push(g);
    bidx.push(bi);
    all.len() - 1
}

/// Forces every constraint edge to appear as a mesh edge by splitting missing
/// edges at their midpoint; each split also bisects the incident triangles so the
/// surface stays a triangulation.
fn recover_edges(db: &mut DelaunayBuilder, all: &mut Vec<Point3>, bidx: &mut Vec<usize>, tris: &mut Vec<[usize; 3]>) {
    let mut steiner = 0usize;
    loop {
        // Collect a missing constraint edge.
        let mut missing: Option<(usize, usize)> = None;
        let mut seen: DMap<(usize, usize), ()> = DMap::default();
        'outer: for t in tris.iter() {
            for e in 0..3 {
                let (a, b) = (t[e], t[(e + 1) % 3]);
                let key = sorted2(a, b);
                if seen.insert(key, ()).is_some() {
                    continue;
                }
                if !db.edge_exists(bidx[a], bidx[b]) {
                    missing = Some((a, b));
                    break 'outer;
                }
            }
        }
        let (a, b) = match missing {
            Some(e) => e,
            None => break,
        };
        let m = split_edge_midpoint(db, all, bidx, a, b);
        bisect_triangles_on_edge(tris, a, b, m);
        steiner += 1;
        assert!(steiner < RECOVER_CAP, "edge recovery did not terminate");
    }
}

/// Replaces every triangle containing edge `(a,b)` by the two triangles obtained
/// by splitting that edge at the new midpoint vertex `m` (preserving winding).
fn bisect_triangles_on_edge(tris: &mut Vec<[usize; 3]>, a: usize, b: usize, m: usize) {
    let mut out: Vec<[usize; 3]> = Vec::with_capacity(tris.len() + 2);
    for &t in tris.iter() {
        let mut e = None;
        for k in 0..3 {
            let (u, v) = (t[k], t[(k + 1) % 3]);
            if (u == a && v == b) || (u == b && v == a) {
                e = Some(k);
                break;
            }
        }
        match e {
            Some(k) => {
                let (u, v, w) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
                // (u,v,w) with edge (u,v) split at m: (u,m,w) and (m,v,w).
                out.push([u, m, w]);
                out.push([m, v, w]);
            }
            None => out.push(t),
        }
    }
    *tris = out;
}

/// Forces every constraint triangle to appear as a mesh face by inserting an
/// interior Steiner point on any missing facet and splitting it into three.
/// Re-runs edge recovery after each split (the new sub-edges may themselves be
/// missing).
fn recover_facets(db: &mut DelaunayBuilder, all: &mut Vec<Point3>, bidx: &mut Vec<usize>, tris: &mut Vec<[usize; 3]>) {
    let mut steiner = 0usize;
    loop {
        let mut missing: Option<usize> = None;
        for (i, t) in tris.iter().enumerate() {
            if !db.face_exists(bidx[t[0]], bidx[t[1]], bidx[t[2]]) {
                missing = Some(i);
                break;
            }
        }
        let i = match missing {
            Some(i) => i,
            None => break,
        };
        let t = tris[i];
        let g = split_facet_centroid(db, all, bidx, t[0], t[1], t[2]);
        tris.swap_remove(i);
        tris.push([t[0], t[1], g]);
        tris.push([t[1], t[2], g]);
        tris.push([t[2], t[0], g]);
        recover_edges(db, all, bidx, tris);
        steiner += 1;
        assert!(steiner < RECOVER_CAP, "facet recovery did not terminate");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_verts() -> Vec<Point3> {
        let mut v = Vec::new();
        for &z in &[0.0, 1.0] {
            for &y in &[0.0, 1.0] {
                for &x in &[0.0, 1.0] {
                    v.push(Point3::explicit(x, y, z));
                }
            }
        }
        v
    }

    /// The 12 outward triangles of the unit cube (2 per face), indices into the
    /// `cube_verts` order (x fastest, then y, then z).
    fn cube_tris() -> Vec<[usize; 3]> {
        // corner index = x + 2y + 4z
        let q = |a, b, c, d| vec![[a, b, c], [a, c, d]];
        let mut t = Vec::new();
        t.extend(q(0, 1, 3, 2)); // z=0 bottom
        t.extend(q(4, 6, 7, 5)); // z=1 top
        t.extend(q(0, 4, 5, 1)); // y=0
        t.extend(q(2, 3, 7, 6)); // y=1
        t.extend(q(0, 2, 6, 4)); // x=0
        t.extend(q(1, 5, 7, 3)); // x=1
        t
    }

    fn total_volume(c: &Constrained) -> f64 {
        let mut vol = 0.0;
        for t in &c.tets {
            let p: [V3; 4] = std::array::from_fn(|j| c.points[t[j]]);
            let d = |i: usize, k: usize| p[i][k] - p[0][k];
            let det = (d(1, 0) * (d(2, 1) * d(3, 2) - d(2, 2) * d(3, 1))
                - d(1, 1) * (d(2, 0) * d(3, 2) - d(2, 2) * d(3, 0))
                + d(1, 2) * (d(2, 0) * d(3, 1) - d(2, 1) * d(3, 0)))
                .abs()
                / 6.0;
            vol += det;
        }
        vol
    }

    #[test]
    fn cube_all_facets_recovered_and_volume_exact() {
        let verts = cube_verts();
        let tris = cube_tris();
        let interior = vec![[0.5, 0.5, 0.5], [0.25, 0.5, 0.7], [0.7, 0.3, 0.4]];
        let c = tetrahedralize_constrained(&verts, &tris, &interior, [0.0; 3], [1.0; 3]);
        // every refined constraint triangle is a tet face: re-check on a fresh
        // builder over the final points would be circular; instead assert the
        // total tet volume equals the cube exactly (a missing facet would leave a
        // void or a straddle, breaking the sum), plus the surface area.
        let vol = total_volume(&c);
        assert!((vol - 1.0).abs() < 1e-9, "cube volume should be 1, got {vol}");
        // prop:watertight on the output: every refined constraint triangle is a
        // face of some tet (so each has a tet on each side).
        let mut faces: std::collections::HashSet<(usize, usize, usize)> = std::collections::HashSet::new();
        for t in &c.tets {
            for f in &[[t[0], t[1], t[2]], [t[0], t[1], t[3]], [t[0], t[2], t[3]], [t[1], t[2], t[3]]] {
                let mut s = *f;
                s.sort_unstable();
                faces.insert((s[0], s[1], s[2]));
            }
        }
        for t in &c.surf_tris {
            let mut s = *t;
            s.sort_unstable();
            assert!(faces.contains(&(s[0], s[1], s[2])), "constraint triangle {t:?} is not a tet face");
        }
        // surface area of the refined constraint triangulation = 6 (cube).
        let mut area = 0.0;
        for t in &c.surf_tris {
            let (a, b, cc) = (c.points[t[0]], c.points[t[1]], c.points[t[2]]);
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [cc[0] - a[0], cc[1] - a[1], cc[2] - a[2]];
            let cr = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            area += 0.5 * (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
        }
        assert!((area - 6.0).abs() < 1e-9, "cube surface area should be 6, got {area}");
    }
}

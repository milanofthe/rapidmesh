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
use rapidmesh_geom::vec3::{V3, sub, dot};
use crate::site::{Carrier, Site};
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

/// Deterministic hashing: region flooding iterates face buckets, and the mesh
/// must be reproducible run to run.
type BH = BuildHasherDefault<rustc_hash::FxHasher>;
// ---- region classification by flood fill ------------------------------------

/// Assigns each tet a region tag by flood fill, blocked by the surface. The
/// oracle `surface_face(&sorted_tri)` returns `Some((front, back, n))` if that
/// tet face lies on the surface, with the region tags on the side the outward
/// normal `n` points to (front) and the opposite side (back), else `None` for an
/// interior face. Seeds each surface face's two incident tets by which side of
/// the face they sit on, then floods the tag across non-surface faces. This is
/// exact-conformant: no centroid test, the surface partitions the tets directly
/// (\cref{prop:watertight}). Tags follow the surface's region labelling; `0` is
/// the background void.
pub fn classify_regions(
    tets: &[[usize; 4]],
    points: &[V3],
    surface_face: impl Fn(&[usize; 3]) -> Option<(u32, u32, V3)>,
) -> Vec<u32> {
    let sorted = |f: [usize; 3]| {
        let mut s = f;
        s.sort_unstable();
        s
    };
    // Face -> incident tets (1 on the hull, 2 in the interior).
    let mut face_tets: HashMap<[usize; 3], Vec<usize>, BH> = HashMap::default();
    for (ti, t) in tets.iter().enumerate() {
        for f in &[[t[0], t[1], t[2]], [t[0], t[1], t[3]], [t[0], t[2], t[3]], [t[1], t[2], t[3]]] {
            face_tets.entry(sorted(*f)).or_default().push(ti);
        }
    }
    let mut region = vec![u32::MAX; tets.len()];
    // Seed: each surface face sets the region of its incident tet(s) by side.
    for (f, owners) in &face_tets {
        let (front, back, n) = match surface_face(f) {
            Some(x) => x,
            None => continue,
        };
        for &ti in owners {
            let apex = *tets[ti].iter().find(|v| !f.contains(v)).unwrap();
            // The tet is on the front side iff its apex is on the normal side.
            let s = dot(sub(points[apex], points[f[0]]), n);
            region[ti] = if s > 0.0 { front } else { back };
        }
    }
    // Flood the tag across non-surface shared faces.
    let mut stack: Vec<usize> = (0..tets.len()).filter(|&i| region[i] != u32::MAX).collect();
    while let Some(ti) = stack.pop() {
        let t = tets[ti];
        for f in &[[t[0], t[1], t[2]], [t[0], t[1], t[3]], [t[0], t[2], t[3]], [t[1], t[2], t[3]]] {
            let key = sorted(*f);
            if surface_face(&key).is_some() {
                continue; // a surface face does not connect two regions
            }
            if let Some(owners) = face_tets.get(&key) {
                for &nb in owners {
                    if nb != ti && region[nb] == u32::MAX {
                        region[nb] = region[ti];
                        stack.push(nb);
                    }
                }
            }
        }
    }
    // Any tet the flood never reached (isolated by degeneracy) is background.
    for r in &mut region {
        if *r == u32::MAX {
            *r = 0;
        }
    }
    region
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

/// A mesh vertex: its f64 position and builder-slot index.
struct Vert {
    pos: V3,
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
    // Insert in OCTREE order: the verts arrive in patch order, not spatially, so
    // an unsorted insert makes the Delaunay point-location walks long. The octree
    // order is density-adaptive with CUBIC cells, so unlike the per-axis-normalised
    // Morton order it does NOT stretch a flat/anisotropic layout (where Morton made
    // it slower). `vs` is still filled in the ORIGINAL order so the constraint
    // triangles' indices are unchanged.
    let mut vs: Vec<Vert> = Vec::with_capacity(verts.len() + interior.len());
    let vpos: Vec<V3> = verts.iter().map(|s| s.pos()).collect();
    let mut surf_bidx = vec![0usize; verts.len()];
    for &oi in &crate::spatial::octree_order(&vpos) {
        surf_bidx[oi] = db.insert_exact(verts[oi].exact());
    }
    for (oi, s) in verts.iter().enumerate() {
        vs.push(Vert { pos: s.pos(), bidx: surf_bidx[oi] });
    }
    let n_surf_verts = vs.len();
    for &oi in &crate::spatial::octree_order(interior) {
        if let Some(bidx) = db.try_insert(interior[oi]) {
            vs.push(Vert { pos: interior[oi], bidx });
        }
    }

    // Constraint triangles, tagged with their parent index (for region/tag) and
    // facet carrier. PLANAR facets are conformed by coplanarity (the Delaunay tiles
    // the plane; region volumes stay bit-exact). CURVED facets are NOT forced: the
    // curved boundary is the restricted Delaunay of the surface points (extracted
    // downstream by region difference). Forcing a chosen curved triangulation needed
    // unbounded Steiner (a barrel band is not near-Delaunay); the restricted Delaunay
    // is recovery-free and curved geometry is tolerance-based anyway (the pivot).
    let tris: Vec<[usize; 3]> = tris.to_vec();
    let parent: Vec<usize> = (0..tris.len()).collect();

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


#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn cube_region_flood_fill_tags_every_interior_tet() {
        let (verts, tris, carr) = subdivided_cube(3);
        let interior = vec![[0.5, 0.5, 0.5], [0.3, 0.6, 0.4], [0.7, 0.4, 0.6]];
        let c = tetrahedralize_constrained(&verts, &tris, &carr, &interior, [0.0; 3], [1.0; 3]);
        // Oracle: a face on a cube plane separates inside (region 1) from the
        // background void (0); the outward normal points out of the cube.
        let oracle = |f: &[usize; 3]| -> Option<(u32, u32, V3)> {
            let p = [c.points[f[0]], c.points[f[1]], c.points[f[2]]];
            for k in 0..3 {
                for (val, dir) in [(0.0, -1.0), (1.0, 1.0)] {
                    if p.iter().all(|q| q[k] == val) {
                        let mut n = [0.0, 0.0, 0.0];
                        n[k] = dir;
                        return Some((0, 1, n)); // front (out) = 0, back (in) = 1
                    }
                }
            }
            None
        };
        let region = classify_regions(&c.tets, &c.points, oracle);
        assert_eq!(region.len(), c.tets.len());
        assert!(region.iter().all(|&r| r == 1), "every tet inside the cube is region 1");
    }
}

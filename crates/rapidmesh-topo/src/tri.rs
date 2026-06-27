//! Triangle complex: dimension-uniform topology (identical for planar MoM and
//! embedded surface meshes) plus coordinate-aware geometry.

use crate::convention::{canonical_edge, NONE, TRI_EDGE_LOCAL};
use crate::csr::Csr;
use crate::source::TriSource;
use std::collections::HashMap;

/// Derived connectivity of a triangle mesh. Pure topology — no coordinates.
#[derive(Debug, Clone, Default)]
pub struct TriTopology {
    pub n_verts: usize,
    /// The triangle → vertex connectivity (the source elements, as `u32`), so the
    /// complex is self-contained and the geometry builders need only coordinates.
    pub tris: Vec<[u32; 3]>,
    /// Unique edges, canonical `(min, max)`.
    pub edges: Vec<[u32; 2]>,
    /// Global edge id per local edge of each triangle (`TRI_EDGE_LOCAL` order).
    pub tri_edges: Vec<[u32; 3]>,
    /// The (≤2) triangles incident to each edge; `NONE` fills a free slot.
    /// Manifold/boundary assumption; for non-manifold edges use `vert_tris`.
    pub edge_tris: Vec<[u32; 2]>,
    /// The tag of each incident triangle (parallel to `edge_tris`); `i64::MIN`
    /// for a free slot. Lets a MoM build pick interior-same-tag (RWG) or
    /// boundary/tag-change edges without re-walking the mesh.
    pub edge_tags: Vec<[i64; 2]>,
    /// Vertex → incident triangles.
    pub vert_tris: Csr,
}

impl TriTopology {
    /// Build the complex in one O(n) pass.
    pub fn build(src: &impl TriSource) -> Self {
        let nt = src.n_tris();
        let mut edge_id: HashMap<[u32; 2], u32> = HashMap::new();
        let mut edges: Vec<[u32; 2]> = Vec::new();
        let mut tris = vec![[0u32; 3]; nt];
        let mut tri_edges = vec![[0u32; 3]; nt];
        let mut vt_pairs: Vec<(u32, u32)> = Vec::with_capacity(nt * 3);

        for t in 0..nt {
            let tri = src.tri(t);
            tris[t] = tri;
            for &v in &tri {
                vt_pairs.push((v, t as u32));
            }
            for (k, &[la, lb]) in TRI_EDGE_LOCAL.iter().enumerate() {
                let (e, _) = canonical_edge(tri[la], tri[lb]);
                let id = *edge_id.entry(e).or_insert_with(|| {
                    edges.push(e);
                    (edges.len() - 1) as u32
                });
                tri_edges[t][k] = id;
            }
        }

        let ne = edges.len();
        let mut edge_tris = vec![[NONE; 2]; ne];
        let mut edge_tags = vec![[i64::MIN; 2]; ne];
        let mut cnt = vec![0u8; ne];
        for t in 0..nt {
            let tag = src.tri_tag(t);
            for k in 0..3 {
                let e = tri_edges[t][k] as usize;
                let c = cnt[e];
                if c < 2 {
                    edge_tris[e][c as usize] = t as u32;
                    edge_tags[e][c as usize] = tag;
                }
                cnt[e] = c.saturating_add(1);
            }
        }

        let vert_tris = Csr::from_pairs(src.n_verts(), &vt_pairs);
        TriTopology { n_verts: src.n_verts(), tris, edges, tri_edges, edge_tris, edge_tags, vert_tris }
    }
}

/// Per-element geometry of a triangle mesh. Coordinate-aware: planar (MoM) via
/// [`build_2d`](TriGeometry::build_2d), 3D-embedded surface via
/// [`build_3d`](TriGeometry::build_3d). All quantities are basis-free facts about
/// the mesh embedding (no reference element, no discretization).
#[derive(Debug, Clone, Default)]
pub struct TriGeometry {
    /// Unsigned triangle area.
    pub area: Vec<f64>,
    /// Triangle centroid (planar: `z = 0`).
    pub centroid: Vec<[f64; 3]>,
    /// Unit face normal. Planar: `[0, 0, ±1]` (sign of the signed area). 3D: the
    /// unit normal of the stored winding.
    pub normal: Vec<[f64; 3]>,
    /// Second area moment about the centroid `[∫dx², ∫dx·dy, ∫dy²]` (the multipole
    /// MoM moment). Populated by `build_2d` only; empty for 3D surfaces (the
    /// in-plane moment has no global frame there).
    pub inertia: Vec<[f64; 3]>,
    /// Per-edge length (parallel to `TriTopology::edges`).
    pub edge_len: Vec<f64>,
    /// Per-edge midpoint (planar: `z = 0`).
    pub edge_mid: Vec<[f64; 3]>,
}

impl TriGeometry {
    /// Planar (z = 0) geometry: area, centroid, ±z normal, second area moment,
    /// edge lengths/midpoints.
    pub fn build_2d(topo: &TriTopology, coords: &[[f64; 2]]) -> Self {
        let nt = topo.tris.len();
        let mut area = vec![0.0; nt];
        let mut centroid = vec![[0.0; 3]; nt];
        let mut normal = vec![[0.0; 3]; nt];
        let mut inertia = vec![[0.0; 3]; nt];
        for t in 0..nt {
            let [ia, ib, ic] = topo.tris[t];
            let (a, b, c) = (coords[ia as usize], coords[ib as usize], coords[ic as usize]);
            let signed = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
            area[t] = signed.abs();
            normal[t] = [0.0, 0.0, if signed >= 0.0 { 1.0 } else { -1.0 }];
            let (cx, cy) = ((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0);
            centroid[t] = [cx, cy, 0.0];
            let d = [[a[0] - cx, a[1] - cy], [b[0] - cx, b[1] - cy], [c[0] - cx, c[1] - cy]];
            let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
            for p in &d {
                sxx += p[0] * p[0];
                sxy += p[0] * p[1];
                syy += p[1] * p[1];
            }
            let k = area[t] / 12.0;
            inertia[t] = [k * sxx, k * sxy, k * syy];
        }
        let ne = topo.edges.len();
        let mut edge_len = vec![0.0; ne];
        let mut edge_mid = vec![[0.0; 3]; ne];
        for (e, &[ia, ib]) in topo.edges.iter().enumerate() {
            let (a, b) = (coords[ia as usize], coords[ib as usize]);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            edge_len[e] = (dx * dx + dy * dy).sqrt();
            edge_mid[e] = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, 0.0];
        }
        TriGeometry { area, centroid, normal, inertia, edge_len, edge_mid }
    }

    /// Surface (3D-embedded) geometry: area, centroid, unit normal, edge
    /// lengths/midpoints. `inertia` is left empty (see the field doc).
    pub fn build_3d(topo: &TriTopology, coords: &[[f64; 3]]) -> Self {
        use crate::math::{add, cross, edge_geom, norm, scale, sub};
        let nt = topo.tris.len();
        let mut area = vec![0.0; nt];
        let mut centroid = vec![[0.0; 3]; nt];
        let mut normal = vec![[0.0; 3]; nt];
        for t in 0..nt {
            let [ia, ib, ic] = topo.tris[t];
            let (a, b, c) = (coords[ia as usize], coords[ib as usize], coords[ic as usize]);
            let n = cross(sub(b, a), sub(c, a));
            let len = norm(n);
            area[t] = 0.5 * len;
            normal[t] = if len > 0.0 { scale(n, 1.0 / len) } else { [0.0; 3] };
            centroid[t] = scale(add(add(a, b), c), 1.0 / 3.0);
        }
        let (edge_len, edge_mid) = edge_geom(&topo.edges, coords);
        TriGeometry { area, centroid, normal, inertia: Vec::new(), edge_len, edge_mid }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::Tris;

    #[test]
    fn single_triangle() {
        let topo = TriTopology::build(&Tris::untagged(&[[0, 1, 2]], 3));
        assert_eq!(topo.edges.len(), 3);
        // every edge is a boundary edge: one incident triangle.
        for e in &topo.edge_tris {
            assert_eq!(e[0], 0);
            assert_eq!(e[1], NONE);
        }
        // each vertex touches the one triangle.
        for v in 0..3 {
            assert_eq!(topo.vert_tris.row(v), &[0]);
        }
    }

    #[test]
    fn shared_edge_is_interior() {
        // two triangles sharing edge (1,2).
        let topo = TriTopology::build(&Tris {
            tris: &[[0, 1, 2], [1, 3, 2]],
            tags: &[7, 9],
            n_verts: 4,
        });
        assert_eq!(topo.edges.len(), 5);
        // find the shared edge id (canonical (1,2)).
        let shared = topo.edges.iter().position(|&e| e == [1, 2]).unwrap();
        let mut tris = topo.edge_tris[shared];
        tris.sort_unstable();
        assert_eq!(tris, [0, 1]);
        // its two sides carry the two triangles' tags.
        let mut tags = topo.edge_tags[shared];
        tags.sort_unstable();
        assert_eq!(tags, [7, 9]);
    }

    #[test]
    fn geometry_2d_unit_right_triangle() {
        let topo = TriTopology::build(&Tris::untagged(&[[0, 1, 2]], 3));
        let g = TriGeometry::build_2d(&topo, &[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        assert!((g.area[0] - 0.5).abs() < 1e-12);
        assert_eq!(g.normal[0], [0.0, 0.0, 1.0]); // CCW
        let c = g.centroid[0];
        assert!((c[0] - 1.0 / 3.0).abs() < 1e-12 && (c[1] - 1.0 / 3.0).abs() < 1e-12 && c[2] == 0.0);
        // symmetric triangle: ∫dx² == ∫dy², cross moment negative.
        assert!((g.inertia[0][0] - g.inertia[0][2]).abs() < 1e-12);
        assert!(g.inertia[0][1] < 0.0);
    }

    #[test]
    fn geometry_3d_normal_and_area() {
        let topo = TriTopology::build(&Tris::untagged(&[[0, 1, 2]], 3));
        let g = TriGeometry::build_3d(&topo, &[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
        assert!((g.area[0] - 2.0).abs() < 1e-12);
        assert_eq!(g.normal[0], [0.0, 0.0, 1.0]);
        assert!(g.inertia.is_empty());
        assert_eq!(g.edge_len.len(), topo.edges.len());
    }
}

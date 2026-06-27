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
        let mut tri_edges = vec![[0u32; 3]; nt];
        let mut vt_pairs: Vec<(u32, u32)> = Vec::with_capacity(nt * 3);

        for t in 0..nt {
            let tri = src.tri(t);
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
        TriTopology { n_verts: src.n_verts(), edges, tri_edges, edge_tris, edge_tags, vert_tris }
    }
}

/// Per-element geometry of a triangle mesh. Coordinate-aware (2D planar or 3D
/// surface). **Stub** — see `DESIGN.md`.
#[derive(Debug, Clone, Default)]
pub struct TriGeometry {
    pub area: Vec<f64>,
    pub centroid: Vec<[f64; 3]>,
    /// Unit normal (3D only; zero in 2D).
    pub normal: Vec<[f64; 3]>,
    /// Second area moment `[Ixx, Ixy, Iyy]` about the centroid (multipole MoM).
    pub inertia: Vec<[f64; 3]>,
    pub edge_len: Vec<f64>,
    pub edge_mid: Vec<[f64; 3]>,
}

impl TriGeometry {
    /// Planar (z = 0) geometry: area, centroid, inertia. **TODO**.
    pub fn build_2d(_topo: &TriTopology, _coords: &[[f64; 2]]) -> Self {
        unimplemented!("TriGeometry::build_2d — Tier-2 stub")
    }

    /// Surface (3D-embedded) geometry: + per-triangle normal. **TODO**.
    pub fn build_3d(_topo: &TriTopology, _coords: &[[f64; 3]]) -> Self {
        unimplemented!("TriGeometry::build_3d — Tier-2 stub")
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
}

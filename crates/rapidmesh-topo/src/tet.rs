//! Tetrahedral complex: full 0/1/2/3-cell incidence with the orientation signs
//! a vector-FEM (Nédélec) assembly needs, plus coordinate-aware geometry.

use crate::convention::{canonical_edge, sort3_sign, NONE, TET_EDGE_LOCAL, TET_FACE_LOCAL, TRI_EDGE_LOCAL};
use crate::csr::Csr;
use crate::source::TetSource;
use std::collections::HashMap;

/// Derived connectivity of a tet mesh. Pure topology — no coordinates.
#[derive(Debug, Clone, Default)]
pub struct TetTopology {
    pub n_verts: usize,
    /// Unique edges, canonical `(min, max)`.
    pub edges: Vec<[u32; 2]>,
    /// Unique faces, canonical (vertex ids ascending).
    pub faces: Vec<[u32; 3]>,
    /// Global edge id per local edge (`TET_EDGE_LOCAL` order).
    pub tet_edges: Vec<[u32; 6]>,
    /// `+1` if the local edge runs min→max (matches canonical), else `-1`.
    pub tet_edge_sign: Vec<[i8; 6]>,
    /// Global face id per local face (`TET_FACE_LOCAL` order).
    pub tet_faces: Vec<[u32; 4]>,
    /// Parity from the local outward order to the canonical face order. Two tets
    /// sharing a face carry opposite signs.
    pub tet_face_sign: Vec<[i8; 4]>,
    /// The 3 edges of each face (`TRI_EDGE_LOCAL` order on the canonical face).
    pub face_edges: Vec<[u32; 3]>,
    /// The (≤2) tets incident to each face; `NONE` marks a boundary face.
    pub face_tets: Vec<[u32; 2]>,
    /// Vertex → incident edges.
    pub vert_edges: Csr,
    /// Vertex → incident tets.
    pub vert_tets: Csr,
}

impl TetTopology {
    /// Build the complex in one O(n) pass.
    pub fn build(src: &impl TetSource) -> Self {
        let nt = src.n_tets();
        let mut edge_id: HashMap<[u32; 2], u32> = HashMap::new();
        let mut edges: Vec<[u32; 2]> = Vec::new();
        let mut face_id: HashMap<[u32; 3], u32> = HashMap::new();
        let mut faces: Vec<[u32; 3]> = Vec::new();
        let mut tet_edges = vec![[0u32; 6]; nt];
        let mut tet_edge_sign = vec![[0i8; 6]; nt];
        let mut tet_faces = vec![[0u32; 4]; nt];
        let mut tet_face_sign = vec![[0i8; 4]; nt];
        let mut vt_pairs: Vec<(u32, u32)> = Vec::with_capacity(nt * 4);

        for t in 0..nt {
            let tet = src.tet(t);
            for &v in &tet {
                vt_pairs.push((v, t as u32));
            }
            for (k, &[la, lb]) in TET_EDGE_LOCAL.iter().enumerate() {
                let (e, s) = canonical_edge(tet[la], tet[lb]);
                let id = *edge_id.entry(e).or_insert_with(|| {
                    edges.push(e);
                    (edges.len() - 1) as u32
                });
                tet_edges[t][k] = id;
                tet_edge_sign[t][k] = s;
            }
            for (k, &[a, b, c]) in TET_FACE_LOCAL.iter().enumerate() {
                let (sorted, s) = sort3_sign([tet[a], tet[b], tet[c]]);
                let id = *face_id.entry(sorted).or_insert_with(|| {
                    faces.push(sorted);
                    (faces.len() - 1) as u32
                });
                tet_faces[t][k] = id;
                tet_face_sign[t][k] = s;
            }
        }

        // Face → its 3 edges (every edge already exists as a tet edge).
        let nf = faces.len();
        let mut face_edges = vec![[0u32; 3]; nf];
        for (fi, &f) in faces.iter().enumerate() {
            for (k, &[la, lb]) in TRI_EDGE_LOCAL.iter().enumerate() {
                let (e, _) = canonical_edge(f[la], f[lb]);
                face_edges[fi][k] = *edge_id.get(&e).expect("face edge must be a tet edge");
            }
        }

        // Face → incident tets (a volume face has ≤2).
        let mut face_tets = vec![[NONE; 2]; nf];
        let mut fcnt = vec![0u8; nf];
        for t in 0..nt {
            for k in 0..4 {
                let f = tet_faces[t][k] as usize;
                let c = fcnt[f];
                if c < 2 {
                    face_tets[f][c as usize] = t as u32;
                }
                fcnt[f] = c.saturating_add(1);
            }
        }

        let mut ve_pairs: Vec<(u32, u32)> = Vec::with_capacity(edges.len() * 2);
        for (ei, &[a, b]) in edges.iter().enumerate() {
            ve_pairs.push((a, ei as u32));
            ve_pairs.push((b, ei as u32));
        }
        let vert_edges = Csr::from_pairs(src.n_verts(), &ve_pairs);
        let vert_tets = Csr::from_pairs(src.n_verts(), &vt_pairs);

        TetTopology {
            n_verts: src.n_verts(),
            edges,
            faces,
            tet_edges,
            tet_edge_sign,
            tet_faces,
            tet_face_sign,
            face_edges,
            face_tets,
            vert_edges,
            vert_tets,
        }
    }
}

/// Per-element geometry of a tet mesh. **Stub** — see `DESIGN.md`. `grad` will
/// hold ∇λ_i (the inverse Jacobian), the only per-element datum a P1/Nédélec
/// assembly needs.
#[derive(Debug, Clone, Default)]
pub struct TetGeometry {
    pub volume: Vec<f64>,
    pub grad: Vec<[[f64; 3]; 4]>,
    pub edge_len: Vec<f64>,
    pub face_area: Vec<f64>,
    pub face_normal: Vec<[f64; 3]>,
    pub face_centroid: Vec<[f64; 3]>,
}

impl TetGeometry {
    pub fn build(_topo: &TetTopology, _coords: &[[f64; 3]]) -> Self {
        unimplemented!("TetGeometry::build — Tier-2 stub")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::Tets;

    #[test]
    fn single_tet() {
        let topo = TetTopology::build(&Tets { tets: &[[0, 1, 2, 3]], n_verts: 4 });
        assert_eq!(topo.edges.len(), 6);
        assert_eq!(topo.faces.len(), 4);
        // ascending vertex labels -> every local edge already runs min→max.
        assert_eq!(topo.tet_edge_sign[0], [1; 6]);
        // all four faces are on the boundary.
        for f in &topo.face_tets {
            assert_eq!(f[1], NONE);
        }
    }

    #[test]
    fn edge_sign_reversed() {
        // local edge 0 = (tet[0], tet[1]) = (3, 1) -> canonical (1,3), reversed.
        let topo = TetTopology::build(&Tets { tets: &[[3, 1, 2, 0]], n_verts: 4 });
        assert_eq!(topo.tet_edge_sign[0][0], -1);
    }

    #[test]
    fn shared_face_opposite_signs() {
        // two tets sharing face (1,2,3).
        let topo = TetTopology::build(&Tets {
            tets: &[[0, 1, 2, 3], [1, 2, 3, 4]],
            n_verts: 5,
        });
        let shared = topo.faces.iter().position(|&f| f == [1, 2, 3]).unwrap();
        let mut tets = topo.face_tets[shared];
        tets.sort_unstable();
        assert_eq!(tets, [0, 1]);
        // each tet lists the shared face; the two carry opposite orientation.
        let sign_in = |t: usize| {
            let k = topo.tet_faces[t].iter().position(|&f| f as usize == shared).unwrap();
            topo.tet_face_sign[t][k]
        };
        assert_eq!(sign_in(0), -sign_in(1));
    }
}

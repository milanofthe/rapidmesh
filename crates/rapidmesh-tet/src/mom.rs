//! MoM/FEM mesh info derived from a [`SurfaceMesh`]: edge adjacency and the RWG /
//! boundary / port / element-quality data a method-of-moments or FEM solver reads
//! off the surface. Pure topology + geometry over `points` and `faces`, so a Rust
//! consumer (rapidmom) gets exactly what the Python wrapper used to compute on
//! numpy -- no Python in the loop.

use crate::conform::SurfaceMesh;
use rapidmesh_geom::vec3::{cross, dot, len, sub};
use rustc_hash::FxHashMap;

/// Undirected edge adjacency of a surface mesh.
pub struct EdgeAdjacency {
    /// Unique edges `[v0, v1]` with `v0 < v1`, sorted lexicographically.
    pub edges: Vec<[u32; 2]>,
    /// The up-to-two incident triangle indices per edge (the first and the last
    /// incident triangle); `-1` for a free side.
    pub tris: Vec<[i64; 2]>,
    /// `face_tag` of those incident triangles; `-1` where none.
    pub tags: Vec<[i64; 2]>,
}

impl SurfaceMesh {
    /// Undirected edge -> incident triangles + their face tags. Manifold per
    /// conductor; a non-manifold edge keeps its first and last triangle.
    pub fn edge_adjacency(&self) -> EdgeAdjacency {
        // Visit the three edge slots in EDGE-TYPE-major order ([0,1] of every
        // face, then [1,2], then [2,0]) so the first/last incident triangle --
        // hence the RWG +/- orientation -- matches a column-stacked numpy build.
        let mut map: FxHashMap<[u32; 2], (i64, i64)> = FxHashMap::default();
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            for (ti, f) in self.faces.iter().enumerate() {
                let (a, b) = (f.tri[i] as u32, f.tri[j] as u32);
                let key = if a < b { [a, b] } else { [b, a] };
                let slot = map.entry(key).or_insert((-1, -1));
                if slot.0 < 0 {
                    slot.0 = ti as i64;
                } else {
                    slot.1 = ti as i64;
                }
            }
        }
        let mut edges: Vec<[u32; 2]> = map.keys().copied().collect();
        edges.sort_unstable();
        let tag_of = |t: i64| -> i64 {
            if t >= 0 {
                self.faces[t as usize].face_tag.0 as i64
            } else {
                -1
            }
        };
        let mut tris = Vec::with_capacity(edges.len());
        let mut tags = Vec::with_capacity(edges.len());
        for e in &edges {
            let (t0, t1) = map[e];
            tris.push([t0, t1]);
            tags.push([tag_of(t0), tag_of(t1)]);
        }
        EdgeAdjacency { edges, tris, tags }
    }

    /// MoM RWG basis edges (the surface degrees of freedom): interior edges shared
    /// by two triangles of the SAME conductor tag. Returns `[v0, v1, tri_plus,
    /// tri_minus]` (current flows `+` -> `-`).
    pub fn rwg_edges(&self) -> Vec<[i64; 4]> {
        let adj = self.edge_adjacency();
        (0..adj.edges.len())
            .filter(|&i| adj.tris[i][1] >= 0 && adj.tags[i][0] == adj.tags[i][1])
            .map(|i| {
                let e = adj.edges[i];
                [e[0] as i64, e[1] as i64, adj.tris[i][0], adj.tris[i][1]]
            })
            .collect()
    }

    /// Conductor outline: edges with only one same-tag triangle (a free side or a
    /// tag change). Returns `[v0, v1, tri]`.
    pub fn boundary_edges(&self) -> Vec<[i64; 3]> {
        let adj = self.edge_adjacency();
        (0..adj.edges.len())
            .filter(|&i| adj.tris[i][1] < 0 || adj.tags[i][0] != adj.tags[i][1])
            .map(|i| {
                let e = adj.edges[i];
                [e[0] as i64, e[1] as i64, adj.tris[i][0]]
            })
            .collect()
    }

    /// Port helper: boundary edges whose BOTH endpoints lie on the line
    /// `{axis = value}` with the first non-`axis` coordinate in `[lo, hi]`
    /// (`axis` is `0`/`1`/`2`). Returns vertex pairs `[v0, v1]`.
    pub fn edges_on_line(&self, axis: usize, value: f64, lo: f64, hi: f64, tol: f64) -> Vec<[u32; 2]> {
        let other = (0..3).find(|&c| c != axis).unwrap_or(0);
        let on = |vi: u32| -> bool {
            let p = self.points[vi as usize];
            (p[axis] - value).abs() <= tol && p[other] >= lo - tol && p[other] <= hi + tol
        };
        self.boundary_edges()
            .iter()
            .filter(|e| on(e[0] as u32) && on(e[1] as u32))
            .map(|e| [e[0] as u32, e[1] as u32])
            .collect()
    }

    /// Per-triangle area (input units), parallel to `faces`.
    pub fn face_areas(&self) -> Vec<f64> {
        let p = &self.points;
        self.faces
            .iter()
            .map(|f| 0.5 * len(cross(sub(p[f.tri[1]], p[f.tri[0]]), sub(p[f.tri[2]], p[f.tri[0]]))))
            .collect()
    }

    /// Per-triangle minimum interior angle in degrees (the element-quality field;
    /// all `>=` the Ruppert bound), parallel to `faces`.
    pub fn face_min_angles(&self) -> Vec<f64> {
        // Angle at `u` between the edges `u->v` and `u->w`.
        let angle = |u: [f64; 3], v: [f64; 3], w: [f64; 3]| -> f64 {
            let (e1, e2) = (sub(v, u), sub(w, u));
            let c = dot(e1, e2) / (len(e1) * len(e2) + 1e-30);
            c.clamp(-1.0, 1.0).acos().to_degrees()
        };
        self.faces
            .iter()
            .map(|f| {
                let (a, b, c) =
                    (self.points[f.tri[0]], self.points[f.tri[1]], self.points[f.tri[2]]);
                angle(a, b, c).min(angle(b, c, a)).min(angle(c, a, b))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::conform::{SurfaceFace, SurfaceMesh};
    use rapidmesh_geom::{FaceTag, RegionTag};

    /// Unit square (0,0),(1,0),(1,1),(0,1) split into two same-tag right triangles
    /// sharing the diagonal (0,2).
    fn square() -> SurfaceMesh {
        let face = |tri| SurfaceFace {
            tri,
            face_tag: FaceTag(7),
            regions: [RegionTag(0); 2],
            patch: 0,
            surface: 0,
        };
        SurfaceMesh {
            points: vec![[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
            faces: vec![face([0, 1, 2]), face([0, 2, 3])],
            surfaces: Vec::new(),
            surface_owners: Vec::new(),
        }
    }

    #[test]
    fn square_mom() {
        let m = square();
        // 5 unique edges: 4 outer + the shared diagonal.
        let adj = m.edge_adjacency();
        assert_eq!(adj.edges.len(), 5);
        // The diagonal (0,2) is the only RWG edge (interior, same tag).
        let rwg = m.rwg_edges();
        assert_eq!(rwg.len(), 1);
        assert_eq!([rwg[0][0], rwg[0][1]], [0, 2]);
        // The 4 outer edges form the boundary.
        assert_eq!(m.boundary_edges().len(), 4);
        // Each right triangle has area 1/2 and a 45deg min angle.
        let ar = m.face_areas();
        assert!((ar[0] - 0.5).abs() < 1e-12 && (ar[1] - 0.5).abs() < 1e-12);
        let an = m.face_min_angles();
        assert!((an[0] - 45.0).abs() < 1e-9 && (an[1] - 45.0).abs() < 1e-9);
        // A line through the bottom edge {y=0} picks up its boundary edge (0,1).
        let on = m.edges_on_line(1, 0.0, -1.0, 2.0, 1e-9);
        assert_eq!(on, vec![[0, 1]]);
    }
}

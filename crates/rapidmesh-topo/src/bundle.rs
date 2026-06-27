//! THE two embedding endpoints: one for 2D/surface, one for 3D/volume.
//!
//! Each returns a complete bundle — mesh + topology + element geometry — so a
//! consumer (rapidmom / rapidfem) gets *everything* from a single call and never
//! has to choose between overlapping accessors. This is the canonical Rust API
//! for embedding rapidmesh:
//!
//! ```ignore
//! let m = rapidmesh_topo::mesh_2d(&plc, &params); // MoM
//! m.mesh;  m.topo;  m.geom;  m.rwg_candidate_edges();
//!
//! let v = rapidmesh_topo::mesh_3d(&plc, &params); // FEM
//! v.mesh;  v.topo;  v.geom;  v.exact_face_normals();
//! ```
//!
//! The basis-aware layer (RWG/Nédélec DOFs, quadrature, assembly) lives in the
//! solver — see `DESIGN.md`.

use crate::convention::NONE;
use crate::{TetGeometry, TetTopology, TriGeometry, TriTopology};
use rapidmesh_geom::TaggedPlc;
use rapidmesh_tet::{mesh_cdt, surface_mesh, MeshParams, SurfaceMesh, TetMesh};

/// Everything about a 2D / surface mesh (the MoM target): the meshed surface, its
/// derived topology, and its element geometry.
pub struct Mesh2D {
    /// The surface mesh: points, tagged faces, analytic surfaces.
    pub mesh: SurfaceMesh,
    /// Edges, incidence, per-edge tags, vertex stars.
    pub topo: TriTopology,
    /// Areas, centroids, normals, edge lengths/midpoints.
    pub geom: TriGeometry,
}

impl Mesh2D {
    /// Bundle an existing surface mesh; topology + geometry are derived once.
    pub fn build(mesh: SurfaceMesh) -> Self {
        let topo = TriTopology::build(&mesh);
        let geom = TriGeometry::build_3d(&topo, &mesh.points);
        Mesh2D { mesh, topo, geom }
    }

    /// Exact analytic outward normals per boundary face (`None` where planar — the
    /// facet normal is already exact — or no closed form). Parallel to
    /// `mesh.faces`.
    pub fn exact_face_normals(&self) -> Vec<Option<[f64; 3]>> {
        crate::mesher::exact_face_normals(&self.mesh.points, &self.mesh.faces, &self.mesh.surfaces)
    }

    /// RWG-eligible edges: interior edges shared by two SAME-tag triangles, as
    /// `[v0, v1, tri_plus, tri_minus]`. A topology query — the solver builds the
    /// actual basis/DOFs; this lives here so everything is in one place.
    pub fn rwg_candidate_edges(&self) -> Vec<[u32; 4]> {
        let t = &self.topo;
        (0..t.edges.len())
            .filter(|&e| t.edge_tris[e][1] != NONE && t.edge_tags[e][0] == t.edge_tags[e][1])
            .map(|e| {
                let [a, b] = t.edges[e];
                [a, b, t.edge_tris[e][0], t.edge_tris[e][1]]
            })
            .collect()
    }

    /// Conductor outline: edges with a free side or a tag change, as
    /// `[v0, v1, tri]`.
    pub fn boundary_edges(&self) -> Vec<[u32; 3]> {
        let t = &self.topo;
        (0..t.edges.len())
            .filter(|&e| t.edge_tris[e][1] == NONE || t.edge_tags[e][0] != t.edge_tags[e][1])
            .map(|e| {
                let [a, b] = t.edges[e];
                [a, b, t.edge_tris[e][0]]
            })
            .collect()
    }

    /// Port helper: boundary edges whose both endpoints lie on `{axis = value}`
    /// with the first other coordinate in `[lo, hi]`. Returns vertex pairs.
    pub fn edges_on_line(&self, axis: usize, value: f64, lo: f64, hi: f64, tol: f64) -> Vec<[u32; 2]> {
        let other = (0..3).find(|&c| c != axis).unwrap_or(0);
        let on = |vi: u32| {
            let p = self.mesh.points[vi as usize];
            (p[axis] - value).abs() <= tol && p[other] >= lo - tol && p[other] <= hi + tol
        };
        self.boundary_edges()
            .iter()
            .filter(|e| on(e[0]) && on(e[1]))
            .map(|e| [e[0], e[1]])
            .collect()
    }
}

/// THE 2D endpoint: mesh a PLC into a complete surface bundle.
pub fn mesh_2d(plc: &TaggedPlc, params: &MeshParams) -> Mesh2D {
    Mesh2D::build(surface_mesh(plc, params))
}

/// Everything about a 3D / volume mesh (the FEM target): the meshed volume, its
/// derived topology (with orientation signs), and its element geometry.
pub struct Mesh3D {
    /// The tet mesh: points, tets, regions, tagged boundary faces, surfaces.
    pub mesh: TetMesh,
    /// Edges, faces, incidence + orientation signs, vertex stars.
    pub topo: TetTopology,
    /// Volumes, ∇λ_i, face areas/normals/centroids, edge lengths.
    pub geom: TetGeometry,
}

impl Mesh3D {
    /// Bundle an existing tet mesh; topology + geometry are derived once.
    pub fn build(mesh: TetMesh) -> Self {
        let topo = TetTopology::build(&mesh);
        let geom = TetGeometry::build(&topo, &mesh.points);
        Mesh3D { mesh, topo, geom }
    }

    /// Exact analytic outward normals per boundary face (`None` where planar / no
    /// closed form). Parallel to `mesh.faces`.
    pub fn exact_face_normals(&self) -> Vec<Option<[f64; 3]>> {
        crate::mesher::exact_face_normals(&self.mesh.points, &self.mesh.faces, &self.mesh.surfaces)
    }
}

/// THE 3D endpoint: mesh a PLC into a complete volume bundle. For an element
/// budget, mesh with [`rapidmesh_tet::mesh_cdt_budgeted`] then [`Mesh3D::build`].
pub fn mesh_3d(plc: &TaggedPlc, params: &MeshParams) -> Mesh3D {
    Mesh3D::build(mesh_cdt(plc, params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapidmesh_geom::{FaceTag, RegionTag, SurfaceKind};
    use rapidmesh_tet::SurfaceFace;

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
            surfaces: vec![SurfaceKind::Plane],
            surface_owners: vec![0],
        }
    }

    #[test]
    fn mesh2d_bundles_everything() {
        let m = Mesh2D::build(square());
        // topology + geometry are present and consistent.
        assert_eq!(m.topo.edges.len(), 5);
        assert_eq!(m.geom.area.len(), 2);
        assert!((m.geom.area[0] - 0.5).abs() < 1e-12);
        // the diagonal is the one RWG-eligible edge; 4 outer edges are boundary.
        let rwg = m.rwg_candidate_edges();
        assert_eq!(rwg.len(), 1);
        assert_eq!([rwg[0][0], rwg[0][1]], [0, 2]);
        assert_eq!(m.boundary_edges().len(), 4);
        // planar faces have no analytic override.
        assert!(m.exact_face_normals().iter().all(|n| n.is_none()));
    }

    #[test]
    fn mesh3d_bundles_everything() {
        let mesh = TetMesh {
            points: vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            tets: vec![[0, 1, 2, 3]],
            tet_regions: vec![RegionTag(1)],
            faces: Vec::new(),
            surfaces: Vec::new(),
            surface_owners: Vec::new(),
            plc_points: 4,
            point_size: Vec::new(),
        };
        let v = Mesh3D::build(mesh);
        assert_eq!(v.topo.edges.len(), 6);
        assert_eq!(v.topo.faces.len(), 4);
        assert_eq!(v.geom.volume.len(), 1);
        assert!((v.geom.volume[0] - 1.0 / 6.0).abs() < 1e-12);
    }
}

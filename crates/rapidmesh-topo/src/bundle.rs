//! THE two embedding endpoints: one for 2D, one for 3D.
//!
//! Each returns a complete bundle — mesh + topology + element geometry — so a
//! consumer (rapidmom / rapidfem) gets *everything* from a single call and never
//! has to choose between overlapping accessors. This is the canonical Rust API
//! for embedding rapidmesh:
//!
//! ```ignore
//! // 2D / MoM — the production 2D path (surf2d, the gmsh-grade mesher the wasm
//! // landing uses). Raw tagged 2D polygons in; a planar mesh bundle out.
//! let m = rapidmesh_topo::mesh_2d(&regions, |_p| 0.05, &Default::default());
//! m.points; m.tris; m.tri_tags; m.topo; m.geom; m.rwg_candidate_edges();
//!
//! // 3D / FEM — the volume path.
//! let v = rapidmesh_topo::mesh_3d(&plc, &params);
//! v.mesh; v.topo; v.geom; v.exact_face_normals();
//! ```
//!
//! The 2D path is *the* 2D path: this endpoint, the wasm landing, and the 3D
//! surface stage's planar patches all run the same `surf2d` core. The basis-aware
//! layer (RWG/Nédélec DOFs, quadrature, assembly) lives in the solver — see
//! `DESIGN.md`.

use crate::source::Tris;
use crate::{TetGeometry, TetTopology, TriGeometry, TriTopology};
use rapidmesh_geom::TaggedPlc;
use rapidmesh_tet::surf2d::mesh_polygon;
use rapidmesh_tet::{mesh_cdt, MeshParams, TetMesh};

// ============================== 2D / surface (MoM) ==========================

/// A tagged 2D region: an outer loop with optional holes, all in the xy plane.
/// The `tag` flows to every triangle of this region (the conductor / layer id a
/// MoM build reads for same-tag RWG edges).
pub struct Region2D {
    /// Outer boundary loop (closed; CCW).
    pub outer: Vec<[f64; 2]>,
    /// Hole loops (closed; CW), nested inside `outer`.
    pub holes: Vec<Vec<[f64; 2]>>,
    /// Conductor / layer tag carried by this region's triangles.
    pub tag: i64,
}

impl Region2D {
    /// A region with no holes.
    pub fn new(outer: Vec<[f64; 2]>, tag: i64) -> Self {
        Region2D { outer, holes: Vec::new(), tag }
    }
}

/// Tuning for the production 2D path. `Default` matches the wasm landing.
#[derive(Debug, Clone, Copy)]
pub struct Mesh2DOptions {
    /// Ruppert minimum-angle bound (degrees).
    pub min_angle_deg: f64,
    /// CVT (Lloyd) seed-relaxation iterations.
    pub cvt_iters: usize,
    /// Maximum Ruppert refinement passes.
    pub max_passes: usize,
}

impl Default for Mesh2DOptions {
    fn default() -> Self {
        Mesh2DOptions { min_angle_deg: 28.0, cvt_iters: 4, max_passes: 12 }
    }
}

/// Everything about a 2D mesh (the MoM target): the gmsh-grade planar
/// triangulation from the production 2D path, its derived topology, and its
/// element geometry. Coordinates are 2D.
pub struct Mesh2D {
    /// Vertices (2D).
    pub points: Vec<[f64; 2]>,
    /// Triangles (CCW).
    pub tris: Vec<[u32; 3]>,
    /// Conductor / layer tag per triangle (parallel to `tris`).
    pub tri_tags: Vec<i64>,
    /// Edges, incidence, per-edge tags, vertex stars.
    pub topo: TriTopology,
    /// Areas, centroids, second moments, edge lengths/midpoints (planar).
    pub geom: TriGeometry,
}

impl Mesh2D {
    /// RWG-eligible edges `[v0, v1, tri_plus, tri_minus]` (interior, same-tag).
    /// The canonical query lives on [`TriTopology`] — shared with the 3D-surface
    /// endpoint; this forwards so the 2D bundle exposes everything in one place.
    pub fn rwg_candidate_edges(&self) -> Vec<[u32; 4]> {
        self.topo.rwg_candidate_edges()
    }

    /// Conductor outline `[v0, v1, tri]` (free side or tag change). Forwards to
    /// [`TriTopology::boundary_edges`].
    pub fn boundary_edges(&self) -> Vec<[u32; 3]> {
        self.topo.boundary_edges()
    }

    /// Port helper: boundary edges whose both endpoints lie on `{axis = value}`
    /// (`axis` 0/1) with the other coordinate in `[lo, hi]`. Returns vertex pairs.
    pub fn edges_on_line(&self, axis: usize, value: f64, lo: f64, hi: f64, tol: f64) -> Vec<[u32; 2]> {
        let other = if axis == 0 { 1 } else { 0 };
        let on = |vi: u32| {
            let p = self.points[vi as usize];
            (p[axis] - value).abs() <= tol && p[other] >= lo - tol && p[other] <= hi + tol
        };
        self.topo
            .boundary_edges()
            .iter()
            .filter(|e| on(e[0]) && on(e[1]))
            .map(|e| [e[0], e[1]])
            .collect()
    }
}

/// THE 2D endpoint: mesh tagged 2D polygons through the production 2D path
/// (`surf2d` — the same gmsh-optimized mesher the wasm landing runs), into one
/// bundle. Each region is meshed (outer + holes) and concatenated, carrying its
/// tag per triangle. `target` is the sizing field `h(x)`.
pub fn mesh_2d(regions: &[Region2D], target: impl Fn([f64; 2]) -> f64, opts: &Mesh2DOptions) -> Mesh2D {
    let mut points: Vec<[f64; 2]> = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut tri_tags: Vec<i64> = Vec::new();

    for r in regions {
        // Resample the raw polygon edges onto the sizing field FIRST. The 2D core
        // protects the boundary it is handed (it assumes a pre-sampled contour,
        // as the wasm landing supplies), so a coarse input edge would otherwise
        // stay one long pinned edge and wreck the triangles against it. Sampling
        // to `h` gives a fine, graded boundary -- and uniform RWG edge lengths
        // along a conductor, which is what MoM wants.
        let mut loops: Vec<Vec<[f64; 2]>> = Vec::with_capacity(1 + r.holes.len());
        loops.push(resample_loop(&r.outer, &target));
        for hl in &r.holes {
            loops.push(resample_loop(hl, &target));
        }
        // Representative CVT seed spacing: the target at the outer centroid.
        let step = target(centroid2(&r.outer)).max(1e-9);
        let (pts, tr3) =
            mesh_polygon(&loops, &target, step, opts.min_angle_deg, opts.cvt_iters, opts.max_passes);
        let base = points.len() as u32;
        points.extend_from_slice(&pts);
        for t in &tr3 {
            tris.push([base + t[0] as u32, base + t[1] as u32, base + t[2] as u32]);
            tri_tags.push(r.tag);
        }
    }

    let topo = TriTopology::build(&Tris { tris: &tris, tags: &tri_tags, n_verts: points.len() });
    let geom = TriGeometry::build_2d(&topo, &points);
    Mesh2D { points, tris, tri_tags, topo, geom }
}

/// Split each edge of a closed loop to the sizing field, so the boundary the
/// protected core meshes against is fine and graded — never a pinned coarse edge.
/// A sub-`h` edge keeps its two endpoints (no over-splitting).
fn resample_loop(lp: &[[f64; 2]], target: &impl Fn([f64; 2]) -> f64) -> Vec<[f64; 2]> {
    let n = lp.len();
    if n < 2 {
        return lp.to_vec();
    }
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let a = lp[i];
        let b = lp[(i + 1) % n];
        out.push(a);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = (dx * dx + dy * dy).sqrt();
        let h = target([(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]).max(1e-9);
        let segs = (len / h).ceil() as usize;
        for k in 1..segs {
            let t = k as f64 / segs as f64;
            out.push([a[0] + dx * t, a[1] + dy * t]);
        }
    }
    out
}

fn centroid2(pts: &[[f64; 2]]) -> [f64; 2] {
    if pts.is_empty() {
        return [0.0, 0.0];
    }
    let (mut x, mut y) = (0.0, 0.0);
    for p in pts {
        x += p[0];
        y += p[1];
    }
    let n = pts.len() as f64;
    [x / n, y / n]
}

// ============================== 3D / volume (FEM) ===========================

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
    use crate::convention::NONE;
    use rapidmesh_geom::RegionTag;

    fn min_angle_deg(p: &[[f64; 2]], tris: &[[u32; 3]]) -> f64 {
        let ang = |u: [f64; 2], v: [f64; 2], w: [f64; 2]| {
            let e1 = [v[0] - u[0], v[1] - u[1]];
            let e2 = [w[0] - u[0], w[1] - u[1]];
            let d = (e1[0] * e2[0] + e1[1] * e2[1])
                / ((e1[0] * e1[0] + e1[1] * e1[1]).sqrt() * (e2[0] * e2[0] + e2[1] * e2[1]).sqrt() + 1e-30);
            d.clamp(-1.0, 1.0).acos().to_degrees()
        };
        tris.iter()
            .map(|t| {
                let (a, b, c) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
                ang(a, b, c).min(ang(b, c, a)).min(ang(c, a, b))
            })
            .fold(180.0, f64::min)
    }

    #[test]
    fn mesh2d_meshes_a_tagged_square() {
        let sq = Region2D::new(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], 7);
        // Coarse input edges (length 1) with h = 0.34: only correct if the
        // endpoint resamples the boundary -- a pinned coarse edge gives slivers.
        let m = mesh_2d(&[sq], |_p| 0.34, &Mesh2DOptions::default());
        assert!(!m.tris.is_empty());
        // every triangle carries the region tag.
        assert!(m.tri_tags.iter().all(|&t| t == 7));
        // the triangulation tiles the square exactly: areas sum to 1.
        let area: f64 = m.geom.area.iter().sum();
        assert!((area - 1.0).abs() < 1e-6, "area sum {area}");
        // QUALITY: the boundary was resampled to h, so no pinned-coarse-edge
        // slivers -- the min angle clears a healthy bound.
        let mn = min_angle_deg(&m.points, &m.tris);
        assert!(mn > 20.0, "min angle {mn} too low (boundary not resampled?)");
        // topology + RWG queries are present and consistent.
        assert!(!m.topo.edges.is_empty());
        assert!(m.rwg_candidate_edges().iter().all(|e| e[2] != NONE && e[3] != NONE));
        assert!(!m.boundary_edges().is_empty());
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

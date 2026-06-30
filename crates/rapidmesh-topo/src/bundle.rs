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
#[derive(Clone, Debug)]
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
#[derive(Clone)]
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
    /// The input regions (polygons), retained so the mesh can re-mesh ITSELF with a
    /// new sizing field — the basis for adaptive (AMR) refinement. See [`Mesh2D::remesh`].
    pub regions: Vec<Region2D>,
    /// The meshing options this mesh was built with (retained for `remesh`).
    pub opts: Mesh2DOptions,
}

impl Mesh2D {
    /// Re-mesh the SAME regions with a new sizing field `target` (e.g. an AMR-refined
    /// size field that shrinks `h` near marked elements). Returns a fresh, equally
    /// self-remeshable `Mesh2D`. The regions and meshing options are this mesh's own.
    pub fn remesh(&self, target: impl Fn([f64; 2]) -> f64) -> Mesh2D {
        mesh_2d(&self.regions, target, &self.opts)
    }

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
    // Merge overlapping/abutting input regions into connected shapes BEFORE triangulating: a
    // conductor handed in as many polygons (e.g. a transformer winding as dozens of abutting arcs)
    // becomes ONE region with a single CONFORMING mesh, so its RWG/edge graph is continuous (no
    // open winding). Non-touching regions stay separate. This robust 2D union (i_overlay) replaces
    // the old coincident-vertex weld, which only handled full-edge abutments and silently left
    // partial-edge / point joints disconnected.
    let merged: Vec<(Vec<[f64; 2]>, Vec<Vec<[f64; 2]>>)> = if regions.len() > 1 {
        union_regions(regions)
    } else {
        regions.iter().map(|r| (r.outer.clone(), r.holes.clone())).collect()
    };

    let mut points: Vec<[f64; 2]> = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();

    for (outer, holes) in &merged {
        if outer.len() < 3 {
            continue;
        }
        // Resample the raw polygon edges onto the sizing field FIRST (the 2D core protects the
        // boundary it is handed), giving a fine, graded boundary and uniform RWG edge lengths.
        let mut loops: Vec<Vec<[f64; 2]>> = Vec::with_capacity(1 + holes.len());
        loops.push(resample_loop(outer, &target));
        for hl in holes {
            loops.push(resample_loop(hl, &target));
        }
        let step = target(centroid2(outer)).max(1e-9);
        let (pts, tr3) =
            mesh_polygon(&loops, &target, step, opts.min_angle_deg, opts.cvt_iters, opts.max_passes);
        let base = points.len() as u32;
        points.extend_from_slice(&pts);
        for t in &tr3 {
            tris.push([base + t[0] as u32, base + t[1] as u32, base + t[2] as u32]);
        }
    }

    // Tags per ORIGINAL region (so a merged multi-conductor layer keeps its conductor labels):
    // one region → its tag directly (fast, exact previous behaviour); several → point-in-polygon
    // of each triangle centroid over the original regions.
    let tri_tags: Vec<i64> = if regions.len() == 1 {
        vec![regions[0].tag; tris.len()]
    } else {
        tris.iter()
            .map(|t| {
                let (a, b, c) = (points[t[0] as usize], points[t[1] as usize], points[t[2] as usize]);
                tag_at([(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0], regions)
            })
            .collect()
    };

    let topo = TriTopology::build(&Tris { tris: &tris, tags: &tri_tags, n_verts: points.len() });
    let geom = TriGeometry::build_2d(&topo, &points);
    Mesh2D { points, tris, tri_tags, topo, geom, regions: regions.to_vec(), opts: *opts }
}

/// Robust 2D union of all region polygons into connected shapes (i_overlay, MIT). Overlapping or
/// abutting regions merge into one shape; separate ones stay separate. Outers are forced CCW and
/// holes CW so the non-zero fill rule unions correctly regardless of the input orientation. Each
/// output is `(outer, holes)` in i_overlay's canonical orientation (outer CCW, holes CW).
fn union_regions(regions: &[Region2D]) -> Vec<(Vec<[f64; 2]>, Vec<Vec<[f64; 2]>>)> {
    use i_overlay::core::fill_rule::FillRule;
    use i_overlay::float::simplify::SimplifyShape;
    let mut contours: Vec<Vec<[f64; 2]>> = Vec::new();
    for r in regions {
        if r.outer.len() >= 3 {
            contours.push(oriented(&r.outer, true));
            for h in &r.holes {
                if h.len() >= 3 {
                    contours.push(oriented(h, false));
                }
            }
        }
    }
    contours
        .simplify_shape(FillRule::NonZero, 0.0)
        .into_iter()
        .map(|shape| {
            let mut it = shape.into_iter();
            let outer = it.next().unwrap_or_default();
            (outer, it.collect())
        })
        .collect()
}

/// Signed area (CCW positive).
fn signed_area(p: &[[f64; 2]]) -> f64 {
    let n = p.len();
    let mut a = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        a += p[i][0] * p[j][1] - p[j][0] * p[i][1];
    }
    0.5 * a
}

/// `p` re-oriented to CCW (`ccw=true`) or CW (`ccw=false`).
fn oriented(p: &[[f64; 2]], ccw: bool) -> Vec<[f64; 2]> {
    let mut v = p.to_vec();
    if (signed_area(p) > 0.0) != ccw {
        v.reverse();
    }
    v
}

/// Even-odd ray cast: is `p` strictly inside the ring?
fn point_in_ring(p: [f64; 2], ring: &[[f64; 2]]) -> bool {
    let n = ring.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (ring[i], ring[j]);
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Tag of the original region containing `p` (outer minus holes); falls back to the nearest
/// region's tag (for a centroid nudged just outside by the union's coordinate snapping).
fn tag_at(p: [f64; 2], regions: &[Region2D]) -> i64 {
    for r in regions {
        if point_in_ring(p, &r.outer) && !r.holes.iter().any(|h| point_in_ring(p, h)) {
            return r.tag;
        }
    }
    regions
        .iter()
        .min_by(|x, y| {
            let d = |r: &Region2D| {
                let c = centroid2(&r.outer);
                (c[0] - p[0]).powi(2) + (c[1] - p[1]).powi(2)
            };
            d(x).partial_cmp(&d(y)).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| r.tag)
        .unwrap_or(0)
}

/// Split each edge of a closed loop to the sizing field, so the boundary the protected core
/// meshes against is fine and **graded** — never a pinned coarse edge, and never an abrupt
/// short-next-to-long jump at a corner (which pins a sliver the Ruppert core cannot fix, since
/// it may not move boundary vertices).
///
/// The grading is the key: `h` is sampled at the two EDGE ENDPOINTS (not the midpoint), so two
/// edges sharing a corner agree on the spacing there — the last segment of one edge and the first
/// of the next are both ≈ `h(corner)`. Within an edge the segment length is graded geometrically
/// from `h(a)` to `h(b)`, so adjacent segment lengths differ by only the size-field ratio, never a
/// jump. Corners are preserved.
fn resample_loop(lp: &[[f64; 2]], target: &impl Fn([f64; 2]) -> f64) -> Vec<[f64; 2]> {
    let n = lp.len();
    if n < 2 {
        return lp.to_vec();
    }
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let a = lp[i];
        let b = lp[(i + 1) % n];
        out.push(a); // keep the corner
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-12 {
            continue;
        }
        // CANONICAL orientation (lexicographic lo→hi): a boundary edge SHARED by two abutting
        // regions is resampled to the IDENTICAL point set from either side (same lo/hi/grading),
        // so the separately-meshed regions weld conformally at the abutment (see `mesh_2d`).
        let rev = (a[0], a[1]) > (b[0], b[1]);
        let (lo, hi) = if rev { (b, a) } else { (a, b) };
        let ha = target(lo).max(1e-9);
        let hb = target(hi).max(1e-9);
        // Segment count = ∫₀ᴸ ds/h(s) with h linear in arc length (= the graded element count).
        let uniform = (ha - hb).abs() <= 1e-9 * ha.max(hb);
        let segs = if uniform {
            (len / ha).ceil().max(1.0) as usize
        } else {
            (len / (hb - ha) * (hb / ha).ln()).abs().ceil().max(1.0) as usize
        };
        // Canonical interior point at fraction k/segs along lo→hi (geometric grading h_k =
        // ha·(hb/ha)^frac ⇒ arc position t; first segment ≈ ha, last ≈ hb).
        let pt = |k: usize| -> [f64; 2] {
            let frac = k as f64 / segs as f64;
            let t = if uniform { frac } else { (ha * (hb / ha).powf(frac) - ha) / (hb - ha) };
            [lo[0] + (hi[0] - lo[0]) * t, lo[1] + (hi[1] - lo[1]) * t]
        };
        // Emit the canonical points in the a→b traversal order (reversed if the edge runs hi→lo).
        if rev {
            for k in (1..segs).rev() {
                out.push(pt(k));
            }
        } else {
            for k in 1..segs {
                out.push(pt(k));
            }
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

    /// RWG-connected components: triangles linked by an inner edge (shared by two triangles).
    fn n_components(m: &Mesh2D) -> usize {
        let nt = m.tris.len();
        let mut parent: Vec<usize> = (0..nt).collect();
        fn find(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        for inc in &m.topo.edge_tris {
            if inc[1] != NONE {
                let (a, b) = (find(&mut parent, inc[0] as usize), find(&mut parent, inc[1] as usize));
                if a != b {
                    parent[a] = b;
                }
            }
        }
        (0..nt).map(|i| find(&mut parent, i)).collect::<std::collections::HashSet<_>>().len()
    }

    /// THE FIX: two conductor regions sharing a full edge must mesh as ONE connected component
    /// (continuous RWG graph) while keeping BOTH conductor tags. Before the per-region union this
    /// produced two disjoint components — an open, non-conducting winding.
    #[test]
    fn abutting_regions_weld_into_one_component() {
        let a = Region2D::new(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], 1);
        let b = Region2D::new(vec![[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]], 2);
        let m = mesh_2d(&[a, b], |_p| 0.34, &Mesh2DOptions::default());
        assert_eq!(n_components(&m), 1, "abutting regions must weld into one component");
        assert!(m.tri_tags.iter().any(|&t| t == 1) && m.tri_tags.iter().any(|&t| t == 2), "both tags kept");
        let area: f64 = m.geom.area.iter().sum();
        assert!((area - 2.0).abs() < 1e-6, "area {area}");
    }

    /// Partial-edge abutment (B meets only the middle of A's edge) — the case the old vertex weld
    /// could not handle — must also weld via the union.
    #[test]
    fn partial_edge_abut_welds() {
        let a = Region2D::new(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], 1);
        let b = Region2D::new(vec![[1.0, 0.25], [2.0, 0.25], [2.0, 0.75], [1.0, 0.75]], 2);
        let m = mesh_2d(&[a, b], |_p| 0.2, &Mesh2DOptions::default());
        assert_eq!(n_components(&m), 1, "partial-edge abutment must weld");
    }

    /// Non-touching regions stay separate (two components, two tags).
    #[test]
    fn separate_regions_stay_disconnected() {
        let a = Region2D::new(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], 1);
        let b = Region2D::new(vec![[3.0, 0.0], [4.0, 0.0], [4.0, 1.0], [3.0, 1.0]], 2);
        let m = mesh_2d(&[a, b], |_p| 0.34, &Mesh2DOptions::default());
        assert_eq!(n_components(&m), 2, "non-touching regions stay separate");
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

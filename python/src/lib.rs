//! Python bindings: a thin native core under the pure-Python builder API in
//! `python_src/rapidmesh/geometry.py`. The builder constructs a
//! [`rapidmesh_geom::Scene`] through [`SceneBuilder`]; `mesh()` runs the full
//! pipeline (exact CSG assembly, conforming tetrahedralization, quality
//! optimization) and hands the result back as numpy arrays.

use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rapidmesh_geom::{
    cylinder, cylinder_iso, extrude_polygon, extrude_spline_profile, frustum, frustum_iso, helix,
    icosphere, loft, mesh_solid, naca0012_profile, pipe, sheet_disk, sheet_polygon, sheet_rect,
    solid_box, torus, wedge, facet_count, FaceTag, Scene,
};
use rapidmesh_brep::{build as brep_build, extract_topology};
use rapidmesh_tet::{
    mesh_cdt, mesh_plc_with, optimize, quality_stats, surface_mesh, MeshParams, OptimizeParams,
    QualityStats, SurfaceMesh, TetMesh,
};

/// Incremental scene builder (one solid/sheet per call); the Python layer
/// owns naming, defaults, and validation.
#[pyclass]
struct SceneBuilder {
    scene: Scene,
    region_maxh: Vec<(u32, f64)>,
    /// The geometry's global target size, so curved primitives can derive a facet
    /// density that tracks the requested mesh (the core owns this -- see
    /// `rapidmesh_geom::facet_count`; the Python layer only forwards `maxh`).
    default_maxh: Option<f64>,
}

impl SceneBuilder {
    /// Azimuthal facet count for a curved primitive of `radius`, from the per-solid
    /// `maxh` (or the geometry default), at the standard 1% surface tolerance.
    fn segs(&self, radius: f64, maxh: Option<f64>, passed: usize) -> usize {
        passed.max(facet_count(radius, maxh.or(self.default_maxh), 1e-2))
    }
}

impl SceneBuilder {
    /// Registers a built shape as a material solid or (void = true) a carved
    /// hole; returns the region tag (0 for voids).
    fn put(&mut self, f: rapidmesh_geom::Faceted, maxh: Option<f64>, void: bool) -> u32 {
        if void {
            self.scene.add_void(f);
            return 0;
        }
        let r = self.scene.add_solid(f);
        if let Some(h) = maxh {
            self.region_maxh.push((r.0, h));
        }
        r.0
    }
}

#[pymethods]
impl SceneBuilder {
    #[new]
    #[pyo3(signature = (default_maxh=None))]
    fn new(default_maxh: Option<f64>) -> SceneBuilder {
        SceneBuilder {
            scene: Scene::new(),
            region_maxh: Vec::new(),
            default_maxh,
        }
    }

    #[pyo3(signature = (min, max, maxh=None, void=false))]
    fn add_box(&mut self, min: [f64; 3], max: [f64; 3], maxh: Option<f64>, void: bool) -> u32 {
        self.put(solid_box(min, max), maxh, void)
    }

    /// Unions solids: retag every solid in region `from` into `into`, so the
    /// boundary between overlapping solids fuses (a boolean union, one material).
    fn merge_regions(&mut self, into: u32, from: u32) {
        self.scene
            .merge_region(rapidmesh_geom::RegionTag(into), rapidmesh_geom::RegionTag(from));
        for (r, _) in &mut self.region_maxh {
            if *r == from {
                *r = into;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (base, axis, radius, segments, maxh=None, void=false))]
    fn add_cylinder(
        &mut self,
        base: [f64; 3],
        axis: [f64; 3],
        radius: f64,
        segments: usize,
        maxh: Option<f64>,
        void: bool,
    ) -> u32 {
        self.put(cylinder(base, axis, radius, segments), maxh, void)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (center, radius, segments, rings, maxh=None, void=false))]
    fn add_sphere(
        &mut self,
        center: [f64; 3],
        radius: f64,
        segments: usize,
        rings: usize,
        maxh: Option<f64>,
        void: bool,
    ) -> u32 {
        // A sphere is faceted GEODESICALLY (icosphere) -- isotropic and pole-free,
        // unlike a UV sphere whose latitude rings cluster at the poles and seed
        // slivers there. Density follows the maxh-driven facet count (`segments`/
        // `rings` only set a floor), so the facets track the requested mesh.
        let _ = rings;
        let n = self.segs(radius, maxh, segments) as f64;
        let level = ((1.0515 * n / std::f64::consts::TAU).log2().ceil() as i64).clamp(1, 6) as usize;
        self.put(icosphere(center, radius, level), maxh, void)
    }

    /// A NACA 0012 airfoil section (chord along +x, leading edge at `origin`)
    /// extruded along `span_axis` by `span`: the curved skin is one analytic
    /// extruded-spline surface (curvature-graded), with a flat blunt trailing
    /// edge and two end caps.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (chord, span, origin=[0.0, 0.0, 0.0], span_axis=[0.0, 0.0, 1.0], n_per_side=40, n_seg=120, maxh=None, void=false))]
    fn add_naca0012(
        &mut self,
        chord: f64,
        span: f64,
        origin: [f64; 3],
        span_axis: [f64; 3],
        n_per_side: usize,
        n_seg: usize,
        maxh: Option<f64>,
        void: bool,
    ) -> u32 {
        let al = (span_axis[0].powi(2) + span_axis[1].powi(2) + span_axis[2].powi(2)).sqrt();
        let h = [span_axis[0] / al * span, span_axis[1] / al * span, span_axis[2] / al * span];
        let profile = naca0012_profile(chord, n_per_side);
        let solid = extrude_spline_profile(profile, n_seg, origin, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], h);
        self.put(solid, maxh, void)
    }

    #[pyo3(signature = (center, radius, subdivisions, maxh=None, void=false))]
    fn add_icosphere(
        &mut self,
        center: [f64; 3],
        radius: f64,
        subdivisions: usize,
        maxh: Option<f64>,
        void: bool,
    ) -> u32 {
        self.put(icosphere(center, radius, subdivisions), maxh, void)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (base, axis, radius, segments, rows, maxh=None, void=false))]
    fn add_cylinder_iso(
        &mut self,
        base: [f64; 3],
        axis: [f64; 3],
        radius: f64,
        segments: usize,
        rows: usize,
        maxh: Option<f64>,
        void: bool,
    ) -> u32 {
        self.put(cylinder_iso(base, axis, radius, segments, rows), maxh, void)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (base, axis, r_base, r_top, segments, rows, maxh=None, void=false))]
    fn add_frustum_iso(
        &mut self,
        base: [f64; 3],
        axis: [f64; 3],
        r_base: f64,
        r_top: f64,
        segments: usize,
        rows: usize,
        maxh: Option<f64>,
        void: bool,
    ) -> u32 {
        self.put(frustum_iso(base, axis, r_base, r_top, segments, rows), maxh, void)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (base, axis, r_base, r_top, segments, maxh=None, void=false))]
    fn add_frustum(
        &mut self,
        base: [f64; 3],
        axis: [f64; 3],
        r_base: f64,
        r_top: f64,
        segments: usize,
        maxh: Option<f64>,
        void: bool,
    ) -> u32 {
        self.put(frustum(base, axis, r_base, r_top, segments), maxh, void)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (outer, holes, base, u, v, h, maxh=None, void=false))]
    fn add_prism(
        &mut self,
        outer: Vec<[f64; 2]>,
        holes: Vec<Vec<[f64; 2]>>,
        base: [f64; 3],
        u: [f64; 3],
        v: [f64; 3],
        h: [f64; 3],
        maxh: Option<f64>,
        void: bool,
    ) -> u32 {
        self.put(extrude_polygon(&outer, &holes, base, u, v, h), maxh, void)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (center, axis, major_radius, minor_radius, segments_major, segments_minor, maxh=None, void=false))]
    fn add_torus(
        &mut self,
        center: [f64; 3],
        axis: [f64; 3],
        major_radius: f64,
        minor_radius: f64,
        segments_major: usize,
        segments_minor: usize,
        maxh: Option<f64>,
        void: bool,
    ) -> u32 {
        self.put(
            torus(center, axis, major_radius, minor_radius, segments_major, segments_minor),
            maxh,
            void,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (position, dx, dy, dz, top_x, maxh=None, void=false))]
    fn add_wedge(
        &mut self,
        position: [f64; 3],
        dx: f64,
        dy: f64,
        dz: f64,
        top_x: f64,
        maxh: Option<f64>,
        void: bool,
    ) -> u32 {
        self.put(wedge(position, dx, dy, dz, top_x), maxh, void)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path, radius, segments, maxh=None, void=false))]
    fn add_pipe(
        &mut self,
        path: Vec<[f64; 3]>,
        radius: f64,
        segments: usize,
        maxh: Option<f64>,
        void: bool,
    ) -> u32 {
        self.put(pipe(&path, radius, segments), maxh, void)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (base, radius, pitch, turns, wire_radius, points_per_turn, segments, maxh=None, void=false))]
    fn add_helix(
        &mut self,
        base: [f64; 3],
        radius: f64,
        pitch: f64,
        turns: f64,
        wire_radius: f64,
        points_per_turn: usize,
        segments: usize,
        maxh: Option<f64>,
        void: bool,
    ) -> u32 {
        self.put(
            helix(base, radius, pitch, turns, wire_radius, points_per_turn, segments),
            maxh,
            void,
        )
    }

    #[pyo3(signature = (profile_a, profile_b, maxh=None, void=false))]
    fn add_loft(
        &mut self,
        profile_a: Vec<[f64; 3]>,
        profile_b: Vec<[f64; 3]>,
        maxh: Option<f64>,
        void: bool,
    ) -> u32 {
        self.put(loft(&profile_a, &profile_b), maxh, void)
    }

    /// Solid from an externally supplied triangle soup (imported STL surface,
    /// marching-cubes iso-surface, ...). `tris` indexes into `verts`. The
    /// surface must be closed and non-self-intersecting; the winding is
    /// normalized to outward internally.
    #[pyo3(signature = (verts, tris, maxh=None, void=false))]
    fn add_mesh(
        &mut self,
        verts: Vec<[f64; 3]>,
        tris: Vec<[u32; 3]>,
        maxh: Option<f64>,
        void: bool,
    ) -> u32 {
        self.put(mesh_solid(&verts, &tris), maxh, void)
    }

    fn add_sheet_rect(&mut self, corner: [f64; 3], u: [f64; 3], v: [f64; 3], tag: u32) {
        self.scene.add_sheet(sheet_rect(corner, u, v), FaceTag(tag));
    }

    fn add_sheet_disk(
        &mut self,
        center: [f64; 3],
        e1: [f64; 3],
        e2: [f64; 3],
        segments: usize,
        tag: u32,
    ) {
        self.scene
            .add_sheet(sheet_disk(center, e1, e2, segments), FaceTag(tag));
    }

    fn add_sheet_polygon(
        &mut self,
        outer: Vec<[f64; 2]>,
        holes: Vec<Vec<[f64; 2]>>,
        base: [f64; 3],
        u: [f64; 3],
        v: [f64; 3],
        tag: u32,
    ) {
        self.scene
            .add_sheet(sheet_polygon(&outer, &holes, base, u, v), FaceTag(tag));
    }

    /// Runs assembly, meshing, and optimization; returns the mesh.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (maxh, radius_edge, max_points, grading, face_maxh=vec![], size_points=vec![], surface_maxh=vec![], density_weighted=false, tol_edge=1e-2, tol_surf=1e-2, maxh_edge=f64::INFINITY, maxh_surf=f64::INFINITY, maxh_vol=f64::INFINITY, edge_maxh=vec![], edge_tol=vec![], surf_maxh=vec![], surf_tol=vec![], region_over=vec![]))]
    fn mesh(
        &self,
        py: Python<'_>,
        maxh: f64,
        radius_edge: f64,
        max_points: usize,
        grading: f64,
        face_maxh: Vec<(u32, f64)>,
        size_points: Vec<([f64; 3], f64)>,
        surface_maxh: Vec<(u32, f64)>,
        density_weighted: bool,
        tol_edge: f64,
        tol_surf: f64,
        maxh_edge: f64,
        maxh_surf: f64,
        maxh_vol: f64,
        edge_maxh: Vec<(u32, f64)>,
        edge_tol: Vec<(u32, f64)>,
        surf_maxh: Vec<(u32, f64)>,
        surf_tol: Vec<(u32, f64)>,
        region_over: Vec<(u32, f64)>,
    ) -> PyMesh {
        let t0 = std::time::Instant::now();
        let timing = std::env::var_os("RAPIDMESH_TIMING").is_some();
        rapidmesh_exact::log::clear();
        // The heavy pipeline runs without the GIL.
        let (mesh, params, q) = py.allow_threads(|| {
            let ta = std::time::Instant::now();
            let plc = self.scene.assemble();
            let t_assemble = ta.elapsed();
            rapidmesh_exact::log::stage("assemble.total", t_assemble.as_secs_f64());
            // Per-region size: the solid-set defaults (g.box(maxh=)) overridden by
            // any hierarchical g.region(tag).maxh entries.
            let mut region_maxh = self.region_maxh.clone();
            for (r, h) in &region_over {
                if let Some(e) = region_maxh.iter_mut().find(|(rr, _)| rr == r) {
                    e.1 = *h;
                } else {
                    region_maxh.push((*r, *h));
                }
            }
            let params = MeshParams {
                maxh,
                region_maxh,
                radius_edge_bound: radius_edge,
                max_points,
                grading,
                face_maxh,
                surface_maxh,
                size_points,
                density_weighted,
                tol_edge,
                tol_surf,
                maxh_edge,
                maxh_surf,
                maxh_vol,
                edge_maxh,
                edge_tol,
                surf_maxh,
                surf_tol,
            };
            let tm = std::time::Instant::now();
            // Opt-in to the new boundary-constrained Stage-3 pipeline for
            // side-by-side inspection; the default stays the proven path.
            let mut mesh: TetMesh = if std::env::var_os("RAPIDMESH_CDT").is_some() {
                mesh_cdt(&plc, &params)
            } else {
                mesh_plc_with(&plc, &params)
            };
            let t_mesh = tm.elapsed();
            rapidmesh_exact::log::stage("mesh.total", t_mesh.as_secs_f64());
            let opt = OptimizeParams {
                maxh: params.maxh,
                region_maxh: params.region_maxh.clone(),
                face_maxh: params.face_maxh.clone(),
                surface_maxh: params.surface_maxh.clone(),
                ..OptimizeParams::default()
            };
            let to = std::time::Instant::now();
            if std::env::var_os("RAPIDMESH_SKIP_OPTIMIZE").is_none() {
                optimize(&mut mesh, &opt);
            }
            // Quality (with the worst element's location/region), logged so a
            // verbose run reports not just timings but where the mesh is worst.
            let q = quality_stats(&mesh);
            rapidmesh_exact::log::stat("quality.min_dihedral_deg", q.min_dihedral_deg);
            rapidmesh_exact::log::stat("quality.max_radius_edge", q.max_radius_edge);
            let lvl = if q.min_dihedral_deg < 5.0 {
                rapidmesh_exact::log::Level::Warn
            } else {
                rapidmesh_exact::log::Level::Info
            };
            rapidmesh_exact::log::event(
                lvl,
                "quality",
                format!(
                    "min dihedral {:.2} deg in region {} near ({:.4}, {:.4}, {:.4}); {} tets",
                    q.min_dihedral_deg,
                    q.worst_region,
                    q.worst_location[0],
                    q.worst_location[1],
                    q.worst_location[2],
                    q.n_tets,
                ),
            );
            if timing {
                eprintln!(
                    "stages: assemble {:?} ({} plc facets), mesh {:?}, optimize {:?}",
                    t_assemble,
                    plc.triangles.len(),
                    t_mesh,
                    to.elapsed(),
                );
            }
            (mesh, params, q)
        });
        let _ = params;
        let (timings, stats, events) = rapidmesh_exact::log::take();
        PyMesh {
            mesh,
            millis: t0.elapsed().as_millis() as u64,
            min_dihedral_deg: q.min_dihedral_deg,
            max_radius_edge: q.max_radius_edge,
            max_edge: q.max_edge,
            timings,
            stats,
            events: events
                .into_iter()
                .map(|e| (e.level.tag().to_string(), e.stage, e.message, e.at))
                .collect(),
            quality: q,
        }
    }

    /// Surface-only export: runs assembly plus the boundary-conforming surface
    /// triangulation (stages 1+2), skipping the volume mesh and optimization.
    /// Returns just the closed boundary surface mesh. Volume-only params
    /// (`radius_edge`, `max_points`) do not apply.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (maxh, grading, face_maxh=vec![], size_points=vec![], surface_maxh=vec![], tol_edge=1e-2, tol_surf=1e-2, maxh_edge=f64::INFINITY, maxh_surf=f64::INFINITY, maxh_vol=f64::INFINITY))]
    fn surface_mesh(
        &self,
        py: Python<'_>,
        maxh: f64,
        grading: f64,
        face_maxh: Vec<(u32, f64)>,
        size_points: Vec<([f64; 3], f64)>,
        surface_maxh: Vec<(u32, f64)>,
        tol_edge: f64,
        tol_surf: f64,
        maxh_edge: f64,
        maxh_surf: f64,
        maxh_vol: f64,
    ) -> PySurfaceMesh {
        let t0 = std::time::Instant::now();
        rapidmesh_exact::log::clear();
        let mesh = py.allow_threads(|| {
            let plc = self.scene.assemble();
            let params = MeshParams {
                maxh,
                region_maxh: self.region_maxh.clone(),
                radius_edge_bound: 0.0,
                max_points: usize::MAX,
                grading,
                face_maxh,
                surface_maxh,
                size_points,
                density_weighted: false,
                tol_edge,
                tol_surf,
                maxh_edge,
                maxh_surf,
                maxh_vol,
                edge_maxh: vec![],
                edge_tol: vec![],
                surf_maxh: vec![],
                surf_tol: vec![],
            };
            surface_mesh(&plc, &params)
        });
        let (timings, _stats, _events) = rapidmesh_exact::log::take();
        PySurfaceMesh {
            mesh,
            millis: t0.elapsed().as_millis() as u64,
            timings,
        }
    }

    /// Region/face/edge topology (stable ids + geometry + incidence) for the
    /// hierarchical per-entity sizing API. Assembles and builds the B-rep on
    /// demand; ids are valid for the current geometry.
    fn topology(&self) -> PyTopology {
        let plc = self.scene.assemble();
        let brep = brep_build::from_plc(&plc);
        PyTopology { topo: extract_topology(&plc, &brep) }
    }
}

/// The B-rep topology read model exposed to the Python hierarchical sizing API.
#[pyclass]
struct PyTopology {
    topo: rapidmesh_brep::Topology,
}

#[pymethods]
impl PyTopology {
    /// Meshed region tags.
    #[getter]
    fn regions(&self) -> Vec<u32> {
        self.topo.regions.clone()
    }
    /// Per face: (centroid, normal, area, region_front, region_back, tag,
    /// surface, owner, edge_ids).
    #[allow(clippy::type_complexity)]
    fn faces(&self) -> Vec<([f64; 3], [f64; 3], f64, u32, u32, u32, u32, u32, Vec<u32>)> {
        self.topo
            .faces
            .iter()
            .map(|f| (f.centroid, f.normal, f.area, f.regions[0], f.regions[1], f.face_tag, f.surface, f.owner, f.edges.clone()))
            .collect()
    }
    /// Per edge: (p0, p1, midpoint, length, kind_code, face_ids).
    fn edges(&self) -> Vec<([f64; 3], [f64; 3], [f64; 3], f64, u8, Vec<u32>)> {
        self.topo
            .edges
            .iter()
            .map(|e| (e.p0, e.p1, e.midpoint, e.length, e.kind as u8, e.faces.clone()))
            .collect()
    }
}

/// A finished tetrahedral mesh. Array properties copy into numpy on access;
/// the Python `Mesh` wrapper caches them.
#[pyclass]
struct PyMesh {
    mesh: TetMesh,
    millis: u64,
    min_dihedral_deg: f64,
    max_radius_edge: f64,
    max_edge: f64,
    /// Ordered (stage, seconds) timings collected during meshing.
    timings: Vec<(String, f64)>,
    /// Ordered (name, value) statistics collected during meshing.
    stats: Vec<(String, f64)>,
    /// Ordered (level, stage, message, seconds-since-start) log events.
    events: Vec<(String, String, String, f64)>,
    /// Quality summary with the worst element's location and per-region data.
    quality: QualityStats,
}

#[pymethods]
impl PyMesh {
    /// Vertex coordinates, shape (n_points, 3).
    fn points<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.mesh.points.len();
        let flat: Vec<f64> = self.mesh.points.iter().flatten().copied().collect();
        numpy::ndarray::Array2::from_shape_vec((n, 3), flat)
            .expect("shape")
            .into_pyarray_bound(py)
    }

    /// Tet connectivity, shape (n_tets, 4).
    fn tets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        let n = self.mesh.tets.len();
        let flat: Vec<u64> = self
            .mesh
            .tets
            .iter()
            .flatten()
            .map(|&v| v as u64)
            .collect();
        numpy::ndarray::Array2::from_shape_vec((n, 4), flat)
            .expect("shape")
            .into_pyarray_bound(py)
    }

    /// Region tag per tet, shape (n_tets,).
    fn tet_regions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let v: Vec<u32> = self.mesh.tet_regions.iter().map(|r| r.0).collect();
        v.into_pyarray_bound(py)
    }

    /// Surface face connectivity, shape (n_faces, 3).
    fn faces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        let n = self.mesh.faces.len();
        let flat: Vec<u64> = self
            .mesh
            .faces
            .iter()
            .flat_map(|f| f.tri.map(|v| v as u64))
            .collect();
        numpy::ndarray::Array2::from_shape_vec((n, 3), flat)
            .expect("shape")
            .into_pyarray_bound(py)
    }

    /// Face tag per surface face (sheet tags; 0 = untagged interface).
    fn face_tags<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let v: Vec<u32> = self.mesh.faces.iter().map(|f| f.face_tag.0).collect();
        v.into_pyarray_bound(py)
    }

    /// The two region tags adjacent to each surface face, shape (n_faces, 2).
    fn face_regions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u32>> {
        let n = self.mesh.faces.len();
        let flat: Vec<u32> = self
            .mesh
            .faces
            .iter()
            .flat_map(|f| [f.regions[0].0, f.regions[1].0])
            .collect();
        numpy::ndarray::Array2::from_shape_vec((n, 2), flat)
            .expect("shape")
            .into_pyarray_bound(py)
    }

    /// Analytic-surface id per surface face, shape (n_faces,). Faces of one
    /// input surface (a box side, a cylinder barrel, a loft flank set) share
    /// one id; together with `surface_owners` this gives B-rep-style face
    /// provenance without a B-rep.
    fn face_surfaces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let v: Vec<u32> = self.mesh.faces.iter().map(|f| f.surface).collect();
        v.into_pyarray_bound(py)
    }

    /// Owner solid index per analytic surface (scene insertion order, voids
    /// included), shape (n_surfaces,); u32::MAX marks sheet surfaces.
    fn surface_owners<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.mesh.surface_owners.clone().into_pyarray_bound(py)
    }

    /// Feature (crease) edges of the surface mesh, shape (n_edges, 2): PLC
    /// creases, patch borders and sheet rims as they exist in the final mesh.
    /// Facet seams of curved analytic surfaces are NOT feature edges.
    fn edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        let e = self.mesh.feature_edges();
        let n = e.len();
        let flat: Vec<u64> = e.iter().flat_map(|p| p.map(|v| v as u64)).collect();
        numpy::ndarray::Array2::from_shape_vec((n, 2), flat)
            .expect("shape")
            .into_pyarray_bound(py)
    }

    /// Quality and timing summary (headline numbers).
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("n_points", self.mesh.points.len())?;
        d.set_item("n_tets", self.mesh.tets.len())?;
        d.set_item("n_faces", self.mesh.faces.len())?;
        d.set_item("min_dihedral_deg", self.min_dihedral_deg)?;
        d.set_item("max_radius_edge", self.max_radius_edge)?;
        d.set_item("max_edge", self.max_edge)?;
        d.set_item("abandoned_patches", self.mesh.abandoned_patches.len())?;
        d.set_item("millis", self.millis)?;
        Ok(d)
    }

    /// Per-stage wall-clock timings in seconds, in pipeline order
    /// (assemble.* / mesh.* / optimize.*), e.g. `mesh.faces`, `mesh.refine`.
    fn timings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        for (k, v) in &self.timings {
            d.set_item(k, v)?;
        }
        Ok(d)
    }

    /// Detailed named statistics collected during meshing (counts, predicate
    /// calls, recovery work, quality), e.g. `predicates.orient3d_exact`,
    /// `recover.facets_swept`, `mesh.rounds`.
    fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        for (k, v) in &self.stats {
            d.set_item(k, v)?;
        }
        Ok(d)
    }

    /// The ordered log of events: a list of dicts
    /// `{level, stage, message, at}` (`at` = seconds since the run started,
    /// `level` in info/warn/error). Warnings flag divergence backstops, budget
    /// caps and slivers.
    fn log<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let rows: Vec<Bound<'py, PyDict>> = self
            .events
            .iter()
            .map(|(level, stage, message, at)| {
                let d = PyDict::new_bound(py);
                d.set_item("level", level).unwrap();
                d.set_item("stage", stage).unwrap();
                d.set_item("message", message).unwrap();
                d.set_item("at", at).unwrap();
                d
            })
            .collect();
        Ok(PyList::new_bound(py, rows))
    }

    /// Quality with location: min dihedral and WHERE it is (worst tet index,
    /// its centroid, its region), the radius-edge bound, and a per-region
    /// breakdown `regions = [{region, min_dihedral_deg, n_tets}, ...]`.
    fn quality<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let q = &self.quality;
        let d = PyDict::new_bound(py);
        d.set_item("n_tets", q.n_tets)?;
        d.set_item("min_dihedral_deg", q.min_dihedral_deg)?;
        d.set_item("max_radius_edge", q.max_radius_edge)?;
        d.set_item("max_edge", q.max_edge)?;
        d.set_item("worst_tet", q.worst_tet)?;
        d.set_item("worst_location", q.worst_location.to_vec())?;
        d.set_item("worst_region", q.worst_region)?;
        let regions: Vec<Bound<'py, PyDict>> = q
            .per_region
            .iter()
            .map(|&(region, min_dih, n)| {
                let r = PyDict::new_bound(py);
                r.set_item("region", region).unwrap();
                r.set_item("min_dihedral_deg", min_dih).unwrap();
                r.set_item("n_tets", n).unwrap();
                r
            })
            .collect();
        d.set_item("regions", PyList::new_bound(py, regions))?;
        Ok(d)
    }

    /// Full mesh diagnostics with LOCATED defects: quality (dihedral histogram,
    /// slivers, radius-edge), conformity (watertight, non-manifold edges, surface
    /// deviation), region volumes, and a `defects` list of
    /// `{kind, pos:[x,y,z], value}` (kind in sliver/nonmanifold_edge/straddler).
    /// The map that drives refinement and the defect overlay.
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dg = rapidmesh_tet::diagnostics::diagnose(&self.mesh);
        let d = PyDict::new_bound(py);
        d.set_item("n_tets", dg.n_tets)?;
        d.set_item("n_points", dg.n_points)?;
        d.set_item("n_faces", dg.n_faces)?;
        d.set_item("min_dihedral_deg", dg.min_dihedral_deg)?;
        d.set_item("mean_min_dihedral_deg", dg.mean_min_dihedral_deg)?;
        d.set_item("dihedral_histogram", dg.dihedral_histogram.to_vec())?;
        d.set_item("n_slivers", dg.n_slivers)?;
        d.set_item("max_radius_edge", dg.max_radius_edge)?;
        d.set_item("watertight", dg.watertight)?;
        d.set_item("n_nonmanifold_edges", dg.n_nonmanifold_edges)?;
        d.set_item("n_straddlers", dg.n_straddlers)?;
        d.set_item("max_surface_deviation", dg.max_surface_deviation)?;
        let rv: Vec<Bound<'py, PyDict>> = dg
            .region_volumes
            .iter()
            .map(|&(region, vol)| {
                let r = PyDict::new_bound(py);
                r.set_item("region", region).unwrap();
                r.set_item("volume", vol).unwrap();
                r
            })
            .collect();
        d.set_item("region_volumes", PyList::new_bound(py, rv))?;
        d.set_item("defects", defects_to_list(py, &dg.defects))?;
        Ok(d)
    }
}

/// Maps the diagnostics defect kinds to their string names for Python.
fn defect_kind_str(k: rapidmesh_tet::diagnostics::DefectKind) -> &'static str {
    use rapidmesh_tet::diagnostics::DefectKind::*;
    match k {
        Sliver => "sliver",
        NonManifoldEdge => "nonmanifold_edge",
        Straddler => "straddler",
    }
}

/// Builds the Python list of `{kind, pos, value}` defect dicts.
fn defects_to_list<'py>(
    py: Python<'py>,
    defects: &[rapidmesh_tet::diagnostics::Defect],
) -> Bound<'py, PyList> {
    let rows: Vec<Bound<'py, PyDict>> = defects
        .iter()
        .map(|f| {
            let d = PyDict::new_bound(py);
            d.set_item("kind", defect_kind_str(f.kind)).unwrap();
            d.set_item("pos", f.pos.to_vec()).unwrap();
            d.set_item("value", f.value).unwrap();
            d
        })
        .collect();
    PyList::new_bound(py, rows)
}

/// A boundary surface mesh (surface-only export): vertices plus the tagged
/// triangulation, without any volume tets.
#[pyclass]
struct PySurfaceMesh {
    mesh: SurfaceMesh,
    millis: u64,
    /// Ordered (stage, seconds) timings collected during meshing.
    timings: Vec<(String, f64)>,
}

#[pymethods]
impl PySurfaceMesh {
    /// Vertex coordinates, shape (n_points, 3).
    fn points<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.mesh.points.len();
        let flat: Vec<f64> = self.mesh.points.iter().flatten().copied().collect();
        numpy::ndarray::Array2::from_shape_vec((n, 3), flat)
            .expect("shape")
            .into_pyarray_bound(py)
    }

    /// Surface face connectivity, shape (n_faces, 3).
    fn faces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        let n = self.mesh.faces.len();
        let flat: Vec<u64> = self
            .mesh
            .faces
            .iter()
            .flat_map(|f| f.tri.map(|v| v as u64))
            .collect();
        numpy::ndarray::Array2::from_shape_vec((n, 3), flat)
            .expect("shape")
            .into_pyarray_bound(py)
    }

    /// Face tag per surface face (sheet tags; 0 = untagged interface).
    fn face_tags<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let v: Vec<u32> = self.mesh.faces.iter().map(|f| f.face_tag.0).collect();
        v.into_pyarray_bound(py)
    }

    /// The two region tags adjacent to each surface face, shape (n_faces, 2).
    fn face_regions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u32>> {
        let n = self.mesh.faces.len();
        let flat: Vec<u32> = self
            .mesh
            .faces
            .iter()
            .flat_map(|f| [f.regions[0].0, f.regions[1].0])
            .collect();
        numpy::ndarray::Array2::from_shape_vec((n, 2), flat)
            .expect("shape")
            .into_pyarray_bound(py)
    }

    /// Analytic-surface id per surface face, shape (n_faces,).
    fn face_surfaces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let v: Vec<u32> = self.mesh.faces.iter().map(|f| f.surface).collect();
        v.into_pyarray_bound(py)
    }

    /// Owner solid index per analytic surface, shape (n_surfaces,).
    fn surface_owners<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.mesh.surface_owners.clone().into_pyarray_bound(py)
    }

    /// Headline counts and wall-clock.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("n_points", self.mesh.points.len())?;
        d.set_item("n_faces", self.mesh.faces.len())?;
        d.set_item("millis", self.millis)?;
        Ok(d)
    }

    /// Per-stage wall-clock timings in seconds, in pipeline order.
    fn timings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        for (k, v) in &self.timings {
            d.set_item(k, v)?;
        }
        Ok(d)
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SceneBuilder>()?;
    m.add_class::<PyMesh>()?;
    m.add_class::<PySurfaceMesh>()?;
    m.add_class::<PyTopology>()?;
    Ok(())
}

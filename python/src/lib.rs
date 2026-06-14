//! Python bindings: a thin native core under the pure-Python builder API in
//! `python_src/rapidmesh/geometry.py`. The builder constructs a
//! [`rapidmesh_geom::Scene`] through [`SceneBuilder`]; `mesh()` runs the full
//! pipeline (exact CSG assembly, conforming tetrahedralization, quality
//! optimization) and hands the result back as numpy arrays.

use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rapidmesh_geom::{
    cylinder, extrude_polygon, frustum, helix, loft, mesh_solid, pipe, sheet_disk, sheet_polygon,
    sheet_rect, solid_box, sphere, torus, wedge, FaceTag, Scene,
};
use rapidmesh_tet::{
    mesh_plc_with, optimize, quality_stats, MeshParams, OptimizeParams, QualityStats, TetMesh,
};

/// Incremental scene builder (one solid/sheet per call); the Python layer
/// owns naming, defaults, and validation.
#[pyclass]
struct SceneBuilder {
    scene: Scene,
    region_maxh: Vec<(u32, f64)>,
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
    fn new() -> SceneBuilder {
        SceneBuilder {
            scene: Scene::new(),
            region_maxh: Vec::new(),
        }
    }

    #[pyo3(signature = (min, max, maxh=None, void=false))]
    fn add_box(&mut self, min: [f64; 3], max: [f64; 3], maxh: Option<f64>, void: bool) -> u32 {
        self.put(solid_box(min, max), maxh, void)
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
        self.put(sphere(center, radius, segments, rings), maxh, void)
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
    #[pyo3(signature = (maxh, radius_edge, max_points, grading, face_maxh=vec![], size_points=vec![], surface_maxh=vec![]))]
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
            let params = MeshParams {
                maxh,
                region_maxh: self.region_maxh.clone(),
                radius_edge_bound: radius_edge,
                max_points,
                grading,
                face_maxh,
                surface_maxh,
                size_points,
            };
            let tm = std::time::Instant::now();
            let mut mesh: TetMesh = mesh_plc_with(&plc, &params);
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
            optimize(&mut mesh, &opt);
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
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SceneBuilder>()?;
    m.add_class::<PyMesh>()?;
    Ok(())
}

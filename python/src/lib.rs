//! Python bindings: a thin native core under the pure-Python builder API in
//! `python_src/rapidmesh/geometry.py`. The builder constructs a
//! [`rapidmesh_geom::Scene`] through [`SceneBuilder`]; `mesh()` runs the full
//! pipeline (exact CSG assembly, conforming tetrahedralization, quality
//! optimization) and hands the result back as numpy arrays.

use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use rapidmesh_geom::{
    cylinder, extrude_polygon, frustum, sheet_disk, sheet_polygon, sheet_rect, solid_box, sphere,
    FaceTag, Scene,
};
use rapidmesh_tet::{
    mesh_plc_with, optimize, quality_stats, MeshParams, OptimizeParams, TetMesh,
};

/// Incremental scene builder (one solid/sheet per call); the Python layer
/// owns naming, defaults, and validation.
#[pyclass]
struct SceneBuilder {
    scene: Scene,
    region_maxh: Vec<(u32, f64)>,
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

    fn add_box(&mut self, min: [f64; 3], max: [f64; 3], maxh: Option<f64>) -> u32 {
        let r = self.scene.add_solid(solid_box(min, max));
        if let Some(h) = maxh {
            self.region_maxh.push((r.0, h));
        }
        r.0
    }

    #[allow(clippy::too_many_arguments)]
    fn add_cylinder(
        &mut self,
        base: [f64; 3],
        axis: [f64; 3],
        radius: f64,
        segments: usize,
        maxh: Option<f64>,
    ) -> u32 {
        let r = self.scene.add_solid(cylinder(base, axis, radius, segments));
        if let Some(h) = maxh {
            self.region_maxh.push((r.0, h));
        }
        r.0
    }

    fn add_sphere(
        &mut self,
        center: [f64; 3],
        radius: f64,
        segments: usize,
        rings: usize,
        maxh: Option<f64>,
    ) -> u32 {
        let r = self.scene.add_solid(sphere(center, radius, segments, rings));
        if let Some(h) = maxh {
            self.region_maxh.push((r.0, h));
        }
        r.0
    }

    #[allow(clippy::too_many_arguments)]
    fn add_frustum(
        &mut self,
        base: [f64; 3],
        axis: [f64; 3],
        r_base: f64,
        r_top: f64,
        segments: usize,
        maxh: Option<f64>,
    ) -> u32 {
        let r = self
            .scene
            .add_solid(frustum(base, axis, r_base, r_top, segments));
        if let Some(h) = maxh {
            self.region_maxh.push((r.0, h));
        }
        r.0
    }

    #[allow(clippy::too_many_arguments)]
    fn add_prism(
        &mut self,
        outer: Vec<[f64; 2]>,
        holes: Vec<Vec<[f64; 2]>>,
        base: [f64; 3],
        u: [f64; 3],
        v: [f64; 3],
        h: [f64; 3],
        maxh: Option<f64>,
    ) -> u32 {
        let r = self
            .scene
            .add_solid(extrude_polygon(&outer, &holes, base, u, v, h));
        if let Some(hh) = maxh {
            self.region_maxh.push((r.0, hh));
        }
        r.0
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
    fn mesh(
        &self,
        py: Python<'_>,
        maxh: f64,
        radius_edge: f64,
        max_points: usize,
    ) -> PyMesh {
        let t0 = std::time::Instant::now();
        // The heavy pipeline runs without the GIL.
        let (mesh, params) = py.allow_threads(|| {
            let plc = self.scene.assemble();
            let params = MeshParams {
                maxh,
                region_maxh: self.region_maxh.clone(),
                radius_edge_bound: radius_edge,
                max_points,
            };
            let mut mesh: TetMesh = mesh_plc_with(&plc, &params);
            let opt = OptimizeParams {
                maxh: params.maxh,
                region_maxh: params.region_maxh.clone(),
                ..OptimizeParams::default()
            };
            optimize(&mut mesh, &opt);
            (mesh, params)
        });
        let _ = params;
        let q = quality_stats(&mesh);
        PyMesh {
            mesh,
            millis: t0.elapsed().as_millis() as u64,
            min_dihedral_deg: q.min_dihedral_deg,
            max_radius_edge: q.max_radius_edge,
            max_edge: q.max_edge,
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

    /// Quality and timing statistics.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use pyo3::types::PyDict;
        let d = PyDict::new_bound(py);
        d.set_item("n_points", self.mesh.points.len())?;
        d.set_item("n_tets", self.mesh.tets.len())?;
        d.set_item("n_faces", self.mesh.faces.len())?;
        d.set_item("min_dihedral_deg", self.min_dihedral_deg)?;
        d.set_item("max_radius_edge", self.max_radius_edge)?;
        d.set_item("max_edge", self.max_edge)?;
        d.set_item("millis", self.millis)?;
        Ok(d)
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SceneBuilder>()?;
    m.add_class::<PyMesh>()?;
    Ok(())
}

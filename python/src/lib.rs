//! Python bindings: a thin layer over the `rapidmesh` crate. Every call maps
//! to one call of the Rust API; this file only converts arguments and turns
//! results into numpy arrays and dicts. The Python classes in
//! `python_src/rapidmesh/geometry.py` add the docstrings and the attribute
//! style on top.

// The pyo3 0.22 method macros convert every `PyResult` error into itself.
#![allow(clippy::useless_conversion)]
// House style of the workspace lints (this crate is its own workspace).
#![allow(clippy::type_complexity, clippy::too_many_arguments)]

#[cfg(feature = "mem")]
#[global_allocator]
static ALLOC: rapidmesh_exact::mem::Counting = rapidmesh_exact::mem::Counting;

// mimalloc: faster than the system allocator under the parallel stages
// and about a fifth less resident memory on large meshes.
#[cfg(not(feature = "mem"))]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::exceptions::{PyIOError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rapidmesh::shapes::{
    Cone, Cuboid, Cylinder, Helix, Icosphere, Import, Loft, Naca0012, Prism, ProfileEdge, Revolve,
    Shape, Sheet, Sphere, Sweep, Torus, Triangles, Wedge,
};
use rapidmesh::{
    EdgeCut, EdgeFilter, EdgePick, Object, SheetRef, Transform, FaceFilter, Level, Mesh2DOptions, MeshOptions, PointClass, Region2D, Scope, Solid,
    SurfaceFace, SurfaceOptions, TriTopology, NONE,
};
use std::collections::BTreeMap;

type P3 = [f64; 3];

/// Edge endpoints, the triangles beside each edge and their tags.
type Adjacency<'py> = (
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
);

fn py_err(e: rapidmesh::Error) -> PyErr {
    match e {
        rapidmesh::Error::Invalid(m) => PyValueError::new_err(m),
        rapidmesh::Error::Io(e) => PyIOError::new_err(e.to_string()),
    }
}

fn arr<'py, T: numpy::Element + Copy, const K: usize>(
    py: Python<'py>,
    rows: &[[T; K]],
) -> Bound<'py, PyArray2<T>> {
    let flat: Vec<T> = rows.iter().flatten().copied().collect();
    numpy::ndarray::Array2::from_shape_vec((rows.len(), K), flat)
        .expect("shape")
        .into_pyarray_bound(py)
}

/// A read-only numpy view of `rows` without a copy, kept alive by `owner`,
/// the Python object holding them, which never changes them.
fn view<'py, T: numpy::Element, const K: usize>(
    owner: &Bound<'py, PyAny>,
    rows: &[[T; K]],
) -> PyResult<Bound<'py, PyArray2<T>>> {
    // SAFETY: `[T; K]` rows are `K` contiguous `T`s each.
    let flat = unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const T, rows.len() * K) };
    let a = numpy::ndarray::ArrayView2::from_shape((rows.len(), K), flat).expect("shape");
    // SAFETY: `owner` holds `rows` unchanged for as long as the view lives.
    let out = unsafe { PyArray2::borrow_from_array_bound(&a, owner.clone()) };
    out.getattr("flags")?.setattr("writeable", false)?;
    Ok(out)
}

/// [`view`] of index rows: `usize` is `u64` on the 64-bit targets the
/// binding is built for.
fn view_u64<'py, const K: usize>(
    owner: &Bound<'py, PyAny>,
    rows: &[[usize; K]],
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    const { assert!(std::mem::size_of::<usize>() == std::mem::size_of::<u64>()) };
    // SAFETY: same size and alignment on these targets.
    let rows = unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const [u64; K], rows.len()) };
    view(owner, rows)
}

fn arr_u64<'py, const K: usize>(py: Python<'py>, rows: &[[usize; K]]) -> Bound<'py, PyArray2<u64>> {
    let rows: Vec<[u64; K]> = rows.iter().map(|r| r.map(|v| v as u64)).collect();
    arr(py, &rows)
}

fn arr_i64<'py, const K: usize>(py: Python<'py>, rows: &[[u32; K]]) -> Bound<'py, PyArray2<i64>> {
    let rows: Vec<[i64; K]> = rows.iter().map(|r| r.map(|v| v as i64)).collect();
    arr(py, &rows)
}

fn arr3<'py, const J: usize>(py: Python<'py>, rows: &[[[f64; 3]; J]]) -> Bound<'py, PyArray3<f64>> {
    let flat: Vec<f64> = rows.iter().flatten().flatten().copied().collect();
    numpy::ndarray::Array3::from_shape_vec((rows.len(), J, 3), flat)
        .expect("shape")
        .into_pyarray_bound(py)
}

/// `-1` for "none".
fn signed(x: u32) -> i64 {
    if x == NONE {
        -1
    } else {
        x as i64
    }
}

fn signed_vec<'py>(py: Python<'py>, v: &[u32]) -> Bound<'py, PyArray1<i64>> {
    v.iter()
        .map(|&x| signed(x))
        .collect::<Vec<_>>()
        .into_pyarray_bound(py)
}

fn pairs<'py>(py: Python<'py>, v: &[(String, f64)]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    for (k, x) in v {
        d.set_item(k, x)?;
    }
    Ok(d)
}

fn index_dict<'py, K: ToPyObject>(
    py: Python<'py>,
    m: &BTreeMap<K, Vec<u32>>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    for (k, v) in m {
        let v: Vec<i64> = v.iter().map(|&x| x as i64).collect();
        d.set_item(k, v.into_pyarray_bound(py))?;
    }
    Ok(d)
}

fn sets_dict<'py>(
    py: Python<'py>,
    s: &rapidmesh::Sets,
    cells: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    if cells {
        d.set_item("cells", index_dict(py, &s.cells)?)?;
    }
    d.set_item("faces", index_dict(py, &s.faces)?)?;
    d.set_item("edges", index_dict(py, &s.edges)?)?;
    d.set_item("patches", index_dict(py, &s.patches)?)?;
    d.set_item("curves", index_dict(py, &s.curves)?)?;
    Ok(d)
}

/// The labels of a mesh as (solids, tag_labels, face_names, edge_names).
fn labels_of<'py>(py: Python<'py>, l: &rapidmesh::Labels) -> PyResult<Bound<'py, PyDict>> {
    let solids = PyList::empty_bound(py);
    for s in &l.solids {
        let d = PyDict::new_bound(py);
        d.set_item("region", s.region)?;
        d.set_item("label", s.label.clone())?;
        solids.append(d)?;
    }
    let tags = PyDict::new_bound(py);
    for (t, n) in &l.tag_labels {
        tags.set_item(t, n)?;
    }
    let names = |list: &[(String, Vec<u32>)]| -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        for (n, ids) in list {
            d.set_item(n, ids.clone())?;
        }
        Ok(d)
    };
    let d = PyDict::new_bound(py);
    d.set_item("solids", solids)?;
    d.set_item("tag_labels", tags)?;
    d.set_item("face_names", names(&l.face_names)?)?;
    d.set_item("edge_names", names(&l.edge_names)?)?;
    Ok(d)
}

fn log_list<'py, 'a>(
    py: Python<'py>,
    log: impl IntoIterator<Item = &'a rapidmesh::Event>,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty_bound(py);
    for e in log {
        let d = PyDict::new_bound(py);
        d.set_item("level", e.level.lower())?;
        d.set_item("stage", &e.stage)?;
        d.set_item("message", &e.message)?;
        d.set_item("at", e.at)?;
        out.append(d)?;
    }
    Ok(out)
}

fn write_to(path: &str, f: impl FnOnce(&str) -> std::io::Result<()>) -> PyResult<()> {
    f(path).map_err(|e| PyIOError::new_err(format!("{path}: {e}")))
}

// ---- the arrays every surface-carrying mesh shares ----------------------------

fn tri_rows(faces: &[SurfaceFace]) -> Vec<[usize; 3]> {
    faces.iter().map(|f| f.tri).collect()
}

fn face_tags<'py>(py: Python<'py>, faces: &[SurfaceFace]) -> Bound<'py, PyArray1<u32>> {
    let v: Vec<u32> = faces.iter().map(|f| f.face_tag.0).collect();
    v.into_pyarray_bound(py)
}

fn face_regions<'py>(py: Python<'py>, faces: &[SurfaceFace]) -> Bound<'py, PyArray2<u32>> {
    let rows: Vec<[u32; 2]> = faces.iter().map(|f| f.regions.map(|r| r.0)).collect();
    arr(py, &rows)
}

fn face_surfaces<'py>(py: Python<'py>, faces: &[SurfaceFace]) -> Bound<'py, PyArray1<u32>> {
    let v: Vec<u32> = faces.iter().map(|f| f.surface).collect();
    v.into_pyarray_bound(py)
}

fn face_patches<'py>(py: Python<'py>, faces: &[SurfaceFace]) -> Bound<'py, PyArray1<u32>> {
    let v: Vec<u32> = faces.iter().map(|f| f.patch).collect();
    v.into_pyarray_bound(py)
}

/// (dimension, entity) per point: 0 vertex, 1 edge, 2 face, 3 interior.
fn point_class<'py>(py: Python<'py>, c: &[PointClass]) -> Bound<'py, PyArray2<u32>> {
    let rows: Vec<[u32; 2]> = c
        .iter()
        .map(|c| match *c {
            PointClass::Vertex(i) => [0, i],
            PointClass::Edge(i) => [1, i],
            PointClass::Face(i) => [2, i],
            PointClass::Interior => [3, u32::MAX],
        })
        .collect();
    arr(py, &rows)
}

/// Edge adjacency flattened for numpy: endpoints, the up to two triangles
/// (`-1` for a free side) and their tags (`-1` for none).
fn edge_adjacency<'py>(py: Python<'py>, topo: &TriTopology) -> Adjacency<'py> {
    let tag = |g: i64| if g == i64::MIN { -1 } else { g };
    let tris: Vec<[i64; 2]> = topo.edge_tris.iter().map(|t| t.map(signed)).collect();
    let tags: Vec<[i64; 2]> = topo.edge_tags.iter().map(|g| g.map(tag)).collect();
    (arr_i64(py, &topo.edges), arr(py, &tris), arr(py, &tags))
}

// ---- scopes ------------------------------------------------------------------

/// A selection built from the filter dicts of the Python `_Scope`.
#[pyclass(name = "Scope")]
struct PyScope {
    scope: Scope,
}

fn get<'py, T: FromPyObject<'py>>(d: &Bound<'py, PyDict>, k: &str) -> PyResult<Option<T>> {
    match d.get_item(k)? {
        Some(v) if !v.is_none() => Ok(Some(v.extract()?)),
        _ => Ok(None),
    }
}

fn only(d: &Bound<'_, PyDict>, keys: &[&str], what: &str) -> PyResult<()> {
    for k in d.keys() {
        let k: String = k.extract()?;
        if !keys.contains(&k.as_str()) {
            return Err(PyValueError::new_err(format!(
                "unknown {what} filter {k:?} (expected one of {keys:?})"
            )));
        }
    }
    Ok(())
}

fn face_filter(d: &Bound<'_, PyDict>) -> PyResult<FaceFilter> {
    only(d, &["id", "tag", "solid", "role", "normal", "normal_tol", "near"], "surf")?;
    let mut f = FaceFilter {
        id: get(d, "id")?,
        tag: get(d, "tag")?,
        solid: get(d, "solid")?,
        role: get(d, "role")?,
        normal: get(d, "normal")?,
        near: get(d, "near")?,
        ..Default::default()
    };
    if let Some(t) = get(d, "normal_tol")? {
        f.normal_tol = t;
    }
    Ok(f)
}

fn edge_filter(d: &Bound<'_, PyDict>) -> PyResult<EdgeFilter> {
    only(d, &["id", "kind", "between", "near"], "edge")?;
    Ok(EdgeFilter {
        id: get(d, "id")?,
        kind: get(d, "kind")?,
        between: get(d, "between")?,
        near: get(d, "near")?,
    })
}

#[pymethods]
impl PyScope {
    /// `level` is "region", "surf" or "edge"; every filter a dict, or
    /// `None` for unfiltered.
    #[new]
    #[pyo3(signature = (level, region=None, face=None, edge=None))]
    fn new(
        level: &str,
        region: Option<&Bound<'_, PyDict>>,
        face: Option<&Bound<'_, PyDict>>,
        edge: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyScope> {
        let level = match level {
            "region" => Level::Region,
            "surf" => Level::Surf,
            "edge" => Level::Edge,
            other => return Err(PyValueError::new_err(format!("unknown level {other:?}"))),
        };
        let region = match region {
            Some(d) => {
                only(d, &["id", "tag"], "region")?;
                get(d, "id")?.or(get(d, "tag")?)
            }
            None => None,
        };
        Ok(PyScope {
            scope: Scope {
                level,
                region,
                face: face.map(face_filter).transpose()?,
                edge: edge.map(edge_filter).transpose()?,
            },
        })
    }
}

// ---- geometry ----------------------------------------------------------------

fn solid(s: Solid) -> (u32, u32) {
    (s.region, s.index)
}

/// The geometry builder (`rapidmesh::Geometry`). A `None` argument takes
/// the default of the Rust shape.
#[pyclass(name = "Geometry")]
struct PyGeometry {
    g: rapidmesh::Geometry,
}

impl PyGeometry {
    fn put(
        &mut self,
        shape: impl Into<Shape>,
        maxh: Option<f64>,
        void: bool,
    ) -> PyResult<(u32, u32)> {
        self.g
            .add_solid(shape, maxh, void)
            .map(solid)
            .map_err(py_err)
    }
}

fn object_of((sheet, a, b): (bool, u32, u32)) -> Object {
    if sheet {
        Object::Sheet(SheetRef { index: a, tag: b })
    } else {
        Object::Solid(Solid {
            region: a,
            index: b,
        })
    }
}

fn object_to(o: Object) -> (bool, u32, u32) {
    match o {
        Object::Solid(s) => (false, s.region, s.index),
        Object::Sheet(s) => (true, s.index, s.tag),
    }
}

fn transform_of(kind: &str, a: P3, b: P3, angle: f64) -> PyResult<Transform> {
    Ok(match kind {
        "translate" => Transform::Translate(a),
        "rotate" => Transform::Rotate {
            angle,
            axis: a,
            center: b,
        },
        "mirror" => Transform::Mirror {
            normal: a,
            point: b,
        },
        "stretch" => Transform::Stretch {
            factors: a,
            center: b,
        },
        other => return Err(PyValueError::new_err(format!("unknown transform {other:?}"))),
    })
}

/// Overrides the fields of a shape the caller gave.
macro_rules! given {
    ($s:ident, $($f:ident),*) => {$(
        if let Some(v) = $f {
            $s.$f = v;
        }
    )*};
}

#[pymethods]
impl PyGeometry {
    #[new]
    #[pyo3(signature = (maxh=None, grading=None))]
    fn new(maxh: Option<f64>, grading: Option<f64>) -> PyGeometry {
        let mut g = rapidmesh::Geometry::new(maxh);
        if let Some(k) = grading {
            g.set_grading(k);
        }
        PyGeometry { g }
    }

    #[getter]
    fn get_maxh(&self) -> Option<f64> {
        self.g.maxh()
    }

    #[setter]
    fn set_maxh(&mut self, h: f64) {
        self.g.set_maxh(h);
    }

    fn set_tol(&mut self, tol: f64) {
        self.g.set_tol(tol);
    }

    #[pyo3(signature = (size, position=None, maxh=None, void=false))]
    fn add_box(
        &mut self,
        size: P3,
        position: Option<P3>,
        maxh: Option<f64>,
        void: bool,
    ) -> PyResult<(u32, u32)> {
        let mut s = Cuboid::new(size);
        given!(s, position);
        self.put(s, maxh, void)
    }

    #[pyo3(signature = (radius, height, position=None, axis=None, segments=None, uniform=false, rows=None, maxh=None, void=false))]
    fn add_cylinder(
        &mut self,
        radius: f64,
        height: f64,
        position: Option<P3>,
        axis: Option<P3>,
        segments: Option<usize>,
        uniform: bool,
        rows: Option<usize>,
        maxh: Option<f64>,
        void: bool,
    ) -> PyResult<(u32, u32)> {
        let mut s = Cylinder {
            uniform,
            rows,
            ..Cylinder::new(radius, height)
        };
        given!(s, position, axis, segments);
        self.put(s, maxh, void)
    }

    #[pyo3(signature = (radius, position=None, segments=None, maxh=None, void=false))]
    fn add_sphere(
        &mut self,
        radius: f64,
        position: Option<P3>,
        segments: Option<usize>,
        maxh: Option<f64>,
        void: bool,
    ) -> PyResult<(u32, u32)> {
        let mut s = Sphere::new(radius);
        given!(s, position, segments);
        self.put(s, maxh, void)
    }

    #[pyo3(signature = (radius, position=None, subdivisions=None, maxh=None, void=false))]
    fn add_icosphere(
        &mut self,
        radius: f64,
        position: Option<P3>,
        subdivisions: Option<usize>,
        maxh: Option<f64>,
        void: bool,
    ) -> PyResult<(u32, u32)> {
        let mut s = Icosphere::new(radius);
        given!(s, position, subdivisions);
        self.put(s, maxh, void)
    }

    #[pyo3(signature = (chord, span, position=None, span_axis=None, n_per_side=None, n_seg=None, maxh=None, void=false))]
    fn add_naca0012(
        &mut self,
        chord: f64,
        span: f64,
        position: Option<P3>,
        span_axis: Option<P3>,
        n_per_side: Option<usize>,
        n_seg: Option<usize>,
        maxh: Option<f64>,
        void: bool,
    ) -> PyResult<(u32, u32)> {
        let mut s = Naca0012::new(chord, span);
        given!(s, position, span_axis, n_per_side, n_seg);
        self.put(s, maxh, void)
    }

    #[pyo3(signature = (r1, r2, height, position=None, axis=None, segments=None, uniform=false, rows=None, maxh=None, void=false))]
    fn add_cone(
        &mut self,
        r1: f64,
        r2: f64,
        height: f64,
        position: Option<P3>,
        axis: Option<P3>,
        segments: Option<usize>,
        uniform: bool,
        rows: Option<usize>,
        maxh: Option<f64>,
        void: bool,
    ) -> PyResult<(u32, u32)> {
        let mut s = Cone {
            uniform,
            rows,
            ..Cone::new(r1, r2, height)
        };
        given!(s, position, axis, segments);
        self.put(s, maxh, void)
    }

    #[pyo3(signature = (points, height, position=None, holes=None, maxh=None, void=false))]
    fn add_prism(
        &mut self,
        points: Vec<[f64; 2]>,
        height: f64,
        position: Option<P3>,
        holes: Option<Vec<Vec<[f64; 2]>>>,
        maxh: Option<f64>,
        void: bool,
    ) -> PyResult<(u32, u32)> {
        let mut s = Prism::new(points, height);
        given!(s, position, holes);
        self.put(s, maxh, void)
    }

    #[pyo3(signature = (major_radius, minor_radius, position=None, axis=None, segments=None, tube_segments=None, maxh=None, void=false))]
    fn add_torus(
        &mut self,
        major_radius: f64,
        minor_radius: f64,
        position: Option<P3>,
        axis: Option<P3>,
        segments: Option<usize>,
        tube_segments: Option<usize>,
        maxh: Option<f64>,
        void: bool,
    ) -> PyResult<(u32, u32)> {
        let mut s = Torus::new(major_radius, minor_radius);
        given!(s, position, axis, segments, tube_segments);
        self.put(s, maxh, void)
    }

    #[pyo3(signature = (size, position=None, top_x=None, maxh=None, void=false))]
    fn add_wedge(
        &mut self,
        size: P3,
        position: Option<P3>,
        top_x: Option<f64>,
        maxh: Option<f64>,
        void: bool,
    ) -> PyResult<(u32, u32)> {
        let mut s = Wedge::new(size);
        given!(s, position, top_x);
        self.put(s, maxh, void)
    }

    #[pyo3(signature = (path, radius, segments=None, maxh=None, void=false))]
    fn add_sweep(
        &mut self,
        path: Vec<P3>,
        radius: f64,
        segments: Option<usize>,
        maxh: Option<f64>,
        void: bool,
    ) -> PyResult<(u32, u32)> {
        let mut s = Sweep::new(path, radius);
        given!(s, segments);
        self.put(s, maxh, void)
    }

    #[pyo3(signature = (radius, pitch, turns, wire_radius, position=None, points_per_turn=None, segments=None, maxh=None, void=false))]
    fn add_helix(
        &mut self,
        radius: f64,
        pitch: f64,
        turns: f64,
        wire_radius: f64,
        position: Option<P3>,
        points_per_turn: Option<usize>,
        segments: Option<usize>,
        maxh: Option<f64>,
        void: bool,
    ) -> PyResult<(u32, u32)> {
        let mut s = Helix::new(radius, pitch, turns, wire_radius);
        given!(s, position, points_per_turn, segments);
        self.put(s, maxh, void)
    }

    #[pyo3(signature = (profile_a, profile_b, maxh=None, void=false))]
    fn add_loft(
        &mut self,
        profile_a: Vec<P3>,
        profile_b: Vec<P3>,
        maxh: Option<f64>,
        void: bool,
    ) -> PyResult<(u32, u32)> {
        self.put(
            Loft {
                profile_a,
                profile_b,
            },
            maxh,
            void,
        )
    }

    /// `edges[i]` is `("line", 0, [])`, `("arc", bulge, [])` or
    /// `("spline", 0, interior points)`.
    #[pyo3(signature = (points, edges, position=None, axis=None, angle=None, segments=None, maxh=None, void=false))]
    #[allow(clippy::too_many_arguments)]
    fn add_revolve(
        &mut self,
        points: Vec<[f64; 2]>,
        edges: Vec<(String, f64, Vec<[f64; 2]>)>,
        position: Option<P3>,
        axis: Option<P3>,
        angle: Option<f64>,
        segments: Option<usize>,
        maxh: Option<f64>,
        void: bool,
    ) -> PyResult<(u32, u32)> {
        let edges = edges
            .into_iter()
            .map(|(kind, bulge, pts)| match kind.as_str() {
                "line" => Ok(ProfileEdge::Line),
                "arc" => Ok(ProfileEdge::Arc(bulge)),
                "spline" => Ok(ProfileEdge::Spline(pts)),
                other => Err(PyValueError::new_err(format!("unknown profile edge {other:?}"))),
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut s = Revolve {
            edges,
            ..Revolve::new(points)
        };
        given!(s, position, axis, angle, segments);
        self.put(s, maxh, void)
    }

    #[pyo3(signature = (verts, tris, maxh=None, void=false))]
    fn add_triangles(
        &mut self,
        verts: Vec<P3>,
        tris: Vec<[u32; 3]>,
        maxh: Option<f64>,
        void: bool,
    ) -> PyResult<(u32, u32)> {
        self.put(Triangles { verts, tris }, maxh, void)
    }

    #[pyo3(signature = (path, crease_deg=None, up=None, maxh=None, void=false))]
    fn add_import(
        &mut self,
        path: std::path::PathBuf,
        crease_deg: Option<f64>,
        up: Option<&str>,
        maxh: Option<f64>,
        void: bool,
    ) -> PyResult<(u32, u32)> {
        let mut s = Import::new(path);
        given!(s, crease_deg);
        if let Some(u) = up {
            s.up = u.parse().map_err(py_err)?;
        }
        self.put(s, maxh, void)
    }

    #[pyo3(signature = (path, maxh=None))]
    fn import_step(
        &mut self,
        path: std::path::PathBuf,
        maxh: Option<f64>,
    ) -> PyResult<Vec<(u32, u32)>> {
        self.g
            .import_step(path, maxh)
            .map(|v| v.into_iter().map(solid).collect())
            .map_err(py_err)
    }

    #[pyo3(signature = (corner, u, v, tag, maxh=None))]
    fn add_sheet_rect(
        &mut self,
        corner: P3,
        u: P3,
        v: P3,
        tag: u32,
        maxh: Option<f64>,
    ) -> PyResult<(u32, u32)> {
        self.g
            .add_sheet(&Sheet::plate(corner, u, v), tag, maxh)
            .map(|r| (r.index, r.tag))
            .map_err(py_err)
    }

    #[pyo3(signature = (radius, center, axis, tag, segments=None, maxh=None))]
    fn add_sheet_disc(
        &mut self,
        radius: f64,
        center: P3,
        axis: P3,
        tag: u32,
        segments: Option<usize>,
        maxh: Option<f64>,
    ) -> PyResult<(u32, u32)> {
        let mut sheet = Sheet::disc(radius, center, axis);
        if let (Sheet::Disc { segments: s, .. }, Some(n)) = (&mut sheet, segments) {
            *s = n;
        }
        self.g
            .add_sheet(&sheet, tag, maxh)
            .map(|r| (r.index, r.tag))
            .map_err(py_err)
    }

    #[pyo3(signature = (points, position, tag, holes=None, maxh=None))]
    fn add_sheet_polygon(
        &mut self,
        points: Vec<[f64; 2]>,
        position: P3,
        tag: u32,
        holes: Option<Vec<Vec<[f64; 2]>>>,
        maxh: Option<f64>,
    ) -> PyResult<(u32, u32)> {
        let sheet = Sheet::Polygon {
            points,
            holes: holes.unwrap_or_default(),
            position,
        };
        self.g
            .add_sheet(&sheet, tag, maxh)
            .map(|r| (r.index, r.tag))
            .map_err(py_err)
    }

    #[pyo3(signature = (ctrl, degree, tag, weights=None, knots=None, maxh=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_sheet_nurbs(
        &mut self,
        ctrl: Vec<Vec<P3>>,
        degree: [usize; 2],
        tag: u32,
        weights: Option<Vec<Vec<f64>>>,
        knots: Option<[Vec<f64>; 2]>,
        maxh: Option<f64>,
    ) -> PyResult<(u32, u32)> {
        let sheet = Sheet::nurbs(ctrl, degree, weights, knots).map_err(py_err)?;
        self.g
            .add_sheet(&sheet, tag, maxh)
            .map(|r| (r.index, r.tag))
            .map_err(py_err)
    }

    /// Chamfers or fillets (`kind`) edges of the solid (region, index) by
    /// `size`. `edges` holds picks:
    /// `("all", 0, 0, 0)`, `("of", role, 0, 0)`, `("between", a, b, 0)` or
    /// `("with", role, other solid, its role)`. Returns the origin
    /// (region, index, role) of every new face.
    #[pyo3(signature = (region, index, edges, kind, size, void=false))]
    fn cut_edges(
        &mut self,
        region: u32,
        index: u32,
        edges: Vec<(String, u32, u32, u32)>,
        kind: &str,
        size: f64,
        void: bool,
    ) -> PyResult<Vec<(u32, u32, u32)>> {
        let cut = match kind {
            "chamfer" => EdgeCut::Chamfer(size),
            "fillet" => EdgeCut::Fillet(size),
            other => return Err(PyValueError::new_err(format!("unknown edge cut {other:?}"))),
        };
        let picks = edges
            .into_iter()
            .map(|(kind, a, b, c)| match kind.as_str() {
                "all" => Ok(EdgePick::All),
                "of" => Ok(EdgePick::Of(a)),
                "between" => Ok(EdgePick::Between(a, b)),
                "with" => Ok(EdgePick::With(a, b, c)),
                other => Err(PyValueError::new_err(format!("unknown edge pick {other:?}"))),
            })
            .collect::<PyResult<Vec<_>>>()?;
        let faces = self
            .g
            .cut_edges(Solid { region, index }, &picks, cut, void)
            .map_err(py_err)?;
        Ok(faces
            .into_iter()
            .map(|(s, role)| (s.region, s.index, role))
            .collect())
    }

    /// Moves (`"translate"`, `a` the offset), turns (`"rotate"`, `angle`
    /// radians about axis `a` through `b`), mirrors (`"mirror"`, normal `a`
    /// through `b`) or stretches (`"stretch"`, factors `a` about `b`) the
    /// object `(is_sheet, first, second)`: a solid `(false, region, index)`
    /// or a sheet `(true, index, tag)`.
    fn transform(&mut self, obj: (bool, u32, u32), kind: &str, a: P3, b: P3, angle: f64) -> PyResult<()> {
        let t = transform_of(kind, a, b, angle)?;
        self.g.transform(object_of(obj), t).map_err(py_err)
    }

    /// A copy of the object, as `(is_sheet, first, second)`.
    fn copy(&mut self, obj: (bool, u32, u32)) -> PyResult<(bool, u32, u32)> {
        self.g.copy(object_of(obj)).map(object_to).map_err(py_err)
    }

    /// `count` objects: the object and copies moved by the step taken once,
    /// twice, ... from it.
    #[pyo3(signature = (obj, count, kind, a, b, angle))]
    fn array(
        &mut self,
        obj: (bool, u32, u32),
        count: u32,
        kind: &str,
        a: P3,
        b: P3,
        angle: f64,
    ) -> PyResult<Vec<(bool, u32, u32)>> {
        let t = transform_of(kind, a, b, angle)?;
        self.g
            .array(object_of(obj), count, t)
            .map(|os| os.into_iter().map(object_to).collect())
            .map_err(py_err)
    }

    /// The target (region, index) cut down to what it has in common with
    /// the tools, which are used up.
    fn intersect(&mut self, target: (u32, u32), tools: Vec<(u32, u32)>) -> PyResult<(u32, u32)> {
        let tools: Vec<Solid> = tools
            .into_iter()
            .map(|(region, index)| Solid { region, index })
            .collect();
        self.g
            .intersect(
                Solid {
                    region: target.0,
                    index: target.1,
                },
                &tools,
            )
            .map(solid)
            .map_err(py_err)
    }

    /// The solid the sheet (index, tag) sweeps along `vector`.
    #[pyo3(signature = (sheet, vector, maxh=None))]
    fn extrude(&mut self, sheet: (u32, u32), vector: P3, maxh: Option<f64>) -> PyResult<(u32, u32)> {
        self.g
            .extrude(
                SheetRef {
                    index: sheet.0,
                    tag: sheet.1,
                },
                vector,
                maxh,
            )
            .map(solid)
            .map_err(py_err)
    }

    /// Fuses the solids given as (region, index); returns the first.
    fn union(&mut self, solids: Vec<(u32, u32)>) -> PyResult<(u32, u32)> {
        let s: Vec<Solid> = solids
            .into_iter()
            .map(|(region, index)| Solid { region, index })
            .collect();
        self.g.union(&s).map(solid).map_err(py_err)
    }

    fn label_solid(&mut self, region: u32, index: u32, name: &str) {
        self.g.label_solid(Solid { region, index }, name);
    }

    /// The names of the solid's faces by role.
    fn roles(&self, region: u32, index: u32) -> Vec<String> {
        self.g.roles(Solid { region, index }).to_vec()
    }

    /// The role of the solid's face called `name`.
    fn role(&self, region: u32, index: u32, name: &str) -> PyResult<u32> {
        self.g.role(Solid { region, index }, name).map_err(py_err)
    }

    fn label_tag(&mut self, tag: u32, name: &str) {
        self.g.label_tag(tag, name);
    }

    fn refine_surface(&mut self, region: u32, index: u32, h: f64) {
        self.g.refine_surface(Solid { region, index }, h);
    }

    fn add_size_points(&mut self, points: Vec<P3>, hs: Vec<f64>) -> PyResult<()> {
        if points.len() != hs.len() {
            return Err(PyValueError::new_err(
                "per-point h must match number of points",
            ));
        }
        for (p, h) in points.into_iter().zip(hs) {
            self.g.add_size_point(p, h);
        }
        Ok(())
    }

    fn resolve(&self, scope: &PyScope) -> PyResult<Vec<u32>> {
        self.g.resolve(&scope.scope).map_err(py_err)
    }

    fn set_maxh_on(&mut self, scope: &PyScope, h: f64) -> PyResult<()> {
        self.g.set_maxh_on(&scope.scope, h).map_err(py_err)
    }

    fn set_tol_on(&mut self, scope: &PyScope, tol: f64) -> PyResult<()> {
        self.g.set_tol_on(&scope.scope, tol).map_err(py_err)
    }

    fn name(&mut self, scope: &PyScope, name: &str) -> PyResult<()> {
        self.g.name(&scope.scope, name).map_err(py_err)
    }

    #[pyo3(signature = (master, slave, shift=None))]
    fn periodic(
        &mut self,
        master: &PyScope,
        slave: &PyScope,
        shift: Option<P3>,
    ) -> PyResult<(f64, f64, f64)> {
        let t = self
            .g
            .periodic(&master.scope, &slave.scope, shift)
            .map_err(py_err)?;
        Ok((t[0], t[1], t[2]))
    }

    /// The region, face and edge topology of the current model.
    fn topology(&self) -> PyResult<PyTopology> {
        Ok(PyTopology {
            topo: self.g.topology().map_err(py_err)?,
        })
    }

    #[pyo3(signature = (maxh=None, radius_edge=None, max_points=None, grading=None, cells_across=None, tol_edge=None, tol_surf=None, maxh_edge=None, maxh_surf=None, maxh_vol=None, optimize=None, optimize_passes=None, target_elements=None, min_h_surf=None, min_h_vol=None, bottom_up=None))]
    #[allow(clippy::too_many_arguments)]
    fn mesh(
        &self,
        py: Python<'_>,
        maxh: Option<f64>,
        radius_edge: Option<f64>,
        max_points: Option<usize>,
        grading: Option<f64>,
        cells_across: Option<f64>,
        tol_edge: Option<f64>,
        tol_surf: Option<f64>,
        maxh_edge: Option<f64>,
        maxh_surf: Option<f64>,
        maxh_vol: Option<f64>,
        optimize: Option<bool>,
        optimize_passes: Option<usize>,
        target_elements: Option<usize>,
        min_h_surf: Option<f64>,
        min_h_vol: Option<f64>,
        bottom_up: Option<bool>,
    ) -> PyResult<PyMesh> {
        // What is not given takes the Rust default.
        let d = MeshOptions::default();
        let opts = MeshOptions {
            maxh,
            radius_edge: radius_edge.unwrap_or(d.radius_edge),
            max_points: max_points.unwrap_or(d.max_points),
            grading,
            cells_across,
            tol_edge,
            tol_surf,
            maxh_edge,
            maxh_surf,
            maxh_vol,
            optimize: optimize.unwrap_or(d.optimize),
            optimize_passes,
            target_elements,
            min_h_surf: min_h_surf.unwrap_or(d.min_h_surf),
            min_h_vol: min_h_vol.unwrap_or(d.min_h_vol),
            bottom_up,
        };
        let m = py.allow_threads(|| self.g.mesh(&opts)).map_err(py_err)?;
        Ok(PyMesh { m })
    }

    #[pyo3(signature = (maxh=None, grading=None, tol_edge=None, tol_surf=None, maxh_edge=None, maxh_surf=None, maxh_vol=None, target_triangles=None, bottom_up=None))]
    #[allow(clippy::too_many_arguments)]
    fn surface_mesh(
        &self,
        py: Python<'_>,
        maxh: Option<f64>,
        grading: Option<f64>,
        tol_edge: Option<f64>,
        tol_surf: Option<f64>,
        maxh_edge: Option<f64>,
        maxh_surf: Option<f64>,
        maxh_vol: Option<f64>,
        target_triangles: Option<usize>,
        bottom_up: Option<bool>,
    ) -> PyResult<PySurfaceMesh> {
        let opts = SurfaceOptions {
            maxh,
            grading,
            tol_edge,
            tol_surf,
            maxh_edge,
            maxh_surf,
            maxh_vol,
            target_triangles,
            bottom_up,
        };
        let m = py
            .allow_threads(|| self.g.surface_mesh(&opts))
            .map_err(py_err)?;
        Ok(PySurfaceMesh { m })
    }
}

/// The region, face and edge topology of a model.
#[pyclass]
struct PyTopology {
    topo: rapidmesh::Topology,
}

#[pymethods]
impl PyTopology {
    /// Meshed region tags.
    #[getter]
    fn regions(&self) -> Vec<u32> {
        self.topo.regions.clone()
    }

    /// Per region, parallel to `regions()`: its bounding box `(min, max)`.
    fn region_bbox(&self) -> Vec<(P3, P3)> {
        self.topo.region_bbox.iter().map(|b| (b[0], b[1])).collect()
    }

    /// Per face: (centroid, normal, area, region_front, region_back, tag,
    /// surface, owner, edge_ids, role, (bbox_min, bbox_max)).
    #[allow(clippy::type_complexity)]
    fn faces(&self) -> Vec<(P3, P3, f64, u32, u32, u32, u32, u32, Vec<u32>, u32, (P3, P3))> {
        self.topo
            .faces
            .iter()
            .map(|f| {
                (
                    f.centroid,
                    f.normal,
                    f.area,
                    f.regions[0],
                    f.regions[1],
                    f.face_tag,
                    f.surface,
                    f.owner,
                    f.edges.clone(),
                    f.role,
                    (f.bbox[0], f.bbox[1]),
                )
            })
            .collect()
    }

    /// Per edge: (p0, p1, midpoint, length, kind_code, face_ids,
    /// (bbox_min, bbox_max)).
    #[allow(clippy::type_complexity)]
    fn edges(&self) -> Vec<(P3, P3, P3, f64, u8, Vec<u32>, (P3, P3))> {
        self.topo
            .edges
            .iter()
            .map(|e| {
                (
                    e.p0,
                    e.p1,
                    e.midpoint,
                    e.length,
                    e.kind as u8,
                    e.faces.clone(),
                    (e.bbox[0], e.bbox[1]),
                )
            })
            .collect()
    }
}

// ---- volume mesh ---------------------------------------------------------------

/// A tetrahedral mesh (`rapidmesh::Mesh`).
#[pyclass]
struct PyMesh {
    m: rapidmesh::Mesh,
}

#[pymethods]
impl PyMesh {
    fn points<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        view(slf.as_any(), &slf.borrow().m.points)
    }

    fn tets<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyArray2<u64>>> {
        view_u64(slf.as_any(), &slf.borrow().m.tets)
    }

    fn tet_regions<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let m = slf.borrow();
        let regions = &m.m.tet_regions;
        // SAFETY: `RegionTag` is a transparent `u32`.
        let flat = unsafe { std::slice::from_raw_parts(regions.as_ptr() as *const u32, regions.len()) };
        let a = numpy::ndarray::ArrayView1::from(flat);
        // SAFETY: the mesh holds its regions unchanged while the view lives.
        let out = unsafe { PyArray1::borrow_from_array_bound(&a, slf.as_any().clone()) };
        out.getattr("flags")?.setattr("writeable", false)?;
        Ok(out)
    }

    fn faces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        arr_u64(py, &tri_rows(&self.m.faces))
    }

    fn face_tags<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        face_tags(py, &self.m.faces)
    }

    fn face_regions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u32>> {
        face_regions(py, &self.m.faces)
    }

    fn face_surfaces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        face_surfaces(py, &self.m.faces)
    }

    fn face_patches<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        face_patches(py, &self.m.faces)
    }

    fn point_class<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u32>> {
        point_class(py, &self.m.point_class)
    }

    fn surface_owners<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.m.surface_owners.clone().into_pyarray_bound(py)
    }

    /// Feature (crease) edges of the surface mesh.
    fn edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        arr_u64(py, &self.m.feature_edges())
    }

    fn periodic_points<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        arr_u64(py, &self.m.periodic_points)
    }

    fn labels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        labels_of(py, &self.m.labels)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (m, q) = (&self.m, &self.m.quality);
        let d = PyDict::new_bound(py);
        d.set_item("n_points", m.points.len())?;
        d.set_item("plc_points", m.plc_points)?;
        d.set_item("n_tets", m.tets.len())?;
        d.set_item("n_faces", m.faces.len())?;
        d.set_item("min_dihedral_deg", q.min_dihedral_deg)?;
        d.set_item("n_slivers", q.n_slivers)?;
        d.set_item("max_radius_edge", q.max_radius_edge)?;
        d.set_item("max_edge", q.max_edge)?;
        d.set_item("millis", m.run.millis)?;
        Ok(d)
    }

    fn timings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pairs(py, &self.m.run.timings)
    }

    fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pairs(py, &self.m.run.metrics)
    }

    fn log<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        log_list(py, &self.m.run.log)
    }

    fn log_text(&self) -> String {
        self.m.run.log_text()
    }

    /// The log events at warn or error level.
    fn warnings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        log_list(py, self.m.run.warnings())
    }

    fn report(&self) -> String {
        self.m.report()
    }

    fn __repr__(&self) -> String {
        self.m.to_string()
    }

    fn quality<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let q = &self.m.quality;
        let d = PyDict::new_bound(py);
        d.set_item("n_tets", q.n_tets)?;
        d.set_item("min_dihedral_deg", q.min_dihedral_deg)?;
        d.set_item("n_slivers", q.n_slivers)?;
        d.set_item("max_radius_edge", q.max_radius_edge)?;
        d.set_item("max_edge", q.max_edge)?;
        d.set_item("worst_tet", q.worst_tet)?;
        d.set_item("worst_location", q.worst_location.to_vec())?;
        d.set_item("worst_region", q.worst_region)?;
        let regions = PyList::empty_bound(py);
        for &(region, min_dih, n) in &q.per_region {
            let r = PyDict::new_bound(py);
            r.set_item("region", region)?;
            r.set_item("min_dihedral_deg", min_dih)?;
            r.set_item("n_tets", n)?;
            regions.append(r)?;
        }
        d.set_item("regions", regions)?;
        Ok(d)
    }

    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dg = py.allow_threads(|| self.m.diagnostics());
        let q = &dg.mesh;
        let d = PyDict::new_bound(py);
        d.set_item("n_tets", q.n_tets)?;
        d.set_item("n_points", q.n_points)?;
        d.set_item("n_faces", q.n_faces)?;
        d.set_item("min_dihedral_deg", q.min_dihedral_deg)?;
        d.set_item("mean_min_dihedral_deg", q.mean_min_dihedral_deg)?;
        d.set_item("dihedral_histogram", q.dihedral_histogram.to_vec())?;
        d.set_item("n_slivers", q.n_slivers)?;
        d.set_item("max_radius_edge", q.max_radius_edge)?;
        d.set_item("watertight", q.watertight)?;
        d.set_item("n_nonmanifold_edges", q.n_nonmanifold_edges)?;
        d.set_item("n_straddlers", q.n_straddlers)?;
        d.set_item("n_bridge_faces", q.n_bridge_faces)?;
        d.set_item("max_surface_deviation", q.max_surface_deviation)?;
        let rv = PyList::empty_bound(py);
        for &(region, vol) in &q.region_volumes {
            let r = PyDict::new_bound(py);
            r.set_item("region", region)?;
            r.set_item("volume", vol)?;
            rv.append(r)?;
        }
        d.set_item("region_volumes", rv)?;
        if let Some(f) = &dg.fidelity {
            let fd = PyDict::new_bound(py);
            fd.set_item("mesh_to_plc", f.mesh_to_plc)?;
            fd.set_item("plc_to_mesh", f.plc_to_mesh)?;
            fd.set_item("excess_area", f.excess_area)?;
            fd.set_item("uncovered_area", f.uncovered_area)?;
            fd.set_item("feature_dev", f.feature_dev)?;
            fd.set_item("feature_missed", f.feature_missed)?;
            fd.set_item("mislabeled_area", f.mislabeled_area)?;
            d.set_item("fidelity", fd)?;
        }
        let defects = PyList::empty_bound(py);
        for f in dg.defects() {
            let e = PyDict::new_bound(py);
            e.set_item("kind", f.kind.name())?;
            e.set_item("pos", f.pos.to_vec())?;
            e.set_item("value", f.value)?;
            defects.append(e)?;
        }
        d.set_item("defects", defects)?;
        Ok(d)
    }

    /// The solver view: topology with signs and face permutations, the
    /// classification of faces and edges, element geometry.
    fn topology<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let v = self.m.view();
        let (t, g, c) = (&v.topo, &v.geom, &v.class);
        let d = PyDict::new_bound(py);
        d.set_item("edges", arr(py, &t.edges))?;
        d.set_item("faces", arr(py, &t.faces))?;
        d.set_item("tet_edges", arr(py, &t.tet_edges))?;
        d.set_item("tet_edge_sign", arr(py, &t.tet_edge_sign))?;
        d.set_item("tet_faces", arr(py, &t.tet_faces))?;
        d.set_item("tet_face_sign", arr(py, &t.tet_face_sign))?;
        d.set_item("tet_face_perm", arr(py, &t.tet_face_perm))?;
        d.set_item("face_edges", arr(py, &t.face_edges))?;
        let face_tets: Vec<[i64; 2]> = t.face_tets.iter().map(|x| x.map(signed)).collect();
        d.set_item("face_tets", arr(py, &face_tets))?;
        d.set_item("face_patch", signed_vec(py, &c.face_patch))?;
        d.set_item("face_tag", c.face_tag.clone().into_pyarray_bound(py))?;
        d.set_item("face_regions", arr(py, &c.face_regions))?;
        d.set_item("edge_curve", signed_vec(py, &c.edge_curve))?;
        d.set_item("volume", g.volume.clone().into_pyarray_bound(py))?;
        d.set_item("grad", arr3(py, &g.grad))?;
        d.set_item("face_area", g.face_area.clone().into_pyarray_bound(py))?;
        d.set_item("face_normal", arr(py, &g.face_normal))?;
        Ok(d)
    }

    fn sets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        sets_dict(py, &self.m.sets(), true)
    }

    fn write_msh(&self, path: &str) -> PyResult<()> {
        write_to(path, |p| self.m.write_msh(p))
    }

    fn write_vtu(&self, path: &str) -> PyResult<()> {
        write_to(path, |p| self.m.write_vtu(p))
    }

    fn viewer_json(&self, py: Python<'_>, name: &str) -> String {
        py.allow_threads(|| self.m.viewer_json(name))
    }

    fn save_viewer_json(
        &self,
        name: &str,
        directory: std::path::PathBuf,
    ) -> PyResult<std::path::PathBuf> {
        self.m
            .save_viewer_json(name, &directory)
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }
}

// ---- surface mesh --------------------------------------------------------------

/// A surface mesh (`rapidmesh::SurfaceMesh`).
#[pyclass]
struct PySurfaceMesh {
    m: rapidmesh::SurfaceMesh,
}

#[pymethods]
impl PySurfaceMesh {
    fn points<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        arr(py, &self.m.points)
    }

    fn faces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        arr_u64(py, &tri_rows(&self.m.faces))
    }

    fn face_tags<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        face_tags(py, &self.m.faces)
    }

    fn face_regions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u32>> {
        face_regions(py, &self.m.faces)
    }

    fn face_surfaces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        face_surfaces(py, &self.m.faces)
    }

    fn face_patches<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        face_patches(py, &self.m.faces)
    }

    fn point_class<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u32>> {
        point_class(py, &self.m.point_class)
    }

    fn surface_owners<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.m.surface_owners.clone().into_pyarray_bound(py)
    }

    fn labels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        labels_of(py, &self.m.labels)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("n_points", self.m.points.len())?;
        d.set_item("n_faces", self.m.faces.len())?;
        d.set_item("millis", self.m.run.millis)?;
        Ok(d)
    }

    fn timings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pairs(py, &self.m.run.timings)
    }

    fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pairs(py, &self.m.run.metrics)
    }

    fn __repr__(&self) -> String {
        self.m.to_string()
    }

    fn edge_adjacency<'py>(&self, py: Python<'py>) -> Adjacency<'py> {
        edge_adjacency(py, &self.m.view().topo)
    }

    #[pyo3(signature = (connect_tags=false))]
    fn rwg_edges<'py>(&self, py: Python<'py>, connect_tags: bool) -> Bound<'py, PyArray2<i64>> {
        arr_i64(py, &self.m.rwg_edges(connect_tags))
    }

    /// The solver view: topology with edge signs and the full
    /// edge-to-triangle incidence, classification, element geometry.
    fn topology<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let v = self.m.view();
        let (t, g, c) = (&v.topo, &v.geom, &v.class);
        let d = PyDict::new_bound(py);
        d.set_item("edges", arr(py, &t.edges))?;
        d.set_item("tri_edges", arr(py, &t.tri_edges))?;
        d.set_item("tri_edge_sign", arr(py, &t.tri_edge_sign))?;
        d.set_item("tri_tags", t.tri_tags.clone().into_pyarray_bound(py))?;
        let (mut offsets, mut index) = (vec![0i64], Vec::new());
        for e in 0..t.edges.len() {
            index.extend(t.edge_tris_all.row(e).iter().map(|&x| x as i64));
            offsets.push(index.len() as i64);
        }
        d.set_item("edge_tris_offsets", offsets.into_pyarray_bound(py))?;
        d.set_item("edge_tris", index.into_pyarray_bound(py))?;
        d.set_item("tri_patch", signed_vec(py, &c.tri_patch))?;
        d.set_item("edge_curve", signed_vec(py, &c.edge_curve))?;
        d.set_item("area", g.area.clone().into_pyarray_bound(py))?;
        d.set_item("normal", arr(py, &g.normal))?;
        d.set_item("grad", arr3(py, &g.grad))?;
        Ok(d)
    }

    fn sets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        sets_dict(py, &self.m.sets(), false)
    }

    fn write_msh(&self, path: &str) -> PyResult<()> {
        write_to(path, |p| self.m.write_msh(p))
    }

    fn write_vtu(&self, path: &str) -> PyResult<()> {
        write_to(path, |p| self.m.write_vtu(p))
    }

    fn boundary_edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        arr_i64(py, &self.m.boundary_edges())
    }

    #[pyo3(signature = (axis, value, lo, hi, tol=1e-7))]
    fn edges_on_line<'py>(
        &self,
        py: Python<'py>,
        axis: usize,
        value: f64,
        lo: f64,
        hi: f64,
        tol: f64,
    ) -> Bound<'py, PyArray2<i64>> {
        arr_i64(py, &self.m.edges_on_line(axis, value, lo, hi, tol))
    }

    fn areas<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.m.view().geom.area.clone().into_pyarray_bound(py)
    }

    fn min_angles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.m.view().geom.min_angle.clone().into_pyarray_bound(py)
    }

    fn viewer_json(&self, name: &str) -> String {
        self.m.viewer_json(name)
    }

    /// Doerfler-marks by per-triangle `eta`: (marked, centroids, sizes).
    #[pyo3(signature = (eta, theta=0.5, factor=2.0, h_min=0.0))]
    fn dorfler_size_points<'py>(
        &self,
        py: Python<'py>,
        eta: Vec<f64>,
        theta: f64,
        factor: f64,
        h_min: f64,
    ) -> (
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<f64>>,
    ) {
        let (marked, cents, hs) = self.m.dorfler_size_points(&eta, theta, factor, h_min);
        let m: Vec<i64> = marked.iter().map(|&i| i as i64).collect();
        (
            m.into_pyarray_bound(py),
            arr(py, &cents),
            hs.into_pyarray_bound(py),
        )
    }
}

/// Doerfler bulk marking; returns the marked indices, ascending.
#[pyfunction]
#[pyo3(signature = (eta, theta=0.5))]
fn dorfler_mark<'py>(py: Python<'py>, eta: Vec<f64>, theta: f64) -> Bound<'py, PyArray1<i64>> {
    let m: Vec<i64> = rapidmesh::dorfler_mark(&eta, theta)
        .iter()
        .map(|&i| i as i64)
        .collect();
    m.into_pyarray_bound(py)
}

/// The level from which the meshing log prints live: "debug", "info",
/// "warn" or "error"; anything else (or `None`) silences it.
/// The volume mesh in a gmsh MSH file (4.1 or 2.2, ASCII), its physical
/// groups as the names; no remeshing.
#[pyfunction]
fn load_msh(path: &str) -> PyResult<PyMesh> {
    rapidmesh::load_msh(path)
        .map(|m| PyMesh { m })
        .map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (level=None))]
fn set_log_level(level: Option<&str>) {
    rapidmesh::set_log_level(level.and_then(rapidmesh::LogLevel::parse));
}

// ---- planar meshes -----------------------------------------------------------

/// A planar mesh (`rapidmesh::Mesh2D`).
#[pyclass]
struct PyMesh2D {
    inner: rapidmesh::Mesh2D,
    millis: u64,
}

#[pymethods]
impl PyMesh2D {
    fn points<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        arr(py, &self.inner.points)
    }

    fn tris<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        let rows: Vec<[u64; 3]> = self
            .inner
            .tris
            .iter()
            .map(|t| t.map(|v| v as u64))
            .collect();
        arr(py, &rows)
    }

    fn tri_tags<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.inner.tri_tags.clone().into_pyarray_bound(py)
    }

    #[pyo3(signature = (connect_tags=false))]
    fn rwg_edges<'py>(&self, py: Python<'py>, connect_tags: bool) -> Bound<'py, PyArray2<i64>> {
        let t = &self.inner.topo;
        arr_i64(py, &t.rwg_edges(&t.tri_tags, connect_tags))
    }

    fn boundary_edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        arr_i64(py, &self.inner.boundary_edges())
    }

    #[pyo3(signature = (axis, value, lo, hi, tol=1e-7))]
    fn edges_on_line<'py>(
        &self,
        py: Python<'py>,
        axis: usize,
        value: f64,
        lo: f64,
        hi: f64,
        tol: f64,
    ) -> Bound<'py, PyArray2<i64>> {
        arr_i64(py, &self.inner.edges_on_line(axis, value, lo, hi, tol))
    }

    fn edge_adjacency<'py>(&self, py: Python<'py>) -> Adjacency<'py> {
        edge_adjacency(py, &self.inner.topo)
    }

    fn areas<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.geom.area.clone().into_pyarray_bound(py)
    }

    fn min_angles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.geom.min_angle.clone().into_pyarray_bound(py)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("n_points", self.inner.points.len())?;
        d.set_item("n_tris", self.inner.tris.len())?;
        d.set_item("millis", self.millis)?;
        Ok(d)
    }
}

/// A region as Python hands it over: `(outer, holes, tag, constraints)`.
type Region2DTuple = (Vec<[f64; 2]>, Vec<Vec<[f64; 2]>>, i64, Vec<Vec<[f64; 2]>>);

fn regions_of(regions: Vec<Region2DTuple>) -> Vec<Region2D> {
    regions
        .into_iter()
        .map(|(outer, holes, tag, constraints)| Region2D {
            outer,
            holes,
            tag,
            constraints,
        })
        .collect()
}

fn region_of(outer: Vec<[f64; 2]>, holes: Vec<Vec<[f64; 2]>>) -> Region2D {
    Region2D {
        outer,
        holes,
        tag: 0,
        constraints: Vec::new(),
    }
}

/// The width of a region at a boundary point, along `inward`.
#[pyfunction]
fn local_width(
    outer: Vec<[f64; 2]>,
    holes: Vec<Vec<[f64; 2]>>,
    point: [f64; 2],
    inward: [f64; 2],
) -> f64 {
    region_of(outer, holes).local_width(point, inward)
}

/// Inward offsets of a region's boundary at `scales` multiples of a
/// distance: a number, or a callable of the local width.
#[pyfunction]
#[pyo3(signature = (outer, holes, pitch, scales, minh=None, grading=None))]
fn offset_chains(
    outer: Vec<[f64; 2]>,
    holes: Vec<Vec<[f64; 2]>>,
    pitch: &Bound<'_, PyAny>,
    scales: Vec<f64>,
    minh: Option<f64>,
    grading: Option<f64>,
) -> PyResult<Vec<Vec<[f64; 2]>>> {
    let opts = Mesh2DOptions {
        minh: minh.unwrap_or(Mesh2DOptions::default().minh),
        grading: grading.unwrap_or(rapidmesh::OFFSET_GRADING),
        ..Default::default()
    };
    let region = region_of(outer, holes);
    if let Ok(d) = pitch.extract::<f64>() {
        return Ok(region.offset_chains(|_w| d, &scales, &opts));
    }
    if !pitch.is_callable() {
        return Err(PyTypeError::new_err(
            "pitch must be a number or a callable taking the local width",
        ));
    }
    // The first error of the callable is carried out; an infinite pitch
    // drops the row the way an unmeasurable width does.
    let err: std::cell::RefCell<Option<PyErr>> = std::cell::RefCell::new(None);
    let chains = region.offset_chains(
        |w| match pitch.call1((w,)).and_then(|v| v.extract::<f64>()) {
            Ok(d) => d,
            Err(e) => {
                if err.borrow().is_none() {
                    *err.borrow_mut() = Some(e);
                }
                f64::INFINITY
            }
        },
        &scales,
        &opts,
    );
    match err.into_inner() {
        Some(e) => Err(e),
        None => Ok(chains),
    }
}

/// Union of 2D regions: `(outer, holes)` per shape.
#[pyfunction]
fn union_regions(regions: Vec<Region2DTuple>) -> Vec<(Vec<[f64; 2]>, Vec<Vec<[f64; 2]>>)> {
    rapidmesh::union_regions(&regions_of(regions))
}

/// Boolean overlay of two region sets by `rule`: "union", "intersect" or
/// "difference".
#[pyfunction]
fn overlay_regions(
    subject: Vec<Region2DTuple>,
    clip: Vec<Region2DTuple>,
    rule: &str,
) -> PyResult<Vec<(Vec<[f64; 2]>, Vec<Vec<[f64; 2]>>)>> {
    let r = match rule {
        "union" => rapidmesh::OverlayRule::Union,
        "intersect" => rapidmesh::OverlayRule::Intersect,
        "difference" => rapidmesh::OverlayRule::Difference,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown rule {other:?}, expected union, intersect or difference"
            )))
        }
    };
    Ok(rapidmesh::overlay_regions(
        &regions_of(subject),
        &regions_of(clip),
        r,
    ))
}

/// The 2D options, each not given at its Rust default.
#[allow(clippy::too_many_arguments)]
fn mesh2d_opts(
    min_angle_deg: Option<f64>,
    cvt_iters: Option<usize>,
    max_passes: Option<usize>,
    target_count: Option<usize>,
    minh: Option<f64>,
    maxh: Option<f64>,
    grading: Option<f64>,
    band_diagonals: Option<&str>,
    width_size: Option<f64>,
    snap: Option<f64>,
) -> PyResult<Mesh2DOptions> {
    let d = Mesh2DOptions::default();
    Ok(Mesh2DOptions {
        min_angle_deg: min_angle_deg.unwrap_or(d.min_angle_deg),
        cvt_iters: cvt_iters.unwrap_or(d.cvt_iters),
        max_passes: max_passes.unwrap_or(d.max_passes),
        target_count: target_count.unwrap_or(d.target_count),
        minh: minh.unwrap_or(d.minh),
        maxh: maxh.unwrap_or(d.maxh),
        grading: grading.unwrap_or(d.grading),
        band_diagonals: match band_diagonals {
            Some(s) => s.parse().map_err(PyValueError::new_err)?,
            None => d.band_diagonals,
        },
        width_size: width_size.unwrap_or(d.width_size),
        snap: snap.unwrap_or(d.snap),
    })
}

/// Meshes tagged 2D regions at target size `h`.
#[pyfunction]
#[pyo3(signature = (regions, h, min_angle_deg=None, cvt_iters=None, max_passes=None, target_count=None, minh=None, maxh=None, grading=None, band_diagonals=None, width_size=None, snap=None))]
fn mesh_2d(
    py: Python<'_>,
    regions: Vec<Region2DTuple>,
    h: f64,
    min_angle_deg: Option<f64>,
    cvt_iters: Option<usize>,
    max_passes: Option<usize>,
    target_count: Option<usize>,
    minh: Option<f64>,
    maxh: Option<f64>,
    grading: Option<f64>,
    band_diagonals: Option<&str>,
    width_size: Option<f64>,
    snap: Option<f64>,
) -> PyResult<PyMesh2D> {
    let t0 = std::time::Instant::now();
    let regs = regions_of(regions);
    let opts = mesh2d_opts(
        min_angle_deg,
        cvt_iters,
        max_passes,
        target_count,
        minh,
        maxh,
        grading,
        band_diagonals,
        width_size,
        snap,
    )?;
    let inner = py.allow_threads(|| rapidmesh::mesh_2d(&regs, |_p| h, &opts));
    Ok(PyMesh2D {
        inner,
        millis: t0.elapsed().as_millis() as u64,
    })
}

/// Meshes groups of tagged 2D regions (one mesh per group) under one
/// triangle budget.
#[pyfunction]
#[pyo3(signature = (groups, h, min_angle_deg=None, cvt_iters=None, max_passes=None, target_count=None, minh=None, maxh=None, grading=None, band_diagonals=None, width_size=None, snap=None))]
fn mesh_layers(
    py: Python<'_>,
    groups: Vec<Vec<Region2DTuple>>,
    h: f64,
    min_angle_deg: Option<f64>,
    cvt_iters: Option<usize>,
    max_passes: Option<usize>,
    target_count: Option<usize>,
    minh: Option<f64>,
    maxh: Option<f64>,
    grading: Option<f64>,
    band_diagonals: Option<&str>,
    width_size: Option<f64>,
    snap: Option<f64>,
) -> PyResult<Vec<PyMesh2D>> {
    let t0 = std::time::Instant::now();
    let groups: Vec<Vec<Region2D>> = groups.into_iter().map(regions_of).collect();
    let opts = mesh2d_opts(
        min_angle_deg,
        cvt_iters,
        max_passes,
        target_count,
        minh,
        maxh,
        grading,
        band_diagonals,
        width_size,
        snap,
    )?;
    let inner = py.allow_threads(|| rapidmesh::mesh_layers(&groups, |_p| h, &opts));
    let millis = t0.elapsed().as_millis() as u64;
    Ok(inner
        .into_iter()
        .map(|m| PyMesh2D { inner: m, millis })
        .collect())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGeometry>()?;
    m.add_class::<PyScope>()?;
    m.add_class::<PyTopology>()?;
    m.add_class::<PyMesh>()?;
    m.add_class::<PySurfaceMesh>()?;
    m.add_class::<PyMesh2D>()?;
    m.add_function(wrap_pyfunction!(mesh_2d, m)?)?;
    m.add_function(wrap_pyfunction!(load_msh, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_layers, m)?)?;
    m.add_function(wrap_pyfunction!(local_width, m)?)?;
    m.add_function(wrap_pyfunction!(offset_chains, m)?)?;
    m.add_function(wrap_pyfunction!(union_regions, m)?)?;
    m.add_function(wrap_pyfunction!(overlay_regions, m)?)?;
    m.add_function(wrap_pyfunction!(dorfler_mark, m)?)?;
    m.add_function(wrap_pyfunction!(set_log_level, m)?)?;
    Ok(())
}

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
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rapidmesh::shapes::{Shape, Sheet};
use rapidmesh::{
    EdgeCut, EdgePick, MeshOptions, Object, PointClass, Scope, SheetRef, Solid, SurfaceFace,
    SurfaceOptions, TriTopology, NONE,
};
use rapidmesh_exact::vector::V3;
use std::collections::BTreeMap;

/// Edge endpoints, the triangles beside each edge and their tags.
type Adjacency<'py> = (
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
);

mod errors {
    // The macro checks a pyo3 feature this crate does not declare.
    #![allow(unexpected_cfgs)]
    pyo3::create_exception!(
        rapidmesh,
        MeshError,
        pyo3::exceptions::PyValueError,
        "A geometry the mesher cannot mesh; the message says where and what to repair."
    );
}
use errors::MeshError;

fn py_err(e: rapidmesh::Error) -> PyErr {
    match e {
        rapidmesh::Error::Invalid(m) => PyValueError::new_err(m),
        rapidmesh::Error::Mesh(m) => MeshError::new_err(m),
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

fn u32_vec<'py>(py: Python<'py>, v: &[u32]) -> Bound<'py, PyArray1<i64>> {
    v.iter()
        .map(|&x| x as i64)
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

/// Each surface (the ids of `face_surfaces`) as a dict: its kind and the
/// parameters of its geometry, directions as unit vectors ("facets" for a
/// face without a carrier).
fn surfaces<'py>(
    py: Python<'py>,
    kinds: &[Option<rapidmesh::Surface>],
) -> PyResult<Bound<'py, PyList>> {
    use rapidmesh::Surface as K;
    let out = PyList::empty_bound(py);
    for k in kinds {
        let d = PyDict::new_bound(py);
        d.set_item("kind", k.as_ref().map_or("facets", K::name))?;
        let frame = k.as_ref().and_then(K::frame);
        match k {
            Some(K::Plane(f)) => {
                d.set_item("point", f.o.to_vec())?;
                d.set_item("normal", f.z.to_vec())?;
            }
            Some(K::Cylinder { radius, .. }) | Some(K::Sphere { radius, .. }) => {
                d.set_item("center", frame.map(|f| f.o.to_vec()))?;
                d.set_item("radius", *radius)?;
            }
            Some(K::Cone {
                frame: f,
                half_angle,
            }) => {
                d.set_item("apex", f.o.to_vec())?;
                d.set_item("half_angle_deg", half_angle.to_degrees())?;
            }
            Some(K::Torus {
                frame: f,
                major,
                minor,
            }) => {
                d.set_item("center", f.o.to_vec())?;
                d.set_item("major_radius", *major)?;
                d.set_item("minor_radius", *minor)?;
            }
            Some(K::Revolved { frame: f, .. }) => d.set_item("origin", f.o.to_vec())?,
            Some(K::Tube { radius, .. }) => d.set_item("radius", *radius)?,
            Some(K::Nurbs(n)) => {
                d.set_item("degree", n.degree.to_vec())?;
                d.set_item("controls", n.n.to_vec())?;
            }
            Some(K::Extruded { .. }) | Some(K::Discrete(_)) | None => {}
        }
        // A surface about an axis (and an extrusion along one) names it.
        if let (Some(f), false) = (frame, matches!(k, Some(K::Plane(_)))) {
            if !matches!(k, Some(K::Sphere { .. })) {
                d.set_item("axis", f.z.to_vec())?;
            }
        }
        out.append(d)?;
    }
    Ok(out)
}

fn face_patches<'py>(py: Python<'py>, faces: &[SurfaceFace]) -> Bound<'py, PyArray1<u32>> {
    let v: Vec<u32> = faces.iter().map(|f| f.patch).collect();
    v.into_pyarray_bound(py)
}

/// (dimension, entity) per point: 0 vertex, 1 edge, 2 face, 3 interior.
fn point_class<'py>(py: Python<'py>, c: &[PointClass]) -> Bound<'py, PyArray2<u32>> {
    let rows: Vec<[u32; 2]> = c
        .iter()
        .map(|c| {
            let (dim, id) = c.dim_id();
            [dim as u32, id]
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

/// A value by name from Python (a dict of fields or options), the ones
/// not given at their Rust defaults.
fn options<T: serde::de::DeserializeOwned>(d: &Bound<'_, PyDict>) -> PyResult<T> {
    value(d.as_any())
}

/// The element order 1 or 2.
fn order_of(order: u8) -> PyResult<rapidmesh::Order> {
    rapidmesh::Order::try_from(order).map_err(PyValueError::new_err)
}

/// A value from any Python object (a name, a list, a dict).
fn value<T: serde::de::DeserializeOwned>(v: &Bound<'_, PyAny>) -> PyResult<T> {
    pythonize::depythonize(v).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pymethods]
impl PyScope {
    /// `level` is "region", "surf" or "edge"; every filter a dict, or
    /// `None` for unfiltered.
    #[new]
    #[pyo3(signature = (level, region=None, face=None, edge=None))]
    fn new(
        level: &Bound<'_, PyAny>,
        region: Option<&Bound<'_, PyDict>>,
        face: Option<&Bound<'_, PyDict>>,
        edge: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyScope> {
        /// A region by its tag (`id` and `tag` are one here).
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Region {
            id: Option<u32>,
            tag: Option<u32>,
        }
        let region = region.map(options::<Region>).transpose()?;
        Ok(PyScope {
            scope: Scope {
                level: value(level)?,
                region: region.and_then(|r| r.id.or(r.tag)),
                face: face.map(options).transpose()?,
                edge: edge.map(options).transpose()?,
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

    /// A solid of kind `kind` (see `rapidmesh::shapes::Shape::of_kind`)
    /// from its fields by name; the fields not given take the Rust defaults.
    #[pyo3(signature = (kind, fields, maxh=None, void=false))]
    fn add_solid(
        &mut self,
        kind: &str,
        fields: &Bound<'_, PyDict>,
        maxh: Option<f64>,
        void: bool,
    ) -> PyResult<(u32, u32)> {
        let mut de = pythonize::Depythonizer::from_object(fields.as_any());
        let shape = Shape::of_kind(kind, &mut de).map_err(py_err)?;
        self.put(shape, maxh, void)
    }

    /// A sheet of kind `kind` (see `rapidmesh::shapes::Sheet::of_kind`)
    /// from its fields by name.
    #[pyo3(signature = (kind, fields, tag, maxh=None))]
    fn add_sheet(
        &mut self,
        kind: &str,
        fields: &Bound<'_, PyDict>,
        tag: u32,
        maxh: Option<f64>,
    ) -> PyResult<(u32, u32)> {
        let mut de = pythonize::Depythonizer::from_object(fields.as_any());
        let sheet = Sheet::of_kind(kind, &mut de).map_err(py_err)?;
        self.g
            .add_sheet(&sheet, tag, maxh)
            .map(|r| (r.index, r.tag))
            .map_err(py_err)
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

    /// Chamfers or fillets edges of the solid (region, index): `cut` is
    /// `{"chamfer": distance}` or `{"fillet": radius}`, `edges` a list of
    /// picks: `"all"`, `{"of": role}`, `{"between": [a, b]}` or
    /// `{"with": [role, other solid, its role]}`. Returns the origin
    /// (region, index, role) of every new face.
    #[pyo3(signature = (region, index, edges, cut, void=false))]
    fn cut_edges(
        &mut self,
        region: u32,
        index: u32,
        edges: &Bound<'_, PyAny>,
        cut: &Bound<'_, PyAny>,
        void: bool,
    ) -> PyResult<Vec<(u32, u32, u32)>> {
        let (picks, cut): (Vec<EdgePick>, EdgeCut) = (value(edges)?, value(cut)?);
        let faces = self
            .g
            .cut_edges(Solid { region, index }, &picks, cut, void)
            .map_err(py_err)?;
        Ok(faces
            .into_iter()
            .map(|(s, role)| (s.region, s.index, role))
            .collect())
    }

    /// Moves (`{"translate": offset}`), turns (`{"rotate": {"angle":
    /// radians, "axis", "center"}}`), mirrors (`{"mirror": {"normal",
    /// "point"}}`) or stretches (`{"stretch": {"factors", "center"}}`) the
    /// object `(is_sheet, first, second)`: a solid `(false, region, index)`
    /// or a sheet `(true, index, tag)`.
    fn transform(&mut self, obj: (bool, u32, u32), t: &Bound<'_, PyAny>) -> PyResult<()> {
        self.g.transform(object_of(obj), value(t)?).map_err(py_err)
    }

    /// A copy of the object, as `(is_sheet, first, second)`.
    fn copy(&mut self, obj: (bool, u32, u32)) -> PyResult<(bool, u32, u32)> {
        self.g.copy(object_of(obj)).map(object_to).map_err(py_err)
    }

    /// `count` objects: the object and copies moved by the step `t` (a
    /// transform as for `transform`) taken once, twice, ... from it.
    fn array(
        &mut self,
        obj: (bool, u32, u32),
        count: u32,
        t: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<(bool, u32, u32)>> {
        self.g
            .array(object_of(obj), count, value(t)?)
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
    fn extrude(
        &mut self,
        sheet: (u32, u32),
        vector: V3,
        maxh: Option<f64>,
    ) -> PyResult<(u32, u32)> {
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

    fn add_size_points(&mut self, points: Vec<V3>, hs: Vec<f64>) -> PyResult<()> {
        self.g.add_size_points(&points, &hs).map_err(py_err)
    }

    #[pyo3(signature = (mesh, eta, theta=None, factor=None, h_min=None))]
    fn mark_dorfler<'py>(
        &mut self,
        py: Python<'py>,
        mesh: &PySurfaceMesh,
        eta: Vec<f64>,
        theta: Option<f64>,
        factor: Option<f64>,
        h_min: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let d = rapidmesh::Dorfler::default();
        let d = rapidmesh::Dorfler {
            theta: theta.unwrap_or(d.theta),
            factor: factor.unwrap_or(d.factor),
            h_min: h_min.unwrap_or(d.h_min),
        };
        let marked = self.g.mark_dorfler(&mesh.m, &eta, &d).map_err(py_err)?;
        let m: Vec<i64> = marked.iter().map(|&i| i as i64).collect();
        Ok(m.into_pyarray_bound(py))
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
        shift: Option<V3>,
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

    /// A volume mesh; `opts` holds the `rapidmesh::MeshOptions` by name,
    /// the ones not given at their Rust defaults.
    fn mesh(&self, py: Python<'_>, opts: &Bound<'_, PyDict>) -> PyResult<PyMesh> {
        let opts: MeshOptions = options(opts)?;
        let m = py.allow_threads(|| self.g.mesh(&opts)).map_err(py_err)?;
        Ok(PyMesh { m })
    }

    /// A surface mesh; `opts` holds the `rapidmesh::SurfaceOptions` by
    /// name.
    fn surface_mesh(&self, py: Python<'_>, opts: &Bound<'_, PyDict>) -> PyResult<PySurfaceMesh> {
        let opts: SurfaceOptions = options(opts)?;
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
    fn region_bbox(&self) -> Vec<(V3, V3)> {
        self.topo.region_bbox.iter().map(|b| (b[0], b[1])).collect()
    }

    /// Per face: (centroid, normal, area, region_front, region_back, tag,
    /// surface, owner, edge_ids, role, (bbox_min, bbox_max)).
    #[allow(clippy::type_complexity)]
    fn faces(
        &self,
    ) -> Vec<(
        V3,
        V3,
        f64,
        u32,
        u32,
        u32,
        u32,
        u32,
        Vec<u32>,
        u32,
        (V3, V3),
    )> {
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

    /// Per edge: (p0, p1, midpoint, length, kind name, face_ids,
    /// (bbox_min, bbox_max)).
    #[allow(clippy::type_complexity)]
    fn edges(&self) -> Vec<(V3, V3, V3, f64, &'static str, Vec<u32>, (V3, V3))> {
        self.topo
            .edges
            .iter()
            .map(|e| {
                (
                    e.p0,
                    e.p1,
                    e.midpoint,
                    e.length,
                    e.kind.name(),
                    e.faces.clone(),
                    (e.bbox[0], e.bbox[1]),
                )
            })
            .collect()
    }
}

// ---- volume mesh ---------------------------------------------------------------

/// A tetrahedral mesh (`rapidmesh::Mesh`).
/// A mesh class with the accessors every mesh has (points, faces and their
/// tags, regions, carriers and patches, point classes, labels, the run's
/// timings and metrics, VTU output) and its own `methods`.
macro_rules! py_mesh {
    ($ty:ident { $($methods:tt)* }) => {
        #[pymethods]
        impl $ty {
            fn points<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyArray2<f64>>> {
                view(slf.as_any(), &slf.borrow().m.points)
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

            fn surfaces<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
                surfaces(py, &self.m.surfaces)
            }

            fn surface_owners<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
                self.m.surface_owners.clone().into_pyarray_bound(py)
            }

            fn labels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
                labels_of(py, &self.m.labels)
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

            $($methods)*
        }
    };
}

#[pyclass]
struct PyMesh {
    m: rapidmesh::Mesh,
}

py_mesh!(PyMesh {

    fn tets<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyArray2<u64>>> {
        view_u64(slf.as_any(), &slf.borrow().m.tets)
    }

    fn tet_regions<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let m = slf.borrow();
        let regions = &m.m.tet_regions;
        // SAFETY: `RegionTag` is a transparent `u32`.
        let flat =
            unsafe { std::slice::from_raw_parts(regions.as_ptr() as *const u32, regions.len()) };
        let a = numpy::ndarray::ArrayView1::from(flat);
        // SAFETY: the mesh holds its regions unchanged while the view lives.
        let out = unsafe { PyArray1::borrow_from_array_bound(&a, slf.as_any().clone()) };
        out.getattr("flags")?.setattr("writeable", false)?;
        Ok(out)
    }

    /// Feature (crease) edges of the surface mesh.
    fn edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        arr_u64(py, &self.m.feature_edges())
    }

    fn periodic_points<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        arr_u64(py, &self.m.periodic_points)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (m, q) = (&self.m, &self.m.quality);
        let d = PyDict::new_bound(py);
        d.set_item("n_points", m.points.len())?;
        d.set_item("plc_points", m.plc_points)?;
        d.set_item("n_tets", m.tets.len())?;
        d.set_item("n_faces", m.faces.len())?;
        d.set_item("min_dihedral_deg", q.min_dihedral_deg)?;
        d.set_item("n_slivers", q.slivers.len())?;
        d.set_item("max_radius_edge", q.max_radius_edge)?;
        d.set_item("max_edge", q.max_edge)?;
        d.set_item("millis", m.run.millis)?;
        Ok(d)
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

    fn quality<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let q = &self.m.quality;
        let d = PyDict::new_bound(py);
        d.set_item("n_tets", q.n_tets)?;
        d.set_item("min_dihedral_deg", q.min_dihedral_deg)?;
        d.set_item("n_slivers", q.slivers.len())?;
        d.set_item("max_radius_edge", q.max_radius_edge)?;
        d.set_item("max_edge", q.max_edge)?;
        d.set_item("worst_tet", q.worst_tet)?;
        d.set_item("worst_location", q.worst_location.to_vec())?;
        d.set_item("worst_region", q.worst_region)?;
        let regions = PyList::empty_bound(py);
        for rq in &q.per_region {
            let r = PyDict::new_bound(py);
            r.set_item("region", rq.region)?;
            r.set_item("min_dihedral_deg", rq.min_dihedral_deg)?;
            r.set_item("n_tets", rq.n_tets)?;
            regions.append(r)?;
        }
        d.set_item("regions", regions)?;
        Ok(d)
    }

    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dg = py.allow_threads(|| self.m.diagnostics());
        let (q, mq) = (&dg.mesh, &dg.mesh.quality);
        let d = PyDict::new_bound(py);
        d.set_item("n_tets", mq.n_tets)?;
        d.set_item("n_points", q.n_points)?;
        d.set_item("n_faces", q.n_faces)?;
        d.set_item("min_dihedral_deg", mq.min_dihedral_deg)?;
        d.set_item("mean_min_dihedral_deg", mq.mean_min_dihedral_deg)?;
        d.set_item("dihedral_histogram", mq.dihedral_histogram.to_vec())?;
        d.set_item("n_slivers", mq.slivers.len())?;
        d.set_item("max_radius_edge", mq.max_radius_edge)?;
        d.set_item("watertight", q.watertight)?;
        d.set_item("n_nonmanifold_edges", q.n_nonmanifold_edges)?;
        d.set_item("n_loose_faces", q.n_loose_faces)?;
        d.set_item("n_straddlers", q.n_straddlers)?;
        d.set_item("n_bridge_faces", q.n_bridge_faces)?;
        d.set_item("max_surface_deviation", q.max_surface_deviation)?;
        let rv = PyList::empty_bound(py);
        for rq in &mq.per_region {
            let r = PyDict::new_bound(py);
            r.set_item("region", rq.region)?;
            r.set_item("volume", rq.volume)?;
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

    #[pyo3(signature = (path, order=1))]
    fn write_msh(&self, path: &str, order: u8) -> PyResult<()> {
        let order = order_of(order)?;
        write_to(path, |p| self.m.write_msh(p, order))
    }

    #[pyo3(signature = (path, order=1))]
    fn write_vtu(&self, path: &str, order: u8) -> PyResult<()> {
        let order = order_of(order)?;
        write_to(path, |p| self.m.write_vtu(p, order))
    }

    /// The second-order mesh as arrays, and its writers.
    fn second_order<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let so = self.m.second_order();
        let d = PyDict::new_bound(py);
        d.set_item("points", arr(py, &so.points))?;
        let tets: Vec<[i64; 10]> = so.tets.iter().map(|t| t.map(|v| v as i64)).collect();
        let faces: Vec<[i64; 6]> = so.faces.iter().map(|t| t.map(|v| v as i64)).collect();
        d.set_item("tets", arr(py, &tets))?;
        d.set_item("faces", arr(py, &faces))?;
        d.set_item("volumes", so.volumes().into_pyarray_bound(py))?;
        d.set_item("curved_tets", so.curved_tets.clone().into_pyarray_bound(py))?;
        d.set_item("curved", so.curved)?;
        d.set_item("straightened", so.straightened)?;
        Ok(d)
    }

    #[pyo3(signature = (path, order=1))]
    fn write_inp(&self, path: &str, order: u8) -> PyResult<()> {
        let order = order_of(order)?;
        write_to(path, |p| self.m.write_inp(p, order))
    }

    #[pyo3(signature = (dir, polyhedral=false))]
    fn write_foam(&self, dir: &str, polyhedral: bool) -> PyResult<()> {
        self.m
            .write_foam(dir, polyhedral)
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    #[pyo3(signature = (polyhedral=false))]
    fn fvm_quality<'py>(&self, py: Python<'py>, polyhedral: bool) -> PyResult<Bound<'py, PyDict>> {
        let m = self.m.poly_mesh(polyhedral);
        let q = m.quality();
        let d = PyDict::new_bound(py);
        d.set_item("cells", m.n_cells())?;
        d.set_item("faces", m.faces.len())?;
        d.set_item(
            "non_orthogonality",
            q.non_orthogonality.into_pyarray_bound(py),
        )?;
        d.set_item("skewness", q.skewness.into_pyarray_bound(py))?;
        d.set_item("max_non_orthogonality", q.max_non_orthogonality)?;
        d.set_item("mean_non_orthogonality", q.mean_non_orthogonality)?;
        d.set_item("max_skewness", q.max_skewness)?;
        d.set_item("severely_non_orthogonal", q.severely_non_orthogonal)?;
        d.set_item("max_openness", q.max_openness)?;
        let (_, vol) = m.cells();
        d.set_item("volumes", vol.into_pyarray_bound(py))?;
        Ok(d)
    }

    #[pyo3(signature = (name, order=1))]
    fn viewer_json(&self, py: Python<'_>, name: &str, order: u8) -> PyResult<String> {
        let order = order_of(order)?;
        Ok(py.allow_threads(|| self.m.viewer_json(name, order)))
    }
});

// ---- surface mesh --------------------------------------------------------------

/// A surface mesh (`rapidmesh::SurfaceMesh`).
#[pyclass]
struct PySurfaceMesh {
    m: rapidmesh::SurfaceMesh,
}

py_mesh!(PySurfaceMesh {

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("n_points", self.m.points.len())?;
        d.set_item("n_faces", self.m.faces.len())?;
        d.set_item("millis", self.m.run.millis)?;
        Ok(d)
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
        let (offsets, index) = t.edge_tris_all.parts();
        d.set_item("edge_tris_offsets", u32_vec(py, offsets))?;
        d.set_item("edge_tris", u32_vec(py, index))?;
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
});

/// Doerfler bulk marking; returns the marked indices, ascending.
#[pyfunction]
#[pyo3(signature = (eta, theta=None))]
fn dorfler_mark<'py>(
    py: Python<'py>,
    eta: Vec<f64>,
    theta: Option<f64>,
) -> Bound<'py, PyArray1<i64>> {
    let theta = theta.unwrap_or(rapidmesh::Dorfler::default().theta);
    let m: Vec<i64> = rapidmesh::dorfler_mark(&eta, theta)
        .iter()
        .map(|&i| i as i64)
        .collect();
    m.into_pyarray_bound(py)
}

/// The volume mesh in a gmsh MSH file (4.1 or 2.2, ASCII), its physical
/// groups as the names; no remeshing.
#[pyfunction]
fn load_msh(path: &str) -> PyResult<PyMesh> {
    rapidmesh::load_msh(path)
        .map(|m| PyMesh { m })
        .map_err(py_err)
}

/// The level from which the meshing log prints live: "debug", "info",
/// "warn" or "error"; anything else (or `None`) silences it.
#[pyfunction]
#[pyo3(signature = (level=None))]
fn set_log_level(level: Option<&str>) {
    rapidmesh::set_log_level(level.and_then(rapidmesh::LogLevel::parse));
}

/// The union of planar polygons, each `(outer, holes)`, into connected shapes
/// `(outer, holes)` (outer counter-clockwise, holes clockwise).
#[pyfunction]
fn polygon_union(
    polygons: Vec<(Vec<[f64; 2]>, Vec<Vec<[f64; 2]>>)>,
) -> Vec<(Vec<[f64; 2]>, Vec<Vec<[f64; 2]>>)> {
    rapidmesh::polygon_union(&polygons)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("MeshError", m.py().get_type_bound::<MeshError>())?;
    m.add_class::<PyGeometry>()?;
    m.add_class::<PyScope>()?;
    m.add_class::<PyTopology>()?;
    m.add_class::<PyMesh>()?;
    m.add_class::<PySurfaceMesh>()?;
    m.add_function(wrap_pyfunction!(load_msh, m)?)?;
    m.add_function(wrap_pyfunction!(polygon_union, m)?)?;
    m.add_function(wrap_pyfunction!(dorfler_mark, m)?)?;
    m.add_function(wrap_pyfunction!(set_log_level, m)?)?;
    Ok(())
}

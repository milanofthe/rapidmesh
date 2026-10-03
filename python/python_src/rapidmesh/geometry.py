"""Geometry builder.

Build a scene from primitives, then call :meth:`Geometry.mesh` to get a
conforming, region-tagged tetrahedral mesh:

.. code-block:: python

    import rapidmesh as rm

    g = rm.Geometry(maxh=0.9)
    air = g.box(4, 4, 4)
    diel = g.box(2, 2, 1, position=(1, 1, 1), maxh=0.45)
    g.xy_plate(1, 1, position=(1.5, 1.5, 2.0), tag=7)

    mesh = g.mesh()
    print(mesh.stats)

Solids overlap by priority: a solid added later carves its region out of
earlier ones (the dielectric above displaces the air it sits in). Sheets are
zero-thickness faces embedded conformally into the volume mesh, carrying an
integer ``tag`` for downstream boundary conditions (PEC traces, ports).
All coordinates are unitless; use one consistent unit (metres, say).

Everything here is a thin layer over the Rust crate ``rapidmesh``: the
builder, the sizing hierarchy, the names and the meshes are the Rust ones,
this module adds the Python conventions (numpy arrays, keyword arguments,
docstrings). A ``None`` argument takes the default of the Rust shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from . import _native


@dataclass(frozen=True)
class Solid:
    """Handle to a solid added to a :class:`Geometry`: its ``region`` tag
    identifies the solid's tets in :attr:`Mesh.tet_regions`, its ``index``
    (insertion order, voids included) identifies the solid's surfaces in
    :attr:`Mesh.surface_owners`. Voids share ``region`` 0 but keep a unique
    ``index``, so their walls stay addressable. ``roles`` names the solid's
    surfaces in their order (empty where they have no names): a face by its
    origin, ``g.surf(solid=s, role="+z")``, which later changes of the
    geometry keep."""

    region: int
    index: int
    roles: tuple = field(default=(), compare=False)



@dataclass(frozen=True)
class Spline:
    """A revolve profile edge: a cubic spline from the vertex before it
    through ``points`` (at least two) to the vertex after it."""
    points: tuple

    def __init__(self, points):
        object.__setattr__(self, "points", tuple((float(r), float(z)) for r, z in points))


def _show(viewer_dict: dict, name: str, **kw) -> None:
    """Opens a viewer dict in the interactive viewer and blocks until the
    window is closed."""
    import os
    import tempfile
    from contextlib import suppress

    from . import _viewer

    fd, path = tempfile.mkstemp(suffix=".json", prefix="rapidmesh_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(viewer_dict, f)
        _viewer.inspect(path, title=f"rapidmesh - {name}", **kw)
    finally:
        with suppress(OSError):
            os.unlink(path)


class _MeshBase:
    """What every mesh has: the points and faces with their tags, regions,
    carriers and patches, the point classes, the labels it carries from its
    geometry and the run's statistics."""

    def _read(self, native) -> None:
        self._native = native
        self.points: np.ndarray = native.points()
        self.faces: np.ndarray = native.faces()
        self.face_tags: np.ndarray = native.face_tags()
        self.face_regions: np.ndarray = native.face_regions()
        self.face_surfaces: np.ndarray = native.face_surfaces()
        self.face_patches: np.ndarray = native.face_patches()
        self.point_class: np.ndarray = native.point_class()
        self.surface_owners: np.ndarray = native.surface_owners()
        self.stats: dict = native.stats()
        #: per-stage wall-clock seconds, pipeline order
        self.timings: dict = native.timings()
        #: named statistics of the run
        self.metrics: dict = native.metrics()
        labels = native.labels()
        #: per input solid (insertion order): {"region": int, "label": str|None}
        self.solids: list[dict] = labels["solids"]
        #: display label per sheet tag
        self.tag_labels: dict[int, str] = labels["tag_labels"]
        #: named geometric faces and edges (``g.surf(..).name = ...``): name
        #: -> entity ids
        self.face_names: dict[str, list[int]] = labels["face_names"]
        self.edge_names: dict[str, list[int]] = labels["edge_names"]


class Mesh(_MeshBase):
    """A finished tetrahedral mesh (numpy views over the native result).

    Attributes
    ----------
    points : (n_points, 3) float64
        vertex coordinates
    tets : (n_tets, 4) uint64
        tetrahedra as point indices, positively oriented in the orient3d
        sense (the fourth point below the counterclockwise first three);
        the .msh and .vtu files write them in gmsh's and VTK's opposite
        convention
    tet_regions : (n_tets,) uint32
        region tag per tet (the :attr:`Solid.region` of the owning solid)
    faces : (n_faces, 3) uint64
        surface faces (region interfaces, outer boundary, embedded sheets)
    face_tags : (n_faces,) uint32
        sheet tag per face (0 for untagged interfaces)
    face_regions : (n_faces, 2) uint32
        the regions on the two sides of each face (0 = outside)
    face_surfaces : (n_faces,) uint32
        analytic-surface id per face: faces of one input surface (a box
        side, a cylinder barrel) share one id
    face_patches : (n_faces,) uint32
        geometric (B-rep) face id per face
    point_class : (n_points, 2) uint32
        what each point lies on, as (dimension, entity): 0 a geometric
        vertex, 1 a geometric edge, 2 a geometric face (ids as in
        ``face_patches``), 3 the interior of a region (entity max uint32)
    periodic_points : (n, 2) uint64
        every point on a master face of a periodic pair with its image
    surface_owners : (n_surfaces,) uint32
        owner solid per surface id (:attr:`Solid.index`); the max uint32
        marks embedded-sheet surfaces
    edges : (n_edges, 2) uint64
        feature (crease) edges of the surface mesh; facet seams of curved
        analytic surfaces are not included
    stats : dict
        n_points, plc_points, n_tets, n_faces, min_dihedral_deg, n_slivers,
        max_radius_edge, max_edge, millis
    timings : dict
        seconds per stage, pipeline order (assemble.* / mesh.*)
    metrics : dict
        named statistics (predicate calls, counts, quality)
    log : list[dict]
        the events of the run, each {level, stage, message, at}
    quality : dict
        min_dihedral_deg, worst_location, worst_region, max_radius_edge,
        max_edge and a per-region breakdown under "regions"
    """

    def __init__(self, native) -> None:
        self._read(native)
        self.tets: np.ndarray = native.tets()
        self.tet_regions: np.ndarray = native.tet_regions()
        self.periodic_points: np.ndarray = native.periodic_points()
        self.edges: np.ndarray = native.edges()
        self.log: list[dict] = native.log()
        self.quality: dict = native.quality()

    @cached_property
    def topology(self) -> dict:
        """The solver view of the mesh, computed on first access: a dict of
        numpy arrays. Local orders and signs follow
        ``rapidmesh_topo::convention`` (local edges (01, 02, 03, 12, 13, 23),
        local face ``k`` opposite vertex ``k`` in outward order); -1 marks
        "none".

        - ``edges`` (E, 2), ``faces`` (F, 3): unique edges and faces, vertex
          ids ascending
        - ``tet_edges``, ``tet_edge_sign`` (T, 6): global edge per local
          edge; +1 where the local edge runs from the lower to the higher id
        - ``tet_faces``, ``tet_face_sign``, ``tet_face_perm`` (T, 4): global
          face per local face, parity and permutation from the ascending to
          the local outward order (``local[i] == face[FACE_PERMS[p][i]]``,
          ``FACE_PERMS = ((0,1,2), (1,2,0), (2,0,1), (0,2,1), (2,1,0),
          (1,0,2))``, ``p < 3`` exactly when the sign is +1)
        - ``face_edges`` (F, 3): edges of each face (01, 12, 20 of the
          ascending face); ``face_tets`` (F, 2): the tets beside each face,
          -1 on the boundary
        - ``face_patch`` (F,): the geometric (B-rep) face each face lies on,
          -1 inside a region; ``face_tag`` (F,): its face tag (0 none);
          ``face_regions`` (F, 2): the regions of ``face_tets`` (0 outside)
        - ``edge_curve`` (E,): the geometric (B-rep) edge each edge lies on,
          -1 elsewhere
        - ``volume`` (T,), ``grad`` (T, 4, 3): tet volumes and barycentric
          gradients; ``face_area`` (F,), ``face_normal`` (F, 3) (unit, by
          the ascending order)
        """
        return self._native.topology()

    def sets(self) -> dict:
        """Named entity sets for boundary conditions, ports and materials,
        by the labels given to solids and face tags:

        - ``cells[name]``: tet indices of a labelled solid group (unlabelled
          regions are ``region_<r>``)
        - ``faces[name]``: topology face indices (see :attr:`topology`) of
          a labelled face tag or of named geometric faces
          (``g.surf(..).name``); ``faces["boundary"]``: every face with one
          tet
        - ``edges[name]``: topology edge indices on named geometric edges
          (``g.edge(..).name``)
        - ``patches[p]``: topology faces on geometric face ``p``;
          ``curves[e]``: topology edges on geometric edge ``e``
        """
        return self._native.sets()

    def write_msh(self, path: str | Path, order: int = 1) -> Path:
        """Writes the mesh as a gmsh MSH 4.1 file: geometric vertices, edges
        and faces become point, curve and surface entities (tag = id + 1),
        regions volume entities; every node sits in the block of what it is
        classified on. Physical groups: the labelled solid groups (volumes),
        the named face tags and the named geometric faces and edges.
        ``order=2`` writes the second-order mesh (see :meth:`second_order`):
        lines with three nodes, triangles with six, tets with ten."""
        self._native.write_msh(str(path), order)
        return Path(path)

    def write_vtu(self, path: str | Path) -> Path:
        """Writes the mesh as a VTK XML unstructured grid (``.vtu``): the
        tets and the geometric faces, with cell data ``region``, ``patch``
        and ``face_tag``."""
        self._native.write_vtu(str(path))
        return Path(path)

    def second_order(self) -> dict:
        """The second-order mesh: a node in the middle of every edge, on the
        true geometry where the edge lies on a curved surface or on a curve
        (a rim between flat faces too, such as the edge of a disc sheet), in
        the middle of the edge elsewhere. Meshed with ``order=2`` (see
        :meth:`Geometry.mesh`) the sizes are made for it.

        Returns ``points`` (the corners, then the mid-edge
        nodes), ``tets`` (n, 10) in the node order of Abaqus C3D10 and VTK's
        quadratic tetra, the surface ``faces`` (m, 6), the exact ``volumes``,
        per tet whether it is curved (``curved_tets``: a mid-edge node off
        its chord; every other tet has its mid-edge nodes in the middle of
        its edges, an affine map like a linear tet, so a solver can map only
        the curved ones isoparametrically), and how many mid-edge nodes went
        onto a curved surface (``curved``) and back on their chord to keep a
        tet valid (``straightened``)."""
        return self._native.second_order()

    def write_inp(self, path: str | Path, order: int = 1) -> Path:
        """Writes a CalculiX / Abaqus input file: C3D4 (``order=1``) or C3D10
        with the mid-edge nodes on the true geometry (``order=2``); the
        region groups as element sets, every named face and sheet tag as a
        node set and, on the boundary, an element-face surface."""
        self._native.write_inp(str(path), order)
        return Path(path)

    def write_vtu_second_order(self, path: str | Path) -> Path:
        """Writes the second-order mesh as a VTK XML unstructured grid
        (quadratic tetra) with cell data ``region``."""
        self._native.write_vtu_second_order(str(path))
        return Path(path)

    def write_foam(self, case: str | Path, polyhedral: bool = False) -> Path:
        """Writes the mesh as an OpenFOAM ``polyMesh`` into
        ``<case>/constant/polyMesh``: every tet a cell, or with
        ``polyhedral`` the median dual (a polyhedral cell per vertex and
        region group, a third to a quarter as many cells). The boundary patches are
        the named faces, then the named sheet tags, the rest ``boundary``;
        the region groups are cell zones, and with tets named faces inside
        the mesh (sheets, interfaces) face zones. Returns the ``polyMesh``
        directory."""
        out = Path(case) / "constant" / "polyMesh"
        self._native.write_foam(str(out), polyhedral)
        return out

    def fvm_quality(self, polyhedral: bool = False) -> dict:
        """The finite volume quality of the cells :meth:`write_foam` writes,
        as OpenFOAM's ``checkMesh`` measures it: per face
        ``non_orthogonality`` (degrees) and ``skewness`` (0 on the boundary),
        with ``max_non_orthogonality``, ``mean_non_orthogonality``,
        ``max_skewness``, ``severely_non_orthogonal`` (faces above 70
        degrees), and the ``cells``, ``faces`` and cell ``volumes``."""
        return self._native.fvm_quality(polyhedral)

    @cached_property
    def diagnostics(self) -> dict:
        """Full diagnostics with located defects, computed on first access:
        dihedral histogram, slivers, watertight, non-manifold edges, surface
        deviation, region volumes, "fidelity" to the input (shares of
        uncovered and excess surface, missed creases, mislabelled faces) and a
        "defects" list of {kind, pos:[x,y,z], value}."""
        return self._native.diagnostics()

    def __repr__(self) -> str:
        return repr(self._native)

    @property
    def warnings(self) -> list[dict]:
        """Log events at warn/error level (divergence backstops, budget caps,
        slivers)."""
        return self._native.warnings()

    def log_text(self) -> str:
        """The log as a human-readable string, one event per line."""
        return self._native.log_text()

    def report(self) -> str:
        """A full human-readable report: per-stage timings, the
        worst-quality location and per-region quality, and any warnings."""
        return self._native.report()

    def to_viewer_dict(self, name: str, *, second_order: bool = False) -> dict:
        """The mesh in the viewer JSON schema (shared by the comparison
        viewer and the showcase site), with the located defects; with
        ``second_order`` the mid-edge nodes off their chords too, so the
        viewer draws the curved tets curved (see :meth:`second_order`)."""
        return json.loads(self._native.viewer_json(name, second_order))

    def show(self, name: str = "mesh", *, clip: float | None = 0.6,
             clip_axis: int = 1, second_order: bool = False, **kw) -> None:
        """Open this mesh in the interactive viewer and block until the window is
        closed: orbit / zoom / pan, the region legend, the crinkle clip (``clip``
        is the fraction along ``clip_axis``; ``None`` disables it), the
        located-defect overlay, and figure export. Uses a native window
        (``pywebview``) if installed, else a Chromium window. The viewer ships in
        the wheel; a window backend does not -- ``pip install pywebview`` (or
        ``playwright``). With ``second_order`` the curved tets of the
        second-order mesh are drawn curved, their faces shaded smoothly."""
        _show(self.to_viewer_dict(name, second_order=second_order), name, clip=clip,
              clip_axis=clip_axis, **kw)


class SurfaceMesh(_MeshBase):
    """A boundary surface mesh (surface-only export): the conforming surface
    triangulation without any volume tets.

    Attributes
    ----------
    points : (n_points, 3) float64
        vertex coordinates
    faces : (n_faces, 3) uint64
        surface faces (region interfaces, outer boundary, embedded sheets)
    face_tags : (n_faces,) uint32
        sheet tag per face (0 for untagged interfaces)
    face_regions : (n_faces, 2) uint32
        the regions on the two sides of each face (0 = outside)
    face_surfaces : (n_faces,) uint32
        analytic-surface id per face
    face_patches : (n_faces,) uint32
        geometric (B-rep) face id per face
    point_class : (n_points, 2) uint32
        what each point lies on (see :attr:`Mesh.point_class`)
    surface_owners : (n_surfaces,) uint32
        owner solid per surface id; the max uint32 marks embedded sheets
    stats : dict
        n_points, n_faces, millis
    metrics : dict
        named statistics of the run
    """

    def __init__(self, native) -> None:
        self._read(native)

    def __repr__(self) -> str:
        return repr(self._native)

    # ---- solver mesh info --------------------------------------------------

    def edge_adjacency(self):
        """Undirected edge -> incident triangles. Returns ``(edges, faces, tags)``,
        each ``(E, 2)`` int64: edge endpoints (``v0 < v1``); the up-to-two incident
        triangle indices (``-1`` for a free side); their tags (``-1`` if none)."""
        return self._native.edge_adjacency()

    def rwg_edges(self, connect_tags: bool = False):
        """MoM RWG basis functions = the surface degrees of freedom, ``(D, 4)``
        int64 ``[v0, v1, tri_plus, tri_minus]`` (current flows + -> -). An
        edge with ``k`` triangles of one conductor tag carries ``k - 1``
        functions, from its lowest triangle to each other one (the junction
        basis where sheets meet in a T); a manifold edge carries one. With
        ``connect_tags`` triangles of different tags at an edge are joined
        too (conductors of several tags touching there)."""
        return self._native.rwg_edges(connect_tags)

    @cached_property
    def topology(self) -> dict:
        """The solver view of the surface mesh, computed on first access: a
        dict of numpy arrays; -1 marks "none".

        - ``edges`` (E, 2): unique edges, vertex ids ascending;
          ``tri_edges``, ``tri_edge_sign`` (T, 3): global edge per local edge
          (01, 12, 20), +1 where it runs from the lower to the higher id
        - ``edge_tris_offsets`` (E+1,), ``edge_tris``: every triangle at
          every edge (CSR: the triangles of edge ``e`` are
          ``edge_tris[offsets[e]:offsets[e+1]]``, three or more at a
          junction); ``tri_tags`` (T,)
        - ``tri_patch`` (T,): the geometric (B-rep) face of each triangle;
          ``edge_curve`` (E,): the geometric (B-rep) edge of each edge, -1
          elsewhere
        - ``area`` (T,), ``normal`` (T, 3), ``grad`` (T, 3, 3): barycentric
          gradients, tangent to the triangle
        """
        return self._native.topology()

    def sets(self) -> dict:
        """Named entity sets: ``faces[name]`` triangle indices of a named face
        tag or of named geometric faces, ``edges[name]`` topology edges on
        named geometric edges, ``patches[p]`` triangles on geometric face ``p``,
        ``curves[e]`` topology edges on geometric edge ``e`` (ports along a
        geometric edge, for example)."""
        return self._native.sets()

    def write_msh(self, path: str | Path) -> Path:
        """Writes the surface mesh as a gmsh MSH 4.1 file (see
        :meth:`Mesh.write_msh`)."""
        self._native.write_msh(str(path))
        return Path(path)

    def write_vtu(self, path: str | Path) -> Path:
        """Writes the surface mesh as a VTK XML unstructured grid (``.vtu``)
        with cell data ``patch`` and ``face_tag``."""
        self._native.write_vtu(str(path))
        return Path(path)

    def boundary_edges(self):
        """Conductor outline = edges with only one same-tag triangle (a free side
        or a tag change). Returns ``(B, 3)`` int64 ``[v0, v1, tri]``."""
        return self._native.boundary_edges()

    def edges_on_line(self, axis, value, lo, hi, tol=1e-7):
        """Port helper: boundary edges whose BOTH endpoints lie on the line
        ``{axis = value}`` within ``[lo, hi]`` (``axis`` is ``'x'``/``'y'``/``'z'``
        or 0/1/2). Returns ``(k, 2)`` int64 vertex pairs."""
        a = {"x": 0, "y": 1, "z": 2}.get(axis, axis)
        return self._native.edges_on_line(a, value, lo, hi, tol)

    def areas(self):
        """Per-triangle area (input units), shape ``(n_faces,)``."""
        return self._native.areas()

    def min_angles(self):
        """Per-triangle minimum interior angle in degrees, shape ``(n_faces,)``."""
        return self._native.min_angles()

    def to_viewer_dict(self, name: str) -> dict:
        """The surface mesh in the viewer JSON schema (no tets)."""
        return json.loads(self._native.viewer_json(name))

    def show(self, name: str = "surface", *, clip: float | None = None, **kw) -> None:
        """Open this surface mesh in the interactive viewer and block until the
        window is closed (see :meth:`Mesh.show`); a flat triangulation needs no
        clip by default."""
        _show(self.to_viewer_dict(name), name, clip=clip, tets=False, **kw)


def load_msh(path) -> Mesh:
    """The volume mesh in a gmsh MSH file (4.1 or 2.2, ASCII) as it is, no
    remeshing: a volume entity is a region labelled by its first physical
    group, the first physical group of a surface entity its face tag, any
    further surface and curve groups named face and edge sets."""
    return Mesh(_native.load_msh(str(path)))


def polygon_union(polygons):
    """The union of planar polygons into connected shapes: overlapping or
    abutting ones merge, separate ones stay separate. Each polygon is a list
    of ``(x, y)`` points or an ``(outer, holes)`` pair, either way round.
    Returns ``(outer, holes)`` per shape, the outer counter-clockwise and the
    holes clockwise; the outlines of layout layers whose rectangles overlap,
    ready for :meth:`Geometry.polygon_plate` or a prism."""
    norm = []
    for p in polygons:
        if len(p) == 2 and not np.isscalar(p[0][0]):
            outer, holes = p
        else:
            outer, holes = p, []
        norm.append(([list(map(float, q)) for q in outer],
                     [[list(map(float, q)) for q in h] for h in holes]))
    return _native.polygon_union(norm)


def _filt(sel, kw):
    """A selector descriptor: ``None`` (unfiltered) or a dict of criteria."""
    if sel is None and not kw:
        return None
    f = dict(kw)
    if sel is not None:
        f["id"] = sel
    return f


class _Scope:
    """A hierarchical sizing scope built by ``g.region(..).surf(..).edge(..)``.

    Setting ``.maxh`` / ``.tol`` applies to every entity the scope selects:
    unfiltered at a dimension sets the global per-dimension knob, a filtered
    scope sets per-entity overrides (the most specific scope wins, because a
    per-entity override beats the dimension default in the mesher). Filters:
    regions by ``id``/``tag``; surfaces by ``id``/``tag``/``normal``
    (``normal_tol``)/``near``; edges by ``id``/``kind``/``between``/``near``.
    """

    __slots__ = ("_g", "_level", "_rf", "_ff", "_ef")

    def __init__(self, g, level, rf=None, ff=None, ef=None):
        self._g, self._level, self._rf, self._ff, self._ef = g, level, rf, ff, ef

    def region(self, sel=None, **kw):
        return _Scope(self._g, "region", _filt(sel, kw), self._ff, self._ef)

    def surf(self, sel=None, **kw):
        return _Scope(self._g, "surf", self._rf, _face_filt(self._g._native, sel, kw), self._ef)

    def edge(self, sel=None, **kw):
        return _Scope(self._g, "edge", self._rf, self._ff, _filt(sel, kw))

    def _native(self):
        return _native.Scope(self._level, self._rf, self._ff, self._ef)

    @property
    def maxh(self):
        raise AttributeError("a sizing scope is write-only")

    @maxh.setter
    def maxh(self, v):
        self._g._native.set_maxh_on(self._native(), float(v))

    @property
    def tol(self):
        raise AttributeError("a sizing scope is write-only")

    @tol.setter
    def tol(self, v):
        self._g._native.set_tol_on(self._native(), float(v))

    @property
    def name(self):
        raise AttributeError("a scope name is write-only")

    @name.setter
    def name(self, v):
        """Names the geometric faces or edges the scope selects (a port, a
        boundary condition): they come out as a set in ``Mesh.sets()`` and
        as a physical group in ``write_msh``. The selection is kept and
        resolved when the mesh is made; select by origin (``solid=``,
        ``role=``) to keep the same face through later changes."""
        self._g._native.name(self._native(), str(v))


def _plain(v):
    """``v`` with numpy arrays and scalars as Python lists and numbers."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


def _given(**fields) -> dict:
    """The fields given, as plain Python: those left at ``None`` take the
    Rust defaults."""
    return {k: _plain(v) for k, v in fields.items() if v is not None}


def _solid(native, pair) -> Solid:
    """The native ``(region, index)`` as a handle, with the names of its
    faces from the geometry."""
    region, index = int(pair[0]), int(pair[1])
    return Solid(region, index, tuple(native.roles(region, index)))


@dataclass(frozen=True)
class Sheet:
    """Handle to a sheet added to a :class:`Geometry`: its ``index`` among
    the sheets (insertion order) and its face ``tag``."""

    index: int
    tag: int


def _obj(o):
    """A solid or sheet as the native ``(is_sheet, first, second)``."""
    if isinstance(o, Solid):
        return (False, o.region, o.index)
    if isinstance(o, Sheet):
        return (True, o.index, o.tag)
    raise TypeError(f"expected a Solid or a Sheet, got {type(o).__name__}")


def _unobj(native, t):
    """The native ``(is_sheet, first, second)`` as a handle."""
    is_sheet, a, b = t
    if is_sheet:
        return Sheet(int(a), int(b))
    return _solid(native, (a, b))


def _face_filt(native, sel, kw):
    """A face selector descriptor (see ``_filt``): ``solid`` may be a
    :class:`Solid`, and ``role`` then the name of one of its faces."""
    kw = dict(kw)
    solid = kw.get("solid")
    if isinstance(solid, Solid):
        kw["solid"] = solid.index
    if isinstance(kw.get("role"), str):
        if not isinstance(solid, Solid):
            raise ValueError("a role by name needs solid= as the Solid")
        kw["role"] = native.role(solid.region, solid.index, kw["role"])
    return _filt(sel, kw)


class Geometry:
    """Top-level geometry builder, without materials and physics: regions
    carry integer tags that downstream tools map to materials.

    Parameters
    ----------
    maxh : float, optional
        global target tet edge length, used by :meth:`mesh` when no
        override is passed (sizing is a target with a documented
        1.5x max-edge contract, like gmsh's mesh size)
    grading : float, optional
        default size-grading Lipschitz constant for :meth:`mesh` (see
        there); the default 0.5 grows neighbor elements by roughly 1.5x

    Notes
    -----
    How the mesh size is set: the finest of what each of these asks for
    wins, and the size grows away from fine places by ``grading``.

    - ``maxh``: the element size, global (here or ``g.maxh``) and per
      region, face, edge or sheet (``g.region(2).maxh = ...``,
      ``g.surf(...).maxh``, ``g.edge(...).maxh``, ``maxh=`` of a solid or a
      sheet). This is the size what lives on the mesh needs; it is never
      exceeded.
    - curved geometry, by one of two measures:

      - a chord tolerance (``g.tol``, ``g.edge(...).tol``,
        ``g.surf(...).tol``, ``tol_edge``/``tol_surf`` of :meth:`mesh`): a
        chord deviates from its curve by at most ``tol`` times the radius,
        the same number of segments per turn however small the curve (the
        default 0.05 is about ten segments round a circle);
      - a geometric error (``geom_error`` and ``order`` of :meth:`mesh`):
        the volume of every region and the area of every sheet within this
        share of the true ones, measured on flat elements (``order=1``) or
        on the quadratic elements of :meth:`Mesh.second_order` (``order=2``),
        which follow a curve with far fewer elements. An explicit ``tol`` on
        an entity still wins there.
    - ``min_angle``: the size shrinks where tets stay below this dihedral
      angle (layers far thinner than the size, features far below it).
    - ``target_elements``: one factor scales every size until the tet count
      lands near it.
    - point sources: :meth:`refine_near_points`.
    """

    def __init__(self, *, maxh: float | None = None, grading: float | None = None) -> None:
        self._native = _native.Geometry(maxh, grading)

    # ---- hierarchical per-entity sizing -----------------------------------
    # g.maxh = ...                      # global size (all dimensions)
    # g.edge().tol = ...                # all edges (== tol_edge)
    # g.surf(normal=(0,0,1)).maxh = ... # selected surfaces
    # g.region(2).surf().edge(near=p).maxh = ...  # specific edges

    @property
    def maxh(self) -> float | None:
        """Global target edge length (all dimensions)."""
        return self._native.maxh

    @maxh.setter
    def maxh(self, v: float) -> None:
        self._native.maxh = float(v)

    @property
    def tol(self) -> float:
        raise AttributeError("write-only; sets both edge and surface tolerance")

    @tol.setter
    def tol(self, v: float) -> None:
        self._native.set_tol(float(v))

    def region(self, sel=None, **kw) -> _Scope:
        """Scope on a region (material) by tag/id, or all regions if unfiltered."""
        return _Scope(self, "region", _filt(sel, kw))

    def surf(self, sel=None, **kw) -> _Scope:
        """Scope on surfaces by ``id=``/``tag=``/``normal=``/``near=``, or
        by origin, ``solid=`` (a :class:`Solid`) and ``role=`` (one of its
        ``roles``, or the surface's index in its shape), or all. A selection
        by origin keeps meaning the same face when the geometry changes."""
        return _Scope(self, "surf", None, _face_filt(self._native, sel, kw))

    def edge(self, sel=None, **kw) -> _Scope:
        """Scope on edges by ``id=``/``near=``/``between=``/``kind=``, or all.
        ``kind`` names the edge's curve: "line", "circle", "ellipse",
        "spline", "profile" (a swept profile's edge), "intersection" (of two
        curved surfaces) or "polyline" (no analytic curve)."""
        return _Scope(self, "edge", None, None, _filt(sel, kw))

    def _topology(self):
        """The B-rep topology of the model every mesh of the current
        geometry uses."""
        return self._native.topology()

    def _resolve(self, scope: _Scope) -> list:
        """The entity ids a scope selects."""
        return list(self._native.resolve(scope._native()))

    def periodic(self, master: _Scope, slave: _Scope, shift=None) -> tuple:
        """Meshes the faces ``slave`` selects with the same triangles as the
        faces ``master`` selects: every slave face is a master face moved
        by ``shift`` (default: the difference of their area-weighted
        centroids). Call it once per direction of a unit cell, after the
        geometry is complete. ``Mesh.periodic_points`` then pairs every
        point on the master faces with its image. Returns the shift.

        Example: ``g.periodic(g.surf(normal=(-1, 0, 0)), g.surf(normal=(1, 0, 0)))``
        """
        return self._native.periodic(master._native(), slave._native(), shift)

    def label(self, target: Solid | int, name: str) -> None:
        """Names a :class:`Solid` (its cells form the set and physical group
        ``name``; solids of one name are one group) or an int sheet tag."""
        if isinstance(target, Solid):
            self._native.label_solid(target.region, target.index, name)
        else:
            self._native.label_tag(int(target), name)

    def union(self, *solids: Solid) -> Solid:
        """Fuse overlapping solids into ONE material (a boolean union): the
        internal boundaries between them become same-region faces and are
        dropped at assembly, leaving the outer union surface. Returns the
        first solid, now representing the merged region."""
        return _solid(self._native, self._native.union([(s.region, s.index) for s in solids]))

    def _add(self, kind: str, maxh, void, **fields) -> Solid:
        """A solid of ``kind`` from the fields given (the others at the Rust
        defaults)."""
        return _solid(self._native, self._native.add_solid(kind, _given(**fields), maxh, void))

    def _sheet(self, kind: str, tag, maxh, **fields) -> "Sheet":
        """A sheet of ``kind`` from the fields given."""
        return Sheet(*self._native.add_sheet(kind, _given(**fields), tag, maxh))

    # ------------------------------------------------------------ solids

    def box(self, width: float, depth: float, height: float,
            position=None, *, maxh: float | None = None, void: bool = False) -> Solid:
        """Axis-aligned box: extents along x, y, z; ``position`` is the
        lower corner. ``void=True`` carves the volume out of everything
        added before it (the cut boolean; the region tag is then 0 and the
        walls become boundary faces)."""
        return self._add("box", maxh, void, size=[width, depth, height], position=position)

    def cylinder(self, radius: float, height: float, position=None, axis=None, *,
                 segments: int | None = None, uniform: bool | None = None, rows: int | None = None,
                 maxh: float | None = None, void: bool = False) -> Solid:
        """Cylinder from the base centre ``position`` along ``axis``. The
        barrel is tessellated with ``segments`` chords (24) but carries the
        exact analytic surface: mesh vertices snap onto the true cylinder.

        With ``uniform=True`` the barrel is a structured grid of height
        ``rows`` (auto-chosen for roughly square cells when ``None``) instead of
        full-height strips, for an isotropic surface mesh."""
        return self._add("cylinder", maxh, void, radius=radius, height=height, position=position,
                         axis=axis, segments=segments, uniform=uniform, rows=rows)

    def sphere(self, radius: float, position=None, *, segments: int | None = None,
               maxh: float | None = None, void: bool = False) -> Solid:
        """Sphere centred at ``position`` (analytic surface, faceted
        geodesically; the facet density follows the target size,
        ``segments`` is a floor)."""
        return self._add("sphere", maxh, void, radius=radius, position=position, segments=segments)

    def icosphere(self, radius: float, position=None, *, subdivisions: int | None = None,
                  maxh: float | None = None, void: bool = False) -> Solid:
        """Geodesic sphere with a fixed facet level: a subdivided icosahedron
        projected onto the analytic sphere, ``20 * 4**subdivisions`` faces
        (3 by default)."""
        return self._add("icosphere", maxh, void, radius=radius, position=position,
                         subdivisions=subdivisions)

    def airfoil_naca0012(self, chord: float, span: float, position=None,
                         span_axis=None, *, n_per_side: int | None = None,
                         n_seg: int | None = None, maxh: float | None = None,
                         void: bool = False) -> Solid:
        """A NACA 0012 airfoil (chord along +x, leading edge at ``position``)
        extruded along ``span_axis`` by ``span``. The curved skin is one
        analytic extruded-spline surface; the trailing edge is a flat blunt
        face. ``n_per_side`` (40) controls profile control points, ``n_seg``
        (120) the facet count along the chord."""
        return self._add("naca0012", maxh, void, chord=chord, span=span, position=position,
                         span_axis=span_axis, n_per_side=n_per_side, n_seg=n_seg)

    def cone(self, r1: float, r2: float, height: float, position=None, axis=None, *,
             segments: int | None = None, uniform: bool | None = None, rows: int | None = None,
             maxh: float | None = None, void: bool = False) -> Solid:
        """Conical frustum: base radius ``r1`` at ``position``, top radius
        ``r2`` (0 for a full cone) at ``position + height * axis``;
        ``uniform`` and ``rows`` as for :meth:`cylinder`."""
        return self._add("cone", maxh, void, r1=r1, r2=r2, height=height, position=position,
                         axis=axis, segments=segments, uniform=uniform, rows=rows)

    def prism(self, points, height: float, position=None, *, holes=None,
              maxh: float | None = None, void: bool = False) -> Solid:
        """Right prism: the 2D polygon ``points`` (in the xy plane, offset by
        ``position``) extruded by ``height`` along z."""
        return self._add("prism", maxh, void, points=points, height=height, position=position,
                         holes=holes)

    def torus(self, major_radius: float, minor_radius: float, position=None,
              axis=None, *, segments: int | None = None,
              tube_segments: int | None = None, maxh: float | None = None,
              void: bool = False) -> Solid:
        """Torus centred at ``position`` with the donut plane normal to
        ``axis`` (analytic surface)."""
        return self._add("torus", maxh, void, major_radius=major_radius,
                         minor_radius=minor_radius, position=position, axis=axis,
                         segments=segments, tube_segments=tube_segments)

    def wedge(self, dx: float, dy: float, dz: float, position=None, *,
              top_x: float | None = None, maxh: float | None = None,
              void: bool = False) -> Solid:
        """Wedge: a ``dx x dy x dz`` box whose top edge is shortened to
        ``top_x`` along x (0, the default, gives a triangular prism); the
        taper runs in the xz plane."""
        return self._add("wedge", maxh, void, size=[dx, dy, dz], position=position, top_x=top_x)

    def sweep(self, path, radius: float, *, segments: int | None = None,
              maxh: float | None = None, void: bool = False) -> Solid:
        """Tube with a circular cross-section swept along the open polyline
        ``path``. Sample curved paths finely; the tube radius must stay
        below the local curvature radius."""
        return self._add("sweep", maxh, void, path=path, radius=radius, segments=segments)

    def helix(self, radius: float, pitch: float, turns: float, wire_radius: float,
              position=None, *, points_per_turn: int | None = None,
              segments: int | None = None, maxh: float | None = None,
              void: bool = False) -> Solid:
        """Helical coil around +z through ``position``: helix ``radius``,
        ``pitch`` advance per turn, round wire of ``wire_radius``."""
        return self._add("helix", maxh, void, radius=radius, pitch=pitch, turns=turns,
                         wire_radius=wire_radius, position=position,
                         points_per_turn=points_per_turn, segments=segments)

    def loft(self, profile_a, profile_b, *, maxh: float | None = None,
             void: bool = False) -> Solid:
        """Ruled loft between two planar profiles with the same vertex
        count, corresponded by index (horn tapers). Profiles must be
        star-shaped about their centroid (convex profiles always are)."""
        return self._add("loft", maxh, void, profile_a=profile_a, profile_b=profile_b)

    def mesh_solid(self, verts, tris, *, maxh: float | None = None,
                   void: bool = False) -> Solid:
        """Solid from an externally supplied triangle soup: ``verts`` is an
        ``(n, 3)`` array of vertex coordinates and ``tris`` an ``(m, 3)``
        array of triangle vertex indices. The surface must be closed and
        non-self-intersecting; the winding is normalized to outward
        internally. The triangles ARE the surface (no analytic back-reference):
        sample organic shapes finely."""
        v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        t = np.asarray(tris, dtype=np.uint32).reshape(-1, 3)
        return self._add("triangles", maxh, void, verts=v, tris=t)

    def revolve(self, profile, *, position=None, axis=None, angle: float | None = None,
                segments: int | None = None, maxh: float | None = None,
                void: bool = False) -> Solid:
        """Solid of revolution: the closed ``profile`` in the half-plane
        ``(r, z)`` of ``axis`` through ``position`` (``r >= 0``), turned by
        ``angle`` degrees. Each entry is a vertex ``(r, z)`` followed by a
        straight edge, ``(r, z, bulge)`` followed by a circular arc
        (``bulge = tan(swept angle / 4)``, positive counterclockwise), or a
        :class:`Spline` that makes the edge before it a spline through its
        points. Every edge is its own surface with an exact carrier; the
        roles are ``edge0``, ``edge1``, ... and, for a part turn, ``start``
        and ``end``."""
        pts, edges = [], []
        for item in profile:
            if isinstance(item, Spline):
                if not edges or edges[-1] != "line":
                    raise ValueError("a Spline must follow a vertex with a straight edge")
                edges[-1] = {"spline": [list(p) for p in item.points]}
                continue
            if len(item) not in (2, 3):
                raise ValueError("a profile vertex is (r, z) or (r, z, bulge)")
            pts.append([float(item[0]), float(item[1])])
            bulge = float(item[2]) if len(item) == 3 else 0.0
            edges.append({"arc": bulge} if bulge else "line")
        return self._add("revolve", maxh, void, points=pts, edges=edges, position=position,
                         axis=axis, angle=angle, segments=segments)

    # ------------------------------------------------ placement, copies

    def translate(self, obj, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
        """Moves a solid or sheet by ``(dx, dy, dz)`` in place; its faces
        keep their roles. Returns ``obj``."""
        self._native.transform(_obj(obj), "translate", [dx, dy, dz], [0, 0, 0], 0.0)
        return obj

    def rotate(self, obj, angle: float, axis=(0, 0, 1), center=(0, 0, 0)):
        """Turns a solid or sheet by ``angle`` radians (right-handed) about
        the axis along ``axis`` through ``center``, in place."""
        self._native.transform(_obj(obj), "rotate", list(axis), list(center), float(angle))
        return obj

    def mirror(self, obj, normal=(1, 0, 0), point=(0, 0, 0)):
        """Mirrors a solid or sheet across the plane through ``point`` with
        normal ``normal``, in place."""
        self._native.transform(_obj(obj), "mirror", list(normal), list(point), 0.0)
        return obj

    def stretch(self, obj, fx: float = 1.0, fy: float = 1.0, fz: float = 1.0, center=(0, 0, 0)):
        """Scales a solid or sheet by ``fx``, ``fy``, ``fz`` about
        ``center``, in place. Unequal factors keep planes, discrete patches
        and NURBS exact; the other curved faces become faceted."""
        self._native.transform(_obj(obj), "stretch", [fx, fy, fz], list(center), 0.0)
        return obj

    def copy(self, obj):
        """A copy of a solid (a region of its own, same target size; a void
        stays a void) or of a sheet (same face tag), in the same place."""
        return _unobj(self._native, self._native.copy(_obj(obj)))

    def array(self, obj, count: int, *, spacing=None, rotation: float | None = None,
              axis=(0, 0, 1), center=(0, 0, 0)) -> list:
        """``count`` objects: ``obj`` first, then copies moved by ``spacing``
        (an offset) or turned by ``rotation`` radians about the axis along
        ``axis`` through ``center``, once, twice, ... from ``obj``."""
        if (spacing is None) == (rotation is None):
            raise ValueError("give exactly one of spacing and rotation")
        if spacing is not None:
            step = ("translate", list(spacing), [0, 0, 0], 0.0)
        else:
            step = ("rotate", list(axis), list(center), float(rotation))
        return [_unobj(self._native, t) for t in self._native.array(_obj(obj), int(count), *step)]

    def intersect(self, target: Solid, *tools: Solid) -> Solid:
        """Cuts ``target`` down to what it has in common with every tool (the
        exact boolean), in place; the tools are used up."""
        self._native.intersect((target.region, target.index),
                               [(t.region, t.index) for t in tools])
        return target

    def extrude(self, face: "Sheet", height: float, axis=(0, 0, 1), *,
                maxh: float | None = None) -> Solid:
        """The solid ``face`` (a flat sheet) sweeps along ``axis * height``,
        in a region of its own; the sheet stays as its bottom face. A disc
        extrudes along its axis only. Roles ``bottom`` and ``top``; the
        walls follow (one cylinder under a disc, a plane per edge else)."""
        if not isinstance(face, Sheet):
            raise TypeError("extrude takes a sheet (a plate, disc or polygon)")
        vector = [float(a) * float(height) for a in axis]
        pair = self._native.extrude((face.index, face.tag), vector, maxh)
        return _solid(self._native, pair)

    def chamfer(self, solid: Solid, distance: float, *, edges=None,
                void: bool = False) -> Solid:
        """Chamfers edges of ``solid`` by ``distance`` on both faces; the
        material comes off that solid alone and what lies under it fills the
        cut. ``edges``
        is ``None`` (every edge of the solid's faces), a role (the edges of
        that face), a pair of roles (the edge between two of its faces), a
        triple ``(role, other_solid, its_role)`` (the rim of a hole cut into
        it), or a list of these. Straight edges between planes and circles
        between a plane square to their axis, a cylinder or a cone take one.
        Returns the solid with the chamfer faces as roles ``chamfer0``,
        ``chamfer1``, ..., numbered on over later cuts. With ``void=True``
        the solid stays and the cut is carved out as voids, left empty (a
        countersink on a hole): returns those voids, each with its chamfer
        face as role ``chamfer``."""
        return self._cut_edges(solid, "chamfer", distance, edges, void)

    def fillet(self, solid: Solid, radius: float, *, edges=None,
               void: bool = False) -> Solid:
        """Rounds edges of ``solid`` with ``radius``, tangent to both faces
        (a cylinder along a straight edge, a torus along a circle); edges,
        ``void`` and what it returns as for :meth:`chamfer`, the round faces
        as roles ``fillet0``, ``fillet1``, ... (``fillet`` on a void). Where
        three rounded edges meet, the rounds cross (no vertex blend)."""
        return self._cut_edges(solid, "fillet", radius, edges, void)

    def _cut_edges(self, solid, kind, size, edges, void):
        def role(r, of=solid):
            if isinstance(r, str):
                return self._native.role(of.region, of.index, r)
            return int(r)

        def pick(e):
            if isinstance(e, (str, int)):
                return ("of", role(e), 0, 0)
            if len(e) == 2:
                return ("between", role(e[0]), role(e[1]), 0)
            if len(e) == 3 and isinstance(e[1], Solid):
                return ("with", role(e[0]), e[1].index, role(e[2], e[1]))
            raise ValueError(f"cannot read the edge pick {e!r}")

        if edges is None:
            picks = [("all", 0, 0, 0)]
        elif isinstance(edges, list):
            picks = [pick(e) for e in edges]
        else:
            picks = [pick(edges)]
        faces = self._native.cut_edges(
            solid.region, solid.index, picks, kind, float(size), void)
        if void:
            return [_solid(self._native, (g, i)) for g, i, _ in faces]
        return _solid(self._native, (solid.region, solid.index))

    def import_stl(self, path, *, crease_deg: float | None = None, up: str | None = None,
                   maxh: float | None = None, void: bool = False) -> Solid:
        """Solid from an STL file (binary or ASCII) on the discrete-envelope
        path: the triangle soup is split into smooth regions at crease edges
        (facet normals turning by more than ``crease_deg``, 40 by default),
        each region becomes one discrete surface, and the mesher remeshes
        that envelope -- unlike :meth:`mesh_solid`, which keeps the input
        facets. The file must describe a closed, consistently oriented
        2-manifold. ``up`` names the file's up axis (``"z"`` by default); a
        ``"y"``-up model is rotated upright."""
        return self._add("import", maxh, void, path=str(path), crease_deg=crease_deg, up=up)

    def import_obj(self, path, *, crease_deg: float | None = None, up: str | None = None,
                   maxh: float | None = None, void: bool = False) -> Solid:
        """Solid from a Wavefront OBJ file (``v``/``f`` records; polygons
        are fan-triangulated), with the semantics of :meth:`import_stl`."""
        return self._add("import", maxh, void, path=str(path), crease_deg=crease_deg, up=up)

    def import_step(self, path, *, maxh: float | None = None) -> list[Solid]:
        """The solids of a STEP file (AP203/AP214), one per solid body of
        the file, each with its faces on their true surfaces (planes,
        cylinders, cones, spheres, tori, B-splines): the mesh is measured
        against those, not against a tessellation. Each solid is labelled
        with the name the file gives its part, so the mesh's sets and
        physical groups carry those names. Coordinates stay in the file's
        unit."""
        return [_solid(self._native, p) for p in self._native.import_step(str(path), maxh)]

    # ------------------------------------------------------------ sheets

    def xy_plate(self, width: float, height: float, position=(0, 0, 0), *,
                 tag: int = 1, maxh: float | None = None) -> "Sheet":
        """Zero-thickness rectangle in an xy plane (a PEC trace, a port
        marker): spans ``width`` along x and ``height`` along y from the
        corner ``position``; conformally embedded with face tag ``tag``."""
        return self._sheet("rect", tag, maxh, corner=position, u=[width, 0, 0], v=[0, height, 0])

    def xz_plate(self, width: float, height: float, position=(0, 0, 0), *,
                 tag: int = 1, maxh: float | None = None) -> "Sheet":
        """Like :meth:`xy_plate` in an xz plane (width along x, height
        along z)."""
        return self._sheet("rect", tag, maxh, corner=position, u=[width, 0, 0], v=[0, 0, height])

    def yz_plate(self, width: float, height: float, position=(0, 0, 0), *,
                 tag: int = 1, maxh: float | None = None) -> "Sheet":
        """Like :meth:`xy_plate` in a yz plane (width along y, height
        along z)."""
        return self._sheet("rect", tag, maxh, corner=position, u=[0, width, 0], v=[0, 0, height])

    def plate(self, p0, du, dv, *, tag: int = 1, maxh: float | None = None) -> "Sheet":
        """General parallelogram sheet from corner ``p0`` spanned by the
        edge vectors ``du`` and ``dv``."""
        return self._sheet("rect", tag, maxh, corner=p0, u=du, v=dv)

    def disc(self, radius: float, position=None, axis=None, *,
             segments: int | None = None, tag: int = 1, maxh: float | None = None) -> "Sheet":
        """Disc sheet centred at ``position``, normal to ``axis``."""
        return self._sheet("disc", tag, maxh, radius=radius, center=position, axis=axis,
                           segments=segments)

    def polygon_plate(self, points, position=None, *, holes=None, tag: int = 1,
                      maxh: float | None = None) -> "Sheet":
        """Polygonal sheet in an xy plane at ``position`` (2D coordinates
        are offset by ``position``'s x, y)."""
        return self._sheet("polygon", tag, maxh, points=points, position=position, holes=holes)

    def nurbs_plate(self, ctrl, *, degree=(3, 3), weights=None, knots=None, tag: int = 1,
                    maxh: float | None = None) -> "Sheet":
        """NURBS sheet over the control net ``ctrl`` (shape ``(nu, nv, 3)``,
        first index along u). ``weights`` (shape ``(nu, nv)``, positive)
        default to 1, ``knots`` (a pair of clamped knot vectors) to clamped
        uniform ones. The mesh snaps to the exact surface."""
        c = np.asarray(ctrl, float)
        if c.ndim != 3 or c.shape[2] != 3:
            raise ValueError("ctrl must have shape (nu, nv, 3)")
        w = None if weights is None else np.asarray(weights, float)
        k = None if knots is None else [list(map(float, knots[0])), list(map(float, knots[1]))]
        return self._sheet("nurbs", tag, maxh, ctrl=c, degree=[int(degree[0]), int(degree[1])],
                           weights=w, knots=k)

    # ------------------------------------------------------------ sizing

    def refine_surface(self, solid: Solid, h: float) -> None:
        """Per-solid surface sizing: the solid's boundary patches mesh at
        ``h`` and the size recovers along the grading into the surrounding
        volume. The only sizing handle that reaches a VOID's walls (a coax
        inner conductor has no region and no face tag)."""
        self._native.refine_surface(solid.region, solid.index, float(h))

    def refine_near_points(self, points, h) -> None:
        """Registers point size sources: the edge-length target shrinks to
        ``h`` at each point and recovers along the grading away from it (the
        hook for error-driven adaptive refinement). ``h`` may be a scalar or
        a per-point sequence, e.g. element-wise sizes from a Doerfler
        marking pass."""
        pts = [[float(c) for c in p] for p in points]
        hs = [float(x) for x in h] if np.ndim(h) else [float(h)] * len(pts)
        self._native.add_size_points(pts, hs)

    # ------------------------------------------------------------- mesh

    def mesh(
        self,
        *,
        maxh: float | None = None,
        max_points: int | None = None,
        grading: float | None = None,
        cells_across: float | None = None,
        tol_edge: float | None = None,
        tol_surf: float | None = None,
        maxh_edge: float | None = None,
        maxh_surf: float | None = None,
        maxh_vol: float | None = None,
        target_elements: int | None = None,
        min_h_surf: float | None = None,
        min_angle: float | None = None,
        geom_error: float | None = None,
        order: int | None = None,
    ) -> Mesh:
        """Assembles the exact conforming arrangement of every solid and
        sheet, meshes it bottom-up (edges, then each face on its surface,
        then each region by its constrained Delaunay tetrahedralization),
        and improves the tets.

        Raises ``rapidmesh.MeshError`` where the geometry defeats the mesher
        (features far below the mesh size, gaps, degenerate faces); its
        message says where and what to repair.

        Parameters
        ----------
        maxh : float, optional
            global target edge length (defaults to the geometry's;
            unbounded if neither is given)
        max_points : int, optional
            best-effort refinement point budget
        grading : float
            size-grading Lipschitz constant: the edge-length target may grow
            by at most this factor per unit distance from finer features
            (0.5 means neighbor elements grow by roughly 1.5x)
        cells_across : float, optional
            elements across the thickness of each region: inside a region of
            thickness ``t = 2 V / S`` (a plate's thickness, a wire's radius)
            the size is at most ``t / cells_across``, so thin plates and
            wires get proper tets through them. Default ``None``: off (stacks
            of layers far thinner than the size take flat tets through each
            layer)
        tol_edge, tol_surf : float
            relative chord (sagitta) tolerance for curved EDGES and SURFACES: an
            entity of radius ``R`` is sized ``h = R * sqrt(8 * tol)``, so the
            chord deviates by at most ``tol * R``. Default: the geometry's
            (5e-2, about ten segments round a circle). There is no volume
            tolerance: the volume follows the surface.
        maxh_edge, maxh_surf, maxh_vol : float
            maximum element edge length per dimension, each combined with
            ``maxh`` as ``min(maxh, maxh_dim)``. Default: the geometry's (inf).
        target_elements : int, optional
            element (tet) budget: the global size scale is retuned over a few
            remeshes so the tet count lands near it, while the relative
            refinement keeps its shape
        min_h_surf : float, optional
            hard minimum element size on surfaces (0 off)
        geom_error : float, optional
            relative geometric error the elements may make, instead of the
            chord tolerances: the volume of every region and the area of every
            sheet within this share of the true ones (1e-2 is one percent).
            Each curved boundary is sized by its curvature and by the
            thickness of what it bounds (a region's volume over its surface, a
            sheet's area over its perimeter), so a thin pin is meshed finer
            than a large body of the same curvature. Flat faces are not
            refined by it; ``maxh`` still caps the size everywhere, and an
            explicit ``tol`` on an entity still wins. Default ``None``: the
            chord tolerances
        order : int, optional
            the elements ``geom_error`` is measured on: 1 flat (default), 2
            quadratic, whose mid-edge nodes lie on the true surfaces and curves
            (:meth:`Mesh.second_order`, ``write_msh(path, order=2)``). Flat
            elements need many small ones on a tight curve (the error falls
            with the square of their size); quadratic ones follow it with a
            few (with the fourth power), so ``geom_error=1e-2, order=2`` meshes
            most CAD parts with fewer tets than the default tolerance and an
            error of a few hundredths of a percent. Use the second-order mesh
            then: its linear corners alone carry the flat error. Example::

                mesh = g.mesh(geom_error=1e-2, order=2)
                so = mesh.second_order()             # tet10, curved boundary
                mesh.write_msh("part.msh", order=2)
        min_angle : float, optional
            smallest dihedral angle (degrees) to aim at, e.g. 15 for a solver
            that needs one: where tets stay below it (flat tets through a
            layer far thinner than the size, around a feature far below it),
            the size there shrinks over a few remeshes. What stays below (a
            wedge sharper than the angle) is a warning in ``mesh.report()``.
            Default ``None``: one mesh
        """
        return Mesh(self._native.mesh(_given(
            maxh=maxh, max_points=max_points, grading=grading, cells_across=cells_across,
            tol_edge=tol_edge, tol_surf=tol_surf, maxh_edge=maxh_edge, maxh_surf=maxh_surf,
            maxh_vol=maxh_vol, target_elements=target_elements, min_h_surf=min_h_surf,
            min_angle=min_angle, geom_error=geom_error, order=order,
        )))

    def surface_mesh(
        self,
        *,
        maxh: float | None = None,
        grading: float | None = None,
        tol_edge: float | None = None,
        tol_surf: float | None = None,
        maxh_edge: float | None = None,
        maxh_surf: float | None = None,
        maxh_vol: float | None = None,
        target_triangles: int | None = None,
        geom_error: float | None = None,
        order: int | None = None,
    ) -> SurfaceMesh:
        """Surface-only export: assembles the exact arrangement and meshes
        only its surfaces (region interfaces, outer boundary, embedded
        sheets), with the full sizing hierarchy of :meth:`mesh`.
        Each face is meshed alone on the shared samples of its edges.
        ``target_triangles`` is a triangle budget: the sizes are coarsened by
        one factor until the count is at most a little over it.
        ``geom_error`` and ``order`` as for :meth:`mesh`."""
        return SurfaceMesh(self._native.surface_mesh(_given(
            maxh=maxh, grading=grading, tol_edge=tol_edge, tol_surf=tol_surf,
            maxh_edge=maxh_edge, maxh_surf=maxh_surf, maxh_vol=maxh_vol,
            target_triangles=target_triangles, geom_error=geom_error, order=order,
        )))

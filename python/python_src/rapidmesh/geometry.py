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
        _viewer.inspect(path, title=f"rapidmesh — {name}", **kw)
    finally:
        with suppress(OSError):
            os.unlink(path)


class _Labelled:
    """The labels a mesh carries from its geometry."""

    def _read_labels(self, native) -> None:
        labels = native.labels()
        #: per input solid (insertion order): {"region": int, "label": str|None}
        self.solids: list[dict] = labels["solids"]
        #: display label per sheet tag
        self.tag_labels: dict[int, str] = labels["tag_labels"]
        #: named geometric faces and edges (``g.surf(..).name = ...``): name
        #: -> entity ids
        self.face_names: dict[str, list[int]] = labels["face_names"]
        self.edge_names: dict[str, list[int]] = labels["edge_names"]


class Mesh(_Labelled):
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
        self._native = native
        self._read_labels(native)
        self.points: np.ndarray = native.points()
        self.tets: np.ndarray = native.tets()
        self.tet_regions: np.ndarray = native.tet_regions()
        self.faces: np.ndarray = native.faces()
        self.face_tags: np.ndarray = native.face_tags()
        self.face_regions: np.ndarray = native.face_regions()
        self.face_surfaces: np.ndarray = native.face_surfaces()
        self.face_patches: np.ndarray = native.face_patches()
        self.point_class: np.ndarray = native.point_class()
        self.periodic_points: np.ndarray = native.periodic_points()
        self.surface_owners: np.ndarray = native.surface_owners()
        self.edges: np.ndarray = native.edges()
        self.stats: dict = native.stats()
        self.timings: dict = native.timings()
        self.metrics: dict = native.metrics()
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

    def write_msh(self, path: str | Path) -> Path:
        """Writes the mesh as a gmsh MSH 4.1 file: geometric vertices, edges
        and faces become point, curve and surface entities (tag = id + 1),
        regions volume entities; every node sits in the block of what it is
        classified on. Physical groups: the labelled solid groups (volumes),
        the named face tags and the named geometric faces and edges."""
        self._native.write_msh(str(path))
        return Path(path)

    def write_vtu(self, path: str | Path) -> Path:
        """Writes the mesh as a VTK XML unstructured grid (``.vtu``): the
        tets and the geometric faces, with cell data ``region``, ``patch``
        and ``face_tag``."""
        self._native.write_vtu(str(path))
        return Path(path)

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

    def to_viewer_dict(self, name: str) -> dict:
        """The mesh in the viewer JSON schema (shared by the comparison
        viewer and the showcase site), with the located defects."""
        return json.loads(self._native.viewer_json(name))

    def save_viewer_json(self, name: str, directory: str | Path) -> Path:
        """Writes ``rapidmesh_<name>.json`` in the viewer schema and
        refreshes the viewer manifest. Returns the written path."""
        return Path(self._native.save_viewer_json(name, str(directory)))

    def show(self, name: str = "mesh", *, clip: float | None = 0.6,
             clip_axis: int = 1, **kw) -> None:
        """Open this mesh in the interactive viewer and block until the window is
        closed: orbit / zoom / pan, the region legend, the crinkle clip (``clip``
        is the fraction along ``clip_axis``; ``None`` disables it), the
        located-defect overlay, and figure export. Uses a native window
        (``pywebview``) if installed, else a Chromium window. The viewer ships in
        the wheel; a window backend does not -- ``pip install pywebview`` (or
        ``playwright``)."""
        _show(self.to_viewer_dict(name), name, clip=clip, clip_axis=clip_axis, **kw)


class SurfaceMesh(_Labelled):
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
        self._native = native
        self._read_labels(native)
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

    def __repr__(self) -> str:
        return repr(self._native)

    # ---- MoM/FEM mesh info -------------------------------------------------
    # The same accessor vocabulary as :class:`Mesh2D`.

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


@dataclass
class Region2D:
    """A tagged 2D region for :func:`mesh_2d`: an outer loop with optional holes,
    all in the xy plane. ``tag`` flows to every triangle of this region (the
    conductor / layer id a MoM build reads for same-tag RWG edges)."""

    outer: list[tuple[float, float]]
    tag: int = 1
    holes: list[list[tuple[float, float]]] | None = None
    #: Open polylines INSIDE the region whose segments become element edges, without changing
    #: the region's extent. Used to put a known feature on the mesh: the outline a conductor
    #: on a neighbouring layer induces, or the rows of a boundary layer from
    #: :func:`offset_chains`.
    constraints: list[list[tuple[float, float]]] | None = None

    def local_width(self, point, inward) -> float:
        """The width of the region at a boundary `point`, measured along `inward`.

        Twice the radius of the largest ball tangent to the boundary there that is stopped by
        the wall ACROSS from it; ``inf`` when nothing faces the point. ``local_width / k`` is
        the sizing field that asks for k elements across the shape wherever it happens to be.
        """
        return _native.local_width(
            [list(p) for p in self.outer],
            [[list(p) for p in hl] for hl in (self.holes or [])],
            list(point),
            list(inward),
        )

    def offset_chains(self, pitch, scales=(1.0,), minh: float | None = None,
                      grading: float | None = None):
        """Inward offsets of the boundary at a distance that VARIES along it.

        `pitch` is either a number (a constant distance, i.e. a plain inward buffer) or a
        callable mapping the local width at a boundary point to the distance there, so
        ``lambda w: w / 16`` lays the first row a sixteenth of the local width inside the
        boundary. `scales` are the multiples to emit, e.g. ``(1, 3, 7)`` for rows of width
        ``d``, ``2d``, ``4d`` growing away from the boundary.

        `minh` floors the distance. A chain is a CONSTRAINT the mesher must reproduce, so a row
        finer than the floor cannot be repaired afterwards.

        Returns the chains, ready to assign to :attr:`constraints`.
        """
        return _native.offset_chains(
            [list(p) for p in self.outer],
            [[list(p) for p in hl] for hl in (self.holes or [])],
            pitch,
            [float(s) for s in scales],
            minh,
            grading,
        )


def load_msh(path) -> Mesh:
    """The volume mesh in a gmsh MSH file (4.1 or 2.2, ASCII) as it is, no
    remeshing: a volume entity is a region labelled by its first physical
    group, the first physical group of a surface entity its face tag, any
    further surface and curve groups named face and edge sets."""
    return Mesh(_native.load_msh(str(path)))


class Mesh2D:
    """A 2D mesh from the production 2D path -- the canonical 2D / MoM
    endpoint.

    Attributes
    ----------
    points : (n_points, 2) float64
        vertex coordinates (2D)
    tris : (n_tris, 3) uint64
        triangles (CCW)
    tri_tags : (n_tris,) int64
        conductor / layer tag per triangle
    """

    def __init__(self, native) -> None:
        self._native = native
        self.points: np.ndarray = native.points()
        self.tris: np.ndarray = native.tris()
        self.tri_tags: np.ndarray = native.tri_tags()
        self.stats: dict = native.stats()

    def __repr__(self) -> str:
        s = self.stats
        return f"Mesh2D({s['n_tris']} tris, {s['n_points']} points, {s['millis']} ms)"

    def edge_adjacency(self):
        """Undirected edge -> incident triangles, see
        :meth:`SurfaceMesh.edge_adjacency`."""
        return self._native.edge_adjacency()

    def rwg_edges(self, connect_tags: bool = False):
        """RWG basis functions, ``(D, 4)`` int64 ``[v0, v1, tri_plus,
        tri_minus]`` (see :meth:`SurfaceMesh.rwg_edges`)."""
        return self._native.rwg_edges(connect_tags)

    def boundary_edges(self):
        """Conductor outline = edges with a free side or a tag change, ``(B, 3)``
        int64 ``[v0, v1, tri]``."""
        return self._native.boundary_edges()

    def edges_on_line(self, axis, value, lo, hi, tol=1e-7):
        """Port helper: boundary edges on the line ``{axis = value}`` within
        ``[lo, hi]`` (``axis`` is ``'x'``/``'y'`` or 0/1). ``(k, 2)`` int64."""
        a = {"x": 0, "y": 1}.get(axis, axis)
        return self._native.edges_on_line(a, value, lo, hi, tol)

    def areas(self):
        """Per-triangle area, ``(n_tris,)`` float64."""
        return self._native.areas()

    def min_angles(self):
        """Per-triangle minimum interior angle in degrees, ``(n_tris,)`` float64."""
        return self._native.min_angles()


def _norm_regions(regions):
    """``Region2D`` / tuple input -> the native ``(outer, holes, tag, constraints)`` form."""
    norm = []
    for r in regions:
        if isinstance(r, Region2D):
            outer, tag, holes = r.outer, r.tag, (r.holes or [])
            chains = r.constraints or []
        else:
            outer = r[0]
            tag = r[1] if len(r) > 1 else 1
            holes = r[2] if len(r) > 2 else []
            chains = r[3] if len(r) > 3 else []
        norm.append((
            [list(p) for p in outer],
            [[list(p) for p in hl] for hl in holes],
            int(tag),
            [[list(p) for p in ch] for ch in chains],
        ))
    return norm


def union_regions(regions):
    """Union of 2D regions: abutting or overlapping ones merge into one shape, separate ones
    stay separate. Returns ``(outer, holes)`` per shape, outer CCW and holes CW."""
    return _native.union_regions(_norm_regions(regions))


def overlay_regions(subject, clip, rule: str = "union"):
    """Boolean overlay of two region sets. `rule` is ``"union"``, ``"intersect"`` or
    ``"difference"`` (subject minus clip). Returns ``(outer, holes)`` per shape."""
    return _native.overlay_regions(_norm_regions(subject), _norm_regions(clip), rule)


def mesh_2d(
    regions,
    h: float,
    *,
    min_angle_deg: float | None = None,
    cvt_iters: int | None = None,
    max_passes: int | None = None,
    target_count: int | None = None,
    minh: float | None = None,
    maxh: float | None = None,
    grading: float | None = None,
    band_diagonals: str | None = None,
    width_size: float | None = None,
    snap: float | None = None,
) -> Mesh2D:
    """THE 2D endpoint: mesh tagged 2D polygons into one bundle, the
    canonical 2D / MoM path.

    Parameters
    ----------
    regions : list[Region2D | tuple]
        the tagged 2D regions; each a :class:`Region2D`, or an
        ``(outer, tag, holes)`` tuple (``outer`` a list of ``(x, y)``).
    h : float
        target edge length (uniform sizing field).
    min_angle_deg, cvt_iters, max_passes : optional
        Ruppert min-angle bound, CVT seed iterations, max refinement passes.
    target_count : optional
        triangle BUDGET: ``> 0`` scales the sizing field by one global factor so
        the mesh lands near this count (0 = field-driven).
    minh, maxh : optional
        hard element-size floor / cap, applied after the budget scaling (0 = off).
    grading : optional
        Lipschitz slope of the sizing field (0 = off).
    band_diagonals : optional
        ``"alternate"`` (default): the diagonals of the edge band's cells
        alternate from cell to cell; ``"along"``: they all lean one way along
        the outline.
    width_size : optional
        the size at most this share of the local width (the width of a trace
        at the nearest point of its outline); ``1.0`` gives cells about as
        long as the trace is wide (0 = off, the default).
    snap : optional
        constraint chains snapped together: a chain point closer than this
        share of the size to another chain or to the outline moves onto it,
        and pieces running along another are dropped (0 = off, the default;
        e.g. 0.25 for the outlines of stacked metal layers).
    """
    return Mesh2D(_native.mesh_2d(
        _norm_regions(regions), float(h), min_angle_deg, cvt_iters, max_passes,
        target_count, minh, maxh, grading, band_diagonals, width_size, snap,
    ))


def mesh_layers(
    groups,
    h: float,
    *,
    min_angle_deg: float | None = None,
    cvt_iters: int | None = None,
    max_passes: int | None = None,
    target_count: int | None = None,
    minh: float | None = None,
    maxh: float | None = None,
    grading: float | None = None,
    band_diagonals: str | None = None,
    width_size: float | None = None,
    snap: float | None = None,
) -> list[Mesh2D]:
    """THE grouped 2D endpoint: each ``group`` is one layer's region list
    (same forms as :func:`mesh_2d`). WITHIN a group, abutting/overlapping
    regions weld into one RWG-connected component; regions of different
    groups never merge; a ``target_count > 0`` is ONE triangle budget shared
    across every patch of every group. Returns one :class:`Mesh2D` per
    group, in input order.
    """
    native = _native.mesh_layers(
        [_norm_regions(g) for g in groups], float(h), min_angle_deg, cvt_iters,
        max_passes, target_count, minh, maxh, grading, band_diagonals, width_size, snap,
    )
    return [Mesh2D(m) for m in native]


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
        """Scope on edges by ``id=``/``near=``/``between=``/``kind=``, or all."""
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

    # ------------------------------------------------------------ solids

    def box(self, width: float, depth: float, height: float,
            position=(0, 0, 0), *, maxh: float | None = None, void: bool = False) -> Solid:
        """Axis-aligned box: extents along x, y, z; ``position`` is the
        lower corner. ``void=True`` carves the volume out of everything
        added before it (the cut boolean; the region tag is then 0 and the
        walls become boundary faces)."""
        return _solid(self._native, self._native.add_box([width, depth, height], position, maxh, void))

    def cylinder(self, radius: float, height: float, position=(0, 0, 0), axis=(0, 0, 1), *,
                 segments: int | None = None, uniform: bool = False, rows: int | None = None,
                 maxh: float | None = None, void: bool = False) -> Solid:
        """Cylinder from the base centre ``position`` along ``axis``. The
        barrel is tessellated with ``segments`` chords (24) but carries the
        exact analytic surface: mesh vertices snap onto the true cylinder.

        With ``uniform=True`` the barrel is a structured grid of height
        ``rows`` (auto-chosen for roughly square cells when ``None``) instead of
        full-height strips, for an isotropic surface mesh."""
        return _solid(self._native, self._native.add_cylinder(
            radius, height, position, axis, segments, uniform, rows, maxh, void))

    def sphere(self, radius: float, position=(0, 0, 0), *, segments: int | None = None,
               maxh: float | None = None, void: bool = False) -> Solid:
        """Sphere centred at ``position`` (analytic surface, faceted
        geodesically; the facet density follows the target size,
        ``segments`` is a floor)."""
        return _solid(self._native, self._native.add_sphere(radius, position, segments, maxh, void))

    def icosphere(self, radius: float, position=(0, 0, 0), *, subdivisions: int | None = None,
                  maxh: float | None = None, void: bool = False) -> Solid:
        """Geodesic sphere with a fixed facet level: a subdivided icosahedron
        projected onto the analytic sphere, ``20 * 4**subdivisions`` faces
        (3 by default)."""
        return _solid(self._native, self._native.add_icosphere(radius, position, subdivisions, maxh, void))

    def airfoil_naca0012(self, chord: float, span: float, position=(0, 0, 0),
                         span_axis=(0, 0, 1), *, n_per_side: int | None = None,
                         n_seg: int | None = None, maxh: float | None = None,
                         void: bool = False) -> Solid:
        """A NACA 0012 airfoil (chord along +x, leading edge at ``position``)
        extruded along ``span_axis`` by ``span``. The curved skin is one
        analytic extruded-spline surface; the trailing edge is a flat blunt
        face. ``n_per_side`` (40) controls profile control points, ``n_seg``
        (120) the facet count along the chord."""
        return _solid(self._native, self._native.add_naca0012(
            chord, span, position, span_axis, n_per_side, n_seg, maxh, void))

    def cone(self, r1: float, r2: float, height: float, position=(0, 0, 0), axis=(0, 0, 1), *,
             segments: int | None = None, uniform: bool = False, rows: int | None = None,
             maxh: float | None = None, void: bool = False) -> Solid:
        """Conical frustum: base radius ``r1`` at ``position``, top radius
        ``r2`` (0 for a full cone) at ``position + height * axis``;
        ``uniform`` and ``rows`` as for :meth:`cylinder`."""
        return _solid(self._native, self._native.add_cone(
            r1, r2, height, position, axis, segments, uniform, rows, maxh, void))

    def prism(self, points, height: float, position=(0, 0, 0), *, holes=None,
              maxh: float | None = None, void: bool = False) -> Solid:
        """Right prism: the 2D polygon ``points`` (in the xy plane, offset by
        ``position``) extruded by ``height`` along z."""
        return _solid(self._native, self._native.add_prism(
            [list(p) for p in points], height, position,
            [[list(q) for q in h] for h in holes] if holes else None, maxh, void))

    def torus(self, major_radius: float, minor_radius: float, position=(0, 0, 0),
              axis=(0, 0, 1), *, segments: int | None = None,
              tube_segments: int | None = None, maxh: float | None = None,
              void: bool = False) -> Solid:
        """Torus centred at ``position`` with the donut plane normal to
        ``axis`` (analytic surface)."""
        return _solid(self._native, self._native.add_torus(
            major_radius, minor_radius, position, axis, segments, tube_segments, maxh, void))

    def wedge(self, dx: float, dy: float, dz: float, position=(0, 0, 0), *,
              top_x: float | None = None, maxh: float | None = None,
              void: bool = False) -> Solid:
        """Wedge: a ``dx x dy x dz`` box whose top edge is shortened to
        ``top_x`` along x (0, the default, gives a triangular prism); the
        taper runs in the xz plane."""
        return _solid(self._native, self._native.add_wedge([dx, dy, dz], position, top_x, maxh, void))

    def sweep(self, path, radius: float, *, segments: int | None = None,
              maxh: float | None = None, void: bool = False) -> Solid:
        """Tube with a circular cross-section swept along the open polyline
        ``path``. Sample curved paths finely; the tube radius must stay
        below the local curvature radius."""
        return _solid(self._native, self._native.add_sweep(
            [list(p) for p in path], radius, segments, maxh, void))

    def helix(self, radius: float, pitch: float, turns: float, wire_radius: float,
              position=(0, 0, 0), *, points_per_turn: int | None = None,
              segments: int | None = None, maxh: float | None = None,
              void: bool = False) -> Solid:
        """Helical coil around +z through ``position``: helix ``radius``,
        ``pitch`` advance per turn, round wire of ``wire_radius``."""
        return _solid(self._native, self._native.add_helix(
            radius, pitch, turns, wire_radius, position, points_per_turn, segments, maxh, void))

    def loft(self, profile_a, profile_b, *, maxh: float | None = None,
             void: bool = False) -> Solid:
        """Ruled loft between two planar profiles with the same vertex
        count, corresponded by index (horn tapers). Profiles must be
        star-shaped about their centroid (convex profiles always are)."""
        return _solid(self._native, self._native.add_loft(
            [list(p) for p in profile_a], [list(p) for p in profile_b], maxh, void))

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
        return _solid(self._native, self._native.add_triangles(v.tolist(), t.tolist(), maxh, void))

    def revolve(self, profile, *, position=(0, 0, 0), axis=(0, 0, 1), angle: float = 360.0,
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
                if not edges or edges[-1][0] != "line":
                    raise ValueError("a Spline must follow a vertex with a straight edge")
                edges[-1] = ("spline", 0.0, [list(p) for p in item.points])
                continue
            if len(item) not in (2, 3):
                raise ValueError("a profile vertex is (r, z) or (r, z, bulge)")
            pts.append([float(item[0]), float(item[1])])
            bulge = float(item[2]) if len(item) == 3 else 0.0
            edges.append(("arc", bulge, []) if bulge else ("line", 0.0, []))
        return _solid(self._native, self._native.add_revolve(
            pts, edges, position, axis, float(angle), segments, maxh, void))

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
        return _solid(self._native, self._native.add_import(str(path), crease_deg, up, maxh, void))

    def import_obj(self, path, *, crease_deg: float | None = None, up: str | None = None,
                   maxh: float | None = None, void: bool = False) -> Solid:
        """Solid from a Wavefront OBJ file (``v``/``f`` records; polygons
        are fan-triangulated), with the semantics of :meth:`import_stl`."""
        return _solid(self._native, self._native.add_import(str(path), crease_deg, up, maxh, void))

    def import_step(self, path, *, maxh: float | None = None) -> list[Solid]:
        """The solids of a STEP file (AP203/AP214), one per solid body of
        the file, each with its faces on their true surfaces (planes,
        cylinders, cones, spheres, tori, B-splines): the mesh is measured
        against those, not against a tessellation. Coordinates stay in the
        file's unit."""
        return [_solid(self._native, p) for p in self._native.import_step(str(path), maxh)]

    # ------------------------------------------------------------ sheets

    def xy_plate(self, width: float, height: float, position=(0, 0, 0), *,
                 tag: int = 1, maxh: float | None = None) -> "Sheet":
        """Zero-thickness rectangle in an xy plane (a PEC trace, a port
        marker): spans ``width`` along x and ``height`` along y from the
        corner ``position``; conformally embedded with face tag ``tag``."""
        return Sheet(*self._native.add_sheet_rect(position, [width, 0, 0], [0, height, 0], tag, maxh))

    def xz_plate(self, width: float, height: float, position=(0, 0, 0), *,
                 tag: int = 1, maxh: float | None = None) -> "Sheet":
        """Like :meth:`xy_plate` in an xz plane (width along x, height
        along z)."""
        return Sheet(*self._native.add_sheet_rect(position, [width, 0, 0], [0, 0, height], tag, maxh))

    def yz_plate(self, width: float, height: float, position=(0, 0, 0), *,
                 tag: int = 1, maxh: float | None = None) -> "Sheet":
        """Like :meth:`xy_plate` in a yz plane (width along y, height
        along z)."""
        return Sheet(*self._native.add_sheet_rect(position, [0, width, 0], [0, 0, height], tag, maxh))

    def plate(self, p0, du, dv, *, tag: int = 1, maxh: float | None = None) -> "Sheet":
        """General parallelogram sheet from corner ``p0`` spanned by the
        edge vectors ``du`` and ``dv``."""
        return Sheet(*self._native.add_sheet_rect(p0, du, dv, tag, maxh))

    def disc(self, radius: float, position=(0, 0, 0), axis=(0, 0, 1), *,
             segments: int | None = None, tag: int = 1, maxh: float | None = None) -> "Sheet":
        """Disc sheet centred at ``position``, normal to ``axis``."""
        return Sheet(*self._native.add_sheet_disc(radius, position, axis, tag, segments, maxh))

    def polygon_plate(self, points, position=(0, 0, 0), *, holes=None, tag: int = 1,
                      maxh: float | None = None) -> "Sheet":
        """Polygonal sheet in an xy plane at ``position`` (2D coordinates
        are offset by ``position``'s x, y)."""
        return Sheet(*self._native.add_sheet_polygon(
            [list(p) for p in points], position, tag,
            [[list(q) for q in h] for h in holes] if holes else None, maxh))

    def nurbs_plate(self, ctrl, *, degree=(3, 3), weights=None, knots=None, tag: int = 1,
                    maxh: float | None = None) -> "Sheet":
        """NURBS sheet over the control net ``ctrl`` (shape ``(nu, nv, 3)``,
        first index along u). ``weights`` (shape ``(nu, nv)``, positive)
        default to 1, ``knots`` (a pair of clamped knot vectors) to clamped
        uniform ones. The mesh snaps to the exact surface."""
        c = np.asarray(ctrl, float)
        if c.ndim != 3 or c.shape[2] != 3:
            raise ValueError("ctrl must have shape (nu, nv, 3)")
        w = None if weights is None else np.asarray(weights, float).tolist()
        k = None if knots is None else [list(map(float, knots[0])), list(map(float, knots[1]))]
        return Sheet(*self._native.add_sheet_nurbs(c.tolist(), [int(degree[0]), int(degree[1])], tag, w, k, maxh))

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
        radius_edge: float | None = None,
        max_points: int | None = None,
        grading: float | None = None,
        cells_across: float | None = None,
        tol_edge: float | None = None,
        tol_surf: float | None = None,
        maxh_edge: float | None = None,
        maxh_surf: float | None = None,
        maxh_vol: float | None = None,
        optimize: bool | None = None,
        optimize_passes: int | None = None,
        target_elements: int | None = None,
        min_h_surf: float | None = None,
        min_h_vol: float | None = None,
        bottom_up: bool | None = None,
    ) -> Mesh:
        """Assembles the exact conforming arrangement of every solid and
        sheet, meshes it, and improves the tets.

        Parameters
        ----------
        maxh : float, optional
            global target edge length (defaults to the geometry's;
            unbounded if neither is given)
        radius_edge : float, optional
            Delaunay quality bound (circumradius / shortest edge); the
            provable refinement regime is >= 2.0
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
            wires get proper tets through them. Default ``None``: off for the
            bottom-up mesher (stacks of layers far thinner than the size take
            flat tets through each layer), 1 for the restricted Delaunay one;
            ``0`` turns it off for both
        tol_edge, tol_surf : float
            relative chord (sagitta) tolerance for curved EDGES and SURFACES: an
            entity of radius ``R`` is sized ``h = R * sqrt(8 * tol)``, so the
            chord deviates by at most ``tol * R``. Default: the geometry's
            (5e-2, about ten segments round a circle). There is no volume
            tolerance: the volume follows the surface.
        maxh_edge, maxh_surf, maxh_vol : float
            maximum element edge length per dimension, each combined with
            ``maxh`` as ``min(maxh, maxh_dim)``. Default: the geometry's (inf).
        optimize : bool, optional
            also run the legacy quality optimizer after the mesher's own
            improvement (slower; not with periodic faces)
        optimize_passes : int, optional
            cap the number of optimization passes
        target_elements : int, optional
            element (tet) budget: the global size scale is retuned over a few
            remeshes so the tet count lands near it, while the relative
            refinement keeps its shape
        min_h_surf, min_h_vol : float, optional
            hard minimum element size on surfaces and in the volume (0 off)
        bottom_up : bool, optional
            the mesher. ``None`` (default): bottom-up (each face alone on the
            shared samples of its edges, then each region by its constrained
            Delaunay tetrahedralization), falling back to restricted Delaunay
            refinement where it fails (logged as a warning). ``True``:
            bottom-up only; ``False``: restricted Delaunay refinement
        """
        return Mesh(self._native.mesh(
            maxh, radius_edge, max_points, grading, cells_across, tol_edge, tol_surf,
            maxh_edge, maxh_surf, maxh_vol, optimize, optimize_passes, target_elements,
            min_h_surf, min_h_vol, bottom_up,
        ))

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
        bottom_up: bool | None = None,
    ) -> SurfaceMesh:
        """Surface-only export: assembles the exact arrangement and meshes
        only its surfaces (region interfaces, outer boundary, embedded
        sheets), with the full sizing hierarchy of :meth:`mesh`.
        ``target_triangles`` is a triangle budget: the refinement stops once
        it is reached. ``bottom_up`` chooses the mesher as in :meth:`mesh`
        (a triangle budget falls back to restricted Delaunay refinement)."""
        return SurfaceMesh(self._native.surface_mesh(
            maxh, grading, tol_edge, tol_surf, maxh_edge, maxh_surf, maxh_vol, target_triangles,
            bottom_up,
        ))

"""Geometry builder in the rapidfem style.

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
All coordinates are unitless; use one consistent unit (rapidfem: metres).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _native


@dataclass(frozen=True)
class Solid:
    """Handle to a solid added to a :class:`Geometry`: its ``region`` tag
    identifies the solid's tets in :attr:`Mesh.tet_regions`, its ``index``
    (insertion order, voids included) identifies the solid's surfaces in
    :attr:`Mesh.surface_owners`. Voids share ``region`` 0 but keep a unique
    ``index``, so their walls stay addressable."""

    region: int
    index: int


class Mesh:
    """A finished tetrahedral mesh (numpy views over the native result).

    Attributes
    ----------
    points : (n_points, 3) float64
        vertex coordinates
    tets : (n_tets, 4) uint64
        positively oriented tetrahedra as point indices
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
    surface_owners : (n_surfaces,) uint32
        owner solid per surface id (:attr:`Solid.index`); the max uint32
        marks embedded-sheet surfaces
    edges : (n_edges, 2) uint64
        feature (crease) edges of the surface mesh; facet seams of curved
        analytic surfaces are not included
    stats : dict
        n_points, n_tets, n_faces, min_dihedral_deg, max_radius_edge,
        max_edge, millis
    """

    def __init__(
        self,
        native,
        solids: list[dict] | None = None,
        tag_labels: dict[int, str] | None = None,
    ) -> None:
        self._native = native
        #: per input solid (insertion order): {"region": int, "label": str|None}
        self.solids: list[dict] = solids or []
        #: display label per sheet tag
        self.tag_labels: dict[int, str] = tag_labels or {}
        self.points: np.ndarray = native.points()
        self.tets: np.ndarray = native.tets()
        self.tet_regions: np.ndarray = native.tet_regions()
        self.faces: np.ndarray = native.faces()
        self.face_tags: np.ndarray = native.face_tags()
        self.face_regions: np.ndarray = native.face_regions()
        self.face_surfaces: np.ndarray = native.face_surfaces()
        self.surface_owners: np.ndarray = native.surface_owners()
        self.edges: np.ndarray = native.edges()
        self.stats: dict = native.stats()
        #: per-stage wall-clock seconds, pipeline order (assemble.* / mesh.* /
        #: optimize.*), e.g. timings["mesh.faces"], timings["mesh.refine"]
        self.timings: dict = native.timings()
        #: detailed named statistics (predicate calls, recovery work, counts,
        #: quality), e.g. metrics["predicates.orient3d_exact"]
        self.metrics: dict = native.metrics()
        #: ordered log events, each {level, stage, message, at} (at = seconds
        #: since the run started; level in info/warn/error)
        self.log: list[dict] = native.log()
        #: quality with location: min_dihedral_deg, worst_location (xyz),
        #: worst_region, max_radius_edge, max_edge, and a per-region breakdown
        #: under "regions"
        self.quality: dict = native.quality()
        #: full diagnostics with located defects: dihedral histogram, slivers,
        #: watertight, non-manifold edges, surface deviation, region volumes, and a
        #: "defects" list of {kind, pos:[x,y,z], value}
        self.diagnostics: dict = native.diagnostics()

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"Mesh({s['n_tets']} tets, {s['n_points']} points, "
            f"min dihedral {s['min_dihedral_deg']:.1f} deg, "
            f"{s['millis']} ms)"
        )

    @property
    def warnings(self) -> list[dict]:
        """Log events at warn/error level (divergence backstops, budget caps,
        slivers)."""
        return [e for e in self.log if e["level"] in ("warn", "error")]

    def log_text(self) -> str:
        """The log as a human-readable string, one event per line."""
        return "\n".join(
            f"[{e['at']:8.3f}s {e['level']:>5} {e['stage']}] {e['message']}"
            for e in self.log
        )

    def report(self) -> str:
        """A full human-readable report: per-stage timings, key metrics, the
        worst-quality location and per-region quality, and any warnings. Use
        ``print(mesh.report())`` to see what happened, how long each stage took,
        and where the mesh quality is worst."""
        q = self.quality
        lines: list[str] = [repr(self), "", "timings (s):"]
        for stage, secs in self.timings.items():
            lines.append(f"  {stage:<22} {secs:8.3f}")
        lines.append("")
        lines.append("quality:")
        loc = q["worst_location"]
        lines.append(
            f"  min dihedral {q['min_dihedral_deg']:.2f} deg in region "
            f"{q['worst_region']} near ({loc[0]:.4g}, {loc[1]:.4g}, {loc[2]:.4g})"
        )
        lines.append(f"  max radius/edge {q['max_radius_edge']:.2f}")
        for r in q["regions"]:
            lines.append(
                f"  region {r['region']:<3} min dihedral {r['min_dihedral_deg']:6.2f} deg "
                f"({r['n_tets']} tets)"
            )
        warn = self.warnings
        if warn:
            lines.append("")
            lines.append("warnings:")
            for e in warn:
                lines.append(f"  [{e['stage']}] {e['message']}")
        return "\n".join(lines)

    def to_viewer_dict(self, name: str) -> dict:
        """The mesh in the viewer JSON schema (shared by the comparison
        viewer and the showcase site)."""
        return {
            "name": name,
            "mesher": "rapidmesh",
            "points": self.points.tolist(),
            "tets": self.tets.astype(int).tolist(),
            "tet_regions": self.tet_regions.astype(int).tolist(),
            "faces": [
                {
                    "tri": [int(a), int(b), int(c)],
                    "tag": int(t),
                    "regions": [int(r0), int(r1)],
                    "surface": int(s),
                }
                for (a, b, c), t, (r0, r1), s in zip(
                    self.faces, self.face_tags, self.face_regions,
                    self.face_surfaces,
                )
            ],
            # owner solid per analytic surface id; -1 marks embedded sheets
            "surface_owners": [
                -1 if int(o) == 0xFFFFFFFF else int(o)
                for o in self.surface_owners
            ],
            # input solids in insertion order with optional display labels
            "solids": self.solids,
            "tag_labels": {str(t): n for t, n in self.tag_labels.items()},
            "edges": self.edges.astype(int).tolist(),
            "stats": {
                "n_points": int(self.stats["n_points"]),
                "n_tets": int(self.stats["n_tets"]),
                "min_dihedral_deg": float(self.stats["min_dihedral_deg"]),
                "max_radius_edge": float(self.stats["max_radius_edge"]),
                "max_edge": float(self.stats["max_edge"]),
                "millis": int(self.stats["millis"]),
            },
            # located mesh defects for the viewer overlay: each {kind, pos, value}
            "defects": self.diagnostics["defects"],
        }

    def save_viewer_json(self, name: str, directory: str | Path) -> Path:
        """Writes ``rapidmesh_<name>.json`` in the comparison-viewer schema
        and refreshes the viewer manifest. Returns the written path."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"rapidmesh_{name}.json"
        path.write_text(json.dumps(self.to_viewer_dict(name)))
        _refresh_manifest(directory)
        return path


class SurfaceMesh:
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
    surface_owners : (n_surfaces,) uint32
        owner solid per surface id; the max uint32 marks embedded sheets
    stats : dict
        n_points, n_faces, millis
    """

    def __init__(
        self,
        native,
        solids: list[dict] | None = None,
        tag_labels: dict[int, str] | None = None,
    ) -> None:
        self._native = native
        self.solids: list[dict] = solids or []
        self.tag_labels: dict[int, str] = tag_labels or {}
        self.points: np.ndarray = native.points()
        self.faces: np.ndarray = native.faces()
        self.face_tags: np.ndarray = native.face_tags()
        self.face_regions: np.ndarray = native.face_regions()
        self.face_surfaces: np.ndarray = native.face_surfaces()
        self.surface_owners: np.ndarray = native.surface_owners()
        self.stats: dict = native.stats()
        #: per-stage wall-clock seconds, pipeline order
        self.timings: dict = native.timings()

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"SurfaceMesh({s['n_faces']} faces, {s['n_points']} points, "
            f"{s['millis']} ms)"
        )


def _refresh_manifest(directory: Path) -> None:
    """Manifest = every geometry with a rapidmesh JSON present, canonical
    comparison scenes first (mirrors the Rust exporter's write_manifest)."""
    canonical = [
        "em_scene",
        "via",
        "microstrip",
        "sphere",
        "l_prism",
        "density_transition",
    ]
    names = sorted(
        p.stem.removeprefix("rapidmesh_")
        for p in directory.glob("rapidmesh_*.json")
    )
    names.sort(key=lambda n: (canonical.index(n) if n in canonical else len(canonical), n))
    (directory / "manifest.json").write_text(json.dumps(names))


def _position_with_center_alias(position, center, *, what):
    if center is not None:
        if position != (0, 0, 0):
            raise ValueError(f"{what}: pass either position or center, not both")
        return center, True
    return position, False


def _filt(sel, kw):
    """A selector descriptor: ``None`` (unfiltered) or a dict of criteria."""
    if sel is None and not kw:
        return None
    f = dict(kw)
    if sel is not None:
        f["id"] = sel
    return f


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


@dataclass
class _Face:
    id: int
    centroid: tuple
    normal: tuple
    area: float
    regions: tuple
    tag: int
    surface: int
    owner: int
    edges: list


@dataclass
class _Edge:
    id: int
    p0: tuple
    p1: tuple
    midpoint: tuple
    length: float
    kind: int
    faces: list


@dataclass
class _Topology:
    regions: list
    faces: list  # list[_Face]
    edges: list  # list[_Edge]


class _Scope:
    """A hierarchical sizing scope built by ``g.region(..).surf(..).edge(..)``.

    Setting ``.maxh`` / ``.tol`` applies to every entity the scope selects:
    unfiltered at a dimension sets the global per-dimension knob, a filtered
    scope sets per-entity overrides (the most specific scope wins, because a
    per-entity override beats the dimension default in the mesher).
    """

    __slots__ = ("_g", "_level", "_rf", "_ff", "_ef")

    def __init__(self, g, level, rf=None, ff=None, ef=None):
        self._g, self._level, self._rf, self._ff, self._ef = g, level, rf, ff, ef

    def region(self, sel=None, **kw):
        return _Scope(self._g, "region", _filt(sel, kw), self._ff, self._ef)

    def surf(self, sel=None, **kw):
        return _Scope(self._g, "surf", self._rf, _filt(sel, kw), self._ef)

    def edge(self, sel=None, **kw):
        return _Scope(self._g, "edge", self._rf, self._ff, _filt(sel, kw))

    @property
    def maxh(self):
        raise AttributeError("a sizing scope is write-only")

    @maxh.setter
    def maxh(self, v):
        self._g._apply_scope(self, "maxh", float(v))

    @property
    def tol(self):
        raise AttributeError("a sizing scope is write-only")

    @tol.setter
    def tol(self, v):
        self._g._apply_scope(self, "tol", float(v))


class Geometry:
    """Top-level geometry builder (the rapidmesh analog of
    ``rapidfem.Geometry``, without materials and physics: regions carry
    integer tags that downstream tools map to materials).

    Parameters
    ----------
    maxh : float, optional
        global target tet edge length, used by :meth:`mesh` when no
        override is passed (sizing is a target with a documented
        1.5x max-edge contract, like gmsh's mesh size)
    grading : float
        default size-grading Lipschitz constant for :meth:`mesh` (see
        there); 0.5 grows neighbor elements by roughly 1.5x
    """

    def __init__(self, *, maxh: float | None = None, grading: float = 0.5) -> None:
        self._builder = _native.SceneBuilder(maxh)
        self._maxh = maxh
        self._grading = grading
        self._face_maxh: dict[int, float] = {}
        self._surface_maxh: dict[int, float] = {}
        self._size_points: list[tuple[tuple[float, float, float], float]] = []
        self._n_solids = 0
        self._solid_regions: list[int] = []
        self._solid_labels: dict[int, str] = {}
        self._tag_labels: dict[int, str] = {}
        # Hierarchical per-entity sizing (set via g.region(..).surf(..).edge(..)).
        # Global per-dimension defaults plus sparse per-brep-id overrides.
        self._tol_edge = 1e-2
        self._tol_surf = 1e-2
        self._maxh_edge = math.inf
        self._maxh_surf = math.inf
        self._maxh_vol = math.inf
        self._edge_maxh: dict[int, float] = {}
        self._edge_tol: dict[int, float] = {}
        self._surf_maxh: dict[int, float] = {}
        self._surf_tol: dict[int, float] = {}
        self._region_maxh: dict[int, float] = {}
        self._topo_cache: _Topology | None = None

    def _solid(self, region: int) -> Solid:
        idx = self._n_solids
        self._n_solids += 1
        self._solid_regions.append(region)
        self._topo_cache = None  # geometry changed: stale topology ids
        return Solid(region, idx)

    # ---- hierarchical per-entity sizing -----------------------------------
    # g.maxh = ...                      # global size (all dimensions)
    # g.edge().tol = ...                # all edges (== tol_edge)
    # g.surf(normal=(0,0,1)).maxh = ... # selected surfaces
    # g.region("diel").surf().edge(near=p).maxh = ...  # specific edges

    @property
    def maxh(self) -> float | None:
        """Global target edge length (all dimensions)."""
        return self._maxh

    @maxh.setter
    def maxh(self, v: float) -> None:
        self._maxh = float(v)

    @property
    def tol(self) -> float:
        raise AttributeError("write-only; sets both edge and surface tolerance")

    @tol.setter
    def tol(self, v: float) -> None:
        self._tol_edge = self._tol_surf = float(v)

    def region(self, sel=None, **kw) -> _Scope:
        """Scope on a region (material) by tag/id, or all regions if unfiltered."""
        return _Scope(self, "region", _filt(sel, kw))

    def surf(self, sel=None, **kw) -> _Scope:
        """Scope on surfaces by ``id=``/``tag=``/``normal=``/``near=``, or all."""
        return _Scope(self, "surf", None, _filt(sel, kw))

    def edge(self, sel=None, **kw) -> _Scope:
        """Scope on edges by ``id=``/``near=``/``between=``/``kind=``, or all."""
        return _Scope(self, "edge", None, None, _filt(sel, kw))

    def _topology(self) -> _Topology:
        if self._topo_cache is None:
            t = self._builder.topology()
            faces = [
                _Face(i, tuple(c), tuple(n), a, (rf, rb), tag, surf, owner, list(edges))
                for i, (c, n, a, rf, rb, tag, surf, owner, edges) in enumerate(t.faces())
            ]
            edges = [
                _Edge(i, tuple(p0), tuple(p1), tuple(mid), ln, kind, list(fs))
                for i, (p0, p1, mid, ln, kind, fs) in enumerate(t.edges())
            ]
            self._topo_cache = _Topology(list(t.regions), faces, edges)
        return self._topo_cache

    @staticmethod
    def _region_ok(face: _Face, rf) -> bool:
        if rf is None:
            return True
        want = rf.get("id", rf.get("tag"))
        return want is None or want in face.regions

    @staticmethod
    def _face_ok(face: _Face, ff) -> bool:
        if ff is None:
            return True
        if "id" in ff and face.id != ff["id"]:
            return False
        if "tag" in ff and face.tag != ff["tag"]:
            return False
        if "normal" in ff:
            n = ff["normal"]
            nl = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5 or 1.0
            d = (face.normal[0] * n[0] + face.normal[1] * n[1] + face.normal[2] * n[2]) / nl
            if d < ff.get("normal_tol", 0.9):
                return False
        return True

    @staticmethod
    def _edge_ok(edge: _Edge, ef, topo: _Topology) -> bool:
        if ef is None:
            return True
        if "id" in ef and edge.id != ef["id"]:
            return False
        if "kind" in ef and edge.kind != ef["kind"]:
            return False
        if "between" in ef:
            a, b = ef["between"]
            regs = set()
            for fid in edge.faces:
                regs.update(topo.faces[fid].regions)
            if not ({a, b} <= regs):
                return False
        return True

    @staticmethod
    def _apply_near(items, filt, pos):
        """If `filt` carries ``near=point``, keep only the single closest item."""
        if filt is None or "near" not in filt:
            return items
        p = filt["near"]
        return [min(items, key=lambda it: _d2(pos(it), p))] if items else []

    def _resolve(self, scope: _Scope) -> list:
        topo = self._topology()
        rf, ff, ef = scope._rf, scope._ff, scope._ef
        if scope._level == "region":
            want = rf.get("id", rf.get("tag")) if rf else None
            return [t for t in topo.regions if want is None or t == want]
        if scope._level == "surf":
            faces = [f for f in topo.faces if self._region_ok(f, rf) and self._face_ok(f, ff)]
            faces = self._apply_near(faces, ff, lambda f: f.centroid)
            return [f.id for f in faces]
        edges = []
        for e in topo.edges:
            incident = [topo.faces[fid] for fid in e.faces]
            if rf is not None and not any(self._region_ok(f, rf) for f in incident):
                continue
            if ff is not None and not any(self._face_ok(f, ff) for f in incident):
                continue
            if not self._edge_ok(e, ef, topo):
                continue
            edges.append(e)
        edges = self._apply_near(edges, ef, lambda e: e.midpoint)
        return [e.id for e in edges]

    def _apply_scope(self, scope: _Scope, what: str, value: float) -> None:
        unfiltered = scope._rf is None and scope._ff is None and scope._ef is None
        if unfiltered:
            # A bare dimension scope sets the global per-dimension knob.
            if scope._level == "edge":
                if what == "maxh":
                    self._maxh_edge = value
                else:
                    self._tol_edge = value
            elif scope._level == "surf":
                if what == "maxh":
                    self._maxh_surf = value
                else:
                    self._tol_surf = value
            else:  # region
                if what != "maxh":
                    raise ValueError("regions have no tolerance (volume follows the surface)")
                self._maxh_vol = value
            return
        for eid in self._resolve(scope):
            if scope._level == "edge":
                (self._edge_maxh if what == "maxh" else self._edge_tol)[eid] = value
            elif scope._level == "surf":
                (self._surf_maxh if what == "maxh" else self._surf_tol)[eid] = value
            else:  # region
                if what != "maxh":
                    raise ValueError("regions have no tolerance (volume follows the surface)")
                self._region_maxh[eid] = value

    def label(self, target: Solid | int, name: str) -> None:
        """Display name for viewer exports: for a :class:`Solid`, the name
        of its surface group (voids without a label merge into a generic
        cavity group); for an int sheet tag, the name of that sheet group."""
        if isinstance(target, Solid):
            self._solid_labels[target.index] = name
        else:
            self._tag_labels[int(target)] = name

    def union(self, *solids: Solid) -> Solid:
        """Fuse overlapping solids into ONE material (a boolean union): the
        internal boundaries between them become same-region faces and are
        dropped at assembly, leaving the outer union surface. Returns the
        first solid, now representing the merged region."""
        if not solids:
            raise ValueError("union needs at least one solid")
        keep = solids[0]
        for s in solids[1:]:
            self._builder.merge_regions(keep.region, s.region)
            for i, r in enumerate(self._solid_regions):
                if r == s.region:
                    self._solid_regions[i] = keep.region
        return keep

    # ------------------------------------------------------------ solids

    def box(
        self,
        width: float,
        depth: float,
        height: float,
        position: tuple[float, float, float] = (0, 0, 0),
        *,
        maxh: float | None = None,
        void: bool = False,
    ) -> Solid:
        """Axis-aligned box: extents along x, y, z; ``position`` is the
        lower corner. ``void=True`` carves the volume out of everything
        added before it (the cut boolean; the region tag is then 0 and the
        walls become boundary faces)."""
        x, y, z = position
        region = self._builder.add_box(
            [x, y, z], [x + width, y + depth, z + height], maxh, void
        )
        return self._solid(region)

    def cylinder(
        self,
        radius: float,
        height: float,
        position: tuple[float, float, float] = (0, 0, 0),
        axis: tuple[float, float, float] = (0, 0, 1),
        *,
        segments: int = 24,
        uniform: bool = False,
        rows: int | None = None,
        maxh: float | None = None,
        void: bool = False,
    ) -> Solid:
        """Cylinder from the base centre ``position`` along ``axis``. The
        barrel is tessellated with ``segments`` chords but carries the exact
        analytic surface: mesh vertices snap onto the true cylinder.

        With ``uniform=True`` the barrel is a structured grid of height
        ``rows`` (auto-chosen for roughly square cells when ``None``) instead of
        full-height strips, giving an isotropic, evenly distributed surface mesh
        like gmsh / tetgen (see :meth:`icosphere`)."""
        ax = [a * height for a in _unit(axis)]
        if uniform:
            if rows is None:
                circ = 2 * math.pi * radius / max(segments, 1)
                rows = max(1, round(height / circ)) if circ > 0 else 1
            region = self._builder.add_cylinder_iso(
                list(position), ax, radius, segments, rows, maxh, void
            )
        else:
            region = self._builder.add_cylinder(
                list(position), ax, radius, segments, maxh, void
            )
        return self._solid(region)

    def sphere(
        self,
        radius: float,
        position: tuple[float, float, float] = (0, 0, 0),
        *,
        segments: int = 24,
        rings: int = 12,
        maxh: float | None = None,
        void: bool = False,
    ) -> Solid:
        """Sphere centred at ``position`` (analytic surface, like
        :meth:`cylinder`)."""
        region = self._builder.add_sphere(
            list(position), radius, segments, rings, maxh, void
        )
        return self._solid(region)

    def icosphere(
        self,
        radius: float,
        position: tuple[float, float, float] = (0, 0, 0),
        *,
        subdivisions: int = 3,
        maxh: float | None = None,
        void: bool = False,
    ) -> Solid:
        """Geodesic sphere: a subdivided icosahedron projected onto the
        analytic sphere. Distributes near-equilateral triangles isotropically
        over the hull (no latitude rings / pole clustering like
        :meth:`sphere`), matching how gmsh and tetgen tessellate a sphere; the
        analytic surface is preserved so vertices still snap onto it.
        ``subdivisions`` sets density (face count ``20 * 4**subdivisions``)."""
        region = self._builder.add_icosphere(
            list(position), radius, subdivisions, maxh, void
        )
        return self._solid(region)

    def airfoil_naca0012(
        self,
        chord: float,
        span: float,
        position: tuple[float, float, float] = (0, 0, 0),
        span_axis: tuple[float, float, float] = (0, 0, 1),
        *,
        n_per_side: int = 40,
        n_seg: int = 120,
        maxh: float | None = None,
        void: bool = False,
    ) -> Solid:
        """A NACA 0012 airfoil (chord along +x, leading edge at ``position``)
        extruded along ``span_axis`` by ``span``. The curved skin is one
        analytic extruded-spline surface, so the surface mesher places vertices
        exactly on it and grades by the profile curvature; the trailing edge is
        a flat blunt face. ``n_per_side`` controls profile control points,
        ``n_seg`` the facet count along the chord."""
        region = self._builder.add_naca0012(
            chord, span, list(position), list(span_axis), n_per_side, n_seg, maxh, void
        )
        return self._solid(region)

    def cone(
        self,
        r1: float,
        r2: float,
        height: float,
        position: tuple[float, float, float] = (0, 0, 0),
        axis: tuple[float, float, float] = (0, 0, 1),
        *,
        segments: int = 24,
        uniform: bool = False,
        rows: int | None = None,
        maxh: float | None = None,
        void: bool = False,
    ) -> Solid:
        """Conical frustum: base radius ``r1`` at ``position``, top radius
        ``r2`` (0 for a full cone) at ``position + height * axis``.

        With ``uniform=True`` the barrel is a structured grid of height
        ``rows`` (auto-chosen for roughly square cells) for an isotropic
        surface like gmsh / tetgen (see :meth:`cylinder`)."""
        ax = [a * height for a in _unit(axis)]
        if uniform:
            if rows is None:
                r_mean = 0.5 * (r1 + r2)
                circ = 2 * math.pi * r_mean / max(segments, 1)
                rows = max(1, round(height / circ)) if circ > 0 else 1
            region = self._builder.add_frustum_iso(
                list(position), ax, r1, r2, segments, rows, maxh, void
            )
        else:
            region = self._builder.add_frustum(
                list(position), ax, r1, r2, segments, maxh, void
            )
        return self._solid(region)

    def prism(
        self,
        points: list[tuple[float, float]],
        height: float,
        position: tuple[float, float, float] = (0, 0, 0),
        *,
        holes: list[list[tuple[float, float]]] | None = None,
        maxh: float | None = None,
        void: bool = False,
    ) -> Solid:
        """Right prism: the 2D polygon ``points`` (in the xy plane, offset by
        ``position``) extruded by ``height`` along z."""
        region = self._builder.add_prism(
            [list(p) for p in points],
            [[list(q) for q in h] for h in (holes or [])],
            list(position),
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, height],
            maxh,
            void,
        )
        return self._solid(region)

    def torus(
        self,
        major_radius: float,
        minor_radius: float,
        position: tuple[float, float, float] = (0, 0, 0),
        axis: tuple[float, float, float] = (0, 0, 1),
        *,
        segments: int = 32,
        tube_segments: int = 16,
        maxh: float | None = None,
        void: bool = False,
    ) -> Solid:
        """Torus centred at ``position`` with the donut plane normal to
        ``axis`` (analytic surface: vertices snap onto the true torus)."""
        region = self._builder.add_torus(
            list(position),
            list(_unit(axis)),
            major_radius,
            minor_radius,
            segments,
            tube_segments,
            maxh,
            void,
        )
        return self._solid(region)

    def wedge(
        self,
        dx: float,
        dy: float,
        dz: float,
        position: tuple[float, float, float] = (0, 0, 0),
        *,
        top_x: float = 0.0,
        maxh: float | None = None,
        void: bool = False,
    ) -> Solid:
        """Wedge: a ``dx x dy x dz`` box whose top edge is shortened to
        ``top_x`` along x (0 gives a triangular prism); the taper runs in
        the xz plane."""
        region = self._builder.add_wedge(
            list(position), dx, dy, dz, top_x, maxh, void
        )
        return self._solid(region)

    def sweep(
        self,
        path: list[tuple[float, float, float]],
        radius: float,
        *,
        segments: int = 16,
        maxh: float | None = None,
        void: bool = False,
    ) -> Solid:
        """Tube with a circular cross-section swept along the open
        polyline ``path`` (rapidfem's ``sweep_along_path``). Sample curved
        paths finely; the tube radius must stay below the local curvature
        radius."""
        region = self._builder.add_pipe(
            [list(p) for p in path], radius, segments, maxh, void
        )
        return self._solid(region)

    def helix(
        self,
        radius: float,
        pitch: float,
        turns: float,
        wire_radius: float,
        position: tuple[float, float, float] = (0, 0, 0),
        *,
        points_per_turn: int = 24,
        segments: int = 12,
        maxh: float | None = None,
        void: bool = False,
    ) -> Solid:
        """Helical coil around +z through ``position``: helix ``radius``,
        ``pitch`` advance per turn, round wire of ``wire_radius``."""
        region = self._builder.add_helix(
            list(position),
            radius,
            pitch,
            turns,
            wire_radius,
            points_per_turn,
            segments,
            maxh,
            void,
        )
        return self._solid(region)

    def loft(
        self,
        profile_a: list[tuple[float, float, float]],
        profile_b: list[tuple[float, float, float]],
        *,
        maxh: float | None = None,
        void: bool = False,
    ) -> Solid:
        """Ruled loft between two planar profiles with the same vertex
        count, corresponded by index (horn tapers). Profiles must be
        star-shaped about their centroid (convex profiles always are)."""
        region = self._builder.add_loft(
            [list(p) for p in profile_a],
            [list(p) for p in profile_b],
            maxh,
            void,
        )
        return self._solid(region)

    def mesh_solid(
        self,
        verts,
        tris,
        *,
        maxh: float | None = None,
        void: bool = False,
    ) -> Solid:
        """Solid from an externally supplied triangle soup: ``verts`` is an
        ``(n, 3)`` array of vertex coordinates and ``tris`` an ``(m, 3)``
        array of triangle vertex indices (an imported STL surface, a
        marching-cubes iso-surface, ...). The surface must be closed and
        non-self-intersecting; the winding is normalized to outward
        internally. The triangles ARE the surface (no analytic back-reference,
        so fidelity snapping is off): sample organic shapes finely."""
        v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        t = np.asarray(tris, dtype=np.uint32).reshape(-1, 3)
        region = self._builder.add_mesh(
            v.tolist(), t.tolist(), maxh, void
        )
        return self._solid(region)

    # ------------------------------------------------------------ sheets

    def xy_plate(
        self,
        width: float,
        height: float,
        position: tuple[float, float, float] = (0, 0, 0),
        *,
        tag: int = 1,
        maxh: float | None = None,
    ) -> None:
        """Zero-thickness rectangle in an xy plane (a PEC trace, a port
        marker): spans ``width`` along x and ``height`` along y from the
        corner ``position``; conformally embedded with face tag ``tag``."""
        self._register_face_maxh(tag, maxh)
        self._builder.add_sheet_rect(
            list(position), [width, 0.0, 0.0], [0.0, height, 0.0], tag
        )

    def xz_plate(
        self,
        width: float,
        height: float,
        position: tuple[float, float, float] = (0, 0, 0),
        *,
        tag: int = 1,
        maxh: float | None = None,
    ) -> None:
        """Like :meth:`xy_plate` in an xz plane (width along x, height
        along z)."""
        self._register_face_maxh(tag, maxh)
        self._builder.add_sheet_rect(
            list(position), [width, 0.0, 0.0], [0.0, 0.0, height], tag
        )

    def yz_plate(
        self,
        width: float,
        height: float,
        position: tuple[float, float, float] = (0, 0, 0),
        *,
        tag: int = 1,
        maxh: float | None = None,
    ) -> None:
        """Like :meth:`xy_plate` in a yz plane (width along y, height
        along z)."""
        self._register_face_maxh(tag, maxh)
        self._builder.add_sheet_rect(
            list(position), [0.0, width, 0.0], [0.0, 0.0, height], tag
        )

    def plate(
        self,
        p0: tuple[float, float, float],
        du: tuple[float, float, float],
        dv: tuple[float, float, float],
        *,
        tag: int = 1,
        maxh: float | None = None,
    ) -> None:
        """General parallelogram sheet from corner ``p0`` spanned by the
        edge vectors ``du`` and ``dv``."""
        self._register_face_maxh(tag, maxh)
        self._builder.add_sheet_rect(list(p0), list(du), list(dv), tag)

    def disc(
        self,
        radius: float,
        position: tuple[float, float, float] = (0, 0, 0),
        axis: tuple[float, float, float] = (0, 0, 1),
        *,
        segments: int = 24,
        tag: int = 1,
        maxh: float | None = None,
    ) -> None:
        """Disc sheet centred at ``position``, normal to ``axis``."""
        self._register_face_maxh(tag, maxh)
        e1, e2 = _disc_basis(axis)
        self._builder.add_sheet_disk(
            list(position),
            [radius * c for c in e1],
            [radius * c for c in e2],
            segments,
            tag,
        )

    def polygon_plate(
        self,
        points: list[tuple[float, float]],
        position: tuple[float, float, float] = (0, 0, 0),
        *,
        holes: list[list[tuple[float, float]]] | None = None,
        tag: int = 1,
        maxh: float | None = None,
    ) -> None:
        """Polygonal sheet in an xy plane at ``position`` (2D coordinates
        are offset by ``position``'s x, y)."""
        self._register_face_maxh(tag, maxh)
        self._builder.add_sheet_polygon(
            [list(p) for p in points],
            [[list(q) for q in h] for h in (holes or [])],
            list(position),
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            tag,
        )

    # ------------------------------------------------------------ sizing

    def _register_face_maxh(self, tag: int, maxh: float | None) -> None:
        if maxh is None:
            return
        cur = self._face_maxh.get(tag)
        self._face_maxh[tag] = maxh if cur is None else min(cur, maxh)

    def refine_surface(self, solid: Solid, h: float) -> None:
        """Per-solid surface sizing: the solid's boundary patches mesh at
        ``h`` and the size recovers along the grading into the surrounding
        volume. The only sizing handle that reaches a VOID's walls (a coax
        inner conductor has no region and no face tag); for field
        concentrations on conductor surfaces generally."""
        cur = self._surface_maxh.get(solid.index)
        self._surface_maxh[solid.index] = h if cur is None else min(cur, h)

    def refine_near_points(
        self,
        points: list[tuple[float, float, float]],
        h: float,
    ) -> None:
        """Registers point size sources: the edge-length target shrinks to
        ``h`` at each point and recovers along the grading away from it
        (rapidfem's ``refine_near_points``; the hook for error-driven
        adaptive refinement)."""
        for pt in points:
            self._size_points.append((tuple(float(c) for c in pt), float(h)))

    # ------------------------------------------------------------- mesh

    def mesh(
        self,
        *,
        maxh: float | None = None,
        radius_edge: float = 2.0,
        max_points: int = 500_000,
        grading: float | None = None,
        density_weighted: bool = False,
        tol_edge: float | None = None,
        tol_surf: float | None = None,
        maxh_edge: float | None = None,
        maxh_surf: float | None = None,
        maxh_vol: float | None = None,
        optimize: bool = True,
        optimize_passes: int | None = None,
        target_elements: int | None = None,
        min_h_surf: float = 0.0,
        min_h_vol: float = 0.0,
    ) -> Mesh:
        """Assembles the exact conforming arrangement of every solid and
        sheet, meshes it, and runs quality optimization.

        Parameters
        ----------
        maxh : float, optional
            global target edge length (defaults to the constructor's;
            unbounded if neither is given)
        radius_edge : float
            Delaunay quality bound (circumradius / shortest edge); the
            provable refinement regime is >= 2.0
        max_points : int
            best-effort refinement point budget
        grading : float
            size-grading Lipschitz constant: the edge-length target may grow
            by at most this factor per unit distance from finer features
            (0.5 means neighbor elements grow by roughly 1.5x; math.inf
            disables grading and sizes jump at region interfaces)
        density_weighted : bool
            density-weighted CVT: relax sites weighted by ``1/h^d`` so a graded
            sizing field settles into a smooth gradient. Off by default (it
            shifts sites near curved boundaries, which the exact-volume path
            cannot absorb); enable for adaptive (curvature/error-bound) meshes.
        tol_edge, tol_surf : float
            relative chord (sagitta) tolerance for curved EDGES and SURFACES: an
            entity of radius ``R`` is sized ``h = R * sqrt(8 * tol)``, so the
            chord deviates by at most ``tol * R`` (scale-invariant, ~constant
            facets per arc). Default 1e-2 (1%); 1e-4 is very fine. There is no
            volume tolerance: the volume size follows from the surface.
        maxh_edge, maxh_surf, maxh_vol : float
            maximum element edge length per dimension (edges / surfaces /
            volume), each combined with ``maxh`` as ``min(maxh, maxh_dim)``.
            Default ``inf`` (only ``maxh`` applies).
        optimize : bool
            run the quality-optimization pass (default ``True``); ``False``
            returns the raw mesh (replaces ``RAPIDMESH_SKIP_OPTIMIZE``).
        optimize_passes : int, optional
            cap the number of optimization passes (default: the optimizer's own
            default); lower for fast iteration.
        target_elements : int, optional
            element (tet) budget. When set, the global size scale is retuned over
            a few remeshes so the tet count lands within ~6% of this target, while
            the RELATIVE refinement (curvature plus ``refine_near_points`` size
            points) keeps its shape -- the budget flows where the error is. This
            is the "optimal mesh for N elements" knob; per-entity caps still apply.
        min_h_surf, min_h_vol : float
            hard minimum element size on surfaces and in the volume: the sizing
            field is never refined below these, so curvature/feature refinement and
            the element budget are both floored. 0 = off.
        """
        h = maxh if maxh is not None else self._maxh
        g = grading if grading is not None else self._grading
        # Call kwargs override the accumulated hierarchical globals.
        te = tol_edge if tol_edge is not None else self._tol_edge
        ts = tol_surf if tol_surf is not None else self._tol_surf
        me = maxh_edge if maxh_edge is not None else self._maxh_edge
        ms = maxh_surf if maxh_surf is not None else self._maxh_surf
        mv = maxh_vol if maxh_vol is not None else self._maxh_vol
        native = self._builder.mesh(
            h if h is not None else math.inf,
            radius_edge,
            max_points,
            g,
            [(t, fh) for t, fh in sorted(self._face_maxh.items())],
            [(list(pt), ph) for pt, ph in self._size_points],
            [(s, sh) for s, sh in sorted(self._surface_maxh.items())],
            density_weighted,
            te,
            ts,
            me,
            ms,
            mv,
            [(int(i), v) for i, v in sorted(self._edge_maxh.items())],
            [(int(i), v) for i, v in sorted(self._edge_tol.items())],
            [(int(i), v) for i, v in sorted(self._surf_maxh.items())],
            [(int(i), v) for i, v in sorted(self._surf_tol.items())],
            [(int(t), v) for t, v in sorted(self._region_maxh.items())],
            optimize,
            optimize_passes,
            target_elements,
            min_h_surf,
            min_h_vol,
        )
        solids = [
            {"region": r, "label": self._solid_labels.get(i)}
            for i, r in enumerate(self._solid_regions)
        ]
        return Mesh(native, solids=solids, tag_labels=dict(self._tag_labels))

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
    ) -> SurfaceMesh:
        """Surface-only export: assembles the exact arrangement and meshes
        only its boundary surface (region interfaces, outer boundary, embedded
        sheets), skipping the volume mesh and quality optimization. Much faster
        than :meth:`mesh` when only the conforming surface triangulation is
        needed.

        Honors the full per-entity sizing hierarchy (``region``/``surf``/``edge``
        scopes) exactly like :meth:`mesh`.

        Parameters
        ----------
        maxh : float, optional
            global target edge length (defaults to the constructor's;
            unbounded if neither is given)
        grading : float
            size-grading Lipschitz constant (see :meth:`mesh`)
        tol_edge, tol_surf, maxh_edge, maxh_surf, maxh_vol : float, optional
            chord tolerances and per-dimension size caps (see :meth:`mesh`);
            default to the accumulated hierarchical globals when omitted.
        """
        h = maxh if maxh is not None else self._maxh
        g = grading if grading is not None else self._grading
        # Call kwargs override the accumulated hierarchical globals (matching
        # :meth:`mesh`); omitted ones fall back to the scoped defaults.
        te = tol_edge if tol_edge is not None else self._tol_edge
        ts = tol_surf if tol_surf is not None else self._tol_surf
        me = maxh_edge if maxh_edge is not None else self._maxh_edge
        ms = maxh_surf if maxh_surf is not None else self._maxh_surf
        mv = maxh_vol if maxh_vol is not None else self._maxh_vol
        native = self._builder.surface_mesh(
            h if h is not None else math.inf,
            g,
            [(t, fh) for t, fh in sorted(self._face_maxh.items())],
            [(list(pt), ph) for pt, ph in self._size_points],
            [(s, sh) for s, sh in sorted(self._surface_maxh.items())],
            te,
            ts,
            me,
            ms,
            mv,
            [(int(i), v) for i, v in sorted(self._edge_maxh.items())],
            [(int(i), v) for i, v in sorted(self._edge_tol.items())],
            [(int(i), v) for i, v in sorted(self._surf_maxh.items())],
            [(int(i), v) for i, v in sorted(self._surf_tol.items())],
            [(int(t), v) for t, v in sorted(self._region_maxh.items())],
            target_triangles,
        )
        solids = [
            {"region": r, "label": self._solid_labels.get(i)}
            for i, r in enumerate(self._solid_regions)
        ]
        return SurfaceMesh(native, solids=solids, tag_labels=dict(self._tag_labels))


def _unit(v: tuple[float, float, float]) -> list[float]:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n == 0.0:
        raise ValueError("axis must be nonzero")
    return [c / n for c in v]


def _disc_basis(axis: tuple[float, float, float]):
    a = _unit(axis)
    pick = (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0)
    e1 = [
        a[1] * pick[2] - a[2] * pick[1],
        a[2] * pick[0] - a[0] * pick[2],
        a[0] * pick[1] - a[1] * pick[0],
    ]
    e1 = _unit((e1[0], e1[1], e1[2]))
    e2 = [
        a[1] * e1[2] - a[2] * e1[1],
        a[2] * e1[0] - a[0] * e1[2],
        a[0] * e1[1] - a[1] * e1[0],
    ]
    return e1, e2

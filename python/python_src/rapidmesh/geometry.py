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
    identifies the solid's tets in :attr:`Mesh.tet_regions`."""

    region: int


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
    stats : dict
        n_points, n_tets, n_faces, min_dihedral_deg, max_radius_edge,
        max_edge, millis
    """

    def __init__(self, native) -> None:
        self._native = native
        self.points: np.ndarray = native.points()
        self.tets: np.ndarray = native.tets()
        self.tet_regions: np.ndarray = native.tet_regions()
        self.faces: np.ndarray = native.faces()
        self.face_tags: np.ndarray = native.face_tags()
        self.face_regions: np.ndarray = native.face_regions()
        self.stats: dict = native.stats()

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"Mesh({s['n_tets']} tets, {s['n_points']} points, "
            f"min dihedral {s['min_dihedral_deg']:.1f} deg, "
            f"{s['millis']} ms)"
        )

    def save_viewer_json(self, name: str, directory: str | Path) -> Path:
        """Writes ``rapidmesh_<name>.json`` in the comparison-viewer schema
        and refreshes the viewer manifest. Returns the written path."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        data = {
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
                }
                for (a, b, c), t, (r0, r1) in zip(
                    self.faces, self.face_tags, self.face_regions
                )
            ],
            "stats": {
                "n_points": int(self.stats["n_points"]),
                "n_tets": int(self.stats["n_tets"]),
                "min_dihedral_deg": float(self.stats["min_dihedral_deg"]),
                "max_radius_edge": float(self.stats["max_radius_edge"]),
                "max_edge": float(self.stats["max_edge"]),
                "millis": int(self.stats["millis"]),
            },
        }
        path = directory / f"rapidmesh_{name}.json"
        path.write_text(json.dumps(data))
        _refresh_manifest(directory)
        return path


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
        self._builder = _native.SceneBuilder()
        self._maxh = maxh
        self._grading = grading

    # ------------------------------------------------------------ solids

    def box(
        self,
        width: float,
        depth: float,
        height: float,
        position: tuple[float, float, float] = (0, 0, 0),
        *,
        maxh: float | None = None,
    ) -> Solid:
        """Axis-aligned box: extents along x, y, z; ``position`` is the
        lower corner."""
        x, y, z = position
        region = self._builder.add_box(
            [x, y, z], [x + width, y + depth, z + height], maxh
        )
        return Solid(region)

    def cylinder(
        self,
        radius: float,
        height: float,
        position: tuple[float, float, float] = (0, 0, 0),
        axis: tuple[float, float, float] = (0, 0, 1),
        *,
        segments: int = 24,
        maxh: float | None = None,
    ) -> Solid:
        """Cylinder from the base centre ``position`` along ``axis``. The
        barrel is tessellated with ``segments`` chords but carries the exact
        analytic surface: mesh vertices snap onto the true cylinder."""
        ax = [a * height for a in _unit(axis)]
        region = self._builder.add_cylinder(list(position), ax, radius, segments, maxh)
        return Solid(region)

    def sphere(
        self,
        radius: float,
        position: tuple[float, float, float] = (0, 0, 0),
        *,
        segments: int = 24,
        rings: int = 12,
        maxh: float | None = None,
    ) -> Solid:
        """Sphere centred at ``position`` (analytic surface, like
        :meth:`cylinder`)."""
        region = self._builder.add_sphere(list(position), radius, segments, rings, maxh)
        return Solid(region)

    def cone(
        self,
        r1: float,
        r2: float,
        height: float,
        position: tuple[float, float, float] = (0, 0, 0),
        axis: tuple[float, float, float] = (0, 0, 1),
        *,
        segments: int = 24,
        maxh: float | None = None,
    ) -> Solid:
        """Conical frustum: base radius ``r1`` at ``position``, top radius
        ``r2`` (0 for a full cone) at ``position + height * axis``."""
        ax = [a * height for a in _unit(axis)]
        region = self._builder.add_frustum(
            list(position), ax, r1, r2, segments, maxh
        )
        return Solid(region)

    def prism(
        self,
        points: list[tuple[float, float]],
        height: float,
        position: tuple[float, float, float] = (0, 0, 0),
        *,
        holes: list[list[tuple[float, float]]] | None = None,
        maxh: float | None = None,
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
        )
        return Solid(region)

    # ------------------------------------------------------------ sheets

    def xy_plate(
        self,
        width: float,
        height: float,
        position: tuple[float, float, float] = (0, 0, 0),
        *,
        tag: int = 1,
    ) -> None:
        """Zero-thickness rectangle in an xy plane (a PEC trace, a port
        marker): spans ``width`` along x and ``height`` along y from the
        corner ``position``; conformally embedded with face tag ``tag``."""
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
    ) -> None:
        """Like :meth:`xy_plate` in an xz plane (width along x, height
        along z)."""
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
    ) -> None:
        """Like :meth:`xy_plate` in a yz plane (width along y, height
        along z)."""
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
    ) -> None:
        """General parallelogram sheet from corner ``p0`` spanned by the
        edge vectors ``du`` and ``dv``."""
        self._builder.add_sheet_rect(list(p0), list(du), list(dv), tag)

    def disc(
        self,
        radius: float,
        position: tuple[float, float, float] = (0, 0, 0),
        axis: tuple[float, float, float] = (0, 0, 1),
        *,
        segments: int = 24,
        tag: int = 1,
    ) -> None:
        """Disc sheet centred at ``position``, normal to ``axis``."""
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
    ) -> None:
        """Polygonal sheet in an xy plane at ``position`` (2D coordinates
        are offset by ``position``'s x, y)."""
        self._builder.add_sheet_polygon(
            [list(p) for p in points],
            [[list(q) for q in h] for h in (holes or [])],
            list(position),
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            tag,
        )

    # ------------------------------------------------------------- mesh

    def mesh(
        self,
        *,
        maxh: float | None = None,
        radius_edge: float = 2.0,
        max_points: int = 500_000,
        grading: float | None = None,
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
        """
        h = maxh if maxh is not None else self._maxh
        g = grading if grading is not None else self._grading
        native = self._builder.mesh(
            h if h is not None else math.inf, radius_edge, max_points, g
        )
        return Mesh(native)


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

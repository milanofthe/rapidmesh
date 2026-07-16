"""Signed-distance expression builders for :meth:`rapidmesh.Geometry.implicit`.

Each function returns a plain nested-dict spec the native layer parses into
its SDF tree. Fields are signed-distance-like (negative inside); smooth
booleans and offsets are the operations exact B-rep CSG cannot express —
fillets, blends, coating/plating shells::

    import rapidmesh as rm
    from rapidmesh import sdf

    g = rm.Geometry()
    rounded = sdf.offset(sdf.box((0, 0, 0), (0.7, 0.7, 0.7)), 0.25)
    g.implicit(rounded, (-1.4, -1.4, -1.4), (1.4, 1.4, 1.4))
    mesh = g.mesh(maxh=0.3)
"""

from __future__ import annotations

V3 = tuple[float, float, float]


def sphere(center: V3, radius: float) -> dict:
    """Sphere around ``center``."""
    return {"op": "sphere", "center": list(center), "radius": float(radius)}


def box(center: V3, half: V3) -> dict:
    """Axis-aligned box around ``center`` with half extents ``half``."""
    return {"op": "box", "center": list(center), "half": list(half)}


def cylinder(a: V3, b: V3, radius: float) -> dict:
    """Capped cylinder between the axis endpoints ``a`` and ``b``."""
    return {"op": "cylinder", "a": list(a), "b": list(b), "radius": float(radius)}


def capsule(a: V3, b: V3, radius: float) -> dict:
    """Capsule (sphere-swept segment) between ``a`` and ``b``."""
    return {"op": "capsule", "a": list(a), "b": list(b), "radius": float(radius)}


def torus(center: V3, axis: V3, major: float, minor: float) -> dict:
    """Torus around ``center`` with plane normal ``axis``."""
    return {
        "op": "torus", "center": list(center), "axis": list(axis),
        "major": float(major), "minor": float(minor),
    }


def half_space(point: V3, normal: V3) -> dict:
    """Half space: inside on the anti-``normal`` side of ``point``."""
    return {"op": "half_space", "point": list(point), "normal": list(normal)}


def union(a: dict, b: dict) -> dict:
    """Boolean union (sharp)."""
    return {"op": "union", "a": a, "b": b}


def intersect(a: dict, b: dict) -> dict:
    """Boolean intersection (sharp)."""
    return {"op": "intersect", "a": a, "b": b}


def difference(a: dict, b: dict) -> dict:
    """Boolean difference ``a - b`` (sharp)."""
    return {"op": "difference", "a": a, "b": b}


def smooth_union(a: dict, b: dict, k: float) -> dict:
    """Filleted union: blends over a band of radius ``k`` (solder menisci,
    glob top, organic fusions)."""
    return {"op": "smooth_union", "a": a, "b": b, "k": float(k)}


def smooth_intersect(a: dict, b: dict, k: float) -> dict:
    """Filleted intersection with blend radius ``k``."""
    return {"op": "smooth_intersect", "a": a, "b": b, "k": float(k)}


def smooth_difference(a: dict, b: dict, k: float) -> dict:
    """Filleted difference ``a - b`` with blend radius ``k``."""
    return {"op": "smooth_difference", "a": a, "b": b, "k": float(k)}


def offset(a: dict, d: float) -> dict:
    """Offset surface: grows the solid by ``d`` (rounds convex edges — the
    fillet/plating/conformal-coating primitive)."""
    return {"op": "offset", "a": a, "d": float(d)}


def shell(a: dict, t: float) -> dict:
    """Shell of half thickness ``t`` around the surface of ``a``."""
    return {"op": "shell", "a": a, "t": float(t)}

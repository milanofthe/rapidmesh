"""The showcase model corpus: general-purpose geometries exercising every
builder feature (primitives, boolean cuts and overlap-unions, embedded
sheets, sizing controls). The showcase site renders these and the bench
harness meshes them, so this registry is the single source for both.

Each builder returns an unmeshed :class:`rapidmesh.Geometry`; callers pick
mesh parameters (the geometry's ``maxh`` defaults are tuned for a web-sized
mesh of roughly 10k-60k tets).

Run directly to mesh every model and print a stats table:

    python python/examples/showcase.py [ids...]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import rapidmesh as rm


# ------------------------------------------------------------------ helpers


def _circle(n: int, r: float, cx: float = 0.0, cy: float = 0.0, a0: float = 0.0):
    return [
        (cx + r * math.cos(a0 + 2 * math.pi * i / n),
         cy + r * math.sin(a0 + 2 * math.pi * i / n))
        for i in range(n)
    ]


def _arc(cx, cy, r, a0, a1, n):
    """Open arc from a0 to a1 (inclusive ends) with n points."""
    return [
        (cx + r * math.cos(a0 + (a1 - a0) * i / (n - 1)),
         cy + r * math.sin(a0 + (a1 - a0) * i / (n - 1)))
        for i in range(n)
    ]


# ------------------------------------------------------------------- models


def pipe_junction() -> rm.Geometry:
    """Pipe tee: two pipes joined at a right angle (overlap-union of the
    barrels), flanged ports, bores carved with cylinder voids. A three-way
    cross with its triple bore intersection is kept out until the exact-CSG
    hotpath gets its performance round."""
    g = rm.Geometry(maxh=0.2)
    L, ro, ri, fr, fh = 3.2, 0.5, 0.3, 0.72, 0.18
    # flanges sit a hair behind the port ends so their caps are not exactly
    # coplanar with the barrel caps (avoids needless coplanar CSG tiling)
    inset = 0.02
    runs = [
        ((-L / 2, 0, 0), (1, 0, 0), L, True),   # main run, both ports
        ((0, 0, 0), (0, 0, 1), L / 2, False),   # branch up, one port
    ]
    for p0, ax, length, both in runs:
        g.label(g.cylinder(ro, length, position=p0, axis=ax, segments=20),
                "pipes")
        pf1 = tuple(p + (length - fh - inset) * a for p, a in zip(p0, ax))
        g.label(g.cylinder(fr, fh, position=pf1, axis=ax, segments=20),
                "flanges")
        if both:
            pf0 = tuple(p + inset * a for p, a in zip(p0, ax))
            g.label(g.cylinder(fr, fh, position=pf0, axis=ax, segments=20),
                    "flanges")
    for p0, ax, length, _ in runs:
        g.label(
            g.cylinder(ri, length, position=p0, axis=ax, segments=20,
                       void=True),
            "bores",
        )
    return g


def dice() -> rm.Geometry:
    """A die: cube with 21 spherical pip dimples carved from its faces
    (boolean cut showcase; the pips get a finer local size)."""
    g = rm.Geometry(maxh=0.24)
    g.label(g.box(2, 2, 2, position=(-1, -1, -1)), "body")
    r, depth, o = 0.26, 0.11, 0.52
    c = 1.0 + (r - depth)  # pip sphere centre distance from cube centre
    pips = {
        "z+": [(0, 0)],
        "z-": [(-0.45, -o), (-0.45, 0), (-0.45, o), (0.45, -o), (0.45, 0), (0.45, o)],
        "x+": [(-o, -o), (o, o)],
        "x-": [(-o, -o), (-o, o), (o, -o), (o, o), (0, 0)],
        "y+": [(-o, -o), (0, 0), (o, o)],
        "y-": [(-o, -o), (-o, o), (o, -o), (o, o)],
    }
    for face, uv in pips.items():
        axis, sign = face[0], 1 if face[1] == "+" else -1
        for u, v in uv:
            if axis == "x":
                p = (sign * c, u, v)
            elif axis == "y":
                p = (u, sign * c, v)
            else:
                p = (u, v, sign * c)
            g.label(
                g.sphere(r, position=p, segments=14, rings=7, maxh=0.09,
                         void=True),
                "pips",
            )
    return g


def gear() -> rm.Geometry:
    """Spur gear blank from a single polygon prism: trapezoid teeth, keyway
    bore and a ring of lightening holes, all via polygon holes."""
    n_teeth, r_root, r_tip, depth = 14, 1.0, 1.2, 0.5
    pts: list[tuple[float, float]] = []
    T = 2 * math.pi / n_teeth
    for i in range(n_teeth):
        a = i * T
        for f, r in ((0.00, r_root), (0.30, r_root), (0.40, r_tip),
                     (0.70, r_tip), (0.80, r_root)):
            pts.append((r * math.cos(a + f * T), r * math.sin(a + f * T)))
    # keyway bore: circular arc with a rectangular notch at angle 0
    rb, kw, kd = 0.40, 0.085, 0.50
    ka = math.asin(kw / rb)
    bore = _arc(0, 0, rb, ka, 2 * math.pi - ka, 30)
    bore += [(rb * math.cos(ka), -kw), (kd, -kw), (kd, kw)]
    holes = [bore]
    for i in range(6):
        a = 2 * math.pi * (i + 0.5) / 6
        holes.append(_circle(16, 0.17, 0.68 * math.cos(a), 0.68 * math.sin(a)))
    g = rm.Geometry(maxh=0.14)
    g.label(g.prism(pts, depth, holes=holes), "gear")
    return g


def bearing() -> rm.Geometry:
    """Ball bearing: outer and inner race rings (cylinders with bore voids)
    and eight free-floating balls in the groove, nothing touches."""
    g = rm.Geometry(maxh=0.12)
    h = 0.5
    g.label(g.cylinder(1.32, h, segments=48), "outer race")
    g.label(g.cylinder(1.06, h, segments=48, void=True), "groove")
    g.label(g.cylinder(0.84, h, segments=48), "inner race")
    g.label(g.cylinder(0.52, h, segments=32, void=True), "shaft bore")
    for i in range(8):
        a = 2 * math.pi * i / 8
        ball = g.sphere(0.09, segments=12, rings=6, maxh=0.05,
                        position=(0.95 * math.cos(a), 0.95 * math.sin(a), h / 2))
        g.label(ball, "balls")
    return g


def rocket() -> rm.Geometry:
    """Sounding rocket: cylinder body, frustum nose cone, flared frustum
    nozzle and four box fins, joined by overlap-union."""
    g = rm.Geometry(maxh=0.15)
    g.label(g.cylinder(0.5, 2.6, segments=24), "body")
    # blunt nose tip: a sharp apex ring (tiny top radius vs segment count)
    # forces micro edges and near-degenerate slivers
    g.label(g.cone(0.5, 0.12, 1.05, position=(0, 0, 2.6), segments=24), "nose")
    g.label(g.cone(0.42, 0.30, 0.45, position=(0, 0, -0.4), segments=24),
            "nozzle")
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        w, t, hgt = 0.78, 0.08, 0.9
        if sx:
            fin = g.box(w, t, hgt,
                        position=(0.38 if sx > 0 else -0.38 - w, -t / 2, 0))
        else:
            fin = g.box(t, w, hgt,
                        position=(-t / 2, 0.38 if sy > 0 else -0.38 - w, 0))
        g.label(fin, "fins")
    return g


def house() -> rm.Geometry:
    """House: hollow box shell with a wedge roof, chimney, a door cut and
    two round window bores (wedge primitive plus mixed cuts)."""
    g = rm.Geometry(maxh=0.16)
    g.label(g.box(2.4, 1.8, 1.1, position=(-1.2, -0.9, 0)), "walls")
    g.label(g.wedge(2.4, 1.8, 0.7, position=(-1.2, -0.9, 1.1), top_x=0.0),
            "roof")
    g.label(g.box(0.3, 0.3, 1.0, position=(0.45, 0.15, 1.15)), "chimney")
    g.label(g.box(2.1, 1.5, 0.85, position=(-1.05, -0.75, 0.12), void=True),
            "interior")
    # the door tunnel overlaps the interior void by 0.05: exactly-tangent
    # void faces would create needless coplanar T-junction work
    g.label(g.box(0.45, 0.3, 0.75, position=(-0.225, -1.0, 0), void=True),
            "door")
    for x in (-0.72, 0.72):
        win = g.cylinder(0.17, 0.4, position=(x, -1.0, 0.6), axis=(0, 1, 0),
                         segments=16, void=True)
        g.label(win, "windows")
    return g


def chain() -> rm.Geometry:
    """Three interlocked torus links with alternating orientation: disjoint
    components threading through each other without contact."""
    g = rm.Geometry(maxh=0.14)
    for i in range(3):
        axis = (0, 0, 1) if i % 2 == 0 else (0, 1, 0)
        link = g.torus(1.0, 0.2, position=(1.5 * i, 0, 0), axis=axis,
                       segments=36, tube_segments=14)
        g.label(link, "odd links" if i % 2 else "even links")
    return g


def spring() -> rm.Geometry:
    """Coil spring: one helix primitive, six and a half turns of round
    wire on the analytic helical surface."""
    g = rm.Geometry(maxh=0.115)
    g.label(g.helix(0.8, 0.42, 6.5, 0.15, points_per_turn=28, segments=12),
            "coil")
    return g


def serpentine() -> rm.Geometry:
    """Heat-exchanger serpentine: a round tube swept along a winding
    polyline path with semicircular turns (sweep showcase)."""
    lanes, lane_len, gap, rturn = 4, 2.4, 0.9, 0.45
    path: list[tuple[float, float, float]] = []
    z = 0.0
    for i in range(lanes):
        y = i * gap
        xs = (0.0, lane_len) if i % 2 == 0 else (lane_len, 0.0)
        path.append((xs[0], y, z))
        path.append((xs[1], y, z))
        if i < lanes - 1:
            # semicircular turn into the next lane, rising slightly
            cx = xs[1]
            sgn = 1.0 if i % 2 == 0 else -1.0
            for k in range(1, 10):
                a = math.pi * k / 10
                path.append((cx + sgn * rturn * math.sin(a),
                             y + rturn - rturn * math.cos(a),
                             z + 0.12 * k / 10))
            z += 0.12
    g = rm.Geometry(maxh=0.13)
    g.label(g.sweep(path, 0.16, segments=14), "tube")
    return g


def square_to_round() -> rm.Geometry:
    """Square-to-round transition duct: ruled loft between a square frame
    and an offset circle (loft showcase), with the duct bore carved by a
    second, inset loft as a void."""
    def profiles(s_half: float, r: float, z0: float, z1: float):
        sq, n_side = [], 6
        corners = [(s_half, s_half), (-s_half, s_half),
                   (-s_half, -s_half), (s_half, -s_half)]
        for c in range(4):
            x0, y0 = corners[c]
            x1, y1 = corners[(c + 1) % 4]
            for k in range(n_side):
                t = k / n_side
                sq.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z0))
        circ = [(0.45 + r * math.cos(2 * math.pi * (i + 0.5) / 24),
                 r * math.sin(2 * math.pi * (i + 0.5) / 24), z1)
                for i in range(24)]
        return sq, circ

    g = rm.Geometry(maxh=0.17)
    # the bore loft pokes well past both end caps: a hair-thin protrusion
    # would create micro intersection slivers and refinement blow-up
    outer = profiles(1.2, 0.85, 0.0, 2.0)
    inner = profiles(1.05, 0.72, -0.08, 2.08)
    g.label(g.loft(outer[0], outer[1]), "duct wall")
    g.label(g.loft(inner[0], inner[1], void=True), "duct bore")
    return g


def perforated_plate() -> rm.Geometry:
    """Perforated plate: a slab with a staggered grid of 33 round bores
    (many-cut robustness benchmark)."""
    g = rm.Geometry(maxh=0.21)
    w, d, t = 4.0, 2.8, 0.28
    g.label(g.box(w, d, t, position=(-w / 2, -d / 2, 0)), "plate")
    r, px, py = 0.16, 0.56, 0.52
    for row in range(5):
        y = -2 * py + row * py
        n = 7 if row % 2 == 0 else 6
        for col in range(n):
            x = -(n - 1) / 2 * px + col * px
            # no per-hole maxh: the 16-gon facets already size the rims,
            # and 33 fine-sized bores blow the refinement budget
            hole = g.cylinder(r, t + 0.1, position=(x, y, -0.05), segments=16,
                              void=True)
            g.label(hole, "holes")
    return g


def mold_block() -> rm.Geometry:
    """CSG cavity sampler: one block with five different cavities carved
    from it (sphere, oblique cylinder, sunken torus, wedge, countersink
    cone), one of each cut type."""
    g = rm.Geometry(maxh=0.17)
    g.label(g.box(4.2, 2.6, 1.2, position=(-2.1, -1.3, 0)), "block")
    g.label(g.sphere(0.55, position=(-1.35, 0, 1.2), segments=18, rings=9,
                     void=True), "sphere cut")
    g.label(g.cylinder(0.28, 1.8, position=(-0.35, -0.95, 0.35),
                       axis=(1, 0.45, 1), segments=18, void=True),
            "oblique bore")
    g.label(g.torus(0.5, 0.17, position=(1.25, 0.35, 1.2), segments=24,
                    tube_segments=12, void=True), "torus groove")
    g.label(g.wedge(0.75, 0.55, 0.55, position=(0.05, -1.05, 0.85), top_x=0.25,
                    void=True), "wedge pocket")
    g.label(g.cone(0.38, 0.12, 0.8, position=(1.25, -0.75, 1.25),
                   axis=(0, 0, -1), segments=18, void=True), "countersink")
    return g


def pressure_vessel() -> rm.Geometry:
    """Pressure vessel: cylindrical shell with spherical heads, a nozzle
    stub and flange on top; the thin-walled interior is one connected void
    (thin-shell stress test)."""
    g = rm.Geometry(maxh=0.16)
    H = 2.4
    # dished heads: sphere radius > cylinder radius so the head meets the
    # barrel transversally; an exact-radius hemisphere would touch the
    # barrel tangentially (degenerate CSG contact, refinement blow-up)
    g.label(g.cylinder(1.0, H, segments=28), "shell")
    g.label(g.sphere(1.081, position=(0, 0, 0.531), segments=28, rings=14),
            "heads")
    g.label(g.sphere(1.081, position=(0, 0, H - 0.531), segments=28,
                     rings=14), "heads")
    g.label(g.cylinder(0.28, 0.7, position=(0, 0, H + 0.45), segments=20),
            "nozzle")
    g.label(g.cylinder(0.46, 0.14, position=(0, 0, H + 1.01), segments=20),
            "flange")
    # connected interior: inner cylinder + inner dished heads + nozzle bore
    for void in (
        g.cylinder(0.82, H, segments=28, void=True),
        g.sphere(0.837, position=(0, 0, 0.467), segments=28, rings=14,
                 void=True),
        g.sphere(0.837, position=(0, 0, H - 0.467), segments=28, rings=14,
                 void=True),
        g.cylinder(0.16, 0.9, position=(0, 0, H + 0.3), segments=16,
                   void=True),
    ):
        g.label(void, "interior")
    return g


def sizing_field() -> rm.Geometry:
    """Sizing-field showcase: a uniform slab whose edge-length target
    collapses near three point sources of different strength and recovers
    along the grading."""
    g = rm.Geometry(maxh=0.55)
    g.label(g.box(4.4, 4.4, 1.2, position=(-2.2, -2.2, 0)), "slab")
    g.refine_near_points([(-1.2, -1.2, 0.6)], 0.05)
    g.refine_near_points([(1.3, 0.4, 0.6)], 0.1)
    g.refine_near_points([(0.2, 1.6, 1.2)], 0.18)
    return g


def baffled_tank() -> rm.Geometry:
    """Baffled tank: a box volume with three zero-thickness baffle sheets
    embedded conformally, alternating the flow gap side."""
    g = rm.Geometry(maxh=0.28)
    g.label(g.box(3.2, 2.0, 1.6), "tank")
    g.yz_plate(1.4, 1.6, position=(0.8, 0.0, 0), tag=1, maxh=0.14)
    g.yz_plate(1.4, 1.6, position=(1.6, 0.6, 0), tag=2, maxh=0.14)
    g.yz_plate(1.4, 1.6, position=(2.4, 0.0, 0), tag=3, maxh=0.14)
    g.label(1, "baffle a")
    g.label(2, "baffle b")
    g.label(3, "baffle c")
    return g


def laminate() -> rm.Geometry:
    """Graded laminate: six stacked layers with alternating per-solid mesh
    size, the grading blends the densities across the interfaces."""
    g = rm.Geometry(maxh=0.5)
    t = 0.28
    for i in range(6):
        ply = g.box(3.2, 3.2, t, position=(-1.6, -1.6, i * t),
                    maxh=0.42 if i % 2 == 0 else 0.16)
        g.label(ply, "coarse plies" if i % 2 == 0 else "fine plies")
    return g


def coax_step() -> rm.Geometry:
    """Coaxial impedance step (the EM classic kept as a benchmark): outer
    dielectric with two stepped inner-conductor voids and finer surface
    sizing on the conductor walls."""
    mm = 1e-3
    ri1, ri2, ro = 1.50 * mm, 0.99 * mm, 3.45 * mm
    l1, l2 = 15.0 * mm, 15.0 * mm
    h = 3.0 * mm
    g = rm.Geometry(maxh=h)
    g.label(g.cylinder(radius=ro, height=l1 + l2, position=(0, 0, 0)),
            "dielectric")
    inner1 = g.cylinder(radius=ri1, height=l1, position=(0, 0, 0), void=True)
    inner2 = g.cylinder(radius=ri2, height=l2, position=(0, 0, l1), void=True)
    g.label(inner1, "inner conductor a")
    g.label(inner2, "inner conductor b")
    g.refine_surface(inner1, h / 2)
    g.refine_surface(inner2, h / 2)
    return g


def microstrip() -> rm.Geometry:
    """Microstrip line (the EM classic kept as a benchmark): substrate and
    air regions with a PEC trace sheet embedded on the interface."""
    mm = 1e-3
    sub_h, line_w, line_l = 0.508 * mm, 1.13 * mm, 30.0 * mm
    sub_w, air_h = 20.0 * mm, 10.0 * mm
    g = rm.Geometry(maxh=9.0 * mm)
    g.label(g.box(sub_w, line_l, air_h + sub_h, position=(-sub_w / 2, 0, 0)),
            "air")
    g.label(g.box(sub_w, line_l, sub_h, position=(-sub_w / 2, 0, 0),
                  maxh=4.8 * mm), "substrate")
    g.xy_plate(line_w, line_l, position=(-line_w / 2, 0, sub_h), tag=7,
               maxh=0.55 * mm)
    g.label(7, "trace")
    return g


# ----------------------------------------------------------------- registry


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    build: Callable[[], rm.Geometry]


MODELS: list[Model] = [
    Model("pipe_junction", "Pipe Tee", pipe_junction),
    Model("gear", "Spur Gear", gear),
    Model("dice", "Dice", dice),
    Model("bearing", "Ball Bearing", bearing),
    Model("rocket", "Rocket", rocket),
    Model("house", "House", house),
    Model("chain", "Chain Links", chain),
    Model("spring", "Coil Spring", spring),
    Model("serpentine", "Serpentine Pipe", serpentine),
    Model("square_to_round", "Square-to-Round Duct", square_to_round),
    Model("perforated_plate", "Perforated Plate", perforated_plate),
    Model("mold_block", "Cavity Sampler", mold_block),
    Model("pressure_vessel", "Pressure Vessel", pressure_vessel),
    Model("sizing_field", "Sizing Field", sizing_field),
    Model("baffled_tank", "Baffled Tank", baffled_tank),
    Model("laminate", "Graded Laminate", laminate),
    Model("coax_step", "Coax Step", coax_step),
    Model("microstrip", "Microstrip", microstrip),
]


def main(argv: list[str]) -> None:
    wanted = set(argv) if argv else None
    for m in MODELS:
        if wanted and m.id not in wanted:
            continue
        try:
            mesh = m.build().mesh()
        except Exception as e:  # noqa: BLE001 - stats table keeps going
            print(f"{m.id:<18} FAILED: {type(e).__name__}: {e}")
            continue
        s = mesh.stats
        print(f"{m.id:<18} {s['n_tets']:>8} tets  {s['n_points']:>7} pts  "
              f"min-dih {s['min_dihedral_deg']:5.1f}  "
              f"re {s['max_radius_edge']:4.2f}  {s['millis']:>6} ms")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])

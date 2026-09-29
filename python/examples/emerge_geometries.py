"""RF geometries ported from the EMerge examples (github.com/FennisRobert/EMerge,
``examples/demo*.py``): the geometry sections only, with conductors as PEC
sheets (tagged plates) or as voids and separate regions, the way the solver
would mark them. Dimensions follow the demos.

    python python/examples/emerge_geometries.py
"""

import math

import rapidmesh as rm

mm = 1e-3
mil = 0.0254 * mm
C0 = 299_792_458.0


def lambda_maxh(f_max: float, er_max: float = 1.0, n: int = 10) -> float:
    """Target edge length of n elements per wavelength at f_max."""
    return C0 / (f_max * n * er_max**0.5)


def _union_outline(rects):
    """Outline and holes of the union of axis-aligned rectangles
    (x0, y0, x1, y1), one entry per connected piece."""
    from shapely.geometry import box
    from shapely.ops import unary_union

    u = unary_union([box(*r) for r in rects]).buffer(0)
    geoms = u.geoms if u.geom_type == "MultiPolygon" else [u]
    return [_rings(p) for p in geoms]


def _rings(p):
    """Exterior counterclockwise and holes clockwise, closing point dropped,
    as ``polygon_plate`` takes them."""
    from shapely.geometry.polygon import orient

    p = orient(p, 1.0)
    return list(p.exterior.coords)[:-1], [list(h.coords)[:-1] for h in p.interiors]


def stepped_impedance_filter() -> rm.Mesh:
    """Microstrip stepped-impedance lowpass (demo1): seven sections on a
    62 mil, er 2.2 board, air above."""
    lengths = [400, 660, 660, 660, 660, 660, 400]
    widths = [50, 128, 8, 224, 8, 128, 50]
    th = 62 * mil
    total = sum(lengths) * mil
    wb = 2 * 200 * mil + max(widths) * mil
    f = 6e9
    g = rm.Geometry(maxh=lambda_maxh(f))
    g.label(g.box(total, wb, th + 8 * th, position=(0, -wb / 2, 0)), "air")
    g.label(g.box(total, wb, th, position=(0, -wb / 2, 0), maxh=lambda_maxh(f, 2.2) / 2), "substrate")
    x, rects = 0.0, []
    for L, W in zip(lengths, widths):
        rects.append((x, -W * mil / 2, x + L * mil, W * mil / 2))
        x += L * mil
    for ext, holes in _union_outline(rects):
        g.polygon_plate(ext, position=(0, 0, th), holes=holes or None, tag=7)
    g.label(7, "trace")
    return g.mesh()


def combline_filter() -> rm.Mesh:
    """Five-resonator combline filter (demo2): rods and the inner conductors
    of the two tapped coax feeds are PEC voids in the air cavity."""
    a, b = 240 * mil, 248 * mil
    W, S1, S2, wi = 84 * mil, 117 * mil, 136 * mil, 84 * mil
    d1, d2, dc, h = 10 * mil, 10 * mil, 8.5 * mil, 74 * mil
    lr1, lr2, C1 = b - d1, b - d2, b - dc
    Lbox = 5 * W + 2 * (S1 + S2 + wi)
    x1 = wi + W / 2
    xs = [x1, x1 + W + S1, x1 + 2 * W + S1 + S2, x1 + 3 * W + S1 + 2 * S2, x1 + 4 * W + 2 * S1 + 2 * S2]
    rout, rin, lfeed = 40.5 * mil, 12.5 * mil, 100 * mil
    g = rm.Geometry(maxh=lambda_maxh(8e9, n=12))
    g.label(g.box(Lbox, a, b, position=(0, -a / 2, 0)), "cavity")
    for x, lr in zip(xs, [C1, lr1, lr2, lr1, C1]):
        g.label(g.cylinder(W / 2, lr, position=(x, 0, 0), segments=20, void=True), "rods")
    for x0 in (-lfeed, Lbox):
        g.label(g.cylinder(rout, lfeed, position=(x0, 0, h), axis=(1, 0, 0), segments=20), "feed")
    for x0 in (-lfeed, Lbox - wi - W / 2):
        pin = g.cylinder(rin, lfeed + wi + W / 2, position=(x0, 0, h), axis=(1, 0, 0), segments=12, void=True)
        g.label(pin, "feed pins")
    return g.mesh()


def coupled_line_filter() -> rm.Mesh:
    """Parallel-coupled-line bandpass (demo3 dimensions): input line, six
    coupled sections of quarter-wave resonators, output line, on a 20 mil
    er 3.55 board with air above."""
    th = 20 * mil
    w0, l0 = 37 * mil, 100 * mil
    lc = [314.22, 301.658, 300.589, 300.589, 301.658, 314.22]
    ws = [18.8, 43.484, 44.331, 44.331, 43.484, 18.8]
    gs = [9.63, 24.84, 41.499, 41.499, 24.84, 9.63]
    rects, x, y = [], 0.0, 0.0
    rects.append((x, y - w0 / 2, x + l0, y + w0 / 2))
    x += l0
    # Each section: the line so far continues for lc, the next resonator starts
    # beside it (gap g) and runs on for another lc.
    prev_w = w0
    for L, W, G in zip(lc, ws, gs):
        L, W, G = L * mil, W * mil, G * mil
        rects.append((x, y - prev_w / 2, x + L, y + prev_w / 2))
        y2 = y + prev_w / 2 + G + W / 2
        rects.append((x, y2 - W / 2, x + 2 * L, y2 + W / 2))
        x, y, prev_w = x + L, y2, W
    rects.append((x, y - prev_w / 2, x + lc[-1] * mil, y + prev_w / 2))
    rects.append((x + lc[-1] * mil, y - w0 / 2, x + lc[-1] * mil + l0, y + w0 / 2))
    xs = [r[0] for r in rects] + [r[2] for r in rects]
    ys = [r[1] for r in rects] + [r[3] for r in rects]
    m = 150 * mil
    x0, y0, x1, y1 = min(xs), min(ys) - m, max(xs), max(ys) + m
    f = 12e9
    g = rm.Geometry(maxh=lambda_maxh(f))
    g.label(g.box(x1 - x0, y1 - y0, 5 * th, position=(x0, y0, 0)), "air")
    g.label(g.box(x1 - x0, y1 - y0, th, position=(x0, y0, 0), maxh=lambda_maxh(f, 3.55) / 2), "substrate")
    for ext, holes in _union_outline(rects):
        g.polygon_plate(ext, position=(0, 0, th), holes=holes or None, tag=7)
    return g.mesh()


def stripline_vias() -> rm.Mesh:
    """Stripline between two ground planes (demo6): a trace with two bends
    in the mid plane of a 1 mm RO4350B board, flanked by a via fence (PEC
    voids through the board)."""
    th, w = 1.0 * mm, 0.4 * mm
    path = [(0, 0), (10 * mm, 0), (10 * mm, 10 * mm), (25 * mm, 10 * mm), (25 * mm, 0), (35 * mm, 0)]
    rects = []
    for (xa, ya), (xb, yb) in zip(path, path[1:]):
        rects.append((min(xa, xb) - w / 2, min(ya, yb) - w / 2, max(xa, xb) + w / 2, max(ya, yb) + w / 2))
    f = 10e9
    g = rm.Geometry(maxh=lambda_maxh(f, 3.66))
    g.label(g.box(35 * mm, 20 * mm, th, position=(0, -5 * mm, -th / 2)), "substrate")
    for ext, holes in _union_outline(rects):
        g.polygon_plate(ext, position=(0, 0, 0), holes=holes or None, tag=7)
    # Via fence 1.5 mm off the trace, every 2 mm along each straight run.
    r, off, step = 0.2 * mm, 1.5 * mm, 2.0 * mm
    for (xa, ya), (xb, yb) in zip(path, path[1:]):
        L = math.hypot(xb - xa, yb - ya)
        ux, uy = (xb - xa) / L, (yb - ya) / L
        for k in range(1, int(L / step)):
            px, py = xa + ux * k * step, ya + uy * k * step
            for s in (-1, 1):
                g.cylinder(r, th, position=(px - s * uy * off, py + s * ux * off, -th / 2), segments=8, void=True)
    return g.mesh()


def horn_antenna() -> rm.Mesh:
    """Pyramidal standard gain horn (demo10): WR-8 feed waveguide flaring to
    a 10 x 7 mm aperture over 21 mm, 1 mm walls as their own region,
    radiating into an air box (five elements per wavelength at 90 GHz)."""
    wga, wgb, WH, HH, Lh, Lf, t = 2.01 * mm, 1.01 * mm, 10 * mm, 7 * mm, 21 * mm, 2 * mm, 1 * mm

    def rect(x, w, h):
        return [(x, -w / 2, -h / 2), (x, w / 2, -h / 2), (x, w / 2, h / 2), (x, -w / 2, h / 2)]

    g = rm.Geometry(maxh=lambda_maxh(90e9, n=5))
    g.label(g.box(Lh + Lf + 8 * mm, 1.6 * WH + 2 * t, 1.6 * HH + 2 * t,
                  position=(-Lf, -0.8 * WH - t, -0.8 * HH - t)), "air")
    g.label(g.loft(rect(0.0, wga + 2 * t, wgb + 2 * t), rect(Lh, WH + 2 * t, HH + 2 * t)), "wall")
    g.label(g.box(Lf, wga + 2 * t, wgb + 2 * t, position=(-Lf, -wga / 2 - t, -wgb / 2 - t)), "wall")
    g.label(g.loft(rect(0.0, wga, wgb), rect(Lh, WH, HH)), "horn")
    g.label(g.box(Lf, wga, wgb, position=(-Lf, -wga / 2, -wgb / 2)), "feed")
    return g.mesh()


def helix_antenna() -> rm.Mesh:
    """Axial-mode helix antenna for 3 GHz (demo13, shortened from 13 to 6
    turns of the same pitch): 1 mm wire, radius lambda / 2 pi, fed by a short
    vertical stub above the ground plane (the air box floor)."""
    f = 3e9
    wl = C0 / f
    rad0, porth = wl / (2 * math.pi), 2 * mm
    pitch, turns = 4 * rad0 / 13, 6
    L = pitch * turns
    g = rm.Geometry(maxh=lambda_maxh(f, n=12))
    g.label(g.box(4 * rad0, 4 * rad0, L + porth + 0.5 * L, position=(-2 * rad0, -2 * rad0, 0)), "air")
    g.label(g.helix(rad0, pitch, turns, 1 * mm, position=(0, 0, porth), maxh=2 * mm), "wire")
    g.label(g.cylinder(1 * mm, porth + 1 * mm, position=(rad0, 0, 0), segments=12, maxh=1.5 * mm), "wire")
    return g.mesh()


def vivaldi_antenna() -> rm.Mesh:
    """Antipodal-free Vivaldi (demo19): an exponentially tapered slot cut from
    the ground plane with a circular cavity and seven corrugation slots, fed
    by a microstrip with a radial stub on the other side of a 0.5 mm FR-4
    board."""
    from shapely.geometry import Point, Polygon, box
    from shapely.ops import unary_union

    gap, L, W, K = 0.3 * mm, 70 * mm, 55 * mm, 200.0
    Rc, th = 20 * mm, 0.5 * mm
    fy = lambda t: (gap / 2) * K**t + (W - gap * K) / (2 - 2 * K) * (1 - K**t)
    ts = [k / 200 for k in range(201)]
    taper = Polygon([(t * L, fy(t)) for t in ts] + [(t * L, -fy(t)) for t in reversed(ts)])
    board = box(-30 * mm, -30 * mm, 70 * mm, 30 * mm)
    slots = unary_union([box(70 * mm - (n + 1) * 8 * mm, -30 * mm, 70 * mm - (n + 1) * 8 * mm + 4 * mm, 30 * mm)
                         for n in range(7)])
    slots = slots.difference(taper.buffer(5 * mm))
    ground = board.difference(taper).difference(Point(-Rc / 2 + 1 * mm, 0).buffer(Rc / 2, 32)).difference(slots)
    f = 10e9
    g = rm.Geometry(maxh=lambda_maxh(f))
    g.label(g.box(120 * mm, 80 * mm, 30 * mm, position=(-40 * mm, -40 * mm, -15 * mm)), "air")
    g.label(g.box(100 * mm, 60 * mm, th, position=(-30 * mm, -30 * mm, -th), maxh=lambda_maxh(f, 4.4) / 2), "substrate")
    geoms = ground.geoms if ground.geom_type == "MultiPolygon" else [ground]
    for p in geoms:
        ext, holes = _rings(p)
        g.polygon_plate(ext, position=(0, 0, -th), holes=holes or None, tag=7)
    # Feed: 0.95 mm line up to the slot, ending in a radial stub.
    w0, stub, ang = 0.95 * mm, 7 * mm, math.radians(80)
    arc = [(2 * mm + stub * math.sin(a), 0.3 * mm + stub * math.cos(a))
           for a in [math.radians(20) - ang / 2 + ang * k / 16 for k in range(17)]]
    feed = unary_union([box(2 * mm - w0 / 2, -10 * mm, 2 * mm + w0 / 2, 0.5 * mm),
                        Polygon([(2 * mm, 0.3 * mm)] + arc)])
    g.polygon_plate(_rings(feed)[0], position=(0, 0, 0), tag=8)
    return g.mesh()


def inverted_f_antenna() -> rm.Mesh:
    """Meandered inverted-F antenna on a 1 mm board edge (demo21): the
    antenna trace and the ground plane as sheets, in an air region."""
    L1, L2, L3, L4, L5, L6 = 3.94 * mm, 2.47 * mm, 4.76 * mm, 2.64 * mm, 1.77 * mm, 4.90 * mm
    W1, W2, D1, D2, D3, D4, D5 = 0.90 * mm, 0.50 * mm, 0.50 * mm, 0.30 * mm, 0.30 * mm, 0.50 * mm, 0.65 * mm
    gw, gl = D1 + L3 + L5 + L2 + L5 + L2 + D3, 30 * mm
    top = L6 - D4
    parts = [
        (0, 0, W1, top), (0, top, L3, W2), (W1 + D5, 0.5 * mm, W2, top - 0.5 * mm),
        (L3 - W2, top - L4, W2, L4), (L3, top - L4, L5, W2), (L3 + L5, top - L4, W2, L4),
        (L3 + L5, top, L2, W2), (L3 + L5 + L2 - W2, top - L4, W2, L4), (L3 + L5 + L2, top - L4, L5, W2),
        (L3 + 2 * L5 + L2, top - L4, W2, L4), (L3 + 2 * L5 + L2, top, L2, W2),
        (L3 + 2 * L5 + 2 * L2 - W2, top - L1, W2, L1),
    ]
    rects = [(x, y, x + w, y + h) for x, y, w, h in parts]
    f = 3e9
    g = rm.Geometry(maxh=lambda_maxh(f, n=20))
    pad = 15 * mm
    g.label(g.box(gw + 2 * pad, gl + top + W2 + D2 + 2 * pad, 1 * mm + 2 * pad,
                  position=(-D1 - pad, -gl - pad, -1 * mm - pad)), "air")
    g.label(g.box(gw, gl + top + W2 + D2, 1 * mm, position=(-D1, -gl, -1 * mm), maxh=1.0 * mm), "substrate")
    for ext, holes in _union_outline(rects):
        g.polygon_plate(ext, position=(0, 0, 0), holes=holes or None, tag=7)
    g.polygon_plate([(-D1, -gl), (-D1 + gw, -gl), (-D1 + gw, 0), (-D1, 0)], position=(0, 0, 0), tag=8)
    g.label(7, "antenna")
    g.label(8, "ground")
    return g.mesh()


EXAMPLES = {
    "stepped_impedance_filter": stepped_impedance_filter,
    "combline_filter": combline_filter,
    "coupled_line_filter": coupled_line_filter,
    "stripline_vias": stripline_vias,
    "horn_antenna": horn_antenna,
    "helix_antenna": helix_antenna,
    "vivaldi_antenna": vivaldi_antenna,
    "inverted_f_antenna": inverted_f_antenna,
}


if __name__ == "__main__":
    import sys

    for name, build in EXAMPLES.items():
        if len(sys.argv) > 1 and name not in sys.argv[1:]:
            continue
        s = build().stats
        print(f"{name:26} {s['n_tets']:7} tets  min-dih {s['min_dihedral_deg']:5.1f}  {s['millis']:6} ms", flush=True)

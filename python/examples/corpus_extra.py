"""More corpus geometries: the EM structures a field solver meets, periodic
unit cells, and the robustness cases (tangent contacts, thin layers, extreme
aspect ratios and scales). Each entry is ``(name, category, kind, build)``
with ``build()`` returning the mesh; ``report/corpus.py`` registers them.

    python python/examples/corpus_extra.py [names...]
"""

from __future__ import annotations

import math

import rapidmesh as rm

mm = 1e-3
um = 1e-6


# ---- helpers -----------------------------------------------------------------


def rects_outline(rects, tag=1):
    """The union of axis-aligned rectangles (x0, y0, x1, y1) as sheet
    outlines: (outer, holes) per piece."""
    regs = [rm.Region2D([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], tag) for x0, y0, x1, y1 in rects]
    return rm.union_regions(regs)


def sheets(g, rects, z, tag, maxh=None):
    """The union of rectangles as sheets at height `z`."""
    for outer, holes in rects_outline(rects):
        g.polygon_plate(outer, position=(0, 0, z), holes=holes or None, tag=tag, maxh=maxh)


def board(g, w, l, h, er_maxh, air=None, name="substrate"):
    """A substrate `w x l x h` centred on the origin in x, y from z = 0, in
    an air box `air` larger on every side (none: no air)."""
    if air is not None:
        a = g.box(w + 2 * air, l + 2 * air, h + 2 * air, position=(-w / 2 - air, -l / 2 - air, -air))
        g.label(a, "air")
    s = g.box(w, l, h, position=(-w / 2, -l / 2, 0), maxh=er_maxh)
    g.label(s, name)
    return s


def square_spiral(turns, w, s, r0):
    """Rectangles of a square spiral: `turns` turns of width `w`, spacing
    `s`, starting at half-side `r0`."""
    rects, x, y, d = [], -r0, -r0, 2 * r0
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for k in range(4 * turns):
        dx, dy = dirs[k % 4]
        L = d + (k // 2) * (w + s)
        x1, y1 = x + dx * L, y + dy * L
        rects.append((min(x, x1) - w / 2, min(y, y1) - w / 2, max(x, x1) + w / 2, max(y, y1) + w / 2))
        x, y = x1, y1
    return rects


def circle(n, r, cx=0.0, cy=0.0, a0=0.0):
    return [(cx + r * math.cos(a0 + 2 * math.pi * k / n), cy + r * math.sin(a0 + 2 * math.pi * k / n)) for k in range(n)]


def arc(n, r, a0, a1, cx=0.0, cy=0.0):
    return [(cx + r * math.cos(a0 + (a1 - a0) * k / (n - 1)), cy + r * math.sin(a0 + (a1 - a0) * k / (n - 1))) for k in range(n)]


def periodic_xy(g):
    g.periodic(g.surf(normal=(-1, 0, 0)), g.surf(normal=(1, 0, 0)))
    g.periodic(g.surf(normal=(0, -1, 0)), g.surf(normal=(0, 1, 0)))


# ---- EM structures -------------------------------------------------------------


def cpw_line():
    """Coplanar waveguide: signal strip between two grounds on a substrate."""
    g = rm.Geometry(maxh=1.2 * mm)
    board(g, 12 * mm, 20 * mm, 0.8 * mm, 0.4 * mm, air=4 * mm)
    w, gap = 1.0 * mm, 0.3 * mm
    g.xy_plate(w, 20 * mm, position=(-w / 2, -10 * mm, 0.8 * mm), tag=1, maxh=0.2 * mm)
    for x0 in (-6 * mm, w / 2 + gap):
        g.xy_plate(6 * mm - w / 2 - gap, 20 * mm, position=(x0, -10 * mm, 0.8 * mm), tag=2)
    g.label(1, "signal")
    g.label(2, "ground")
    return g.mesh()


def microstrip_via_fence():
    """Microstrip with a row of grounding vias on each side."""
    g = rm.Geometry(maxh=1.5 * mm)
    board(g, 12 * mm, 24 * mm, 1.0 * mm, 0.6 * mm, air=5 * mm)
    g.xy_plate(1.8 * mm, 24 * mm, position=(-0.9 * mm, -12 * mm, 1.0 * mm), tag=1, maxh=0.3 * mm)
    g.xy_plate(12 * mm, 24 * mm, position=(-6 * mm, -12 * mm, 0.0), tag=2)
    for k in range(8):
        y = -10.5 * mm + 3 * mm * k
        for x in (-3 * mm, 3 * mm):
            g.label(g.cylinder(0.25 * mm, 1.0 * mm, position=(x, y, 0.0), segments=16, maxh=0.15 * mm), "via")
    return g.mesh()


def stripline():
    """Stripline: a trace between two dielectric layers, grounds outside."""
    g = rm.Geometry(maxh=1.0 * mm)
    g.label(g.box(10 * mm, 16 * mm, 0.8 * mm, position=(-5 * mm, -8 * mm, 0.0)), "core")
    g.label(g.box(10 * mm, 16 * mm, 0.8 * mm, position=(-5 * mm, -8 * mm, 0.8 * mm)), "prepreg")
    g.xy_plate(0.6 * mm, 16 * mm, position=(-0.3 * mm, -8 * mm, 0.8 * mm), tag=1, maxh=0.15 * mm)
    g.label(1, "trace")
    return g.mesh()


def differential_pair():
    """Two coupled microstrip lines."""
    g = rm.Geometry(maxh=1.2 * mm)
    board(g, 10 * mm, 20 * mm, 0.5 * mm, 0.3 * mm, air=3 * mm)
    for x in (-0.55 * mm, 0.15 * mm):
        g.xy_plate(0.4 * mm, 20 * mm, position=(x, -10 * mm, 0.5 * mm), tag=1, maxh=0.12 * mm)
    return g.mesh()


def branch_line_coupler():
    """Branch-line coupler: a square ring of lines with four arms."""
    g = rm.Geometry(maxh=2.0 * mm)
    board(g, 40 * mm, 40 * mm, 0.8 * mm, 1.0 * mm, air=6 * mm)
    a, w, wz = 8 * mm, 1.5 * mm, 2.5 * mm
    rects = [(-a, -a - wz / 2, a, -a + wz / 2), (-a, a - wz / 2, a, a + wz / 2),
             (-a - w / 2, -a, -a + w / 2, a), (a - w / 2, -a, a + w / 2, a)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            x0, x1 = sorted((sx * a, sx * 18 * mm))
            rects.append((x0, sy * a - w / 2, x1, sy * a + w / 2))
    sheets(g, rects, 0.8 * mm, 1, maxh=0.5 * mm)
    return g.mesh()


def hairpin_filter():
    """Three coupled hairpin resonators."""
    g = rm.Geometry(maxh=1.5 * mm)
    board(g, 30 * mm, 20 * mm, 0.6 * mm, 0.8 * mm, air=5 * mm)
    rects, w = [], 0.8 * mm
    for k in range(3):
        x = -9 * mm + 6.5 * mm * k
        rects += [(x, -6 * mm, x + w, 6 * mm), (x + 3 * mm, -6 * mm, x + 3 * mm + w, 6 * mm),
                  (x, 6 * mm - w, x + 3 * mm + w, 6 * mm)]
    sheets(g, rects, 0.6 * mm, 1, maxh=0.3 * mm)
    return g.mesh()


def interdigital_capacitor():
    """Interdigital capacitor on a substrate: two combs of fingers."""
    g = rm.Geometry(maxh=0.6 * mm)
    board(g, 6 * mm, 6 * mm, 0.5 * mm, 0.3 * mm, air=2 * mm)
    fw, gap, n = 0.15 * mm, 0.1 * mm, 12
    bottom = [(-2 * mm, -2 * mm, 2 * mm, -1.7 * mm)]
    top = [(-2 * mm, 1.7 * mm, 2 * mm, 2 * mm)]
    for k in range(n):
        x = -1.9 * mm + k * (fw + gap)
        if k % 2 == 0:
            bottom.append((x, -1.7 * mm, x + fw, 1.5 * mm))
        else:
            top.append((x, -1.5 * mm, x + fw, 1.7 * mm))
    sheets(g, bottom, 0.5 * mm, 1, maxh=0.06 * mm)
    sheets(g, top, 0.5 * mm, 2, maxh=0.06 * mm)
    return g.mesh()


def spiral_on_substrate():
    """Square spiral inductor on a substrate, volume mesh."""
    g = rm.Geometry(maxh=0.6 * mm)
    board(g, 8 * mm, 8 * mm, 0.4 * mm, 0.3 * mm, air=2 * mm)
    sheets(g, square_spiral(3, 0.2 * mm, 0.15 * mm, 0.6 * mm), 0.4 * mm, 1, maxh=0.08 * mm)
    return g.mesh()


def patch_array():
    """2 x 2 patch array with a corporate feed."""
    g = rm.Geometry(maxh=3.0 * mm)
    board(g, 80 * mm, 80 * mm, 1.6 * mm, 2.0 * mm, air=15 * mm)
    rects, P, a = [], 20 * mm, 14 * mm
    for sx in (-1, 1):
        for sy in (-1, 1):
            rects.append((sx * P - a / 2, sy * P - a / 2, sx * P + a / 2, sy * P + a / 2))
        rects.append((sx * P - 0.75 * mm, -P + a / 2, sx * P + 0.75 * mm, P - a / 2))
    rects.append((-P, -0.75 * mm, P, 0.75 * mm))
    rects.append((-0.75 * mm, -35 * mm, 0.75 * mm, 0.75 * mm))
    sheets(g, rects, 1.6 * mm, 1, maxh=1.0 * mm)
    g.xy_plate(80 * mm, 80 * mm, position=(-40 * mm, -40 * mm, 0.0), tag=2)
    return g.mesh()


def bowtie_antenna():
    """Bowtie dipole: two triangles meeting at a feed gap, in air."""
    g = rm.Geometry(maxh=4.0 * mm)
    g.label(g.box(80 * mm, 60 * mm, 40 * mm, position=(-40 * mm, -30 * mm, -20 * mm)), "air")
    for s in (-1, 1):
        g.polygon_plate([(s * 1 * mm, 0), (s * 25 * mm, -15 * mm), (s * 25 * mm, 15 * mm)], tag=1, maxh=1.0 * mm)
    return g.mesh()


def slot_antenna():
    """Slot in a ground plane, fed by a microstrip line underneath."""
    g = rm.Geometry(maxh=2.5 * mm)
    board(g, 50 * mm, 50 * mm, 1.6 * mm, 1.5 * mm, air=12 * mm)
    slot = [(-20 * mm, -1 * mm), (20 * mm, -1 * mm), (20 * mm, 1 * mm), (-20 * mm, 1 * mm)]
    g.polygon_plate([(-25 * mm, -25 * mm), (25 * mm, -25 * mm), (25 * mm, 25 * mm), (-25 * mm, 25 * mm)],
                    position=(0, 0, 1.6 * mm), holes=[slot[::-1]], tag=2, maxh=0.8 * mm)
    g.xy_plate(3 * mm, 30 * mm, position=(5 * mm, -25 * mm, 0.0), tag=1, maxh=0.8 * mm)
    return g.mesh()


def dipole_wire():
    """Half-wave wire dipole with a feed gap, in air."""
    g = rm.Geometry(maxh=6.0 * mm)
    g.label(g.box(60 * mm, 60 * mm, 140 * mm, position=(-30 * mm, -30 * mm, -70 * mm)), "air")
    for z0 in (-50 * mm, 1 * mm):
        g.label(g.cylinder(1.0 * mm, 49 * mm, position=(0, 0, z0), segments=16, maxh=1.0 * mm), "wire")
    return g.mesh()


def yagi_antenna():
    """Three element Yagi: reflector, driven element, director."""
    g = rm.Geometry(maxh=8.0 * mm)
    g.label(g.box(160 * mm, 100 * mm, 180 * mm, position=(-60 * mm, -50 * mm, -90 * mm)), "air")
    for x, L in ((-25 * mm, 150 * mm), (0.0, 140 * mm), (30 * mm, 125 * mm)):
        g.label(g.cylinder(1.5 * mm, L, position=(x, 0, -L / 2), segments=12, maxh=2.0 * mm), "element")
    return g.mesh()


def loop_antenna():
    """Small loop antenna: a torus wire in air."""
    g = rm.Geometry(maxh=4.0 * mm)
    g.label(g.box(80 * mm, 80 * mm, 50 * mm, position=(-40 * mm, -40 * mm, -25 * mm)), "air")
    g.label(g.torus(20 * mm, 1.5 * mm, maxh=1.0 * mm), "loop")
    return g.mesh()


def helical_antenna():
    """Axial-mode helix above a ground disc."""
    g = rm.Geometry(maxh=6.0 * mm)
    g.label(g.box(100 * mm, 100 * mm, 140 * mm, position=(-50 * mm, -50 * mm, -10 * mm)), "air")
    g.label(g.helix(12 * mm, 9 * mm, 6, 0.8 * mm, position=(0, 0, 5 * mm), maxh=1.0 * mm), "helix")
    g.disc(30 * mm, position=(0, 0, 0), tag=2, maxh=3 * mm)
    return g.mesh()


def monopole_on_ground():
    """Quarter-wave monopole on a finite ground plane."""
    g = rm.Geometry(maxh=5.0 * mm)
    g.label(g.box(100 * mm, 100 * mm, 80 * mm, position=(-50 * mm, -50 * mm, -20 * mm)), "air")
    g.xy_plate(60 * mm, 60 * mm, position=(-30 * mm, -30 * mm, 0), tag=2)
    g.label(g.cylinder(1.0 * mm, 30 * mm, position=(0, 0, 0.5 * mm), segments=16, maxh=0.8 * mm), "monopole")
    return g.mesh()


def pyramidal_horn():
    """Pyramidal horn: a lofted interior on a waveguide section, in air."""
    g = rm.Geometry(maxh=5.0 * mm)
    g.label(g.box(120 * mm, 100 * mm, 140 * mm, position=(-60 * mm, -50 * mm, -30 * mm)), "air")
    a, b = 22.86 * mm, 10.16 * mm
    wg = g.box(a, b, 20 * mm, position=(-a / 2, -b / 2, -20 * mm), maxh=3 * mm)
    A, B = 80 * mm, 60 * mm
    lo = [(-a / 2, -b / 2, 0), (a / 2, -b / 2, 0), (a / 2, b / 2, 0), (-a / 2, b / 2, 0)]
    hi = [(-A / 2, -B / 2, 70 * mm), (A / 2, -B / 2, 70 * mm), (A / 2, B / 2, 70 * mm), (-A / 2, B / 2, 70 * mm)]
    horn = g.loft(lo, hi, maxh=4 * mm)
    g.label(g.union(wg, horn), "horn")
    return g.mesh()


def waveguide_bend():
    """E-plane waveguide bend: an L-shaped prism."""
    a, b = 22.86 * mm, 10.16 * mm
    g = rm.Geometry(maxh=3.0 * mm)
    L = 40 * mm
    g.label(g.prism([(0, 0), (L + a, 0), (L + a, L + a), (L, L + a), (L, a), (0, a)], b), "waveguide")
    return g.mesh()


def waveguide_tee():
    """H-plane tee: three waveguide arms."""
    a, b = 22.86 * mm, 10.16 * mm
    g = rm.Geometry(maxh=3.0 * mm)
    main = g.box(100 * mm, a, b, position=(-50 * mm, -a / 2, 0))
    arm = g.box(a, 40 * mm, b, position=(-a / 2, a / 2 - 1e-9, 0))
    g.label(g.union(main, arm), "waveguide")
    return g.mesh()


def ridge_waveguide():
    """Double-ridged waveguide section: two ridges cut into the box."""
    g = rm.Geometry(maxh=2.0 * mm)
    g.label(g.box(30 * mm, 16 * mm, 40 * mm, position=(-15 * mm, -8 * mm, 0)), "waveguide")
    for y0 in (-8 * mm, 4 * mm):
        g.box(8 * mm, 4 * mm, 40 * mm, position=(-4 * mm, y0, 0), void=True)
    return g.mesh()


def coax_to_waveguide():
    """Coax probe feeding a rectangular waveguide."""
    a, b = 22.86 * mm, 10.16 * mm
    g = rm.Geometry(maxh=3.0 * mm)
    g.label(g.box(a, 60 * mm, b, position=(-a / 2, 0, 0)), "waveguide")
    g.label(g.cylinder(2.05 * mm, 10 * mm, position=(0, 15 * mm, -10 * mm), segments=24, maxh=1.0 * mm), "ptfe")
    g.label(g.cylinder(0.65 * mm, 16 * mm, position=(0, 15 * mm, -10 * mm), segments=16, maxh=0.5 * mm), "probe")
    return g.mesh()


def sma_connector():
    """SMA-like connector: flange, PTFE dielectric, centre pin."""
    g = rm.Geometry(maxh=1.0 * mm)
    g.label(g.box(12 * mm, 12 * mm, 2 * mm, position=(-6 * mm, -6 * mm, 0)), "flange")
    g.label(g.cylinder(2.05 * mm, 10 * mm, position=(0, 0, -8 * mm), segments=32, maxh=0.4 * mm), "ptfe")
    g.label(g.cylinder(0.64 * mm, 12 * mm, position=(0, 0, -8 * mm), segments=16, maxh=0.2 * mm), "pin")
    return g.mesh()


def cavity_with_post():
    """Rectangular cavity with a tuning post."""
    g = rm.Geometry(maxh=3.0 * mm)
    g.label(g.box(40 * mm, 30 * mm, 20 * mm), "cavity")
    g.cylinder(2 * mm, 12 * mm, position=(20 * mm, 15 * mm, 0), segments=24, void=True, maxh=1 * mm)
    return g.mesh()


def dielectric_rod_antenna():
    """Tapered dielectric rod: cylinder with a cone tip, in air."""
    g = rm.Geometry(maxh=4.0 * mm)
    g.label(g.box(50 * mm, 50 * mm, 140 * mm, position=(-25 * mm, -25 * mm, -20 * mm)), "air")
    rod = g.cylinder(6 * mm, 40 * mm, position=(0, 0, 0), maxh=1.5 * mm)
    tip = g.cone(6 * mm, 1 * mm, 60 * mm, position=(0, 0, 40 * mm), maxh=1.5 * mm)
    g.label(g.union(rod, tip), "rod")
    return g.mesh()


def luneburg_lens():
    """Luneburg lens: five concentric dielectric shells."""
    g = rm.Geometry(maxh=4.0 * mm)
    g.label(g.box(80 * mm, 80 * mm, 80 * mm, position=(-40 * mm, -40 * mm, -40 * mm)), "air")
    for k, r in enumerate((30, 24, 18, 12, 6)):
        g.label(g.sphere(r * mm, maxh=(1.5 + 0.2 * k) * mm), f"shell_{k}")
    return g.mesh()


def bga_package():
    """Package substrate on a board through a 4 x 4 array of solder balls."""
    g = rm.Geometry(maxh=0.5 * mm)
    g.label(g.box(8 * mm, 8 * mm, 0.8 * mm, position=(-4 * mm, -4 * mm, 0)), "pcb")
    g.label(g.box(6 * mm, 6 * mm, 0.4 * mm, position=(-3 * mm, -3 * mm, 1.2 * mm)), "package")
    for i in range(4):
        for j in range(4):
            c = (-2.25 * mm + 1.5 * i * mm, -2.25 * mm + 1.5 * j * mm, 1.0 * mm)
            g.label(g.sphere(0.3 * mm, position=c, maxh=0.12 * mm), "ball")
    return g.mesh()


def bondwire_package():
    """A die on a substrate, bond wires arching to the pads."""
    g = rm.Geometry(maxh=0.4 * mm)
    g.label(g.box(6 * mm, 6 * mm, 0.3 * mm, position=(-3 * mm, -3 * mm, 0)), "substrate")
    g.label(g.box(2 * mm, 2 * mm, 0.3 * mm, position=(-1 * mm, -1 * mm, 0.3 * mm)), "die")
    for k in range(4):
        y = -0.6 * mm + 0.4 * mm * k
        path = [(0.8 * mm + 1.7 * mm * t, y, 0.6 * mm + 0.5 * mm * math.sin(math.pi * t)
                 - 0.3 * mm * t) for t in (i / 12 for i in range(13))]
        g.label(g.sweep(path, 0.025 * mm, segments=8, maxh=0.03 * mm), "bondwire")
    return g.mesh()


def solenoid_pair():
    """Two coaxial solenoids around a ferrite rod."""
    g = rm.Geometry(maxh=2.0 * mm)
    g.label(g.box(30 * mm, 30 * mm, 60 * mm, position=(-15 * mm, -15 * mm, -5 * mm)), "air")
    g.label(g.cylinder(3 * mm, 50 * mm, maxh=0.8 * mm), "core")
    for z0, r in ((2 * mm, 5 * mm), (28 * mm, 5 * mm)):
        g.label(g.helix(r, 1.5 * mm, 12, 0.4 * mm, position=(0, 0, z0), maxh=0.3 * mm), "winding")
    return g.mesh()


def via_array():
    """A 6 x 6 via array through a substrate between two ground planes."""
    g = rm.Geometry(maxh=0.8 * mm)
    g.label(g.box(10 * mm, 10 * mm, 1.0 * mm, position=(-5 * mm, -5 * mm, 0)), "substrate")
    for i in range(6):
        for j in range(6):
            g.label(g.cylinder(0.2 * mm, 1.0 * mm, position=(-3.75 * mm + 1.5 * i * mm, -3.75 * mm + 1.5 * j * mm, 0),
                               segments=12, maxh=0.12 * mm), "via")
    return g.mesh()


def multilayer_pcb():
    """Four dielectric layers with a trace on every interface."""
    g = rm.Geometry(maxh=1.0 * mm)
    for k in range(4):
        g.label(g.box(12 * mm, 12 * mm, 0.3 * mm, position=(-6 * mm, -6 * mm, 0.3 * mm * k)), f"layer_{k}")
    for k in range(1, 4):
        a = math.pi / 4 * k
        L, w = 10 * mm, 0.3 * mm
        c, s = math.cos(a), math.sin(a)
        pts = [(-L / 2 * c + w / 2 * s, -L / 2 * s - w / 2 * c), (L / 2 * c + w / 2 * s, L / 2 * s - w / 2 * c),
               (L / 2 * c - w / 2 * s, L / 2 * s + w / 2 * c), (-L / 2 * c - w / 2 * s, -L / 2 * s + w / 2 * c)]
        g.polygon_plate(pts, position=(0, 0, 0.3 * mm * k), tag=k, maxh=0.1 * mm)
    return g.mesh()


def meander_line():
    """Meandered delay line on a substrate."""
    g = rm.Geometry(maxh=1.0 * mm)
    board(g, 16 * mm, 12 * mm, 0.5 * mm, 0.4 * mm, air=3 * mm)
    rects, w = [], 0.3 * mm
    for k in range(9):
        x = -6 * mm + 1.5 * mm * k
        rects.append((x, -4 * mm, x + w, 4 * mm))
        y = 4 * mm - w if k % 2 == 0 else -4 * mm
        if k < 8:
            rects.append((x, y, x + 1.5 * mm + w, y + w))
    sheets(g, rects, 0.5 * mm, 1, maxh=0.1 * mm)
    return g.mesh()


def radial_stub():
    """Microstrip line ending in a radial stub."""
    g = rm.Geometry(maxh=1.0 * mm)
    board(g, 16 * mm, 16 * mm, 0.5 * mm, 0.4 * mm, air=3 * mm)
    fan = [(0.0, 0.0)] + arc(16, 6 * mm, math.radians(60), math.radians(120))
    g.polygon_plate(fan, position=(0, 0, 0.5 * mm), tag=1, maxh=0.2 * mm)
    g.xy_plate(0.6 * mm, 8 * mm, position=(-0.3 * mm, -8 * mm, 0.5 * mm), tag=1, maxh=0.2 * mm)
    return g.mesh()


def radome():
    """Hemispherical radome shell over a ground plane."""
    g = rm.Geometry(maxh=4.0 * mm)
    g.label(g.box(100 * mm, 100 * mm, 60 * mm, position=(-50 * mm, -50 * mm, 0)), "air")
    g.label(g.sphere(35 * mm, maxh=1.5 * mm), "radome")
    g.label(g.sphere(32 * mm, maxh=1.5 * mm), "inside")
    g.box(100 * mm, 100 * mm, 40 * mm, position=(-50 * mm, -50 * mm, -40 * mm), void=True)
    return g.mesh()


EM = [
    cpw_line, microstrip_via_fence, stripline, differential_pair, branch_line_coupler, hairpin_filter,
    interdigital_capacitor, spiral_on_substrate, patch_array, bowtie_antenna, slot_antenna, dipole_wire,
    yagi_antenna, loop_antenna, helical_antenna, monopole_on_ground, pyramidal_horn, waveguide_bend,
    waveguide_tee, ridge_waveguide, coax_to_waveguide, sma_connector, cavity_with_post,
    dielectric_rod_antenna, luneburg_lens, bga_package, bondwire_package, solenoid_pair, via_array,
    multilayer_pcb, meander_line, radial_stub, radome,
]


# ---- periodic unit cells ---------------------------------------------------------


def _cell(a, sub_h, air_h, er_maxh, h):
    """A unit cell: air above and below a substrate, `a` wide."""
    g = rm.Geometry(maxh=h)
    g.label(g.box(a, a, 2 * air_h + sub_h), "air")
    g.label(g.box(a, a, sub_h, position=(0, 0, air_h), maxh=er_maxh), "substrate")
    return g


def fss_square_loop():
    a, t = 10 * mm, 1.6 * mm
    g = _cell(a, t, 8 * mm, 1.0 * mm, 1.8 * mm)
    outer = [(1 * mm, 1 * mm), (9 * mm, 1 * mm), (9 * mm, 9 * mm), (1 * mm, 9 * mm)]
    inner = [(2 * mm, 2 * mm), (2 * mm, 8 * mm), (8 * mm, 8 * mm), (8 * mm, 2 * mm)]
    g.polygon_plate(outer, position=(0, 0, 8 * mm + t), holes=[inner], tag=1, maxh=0.4 * mm)
    periodic_xy(g)
    return g.mesh()


def fss_jerusalem_cross():
    a, t = 12 * mm, 1.0 * mm
    g = _cell(a, t, 8 * mm, 0.8 * mm, 1.8 * mm)
    c, w, L, cap = a / 2, 0.6 * mm, 4.5 * mm, 3 * mm
    rects = [(c - w / 2, c - L, c + w / 2, c + L), (c - L, c - w / 2, c + L, c + w / 2)]
    for s in (-1, 1):
        rects.append((c - cap, c + s * L - w / 2, c + cap, c + s * L + w / 2))
        rects.append((c + s * L - w / 2, c - cap, c + s * L + w / 2, c + cap))
    sheets(g, rects, 8 * mm + t, 1, maxh=0.25 * mm)
    periodic_xy(g)
    return g.mesh()


def split_ring_resonator():
    """Two concentric split rings on a substrate (a metamaterial cell)."""
    a, t = 5 * mm, 0.5 * mm
    g = _cell(a, t, 4 * mm, 0.5 * mm, 0.9 * mm)
    for r, gap_side in ((2.0 * mm, 0.0), (1.4 * mm, math.pi)):
        w, gap = 0.25 * mm, math.radians(20)
        a0, a1 = gap_side + gap / 2, gap_side + 2 * math.pi - gap / 2
        pts = arc(40, r, a0, a1, a / 2, a / 2) + arc(40, r - w, a1, a0, a / 2, a / 2)
        g.polygon_plate(pts, position=(0, 0, 4 * mm + t), tag=1, maxh=0.1 * mm)
    periodic_xy(g)
    return g.mesh()


def fss_dipole_array():
    a, t = 8 * mm, 0.8 * mm
    g = _cell(a, t, 6 * mm, 0.8 * mm, 1.5 * mm)
    g.xy_plate(6 * mm, 0.8 * mm, position=(1 * mm, 3.6 * mm, 6 * mm + t), tag=1, maxh=0.3 * mm)
    periodic_xy(g)
    return g.mesh()


def fss_ring():
    a, t = 10 * mm, 1.0 * mm
    g = _cell(a, t, 8 * mm, 1.0 * mm, 1.8 * mm)
    g.polygon_plate(circle(48, 4 * mm, a / 2, a / 2), position=(0, 0, 8 * mm + t),
                    holes=[circle(48, 3.2 * mm, a / 2, a / 2)[::-1]], tag=1, maxh=0.3 * mm)
    periodic_xy(g)
    return g.mesh()


def reflectarray_cell():
    """Reflectarray element: patch over a ground plane, one side open."""
    a, t = 15 * mm, 1.5 * mm
    g = rm.Geometry(maxh=2.0 * mm)
    g.label(g.box(a, a, 20 * mm), "air")
    g.label(g.box(a, a, t, maxh=1.2 * mm), "substrate")
    g.xy_plate(9 * mm, 9 * mm, position=(3 * mm, 3 * mm, t), tag=1, maxh=0.6 * mm)
    g.xy_plate(a, a, position=(0, 0, 0), tag=2)
    periodic_xy(g)
    return g.mesh()


def wire_medium_cell():
    """Wire medium: a metal rod through the cell, periodic in x and y."""
    a = 6 * mm
    g = rm.Geometry(maxh=1.0 * mm)
    g.label(g.box(a, a, 10 * mm), "air")
    g.label(g.cylinder(0.5 * mm, 10 * mm, position=(a / 2, a / 2, 0), maxh=0.3 * mm), "rod")
    periodic_xy(g)
    return g.mesh()


def sphere_lattice_cell():
    """A dielectric sphere in a cube, periodic in x, y and z."""
    g = rm.Geometry(maxh=0.8 * mm)
    g.label(g.box(8 * mm, 8 * mm, 8 * mm), "host")
    g.label(g.sphere(2.5 * mm, position=(4 * mm, 4 * mm, 4 * mm), maxh=0.5 * mm), "sphere")
    periodic_xy(g)
    g.periodic(g.surf(normal=(0, 0, -1)), g.surf(normal=(0, 0, 1)))
    return g.mesh()


def double_layer_fss():
    """Two crossed dipole layers in a two-substrate stack."""
    a = 10 * mm
    g = rm.Geometry(maxh=1.8 * mm)
    g.label(g.box(a, a, 20 * mm), "air")
    g.label(g.box(a, a, 1 * mm, position=(0, 0, 8 * mm), maxh=0.8 * mm), "layer_a")
    g.label(g.box(a, a, 1 * mm, position=(0, 0, 10 * mm), maxh=0.8 * mm), "layer_b")
    g.xy_plate(8 * mm, 1 * mm, position=(1 * mm, 4.5 * mm, 9 * mm), tag=1, maxh=0.3 * mm)
    g.xy_plate(1 * mm, 8 * mm, position=(4.5 * mm, 1 * mm, 11 * mm), tag=2, maxh=0.3 * mm)
    periodic_xy(g)
    return g.mesh()


PERIODIC = [
    fss_square_loop, fss_jerusalem_cross, split_ring_resonator, fss_dipole_array, fss_ring,
    reflectarray_cell, wire_medium_cell, sphere_lattice_cell, double_layer_fss,
]


# ---- robustness ------------------------------------------------------------------


def tangent_spheres():
    """Two spheres touching in one point, in a box."""
    g = rm.Geometry(maxh=0.25)
    g.box(4, 3, 3, position=(-2, -1.5, -1.5))
    g.sphere(0.8, position=(-0.8, 0, 0), maxh=0.12)
    g.sphere(0.8, position=(0.8, 0, 0), maxh=0.12)
    return g.mesh()


def sphere_touching_box():
    """A sphere resting on the face of a box."""
    g = rm.Geometry(maxh=0.25)
    g.box(3, 3, 1, position=(-1.5, -1.5, -1))
    g.sphere(0.7, position=(0, 0, 0.7), maxh=0.12)
    return g.mesh()


def touching_cylinders():
    """Two parallel cylinders touching along a line."""
    g = rm.Geometry(maxh=0.25)
    g.box(3, 3, 2, position=(-1.5, -1.5, 0))
    g.cylinder(0.5, 2, position=(-0.5, 0, 0), maxh=0.12)
    g.cylinder(0.5, 2, position=(0.5, 0, 0), maxh=0.12)
    return g.mesh()


def thin_layer():
    """A layer a hundredth of the box thick."""
    g = rm.Geometry(maxh=0.3)
    g.box(3, 3, 2)
    g.box(3, 3, 0.02, position=(0, 0, 1))
    return g.mesh()


def needle():
    """A cylinder a hundred times longer than wide."""
    g = rm.Geometry(maxh=0.5)
    g.box(4, 4, 12, position=(-2, -2, -1))
    g.cylinder(0.05, 10, segments=12, maxh=0.03)
    return g.mesh()


def hole_grid_plate():
    """A plate with a 10 x 10 grid of holes."""
    g = rm.Geometry(maxh=0.2)
    g.box(5, 5, 0.3)
    for i in range(10):
        for j in range(10):
            g.cylinder(0.12, 0.3, position=(0.25 + 0.5 * i, 0.25 + 0.5 * j, 0), segments=12, void=True, maxh=0.06)
    return g.mesh()


def shared_partial_face():
    """Two boxes sharing part of a face exactly."""
    g = rm.Geometry(maxh=0.2)
    g.box(2, 2, 1)
    g.box(1, 1, 1, position=(0.5, 0.5, 1))
    return g.mesh()


def sheet_t_junction():
    """Three sheets meeting along one line."""
    g = rm.Geometry(maxh=0.25)
    g.box(3, 3, 3, position=(-1.5, -1.5, -1.5))
    g.xy_plate(2, 2, position=(-1, -1, 0), tag=1)
    g.xz_plate(2, 1, position=(-1, 0, 0), tag=2)
    g.xz_plate(2, 1, position=(-1, 0, -1), tag=3)
    return g.mesh()


def crossing_sheets():
    """Two perpendicular sheets crossing each other."""
    g = rm.Geometry(maxh=0.25)
    g.box(3, 3, 3, position=(-1.5, -1.5, -1.5))
    g.xy_plate(2, 2, position=(-1, -1, 0), tag=1)
    g.xz_plate(2, 2, position=(-1, 0, -1), tag=2)
    return g.mesh()


def disc_through_sphere():
    """A sheet cutting through a sphere."""
    g = rm.Geometry(maxh=0.2)
    g.box(3, 3, 3, position=(-1.5, -1.5, -1.5))
    g.sphere(0.8, maxh=0.12)
    g.disc(1.2, position=(0, 0, 0.2), tag=1, maxh=0.1)
    return g.mesh()


def small_feature():
    """A box a hundred times smaller than its neighbour, at its corner."""
    g = rm.Geometry(maxh=0.5)
    g.box(4, 4, 4)
    g.box(0.04, 0.04, 0.04, position=(1, 1, 1))
    return g.mesh()


def near_coincident_faces():
    """Two boxes whose faces are a millionth of the size apart."""
    g = rm.Geometry(maxh=0.25)
    g.box(2, 2, 1)
    g.box(2, 2, 1, position=(0, 0, 1 + 1e-6))
    return g.mesh()


def patch_um():
    """A patch antenna at micrometre scale (scale invariance)."""
    g = rm.Geometry(maxh=40 * um)
    board(g, 400 * um, 400 * um, 20 * um, 20 * um, air=100 * um)
    g.xy_plate(200 * um, 150 * um, position=(-100 * um, -75 * um, 20 * um), tag=1, maxh=10 * um)
    return g.mesh()


def patch_km():
    """The same patch antenna at kilometre scale."""
    s = 1e3 / um * 1e-6
    g = rm.Geometry(maxh=40 * s)
    board(g, 400 * s, 400 * s, 20 * s, 20 * s, air=100 * s)
    g.xy_plate(200 * s, 150 * s, position=(-100 * s, -75 * s, 20 * s), tag=1, maxh=10 * s)
    return g.mesh()


def many_regions():
    """A 5 x 5 x 2 block of cells, each its own region."""
    g = rm.Geometry(maxh=0.25)
    for i in range(5):
        for j in range(5):
            for k in range(2):
                g.box(0.5, 0.5, 0.5, position=(0.5 * i, 0.5 * j, 0.5 * k))
    return g.mesh()


def nested_cylinders():
    """Ten coaxial cylinder layers."""
    g = rm.Geometry(maxh=0.2)
    for k in range(10):
        g.cylinder(1.0 - 0.09 * k, 2.0, maxh=0.08)
    return g.mesh()


def sharp_cone():
    """A cone with a 10 degree tip in a box."""
    g = rm.Geometry(maxh=0.25)
    g.box(3, 3, 4, position=(-1.5, -1.5, -0.5))
    g.cone(0.3, 0.0, 3.0, maxh=0.08)
    return g.mesh()


def sharp_wedge():
    """A wedge with a 6 degree edge."""
    g = rm.Geometry(maxh=0.25)
    g.box(3, 3, 3, position=(-0.5, -0.5, -0.5))
    g.wedge(2, 2, 0.2, top_x=0.0)
    return g.mesh()


def tight_helix():
    """A helix whose turns nearly touch."""
    g = rm.Geometry(maxh=0.3)
    g.box(4, 4, 5, position=(-2, -2, -0.5))
    g.helix(1.0, 0.25, 12, 0.1, maxh=0.05)
    return g.mesh()


def thin_torus():
    """A torus fifty times wider than thick."""
    g = rm.Geometry(maxh=0.3)
    g.box(3, 3, 1, position=(-1.5, -1.5, -0.5))
    g.torus(1.0, 0.02, maxh=0.02)
    return g.mesh()


def rotated_box():
    """A box turned 30 degrees about two axes, given as triangles."""
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    corners = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]

    def rot(p):
        x, y, z = p
        x, y = c * x - s * y, s * x + c * y
        y, z = c * y - s * z, s * y + c * z
        return (x, y, z)

    verts = [rot(p) for p in corners]
    tris = [(0, 2, 1), (1, 2, 3), (4, 5, 6), (5, 7, 6), (0, 1, 4), (1, 5, 4),
            (2, 6, 3), (3, 6, 7), (0, 4, 2), (2, 4, 6), (1, 3, 5), (3, 7, 5)]
    g = rm.Geometry(maxh=0.12)
    g.box(3, 3, 3, position=(-1.5, -1.0, -0.5))
    g.mesh_solid(verts, tris)
    return g.mesh()


def star_in_box():
    """A concave star prism in a box."""
    pts = []
    for k in range(10):
        r = 1.0 if k % 2 == 0 else 0.4
        a = math.pi / 2 + math.pi * k / 5
        pts.append((r * math.cos(a), r * math.sin(a)))
    g = rm.Geometry(maxh=0.15)
    g.box(3, 3, 1.5, position=(-1.5, -1.5, -0.5))
    g.prism(pts, 0.5)
    return g.mesh()


def island_in_hole():
    """A prism with a hole, and an island prism inside the hole."""
    g = rm.Geometry(maxh=0.15)
    g.box(3, 3, 1.5, position=(-1.5, -1.5, -0.5))
    g.prism(circle(32, 1.0), 0.5, holes=[circle(32, 0.6)[::-1]])
    g.prism(circle(24, 0.3), 0.5)
    return g.mesh()


ROBUST = [
    tangent_spheres, sphere_touching_box, touching_cylinders, thin_layer, needle, hole_grid_plate,
    shared_partial_face, sheet_t_junction, crossing_sheets, disc_through_sphere, small_feature,
    near_coincident_faces, patch_um, patch_km, many_regions, nested_cylinders, sharp_cone, sharp_wedge,
    tight_helix, thin_torus, rotated_box, star_in_box, island_in_hole,
]

ENTRIES = (
    [(f.__name__, "EM", "vol", f) for f in EM]
    + [(f.__name__, "Periodic", "vol", f) for f in PERIODIC]
    + [(f.__name__, "Robust", "vol", f) for f in ROBUST]
)


if __name__ == "__main__":
    import sys
    import time

    for name, cat, kind, build in ENTRIES:
        if len(sys.argv) > 1 and name not in sys.argv[1:]:
            continue
        t0 = time.perf_counter()
        try:
            m = build()
            d = m.diagnostics
            print(f"{cat:9} {name:26} {len(m.tets):7} tets  min {m.stats['min_dihedral_deg']:5.1f}  "
                  f"wt {d['watertight']!s:5}  defects {len(d['defects']):4}  {1e3 * (time.perf_counter() - t0):7.0f} ms",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{cat:9} {name:26} FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
